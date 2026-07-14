from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path
import shutil
import re
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import cm
from PIL import Image
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf, open_dict

from rlinf.config import validate_cfg
from rlinf.envs import get_env_cls
from rlinf.envs.action_utils import prepare_actions
from rlinf.models import get_model

plt.rcParams["font.family"] = "Libre Baskerville"
plt.rcParams["font.weight"] = "medium"


def _as_2d(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3:
        return x[0]
    if x.ndim != 2:
        raise ValueError(f"Expected 2D tensor, got shape={tuple(x.shape)}")
    return x


def _pca_2d(features: torch.Tensor) -> torch.Tensor:
    feats = features.float()
    feats = feats - feats.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(feats, full_matrices=False)
    return feats @ vh[:2].T


def _pca_shared_proj(token_list: list[torch.Tensor], n_components: int = 2) -> list[torch.Tensor]:
    """
    Fit PCA (via SVD) on the concatenation of `token_list` and project each token set
    onto the shared top-n_components basis. Returns a list of projected tensors
    corresponding to inputs in token_list.
    """
    toks = [t.float() for t in token_list]
    concat = torch.cat(toks, dim=0)
    mean = concat.mean(dim=0, keepdim=True)
    centered = concat - mean
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    basis = vh[:n_components].T
    projs = []
    start = 0
    for t in toks:
        L = t.shape[0]
        centered_t = (t - mean)
        projs.append(centered_t @ basis)
        start += L
    return projs


def _pairwise_geometry_metrics(
    features: torch.Tensor,
    coords: torch.Tensor,
    k: int = 5,
    normalize_features: bool = False,
) -> dict[str, float]:
    feats = features.float()
    if normalize_features:
        feats = F.normalize(feats, dim=-1)
    coords = coords.float()
    feat_dist = torch.cdist(feats, feats)
    coord_dist = torch.cdist(coords, coords)

    tri = torch.triu(torch.ones_like(feat_dist, dtype=torch.bool), diagonal=1)
    feat_vec = feat_dist[tri].cpu().numpy()
    coord_vec = coord_dist[tri].cpu().numpy()
    corr = float(np.corrcoef(feat_vec, coord_vec)[0, 1])

    k = max(1, min(k, feats.shape[0] - 1))
    feat_knn = torch.topk(feat_dist, k + 1, largest=False).indices[:, 1:]
    coord_knn = torch.topk(coord_dist, k + 1, largest=False).indices[:, 1:]
    recall = []
    for i in range(feats.shape[0]):
        recall.append(len(set(feat_knn[i].tolist()) & set(coord_knn[i].tolist())) / k)

    return {
        "pairwise_distance_corr": corr,
        f"knn_recall@{k}": float(np.mean(recall)),
    }


def _get_payload_tokens(payload: dict[str, Any], key: str) -> torch.Tensor:
    if key in payload and payload[key] is not None:
        return _as_2d(payload[key])
    if key == "fused_tokens" and payload.get("baseline_tokens") is not None:
        return _as_2d(payload["baseline_tokens"])
    if key == "spatial_tokens" and payload.get("fused_tokens") is not None:
        return _as_2d(payload["fused_tokens"])
    raise KeyError(f"Could not resolve token key '{key}' from payload keys: {sorted(payload.keys())}")


def _plot_pca_grid(payload: dict[str, Any], output_path: Path) -> None:
    names = [
        ("baseline_tokens", "Original visual tokens"),
        ("spatial_tokens", "VGGT spatial tokens"),
        ("fused_tokens", "After spatial fusion"),
    ]
    coords = _as_2d(payload["coords"])
    coord_views = [coords[:, 0], coords[:, 1]]
    coord_titles = ["color = x coordinate", "color = y coordinate"]

    fig, axes = plt.subplots(len(names), 2, figsize=(10, 11), constrained_layout=True)
    for row, (key, title) in enumerate(names):
        tokens = _as_2d(payload[key])
        proj = _pca_2d(tokens).cpu().numpy()
        for col, (color_vals, ctitle) in enumerate(zip(coord_views, coord_titles, strict=True)):
            ax = axes[row, col]
            sc = ax.scatter(
                proj[:, 0],
                proj[:, 1],
                c=color_vals.cpu().numpy(),
                cmap="viridis",
                s=18,
                alpha=0.95,
                linewidths=0.0,
            )
            ax.set_title(f"{title}\n{ctitle}")
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)

            fig.savefig(output_path, dpi=400, bbox_inches="tight")
            plt.close(fig)


def _robust_unit_interval(x: torch.Tensor, lower_q: float = 0.05, upper_q: float = 0.95) -> torch.Tensor:
    """Map a tensor to [0, 1] using robust per-tensor quantile bounds.

    This keeps the relative spatial pattern while reducing the effect of
    absolute scale differences across runs.
    """
    x = x.float()
    lo = torch.quantile(x.reshape(-1), lower_q)
    hi = torch.quantile(x.reshape(-1), upper_q)
    if float(hi.item() - lo.item()) < 1e-8:
        return torch.zeros_like(x)
    return ((x - lo) / (hi - lo)).clamp(0.0, 1.0)


def _plot_scan_heatmap(payload: dict[str, Any], output_path: Path) -> dict[str, float]:
    std = _as_2d(payload["action_noise_std"]).float()
    mean_std = float(std.mean().item())
    cv = float(std.std().item() / (mean_std + 1e-8))
    iso = torch.full_like(std, mean_std)

    vmin = float(min(std.min().item(), iso.min().item()))
    vmax = float(max(std.max().item(), iso.max().item()))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    im0 = axes[0].imshow(std.cpu().numpy(), cmap="magma", aspect="auto", vmin=vmin, vmax=vmax)
    axes[0].set_title("SCAN learned std")
    axes[0].set_xlabel("Action dim")
    axes[0].set_ylabel("Denoise token")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.02)

    im1 = axes[1].imshow(iso.cpu().numpy(), cmap="magma", aspect="auto", vmin=vmin, vmax=vmax)
    axes[1].set_title("Isotropic baseline (same mean std)")
    axes[1].set_xlabel("Action dim")
    axes[1].set_ylabel("Denoise token")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.02)

    fig.savefig(output_path, dpi=400, bbox_inches="tight")
    plt.close(fig)

    return {"scan_std_mean": mean_std, "scan_std_cv": cv}


def _plot_compare_scan_noise(
    left_payload: dict[str, Any],
    right_payload: dict[str, Any],
    raw_output_path: Path,
    normalized_output_path: Path,
    residual_output_path: Path,
    distribution_output_path: Path | None,
    left_label: str,
    right_label: str,
) -> dict[str, float]:
    left_std = _as_2d(left_payload["action_noise_std"]).float()
    right_std = _as_2d(right_payload["action_noise_std"]).float()
    if left_std.shape != right_std.shape:
        raise ValueError(
            f"action_noise_std shape mismatch: left={tuple(left_std.shape)} right={tuple(right_std.shape)}"
        )

    left_np = left_std.cpu().numpy()
    right_np = right_std.cpu().numpy()
    delta_np = right_np - left_np

    left_norm = _robust_unit_interval(left_std)
    right_norm = _robust_unit_interval(right_std)
    left_norm_np = left_norm.cpu().numpy()
    right_norm_np = right_norm.cpu().numpy()
    delta_norm_np = right_norm_np - left_norm_np
    left_flat = left_norm_np.reshape(-1)
    right_flat = right_norm_np.reshape(-1)

    # Residuals are also computed in the normalized domain so that all
    # non-raw panels share the same scale and interpretation.
    left_resid = (left_norm - left_norm.mean())
    right_resid = (right_norm - right_norm.mean())
    left_resid_np = left_resid.cpu().numpy()
    right_resid_np = right_resid.cpu().numpy()
    delta_resid_np = right_resid_np - left_resid_np

    left_mean = float(left_std.mean().item())
    right_mean = float(right_std.mean().item())
    left_cv = float(left_std.std().item() / (left_mean + 1e-8))
    right_cv = float(right_std.std().item() / (right_mean + 1e-8))

    raw_lim = max(
        abs(float(left_np.min())),
        abs(float(left_np.max())),
        abs(float(right_np.min())),
        abs(float(right_np.max())),
        abs(float(delta_np.min())),
        abs(float(delta_np.max())),
    )
    norm_lim = max(abs(float(delta_norm_np.min())), abs(float(delta_norm_np.max())))
    resid_lim = max(abs(float(left_resid_np.min())), abs(float(left_resid_np.max())), abs(float(right_resid_np.min())), abs(float(right_resid_np.max())), abs(float(delta_resid_np.min())), abs(float(delta_resid_np.max())))

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), constrained_layout=True)
    panels = [
        (left_np, left_label, "magma", -raw_lim, raw_lim),
        (right_np, right_label, "magma", -raw_lim, raw_lim),
        (delta_np, f"{right_label} minus {left_label}", "bwr", -raw_lim, raw_lim),
    ]
    for ax, (panel, title, cmap_name, lo, hi) in zip(axes, panels, strict=True):
        im = ax.imshow(panel, cmap=cmap_name, aspect="auto", vmin=lo, vmax=hi)
        ax.set_title(title)
        ax.set_xlabel("Action dim")
        ax.set_ylabel("Denoise token")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.savefig(raw_output_path, dpi=400, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), constrained_layout=True)
    panels = [
        (left_norm_np, f"{left_label} (robust norm)", "magma", 0.0, 1.0),
        (right_norm_np, f"{right_label} (robust norm)", "magma", 0.0, 1.0),
        (delta_norm_np, f"{right_label} minus {left_label} (normed)", "bwr", -norm_lim, norm_lim),
    ]
    for ax, (panel, title, cmap_name, lo, hi) in zip(axes, panels, strict=True):
        im = ax.imshow(panel, cmap=cmap_name, aspect="auto", vmin=lo, vmax=hi)
        ax.set_title(title)
        ax.set_xlabel("Action dim")
        ax.set_ylabel("Denoise token")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.savefig(normalized_output_path, dpi=400, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), constrained_layout=True)
    panels = [
        (left_resid_np, f"{left_label} residual", "coolwarm", -resid_lim, resid_lim),
        (right_resid_np, f"{right_label} residual", "coolwarm", -resid_lim, resid_lim),
        (delta_resid_np, f"{right_label} minus {left_label} residual", "bwr", -resid_lim, resid_lim),
    ]
    for ax, (panel, title, cmap_name, lo, hi) in zip(axes, panels, strict=True):
        im = ax.imshow(panel, cmap=cmap_name, aspect="auto", vmin=lo, vmax=hi)
        ax.set_title(title)
        ax.set_xlabel("Action dim")
        ax.set_ylabel("Denoise token")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.savefig(residual_output_path, dpi=400, bbox_inches="tight")
    plt.close(fig)

    if distribution_output_path is not None:
        bins = np.linspace(0.0, 1.0, 31)
        left_hist, _ = np.histogram(left_flat, bins=bins, density=True)
        right_hist, _ = np.histogram(right_flat, bins=bins, density=True)
        left_mass = left_hist / (left_hist.sum() + 1e-8)
        right_mass = right_hist / (right_hist.sum() + 1e-8)
        mix = 0.5 * (left_mass + right_mass)
        eps = 1e-12
        js_div = 0.5 * (
            np.sum(left_mass * np.log((left_mass + eps) / (mix + eps)))
            + np.sum(right_mass * np.log((right_mass + eps) / (mix + eps)))
        )
        l1_hist = float(np.abs(left_mass - right_mass).sum())

        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
        axes[0].hist(
            left_flat,
            bins=bins,
            density=True,
            alpha=0.55,
            color="C3",
            label=left_label,
        )
        axes[0].hist(
            right_flat,
            bins=bins,
            density=True,
            alpha=0.55,
            color="C0",
            label=right_label,
        )
        axes[0].set_title("Normalized std distribution")
        axes[0].set_xlabel("Normalized std")
        axes[0].set_ylabel("Density")
        axes[0].grid(alpha=0.25)
        axes[0].legend(frameon=False)

        stats_text = (
            f"JS={js_div:.4f}\n"
            f"L1={l1_hist:.4f}\n"
            f"median: {np.median(left_flat):.3f} / {np.median(right_flat):.3f}\n"
            f"p90: {np.percentile(left_flat, 90):.3f} / {np.percentile(right_flat, 90):.3f}"
        )
        axes[1].axis("off")
        axes[1].text(
            0.02,
            0.98,
            stats_text,
            va="top",
            ha="left",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, edgecolor="0.8"),
        )
        axes[1].set_title("Distribution summary")

        fig.savefig(distribution_output_path, dpi=400, bbox_inches="tight")
        plt.close(fig)

    return {
        "left_scan_std_mean": left_mean,
        "left_scan_std_cv": left_cv,
        "right_scan_std_mean": right_mean,
        "right_scan_std_cv": right_cv,
        "left_norm_mean": float(left_norm.mean().item()),
        "right_norm_mean": float(right_norm.mean().item()),
        "left_norm_q05": float(np.percentile(left_flat, 5)),
        "left_norm_q25": float(np.percentile(left_flat, 25)),
        "left_norm_median": float(np.median(left_flat)),
        "left_norm_q75": float(np.percentile(left_flat, 75)),
        "left_norm_q95": float(np.percentile(left_flat, 95)),
        "right_norm_q05": float(np.percentile(right_flat, 5)),
        "right_norm_q25": float(np.percentile(right_flat, 25)),
        "right_norm_median": float(np.median(right_flat)),
        "right_norm_q75": float(np.percentile(right_flat, 75)),
        "right_norm_q95": float(np.percentile(right_flat, 95)),
        "delta_scan_std_mean": float(delta_np.mean()),
        "delta_scan_std_abs_mean": float(np.abs(delta_np).mean()),
        "delta_norm_mean": float(delta_norm_np.mean()),
        "delta_norm_abs_mean": float(np.abs(delta_norm_np).mean()),
        "delta_residual_mean": float(delta_resid_np.mean()),
        "delta_residual_abs_mean": float(np.abs(delta_resid_np).mean()),
        "residual_domain": "normalized",
        "hist_js_divergence": float(js_div) if distribution_output_path is not None else float("nan"),
        "hist_l1_distance": float(l1_hist) if distribution_output_path is not None else float("nan"),
    }


def _plot_scan_hist(payload: dict[str, Any], output_path: Path) -> None:
    std = _as_2d(payload["action_noise_std"]).float().reshape(-1).cpu().numpy()
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.hist(std, bins=40, color="C2", alpha=0.9)
    ax.set_title("SCAN std distribution")
    ax.set_xlabel("Std value")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.3)
    fig.savefig(output_path, dpi=400, bbox_inches="tight")
    plt.close(fig)

    return {"pca_grid_status": "ok"}


def _plot_scan_hist_normalized(payload: dict[str, Any], output_path: Path) -> None:
    std = _robust_unit_interval(_as_2d(payload["action_noise_std"]).float()).reshape(-1).cpu().numpy()
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.hist(std, bins=40, color="C2", alpha=0.9, range=(0.0, 1.0))
    ax.set_title("Normalized SCAN std distribution")
    ax.set_xlabel("Normalized std")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.3)
    fig.savefig(output_path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def _plot_token_qualitative(payload: dict[str, Any], output_path: Path) -> dict[str, float]:
    grid_h, grid_w = map(int, payload["grid_hw"])
    names = [
        ("baseline_tokens", "Original visual tokens"),
        ("spatial_tokens", "VGGT spatial tokens"),
        ("fused_tokens", "After spatial fusion"),
    ]

    norm_maps = []
    pca_maps = []
    for key, _title in names:
        tok = _as_2d(payload[key]).float()
        if tok.shape[0] != grid_h * grid_w:
            raise ValueError(
                f"Token count {tok.shape[0]} does not match grid size {grid_h}x{grid_w}."
            )
        norm_maps.append(tok.norm(dim=-1).view(grid_h, grid_w))
        pca_maps.append(_pca_2d(tok)[:, 0].view(grid_h, grid_w))

    norm_min = min(float(m.min().item()) for m in norm_maps)
    norm_max = max(float(m.max().item()) for m in norm_maps)
    pca_min = min(float(m.min().item()) for m in pca_maps)
    pca_max = max(float(m.max().item()) for m in pca_maps)

    fig, axes = plt.subplots(len(names), 2, figsize=(9.5, 10.5), constrained_layout=True)
    for row, ((_, title), norm_map, pca_map) in enumerate(zip(names, norm_maps, pca_maps, strict=True)):
        ax_norm = axes[row, 0]
        ax_pca = axes[row, 1]

        im0 = ax_norm.imshow(norm_map.cpu().numpy(), cmap="viridis", vmin=norm_min, vmax=norm_max)
        ax_norm.set_title(f"{title} | token norm")
        ax_norm.set_xticks([])
        ax_norm.set_yticks([])
        fig.colorbar(im0, ax=ax_norm, fraction=0.046, pad=0.02)

        im1 = ax_pca.imshow(pca_map.cpu().numpy(), cmap="coolwarm", vmin=pca_min, vmax=pca_max)
        ax_pca.set_title(f"{title} | PCA-1 map")
        ax_pca.set_xticks([])
        ax_pca.set_yticks([])
        fig.colorbar(im1, ax=ax_pca, fraction=0.046, pad=0.02)

    fig.savefig(output_path, dpi=400, bbox_inches="tight")
    plt.close(fig)

    return {
        "baseline_norm_mean": float(norm_maps[0].mean().item()),
        "spatial_norm_mean": float(norm_maps[1].mean().item()),
        "fused_norm_mean": float(norm_maps[2].mean().item()),
    }


def _plot_compare_metrics(
    left_metrics: dict[str, float],
    right_metrics: dict[str, float],
    output_path: Path,
    left_label: str,
    right_label: str,
) -> dict[str, float]:
    knn_key = next(k for k in left_metrics.keys() if k.startswith("knn_recall@"))
    labels = [left_label, right_label]
    corr_vals = [left_metrics["pairwise_distance_corr"], right_metrics["pairwise_distance_corr"]]
    knn_vals = [left_metrics[knn_key], right_metrics[knn_key]]
    delta_corr = corr_vals[1] - corr_vals[0]
    delta_knn = knn_vals[1] - knn_vals[0]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)

    axes[0].bar(labels, corr_vals, color=["C3", "C0"])
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("Pearson corr.")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(labels, knn_vals, color=["C3", "C0"])
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title(knn_key)
    axes[1].grid(axis="y", alpha=0.3)

    deltas = [delta_corr, delta_knn]
    delta_labels = ["Δ corr.", f"Δ {knn_key}"]
    axes[2].bar(delta_labels, deltas, color=["C2", "C2"])
    axes[2].axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    axes[2].set_title(f"{right_label} - {left_label}")
    axes[2].grid(axis="y", alpha=0.3)

    for ax in axes:
        ax.tick_params(axis="x", rotation=18)

    fig.savefig(output_path, dpi=400, bbox_inches="tight")
    plt.close(fig)

    return {
        "corr_delta": float(delta_corr),
        "knn_delta": float(delta_knn),
    }


def _plot_compare_token_qualitative(
    left_payload: dict[str, Any],
    right_payload: dict[str, Any],
    output_path: Path | None,
    delta_output_path: Path | None,
    left_label: str,
    right_label: str,
    left_key: str = "fused_tokens",
    right_key: str = "fused_tokens",
) -> None:
    entries = [
        (left_label, _get_payload_tokens(left_payload, left_key)),
        (right_label, _get_payload_tokens(right_payload, right_key)),
    ]

    norm_maps = []
    pca_maps = []
    titles = []
    for label, tok in entries:
        grid_h, grid_w = map(int, left_payload["grid_hw"])
        if tok.shape[0] != grid_h * grid_w:
            grid_h, grid_w = map(int, right_payload["grid_hw"])
        if tok.shape[0] != grid_h * grid_w:
            raise ValueError(f"Token count {tok.shape[0]} does not match either payload grid size.")
        norm_maps.append(tok.norm(dim=-1).view(grid_h, grid_w))
        # collect tokens for later PCA projection
        pca_maps.append(tok)
        titles.append(label)

    norm_min = min(float(m.min().item()) for m in norm_maps)
    norm_max = max(float(m.max().item()) for m in norm_maps)
    # Fit PCA on the concatenation of the two token sets to get a shared basis
    projs = _pca_shared_proj([pca_maps[0], pca_maps[1]], n_components=1)
    # projs are lists of shape (N, n_components); reshape to grid
    pca_maps = [p.view(grid_h, grid_w) for p in projs]
    pca_min = min(float(m.min().item()) for m in pca_maps)
    pca_max = max(float(m.max().item()) for m in pca_maps)

    # Also compute delta maps (right - left) for norms and PCA-1 to highlight differences.
    left_norm, right_norm = norm_maps
    left_pca, right_pca = pca_maps
    delta_norm = (right_norm - left_norm)
    delta_pca = (right_pca - left_pca)

    # Shared color limits for comparability
    norm_vmin, norm_vmax = float(min(left_norm.min().item(), right_norm.min().item())), float(
        max(left_norm.max().item(), right_norm.max().item())
    )
    pca_vmin, pca_vmax = float(min(left_pca.min().item(), right_pca.min().item())), float(
        max(left_pca.max().item(), right_pca.max().item())
    )
    # Delta symmetric limits
    dnorm_lim = max(abs(float(delta_norm.min().item())), abs(float(delta_norm.max().item())))
    dpca_lim = max(abs(float(delta_pca.min().item())), abs(float(delta_pca.max().item())))

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 8.8), constrained_layout=True)
    rows = [
        (left_label, left_norm, left_pca),
        (right_label, right_norm, right_pca),
    ]
    for row_idx, (title, nmap, pmap) in enumerate(rows):
        ax_norm = axes[row_idx, 0]
        ax_pca = axes[row_idx, 1]

        im0 = ax_norm.imshow(nmap.cpu().numpy(), cmap="viridis", vmin=norm_vmin, vmax=norm_vmax)
        ax_norm.set_title(f"{title} | token norm")
        ax_norm.set_xticks([])
        ax_norm.set_yticks([])
        fig.colorbar(im0, ax=ax_norm, fraction=0.046, pad=0.02)

        im1 = ax_pca.imshow(pmap.cpu().numpy(), cmap="coolwarm", vmin=pca_vmin, vmax=pca_vmax)
        ax_pca.set_title(f"{title} | PCA-1 map")
        ax_pca.set_xticks([])
        ax_pca.set_yticks([])
        fig.colorbar(im1, ax=ax_pca, fraction=0.046, pad=0.02)

    if output_path is not None:
        fig.savefig(output_path, dpi=400, bbox_inches="tight")
    plt.close(fig)

    if delta_output_path is not None:
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.6), constrained_layout=True)
        im0 = axes[0].imshow(delta_norm.cpu().numpy(), cmap="bwr", vmin=-dnorm_lim, vmax=dnorm_lim)
        axes[0].set_title(f"{right_label} - {left_label} | token norm delta")
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.02)

        im1 = axes[1].imshow(delta_pca.cpu().numpy(), cmap="bwr", vmin=-dpca_lim, vmax=dpca_lim)
        axes[1].set_title(f"{right_label} - {left_label} | PCA-1 delta")
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.02)
        fig.savefig(delta_output_path, dpi=400, bbox_inches="tight")
        plt.close(fig)


def _plot_pairwise_scatter_for_compare(
    left_payload: dict[str, Any],
    right_payload: dict[str, Any],
    left_tokens: torch.Tensor,
    right_tokens: torch.Tensor,
    output_path: Path,
    sample_pairs: int = 4000,
) -> None:
    # Plot sampled pairwise feature vs coordinate distances for left and right and show regression + corr.
    coords = _as_2d(left_payload["coords"]).float()
    feats_l = left_tokens.float()
    feats_r = right_tokens.float()
    n = feats_l.shape[0]
    # compute full pairwise distances (n up to ~256 ok)
    fd_l = torch.cdist(feats_l, feats_l).cpu().numpy()
    fd_r = torch.cdist(feats_r, feats_r).cpu().numpy()
    cd = torch.cdist(coords, coords).cpu().numpy()

    # sample pairs (i<j)
    iu, ju = np.triu_indices(n, k=1)
    total_pairs = iu.size
    idx = np.random.default_rng(0).choice(total_pairs, size=min(sample_pairs, total_pairs), replace=False)
    si = iu[idx]
    sj = ju[idx]

    xl = fd_l[si, sj]
    xr = fd_r[si, sj]
    xc = cd[si, sj]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    for ax, (feat_dist, label, color) in zip(
        axes,
        [(xl, "Left"), (xr, "Right")],
    ):
        # scatter feature vs coord distance
        ys = feat_dist
        xs = xc
        ax.scatter(xs, ys, s=6, alpha=0.5)
        # regression line
        m, b = np.polyfit(xs, ys, 1)
        xs_lin = np.linspace(xs.min(), xs.max(), 100)
        ax.plot(xs_lin, m * xs_lin + b, color="C2", lw=1.5)
        # compute pearson
        corr = float(np.corrcoef(xs, ys)[0, 1])
        ax.set_title(f"{label}: feat vs coord dist (r={corr:.3f})")
        ax.set_xlabel("coord dist")
        ax.set_ylabel("feature dist")
        ax.grid(alpha=0.2)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _overlay_heatmap_on_image(image_path: Path, heatmap_grid: np.ndarray, out_path: Path, cmap_name: str = "magma", alpha: float = 0.5) -> None:
    """Overlay a heatmap (h x w) onto the image at image_path and save to out_path.
    Uses PIL for resizing and matplotlib colormap for coloring.
    """
    _make_overlay_image(image_path, heatmap_grid, cmap_name=cmap_name, alpha=alpha).save(out_path)


def _make_overlay_image(image_path: Path, heatmap_grid: np.ndarray, cmap_name: str = "magma", alpha: float = 0.5) -> Image.Image:
    """Create a PIL RGB image that overlays a heatmap onto an image without saving."""
    img = Image.open(image_path).convert("RGB")
    img_w, img_h = img.size

    # Normalize heatmap to 0..1
    hmin, hmax = float(np.nanmin(heatmap_grid)), float(np.nanmax(heatmap_grid))
    if hmax - hmin < 1e-8:
        norm = np.zeros_like(heatmap_grid, dtype=np.float32)
    else:
        norm = (heatmap_grid - hmin) / (hmax - hmin)

    # Convert to RGBA via colormap
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    heat_rgba = (cmap(norm) * 255).astype(np.uint8)  # HxWx4

    # Resize to image size
    heat_img = Image.fromarray(heat_rgba).resize((img_w, img_h), resample=Image.BILINEAR)

    # Composite: convert heat to RGB with alpha
    heat_rgb = heat_img.convert("RGBA")
    base = img.convert("RGBA")

    # apply global alpha
    alpha_layer = Image.new("L", heat_rgb.size, int(alpha * 255))
    r, g, b, a = heat_rgb.split()
    heat_rgb = Image.merge("RGBA", (r, g, b, alpha_layer))

    comp = Image.alpha_composite(base, heat_rgb)
    return comp.convert("RGB")


def _save_overlay_image_with_colorbar(
    image_path: Path,
    heatmap_grid: np.ndarray,
    out_path: Path,
    cmap_name: str = "bwr",
    alpha: float = 0.6,
    title: str | None = None,
    cbar_label: str | None = None,
) -> None:
    """Save an overlay image with an explicit colorbar for signed heatmaps."""
    img = Image.open(image_path).convert("RGB")
    img_arr = np.asarray(img)
    hmin = float(np.nanmin(heatmap_grid))
    hmax = float(np.nanmax(heatmap_grid))
    lim = max(abs(hmin), abs(hmax), 1e-8)
    fig, ax = plt.subplots(figsize=(5.2, 4.4), constrained_layout=True)
    ax.imshow(img_arr)
    im = ax.imshow(heatmap_grid, cmap=cmap_name, alpha=alpha, vmin=-lim, vmax=lim)
    ax.axis("off")
    if title:
        ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    if cbar_label:
        cbar.set_label(cbar_label)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _extract_episode_frame_index(path: Path) -> int:
    m = re.search(r"obs_step_(\d+)_main_raw\.png$", path.name)
    if m is None:
        raise ValueError(f"Could not parse frame index from {path}")
    return int(m.group(1))


def _collect_episode_frame_paths(payload_path: Path) -> list[Path]:
    frame_paths = sorted(
        payload_path.parent.glob("obs_step_*_main_raw.png"),
        key=_extract_episode_frame_index,
    )
    if not frame_paths:
        raise FileNotFoundError(f"No raw frame images found next to {payload_path}")
    return frame_paths


def _uniform_sample_paths(paths: list[Path], num_samples: int = 4) -> tuple[list[Path], list[int]]:
    if not paths:
        raise ValueError("Cannot sample from an empty path list.")
    if len(paths) <= num_samples:
        return paths, list(range(len(paths)))
    if num_samples == 4:
        sample_idx = [0, int(round((len(paths) - 1) / 3)), int(round(2 * (len(paths) - 1) / 3)), len(paths) - 1]
    else:
        sample_idx = np.linspace(0, len(paths) - 1, num_samples)
        sample_idx = np.round(sample_idx).astype(int).tolist()
    sample_idx = list(dict.fromkeys(sample_idx))
    if len(sample_idx) < num_samples:
        for idx in range(len(paths)):
            if idx not in sample_idx:
                sample_idx.append(idx)
            if len(sample_idx) == num_samples:
                break
    sample_idx = sorted(sample_idx[:num_samples])
    return [paths[i] for i in sample_idx], sample_idx


def _payload_scalar_response_heatmap(payload: dict[str, Any], key: str = "spatial_tokens") -> np.ndarray:
    tok = _get_payload_tokens(payload, key).float()
    grid_h, grid_w = map(int, payload["grid_hw"])
    if tok.shape[0] != grid_h * grid_w:
        raise ValueError(f"Token count {tok.shape[0]} does not match grid size {grid_h}x{grid_w}.")
    return tok.norm(dim=-1).view(grid_h, grid_w).cpu().numpy()


def _save_overlay_sequence(
    frame_paths: list[Path],
    heatmap_grid: np.ndarray,
    output_path: Path,
    title: str,
    cmap_name: str = "magma",
    alpha: float = 0.55,
    max_cols: int = 4,
) -> None:
    if not frame_paths:
        raise ValueError("No frame paths provided for overlay sequence.")
    cols = min(max_cols, len(frame_paths))
    rows = int(np.ceil(len(frame_paths) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.0 * rows), constrained_layout=True)
    axes = np.atleast_1d(axes).reshape(rows, cols)
    for i, ax in enumerate(axes.reshape(-1)):
        if i >= len(frame_paths):
            ax.axis("off")
            continue
        overlay = _make_overlay_image(frame_paths[i], heatmap_grid, cmap_name=cmap_name, alpha=alpha)
        ax.imshow(overlay)
        ax.set_title(f"frame {i}: {frame_paths[i].stem.split('_')[2]}")
        ax.axis("off")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.savefig(output_path, dpi=350, bbox_inches="tight")
    plt.close(fig)


def _save_overlay_sequence_pair(
    left_frame_paths: list[Path],
    right_frame_paths: list[Path],
    left_heatmap_grid: np.ndarray,
    right_heatmap_grid: np.ndarray,
    output_path: Path,
    left_label: str,
    right_label: str,
    title: str,
    cmap_name: str = "magma",
    alpha: float = 0.55,
) -> None:
    if len(left_frame_paths) != len(right_frame_paths):
        raise ValueError("Left/right frame sequences must have the same length.")
    if not left_frame_paths:
        raise ValueError("No frames provided for overlay pair sequence.")

    cols = len(left_frame_paths)
    fig, axes = plt.subplots(
        2,
        cols,
        figsize=(3.8 * cols, 6.2),
        constrained_layout=True,
        gridspec_kw={"wspace": 0.01, "hspace": 0.02},
    )
    axes = np.asarray(axes)
    for col, (lpath, rpath) in enumerate(zip(left_frame_paths, right_frame_paths, strict=True)):
        left_overlay = _make_overlay_image(lpath, left_heatmap_grid, cmap_name=cmap_name, alpha=alpha)
        right_overlay = _make_overlay_image(rpath, right_heatmap_grid, cmap_name=cmap_name, alpha=alpha)

        ax_l = axes[0, col]
        ax_r = axes[1, col]
        ax_l.imshow(left_overlay)
        ax_l.set_title(f"{left_label} | frame {col}", fontsize=10)
        ax_l.axis("off")

        ax_r.imshow(right_overlay)
        ax_r.set_title(f"{right_label} | frame {col}", fontsize=10)
        ax_r.axis("off")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.savefig(output_path, dpi=350, bbox_inches="tight")
    plt.close(fig)


def _save_overlay_strip(
    frame_paths: list[Path],
    heatmap_grid: np.ndarray,
    output_path: Path,
    cmap_name: str = "magma",
    alpha: float = 0.55,
) -> None:
    if not frame_paths:
        raise ValueError("No frames provided for overlay strip.")
    overlays = [_make_overlay_image(path, heatmap_grid, cmap_name=cmap_name, alpha=alpha) for path in frame_paths]
    _save_overlay_strip_from_overlays(overlays, output_path)


def _save_overlay_strip_from_overlays(overlays: list[Image.Image], output_path: Path) -> None:
    if not overlays:
        raise ValueError("No overlays provided for strip composition.")
    widths = [img.width for img in overlays]
    heights = [img.height for img in overlays]
    total_w = int(sum(widths))
    max_h = int(max(heights))
    canvas = Image.new("RGB", (total_w, max_h), color=(255, 255, 255))
    x = 0
    for img in overlays:
        canvas.paste(img, (x, 0))
        x += img.width
    canvas.save(output_path)


def _collect_payload_step_paths(payload_path: Path) -> dict[int, Path]:
    paths = payload_path.parent.glob("payload_step_*.pt")
    step_map: dict[int, Path] = {}
    for path in paths:
        m = re.search(r"payload_step_(\d+)\.pt$", path.name)
        if m is None:
            continue
        step_map[int(m.group(1))] = path
    return dict(sorted(step_map.items()))


def _load_payload_heatmap_for_step(
    payload_step_paths: dict[int, Path],
    step_idx: int,
    key: str = "spatial_tokens",
) -> np.ndarray | None:
    """Load a per-step payload heatmap if an aligned payload_step file exists."""
    path = payload_step_paths.get(step_idx)
    if path is None or not path.exists():
        return None
    payload = torch.load(path, map_location="cpu")
    return _payload_scalar_response_heatmap(payload, key=key)


def _resolve_payload_step_path(payload_step_paths: dict[int, Path], step_idx: int) -> Path | None:
    """Resolve a payload-step file using a few common step-index conventions."""
    for candidate in (step_idx, step_idx + 1, step_idx - 1):
        path = payload_step_paths.get(candidate)
        if path is not None and path.exists():
            return path
    return None


def _compute_local_neighbor_delta(diff_grid: np.ndarray, h: int, w: int) -> np.ndarray:
    """Compute local neighborhood aggregated delta via 3x3 averaging of diff_grid."""
    # pad and average
    pad = np.pad(diff_grid, pad_width=1, mode="reflect")
    out = np.zeros_like(diff_grid, dtype=np.float32)
    for i in range(h):
        for j in range(w):
            window = pad[i : i + 3, j : j + 3]
            out[i, j] = float(np.mean(window))
    return out


def _tensor_to_image(x: torch.Tensor) -> np.ndarray:
    img = x.detach().cpu().float()
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = img.permute(1, 2, 0)
    elif img.ndim != 3:
        raise ValueError(f"Expected CHW or HWC image tensor, got shape={tuple(img.shape)}")
    # Handle common image ranges:
    # - uint8 / [0, 255]
    # - normalized float [0, 1]
    # - normalized float [-1, 1]
    min_v = float(img.min().item())
    max_v = float(img.max().item())
    if max_v > 1.5:
        img = img / 255.0
    elif min_v < 0.0 and max_v <= 1.5:
        img = (img + 1.0) / 2.0
    img = img.clamp(0.0, 1.0).numpy()
    return img


def _save_tensor_image_no_border(x: torch.Tensor, output_path: Path) -> None:
    """Save a tensor image directly with PIL, no matplotlib padding or borders."""
    arr = _tensor_to_image(x)
    if arr.ndim != 3:
        raise ValueError(f"Expected HWC image array, got shape={arr.shape}")
    arr_uint8 = (arr * 255.0).round().clip(0, 255).astype(np.uint8)
    Image.fromarray(arr_uint8).save(output_path)


def _save_obs_panel(obs: dict[str, Any], output_path: Path, title: str | None = None) -> None:
    main_img = obs["images"][0]
    wrist_img = obs.get("wrist_images", None)
    imgs = [("main", _tensor_to_image(main_img))]
    if wrist_img is not None:
        imgs.append(("wrist", _tensor_to_image(wrist_img[0])))

    fig, axes = plt.subplots(1, len(imgs), figsize=(5.5 * len(imgs), 5.0), constrained_layout=True)
    if len(imgs) == 1:
        axes = [axes]
    for ax, (name, image) in zip(axes, imgs, strict=True):
        ax.imshow(image)
        ax.set_title(f"{name} view")
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.savefig(output_path, dpi=350, bbox_inches="tight")
    plt.close(fig)


def _save_raw_obs_images(obs: dict[str, Any], output_dir: Path, prefix: str) -> None:
    """Save raw observation images directly as borderless PNGs for overlay use."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_tensor_image_no_border(obs["images"][0], output_dir / f"{prefix}_main_raw.png")
    wrist_img = obs.get("wrist_images", None)
    if wrist_img is not None:
        _save_tensor_image_no_border(wrist_img[0], output_dir / f"{prefix}_wrist_raw.png")


def _compose_cfg(config_dir: Path, config_name: str, overrides: list[str]) -> DictConfig:
    with initialize_config_dir(version_base="1.1", config_dir=str(config_dir)):
        cfg = compose(config_name=config_name, overrides=overrides)
    return validate_cfg(cfg)


def _run_direct_visualization(
    cfg: DictConfig,
    output_dir: Path,
    rollout_steps: int,
    noise_mode: str,
    k: int,
    specific_reset_id: int | None = None,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Keep the visualization run tiny and deterministic.
    with open_dict(cfg):
        cfg.runner.only_eval = True
        cfg.env.eval.total_num_envs = 1
        cfg.env.eval.group_size = 1
        cfg.env.eval.auto_reset = False
        cfg.env.eval.ignore_terminations = True
        cfg.env.eval.video_cfg.save_video = False
        cfg.algorithm.eval_rollout_epoch = 1
        if specific_reset_id is not None:
            cfg.env.eval.specific_reset_id = int(specific_reset_id)
        if hasattr(cfg.runner, "eval_policy_path") and cfg.runner.eval_policy_path is None:
            cfg.runner.eval_policy_path = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_cls = get_env_cls(cfg.env.eval.simulator_type, cfg.env.eval)
    env = env_cls(cfg.env.eval, num_envs=1, seed_offset=0, total_num_processes=1)
    model = get_model(cfg.actor.model).to(device)
    model.eval()

    obs, _ = env.reset()
    _save_obs_panel(obs, output_dir / "obs_step_000.png", title="Reset observation")
    _save_raw_obs_images(obs, output_dir, prefix="obs_step_000")

    sample_steps = sorted({1, max(1, rollout_steps // 3), max(1, (2 * rollout_steps) // 3), rollout_steps})
    payload_samples: dict[int, dict[str, Any]] = {}
    payload_sample_paths: dict[int, str] = {}
    rollout_records: list[dict[str, Any]] = []

    for step_idx in range(max(1, rollout_steps)):
        actions, _meta = model.predict_action_batch(obs, mode="eval", compute_values=False)
        prepared_actions = prepare_actions(
            raw_chunk_actions=actions,
            simulator_type=cfg.env.eval.simulator_type,
            model_type=cfg.actor.model.model_type,
            num_action_chunks=cfg.actor.model.num_action_chunks,
            action_dim=cfg.actor.model.action_dim,
            policy=cfg.actor.model.get("policy_setup", None),
        )
        obs, rewards, terminations, truncations, infos = env.chunk_step(prepared_actions)

        _save_obs_panel(
            obs,
            output_dir / f"obs_step_{step_idx + 1:03d}.png",
            title=f"Rollout step {step_idx + 1}",
        )
        _save_raw_obs_images(obs, output_dir, prefix=f"obs_step_{step_idx + 1:03d}")

        current_step = step_idx + 1
        if current_step in sample_steps and current_step not in payload_samples:
            payload = model.export_spatial_geometry_payload(
                env_obs=obs,
                view_rank=0,
                batch_index=0,
                denoise_idx=0,
                mode=noise_mode,
                compute_values=False,
            )
            try:
                img_tensor = obs["images"][0].detach().cpu()
                payload["_obs_image"] = img_tensor
                if obs.get("wrist_images", None) is not None:
                    payload["_wrist_image"] = obs["wrist_images"][0].detach().cpu()
            except Exception:
                pass
            payload_path = output_dir / f"payload_step_{current_step:03d}.pt"
            torch.save(payload, payload_path)
            payload_samples[current_step] = payload
            payload_sample_paths[current_step] = str(payload_path)

        rollout_records.append(
            {
                "step": step_idx,
                "reward": float(
                    rewards[0].float().mean().item() if torch.is_tensor(rewards) else np.mean(rewards[0])
                ),
                "termination": bool(
                    terminations[0].bool().any().item() if torch.is_tensor(terminations) else bool(np.any(terminations[0]))
                ),
                "truncation": bool(
                    truncations[0].bool().any().item() if torch.is_tensor(truncations) else bool(np.any(truncations[0]))
                ),
                "task": str(obs["task_descriptions"][0]),
            }
        )

        if bool(terminations[0].bool().any().item()) or bool(truncations[0].bool().any().item()):
            break

    if not payload_samples:
        raise RuntimeError("Failed to collect visualization payload from the rollout.")

    first_step = sorted(payload_samples.keys())[0]
    first_payload = payload_samples[first_step]
    payload_path = Path(payload_sample_paths[first_step])

    token_qual_metrics = _plot_token_qualitative(first_payload, output_dir / "token_qualitative.png")
    _plot_pca_grid(first_payload, output_dir / "spatial_geometry_pca.png")
    scan_metrics = _plot_scan_heatmap(first_payload, output_dir / "scan_noise_heatmap.png")
    _plot_scan_hist(first_payload, output_dir / "scan_noise_hist.png")

    summary = {
        "token_qual_metrics": token_qual_metrics,
        "scan_metrics": scan_metrics,
        "payload_keys": sorted(first_payload.keys()),
        "rollout_records": rollout_records,
        "payload_path": str(payload_path),
        "payload_sample_steps": sample_steps,
        "payload_sample_paths": payload_sample_paths,
        "specific_reset_id": specific_reset_id,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # A compact final panel that combines raw observation and the first-step summary.
    overview_path = output_dir / "overview.png"
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    axes = np.asarray(axes)
    panels = [
        (output_dir / "obs_step_000_main_raw.png", "Reset observation (raw)"),
        (output_dir / "token_qualitative.png", "Token qualitative maps"),
        (output_dir / "spatial_geometry_pca.png", "Token geometry"),
        (output_dir / "scan_noise_heatmap.png", "SCAN anisotropic noise"),
    ]
    flat_axes = axes.reshape(-1)
    for ax, (path, title) in zip(flat_axes, panels):
        img = plt.imread(path)
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    for ax in flat_axes[len(panels):]:
        ax.axis("off")
        ax.text(0.5, 0.5, "Supplementary\nanalysis", ha="center", va="center", fontsize=14, fontweight="bold")
    fig.savefig(overview_path, dpi=350, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved rollout visualization to {output_dir}")
    print(f"Saved payload: {payload_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved overview: {overview_path}")
    return 0


def _run_compare_visualization(
    left_payload_path: Path,
    right_payload_path: Path,
    output_dir: Path,
    left_label: str,
    right_label: str,
    left_key: str,
    right_key: str,
    k: int,
    compare_target: str,
) -> int:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    left_payload = torch.load(left_payload_path, map_location="cpu")
    right_payload = torch.load(right_payload_path, map_location="cpu")
    full_summary = {
        "left_payload": str(left_payload_path),
        "right_payload": str(right_payload_path),
        "left_label": left_label,
        "right_label": right_label,
        "comparisons": {},
    }

    if compare_target in {"noise", "both"} and "action_noise_std" in left_payload and "action_noise_std" in right_payload:
        key = "action_noise_std"
        subdir = output_dir / "scan_noise"
        subdir.mkdir(parents=True, exist_ok=True)
        entry = {"status": "ok"}
        try:
            noise_stats = _plot_compare_scan_noise(
                left_payload=left_payload,
                right_payload=right_payload,
                raw_output_path=subdir / "compare_scan_noise_raw.png",
                normalized_output_path=subdir / "compare_scan_noise_normalized.png",
                residual_output_path=subdir / "compare_scan_noise_residual.png",
                distribution_output_path=subdir / "compare_scan_noise_distribution.png",
                left_label=left_label,
                right_label=right_label,
            )
            entry.update(noise_stats)
            _plot_scan_hist_normalized(left_payload, subdir / "left_scan_noise_hist.png")
            _plot_scan_hist_normalized(right_payload, subdir / "right_scan_noise_hist.png")
        except Exception as e:
            entry["status"] = "failed"
            entry["error"] = str(e)
        full_summary["comparisons"][key] = entry
    if compare_target in {"feature", "both"}:
        compare_keys = ["fused_tokens"]
        for key in compare_keys:
            subdir = output_dir / key
            subdir.mkdir(parents=True, exist_ok=True)
            entry = {"status": "ok"}
            try:
                ltok = _get_payload_tokens(left_payload, key)
                rtok = _get_payload_tokens(right_payload, key)
                lcoords = _as_2d(left_payload["coords"])
                rcoords = _as_2d(right_payload["coords"])
                if lcoords.shape != rcoords.shape:
                    entry["status"] = "coords_mismatch"
                    entry["error"] = f"coords shape left={tuple(lcoords.shape)} right={tuple(rcoords.shape)}"
                    full_summary["comparisons"][key] = entry
                    continue

                try:
                    left_frames = _collect_episode_frame_paths(left_payload_path)
                    right_frames = _collect_episode_frame_paths(right_payload_path)
                    shared_len = min(len(left_frames), len(right_frames))
                    sample_n = min(4, shared_len)
                    if sample_n > 0:
                        _, sample_idx = _uniform_sample_paths(left_frames[:shared_len], num_samples=sample_n)
                        left_sample_paths = [left_frames[i] for i in sample_idx]
                        right_sample_paths = [right_frames[i] for i in sample_idx]
                        left_step_payloads = _collect_payload_step_paths(left_payload_path)
                        right_step_payloads = _collect_payload_step_paths(right_payload_path)

                        left_heatmaps: list[np.ndarray] = []
                        right_heatmaps: list[np.ndarray] = []
                        local_deltas: list[np.ndarray] = []
                        heatmap_source = "base_payload"
                        for step_idx in sample_idx:
                            if step_idx == 0:
                                lhm = None
                                rhm = None
                            else:
                                lhm = _load_payload_heatmap_for_step(left_step_payloads, step_idx, key="spatial_tokens")
                                rhm = _load_payload_heatmap_for_step(right_step_payloads, step_idx, key="spatial_tokens")
                            if lhm is None:
                                lhm = _payload_scalar_response_heatmap(left_payload, key="spatial_tokens")
                            else:
                                heatmap_source = "per_step_payloads"
                            if rhm is None:
                                rhm = _payload_scalar_response_heatmap(right_payload, key="spatial_tokens")
                            else:
                                heatmap_source = "per_step_payloads"
                            # keep raw heatmaps for scalar overlays
                            left_heatmaps.append(lhm)
                            right_heatmaps.append(rhm)
                            # normalize each heatmap robustly to reduce global offsets before differencing
                            try:
                                lhm_t = torch.from_numpy(lhm)
                                rhm_t = torch.from_numpy(rhm)
                                lhm_norm = _robust_unit_interval(lhm_t).cpu().numpy()
                                rhm_norm = _robust_unit_interval(rhm_t).cpu().numpy()
                            except Exception:
                                # fallback to raw difference if normalization fails
                                lhm_norm = lhm
                                rhm_norm = rhm
                            local_deltas.append(_compute_local_neighbor_delta(rhm_norm - lhm_norm, lhm_norm.shape[0], lhm_norm.shape[1]))

                        left_first_heatmap = left_heatmaps[0]
                        right_first_heatmap = right_heatmaps[0]
                        # compute normalized first-heatmap pair for summary signed-delta
                        try:
                            left_first_norm = _robust_unit_interval(torch.from_numpy(left_first_heatmap)).cpu().numpy()
                            right_first_norm = _robust_unit_interval(torch.from_numpy(right_first_heatmap)).cpu().numpy()
                        except Exception:
                            left_first_norm = left_first_heatmap
                            right_first_norm = right_first_heatmap
                        if left_first_heatmap.shape == right_first_heatmap.shape:
                            rep_rel_idx = int(np.argmax([float(np.mean(np.abs(delta))) for delta in local_deltas]))
                            rep_step_idx = int(sample_idx[rep_rel_idx])
                            rep_left_path = left_sample_paths[rep_rel_idx]
                            rep_right_path = right_sample_paths[rep_rel_idx]
                            rep_local_delta = local_deltas[rep_rel_idx]
                            la = ltok.detach().cpu().numpy()
                            ra = rtok.detach().cpu().numpy()
                            d = np.linalg.norm(ra - la, axis=1)
                            grid_h, grid_w = map(int, left_payload.get("grid_hw", right_payload.get("grid_hw")))
                            if d.size != grid_h * grid_w:
                                raise ValueError(
                                    f"Per-token difference size {d.size} does not match grid size {grid_h}x{grid_w}."
                                )
                            left_overlays = [
                                _make_overlay_image(path, hm, cmap_name="magma", alpha=0.55)
                                for path, hm in zip(left_sample_paths, left_heatmaps, strict=True)
                            ]
                            right_overlays = [
                                _make_overlay_image(path, hm, cmap_name="magma", alpha=0.55)
                                for path, hm in zip(right_sample_paths, right_heatmaps, strict=True)
                            ]
                            delta_overlays = [
                                _make_overlay_image(path, delta, cmap_name="bwr", alpha=0.6)
                                for path, delta in zip(right_sample_paths, local_deltas, strict=True)
                            ]
                            _save_overlay_strip_from_overlays(
                                right_overlays,
                                subdir / "compare_with_spatial_scalar_response_strip.png",
                            )
                            _save_overlay_strip_from_overlays(
                                left_overlays,
                                subdir / "compare_no_spatial_scalar_response_strip.png",
                            )
                            _save_overlay_strip_from_overlays(
                                delta_overlays,
                                subdir / "compare_local_delta_overlay_strip.png",
                            )

                            # second-frame outputs (explicit standalone images)
                            second_rel_idx = 1 if len(sample_idx) > 1 else 0
                            second_step_rel = sample_idx[second_rel_idx]
                            second_step_num = int(second_step_rel)
                            second_left_path = left_sample_paths[second_rel_idx]
                            second_right_path = right_sample_paths[second_rel_idx]
                            second_left_hm = left_heatmaps[second_rel_idx]
                            second_right_hm = right_heatmaps[second_rel_idx]
                            second_local_delta = local_deltas[second_rel_idx]

                            left_overlays[second_rel_idx].save(subdir / "compare_no_spatial_scalar_response_frame2.png")
                            right_overlays[second_rel_idx].save(subdir / "compare_with_spatial_scalar_response_frame2.png")
                            delta_overlays[second_rel_idx].save(subdir / "compare_local_delta_overlay_frame2.png")

                            # second-frame compare_token_qualitative_delta and L2 outputs
                            try:
                                third_left_payload = (
                                    torch.load(_resolve_payload_step_path(left_step_payloads, second_step_num), map_location="cpu")
                                    if _resolve_payload_step_path(left_step_payloads, second_step_num) is not None
                                    else left_payload
                                )
                                third_right_payload = (
                                    torch.load(_resolve_payload_step_path(right_step_payloads, second_step_num), map_location="cpu")
                                    if _resolve_payload_step_path(right_step_payloads, second_step_num) is not None
                                    else right_payload
                                )
                                _plot_compare_token_qualitative(
                                    left_payload=third_left_payload,
                                    right_payload=third_right_payload,
                                    output_path=None,
                                    delta_output_path=subdir / "compare_token_qualitative_delta_frame2.png",
                                    left_label=left_label,
                                    right_label=right_label,
                                    left_key=key,
                                    right_key=key,
                                )
                                third_left_path_payload = _resolve_payload_step_path(left_step_payloads, second_step_num)
                                third_right_path_payload = _resolve_payload_step_path(right_step_payloads, second_step_num)
                                if third_left_path_payload is not None and third_right_path_payload is not None:
                                    pL = torch.load(third_left_path_payload, map_location="cpu")
                                    pR = torch.load(third_right_path_payload, map_location="cpu")
                                    la_th = _get_payload_tokens(pL, key).detach().cpu().numpy()
                                    ra_th = _get_payload_tokens(pR, key).detach().cpu().numpy()
                                    d_th = np.linalg.norm(ra_th - la_th, axis=1)
                                    if d_th.size == grid_h * grid_w:
                                        fig, ax = plt.subplots(figsize=(4, 4))
                                        im = ax.imshow(d_th.reshape(grid_h, grid_w), cmap="magma")
                                        ax.set_title(f"Per-token L2 | frame {second_step_num}")
                                        ax.axis("off")
                                        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
                                        fig.savefig(subdir / "compare_per_token_l2_frame2.png", bbox_inches="tight", dpi=200)
                                        plt.close(fig)
                            except Exception:
                                pass
                            entry["overlay_frame_indices"] = sample_idx
                            entry["overlay_num_frames"] = sample_n
                            entry["overlay_representative_frame_index"] = rep_step_idx
                            entry["overlay_heatmap_key"] = "spatial_tokens_scalar_response"
                            entry["overlay_heatmap_source"] = heatmap_source
                            entry["overlay_delta_source"] = "per_frame_signed_local_delta"
                            entry["overlay_outputs"] = [
                                "compare_with_spatial_scalar_response_strip.png",
                                "compare_no_spatial_scalar_response_strip.png",
                                "compare_local_delta_overlay_strip.png",
                                "compare_with_spatial_scalar_response_frame2.png",
                                "compare_no_spatial_scalar_response_frame2.png",
                                "compare_local_delta_overlay_frame2.png",
                                "compare_token_qualitative_delta_frame2.png",
                                "compare_per_token_l2_frame2.png",
                            ]
                except Exception as e:
                    entry["overlay_sequence_status"] = "failed"
                    entry["overlay_sequence_error"] = str(e)

                # per-token L2 map and overlay on raw obs images (no white border)
                try:
                    la = ltok.detach().cpu().numpy()
                    ra = rtok.detach().cpu().numpy()
                    if la.shape == ra.shape:
                        d = np.linalg.norm(ra - la, axis=1)
                        h, w = left_payload.get("grid_hw", right_payload.get("grid_hw", (int(np.sqrt(d.size)), int(np.sqrt(d.size)))))
                        if d.size == h * w:
                            # save a clean heatmap
                            fig, ax = plt.subplots(figsize=(4, 4))
                            im = ax.imshow(d.reshape(h, w), cmap="magma")
                            ax.set_title(f"Per-token L2: {right_label} - {left_label}")
                            ax.axis("off")
                            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
                            fig.savefig(subdir / "compare_per_token_l2.png", bbox_inches="tight", dpi=200)
                            plt.close(fig)

                        # numeric summary for this key
                        entry["per_token_l2_mean"] = float(d.mean())
                        entry["per_token_l2_median"] = float(np.median(d))
                        entry["per_token_l2_std"] = float(d.std())
                        entry["per_token_l2_pct95"] = float(np.percentile(d, 95))
                    else:
                        entry["status"] = "shape_mismatch"
                        entry["error"] = f"token shapes differ left={la.shape} right={ra.shape}"
                except Exception as e:
                    entry["status"] = "per_token_failed"
                    entry["error"] = str(e)

            except KeyError as e:
                entry["status"] = "missing_key"
                entry["error"] = str(e)
            except Exception as e:
                entry["status"] = "failed"
                entry["error"] = str(e)

            full_summary["comparisons"][key] = entry

    if not full_summary["comparisons"]:
        raise ValueError(
            f"No comparisons were generated for compare_target='{compare_target}'. "
            "Check that the payloads contain the required fields."
        )

    # write master summary
    summary_path = output_dir / "compare_summary.json"
    summary_path.write_text(json.dumps(full_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved comparison figures to {output_dir}")
    print(f"Saved comparison summary: {summary_path}")
    return 0


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description=(
            "Visualize supplementary geometry payloads or run a direct full rollout visualization."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["pt", "direct", "compare"],
        default="direct",
        help="'direct' runs an environment/model rollout; 'pt' reads one payload; 'compare' compares two payloads.",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=None,
        help="Path to a .pt payload created by export_spatial_geometry_payload(). Used in pt mode.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=repo_root / "outputs" / "spatial_geometry",
        help="Directory to store the figures and summary JSON.",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=repo_root / "examples" / "embodiment" / "config",
        help="Hydra config directory used in direct mode.",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="libero_spatial_ppo_openpi_pi05_eval",
        help="Hydra config name used in direct mode.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Hydra override applied in direct mode; can be passed multiple times.",
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=48,
        help="How many environment chunk steps to run in direct mode.",
    )
    parser.add_argument(
        "--specific-reset-id",
        type=int,
        default=893,
        help="Fix the LIBERO reset state id in direct mode so the same task is reproduced.",
    )
    parser.add_argument(
        "--noise-mode",
        type=str,
        default="train",
        choices=["train", "eval"],
        help="Mode passed to the payload export. train shows SCAN noise; eval zeros it out.",
    )
    parser.add_argument(
        "--left-input",
        type=Path,
        default=None,
        help="Left payload for compare mode (usually the no-spatial run).",
    )
    parser.add_argument(
        "--right-input",
        type=Path,
        default=None,
        help="Right payload for compare mode (usually the spatial run).",
    )
    parser.add_argument(
        "--left-label",
        type=str,
        default="No spatial",
        help="Label for the left payload in compare mode.",
    )
    parser.add_argument(
        "--right-label",
        type=str,
        default="With spatial",
        help="Label for the right payload in compare mode.",
    )
    parser.add_argument(
        "--left-key",
        type=str,
        default="fused_tokens",
        help="Token field to analyze from the left payload in compare mode.",
    )
    parser.add_argument(
        "--right-key",
        type=str,
        default="fused_tokens",
        help="Token field to analyze from the right payload in compare mode.",
    )
    parser.add_argument("--k", type=int, default=5, help="k for spatial neighbor recall.")
    parser.add_argument(
        "--compare-target",
        type=str,
        default="noise",
        choices=["noise", "feature", "both"],
        help="Which comparison(s) to generate in compare mode.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "direct":
        cfg = _compose_cfg(args.config_dir, args.config_name, args.override)
        return _run_direct_visualization(
            cfg=cfg,
            output_dir=args.output_dir,
            rollout_steps=args.rollout_steps,
            noise_mode=args.noise_mode,
            k=args.k,
            specific_reset_id=args.specific_reset_id,
        )

    if args.mode == "compare":
        if args.left_input is None or args.right_input is None:
            raise ValueError("--left-input and --right-input are required in compare mode.")
        return _run_compare_visualization(
            left_payload_path=args.left_input,
            right_payload_path=args.right_input,
            output_dir=args.output_dir,
            left_label=args.left_label,
            right_label=args.right_label,
            left_key=args.left_key,
            right_key=args.right_key,
            k=args.k,
            compare_target=args.compare_target,
        )

    if args.input is None:
        raise ValueError("--input is required in pt mode.")

    payload = torch.load(args.input, map_location="cpu")
    _plot_pca_grid(payload, args.output_dir / "spatial_geometry_pca.png")

    scan_metrics = _plot_scan_heatmap(payload, args.output_dir / "scan_noise_heatmap.png")
    _plot_scan_hist(payload, args.output_dir / "scan_noise_hist.png")

    summary = {
        "scan_metrics": scan_metrics,
        "payload_keys": sorted(payload.keys()),
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved figures and summary to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())