from __future__ import annotations

import argparse
import csv
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Libre Baskerville"]
plt.rcParams["font.size"] = 11.5
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["legend.fontsize"] = 10.0


def read_radar_csv(csv_path: Path) -> tuple[list[str], list[str], np.ndarray]:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"CSV has no data rows: {csv_path}")

    headers = [h for h in reader.fieldnames or [] if h and h != "method"]
    methods = [r["method"].strip() for r in rows]

    data = []
    for r in rows:
        vals = []
        for h in headers:
            vals.append(float(r[h]))
        data.append(vals)

    return methods, headers, np.asarray(data, dtype=np.float64)


def _close_polygon(vals: np.ndarray) -> np.ndarray:
    return np.concatenate([vals, vals[:1]])


def _display_name(category: str) -> str:
    return category.replace(" (%)", "").strip()


def _dimension_type(category: str) -> str:
    c = category.lower()
    if c.startswith("total"):
        return "aggregate"
    if c.startswith("camera view") or c.startswith("robot initial states"):
        return "few-shot"
    return "zero-shot"


def _wrap_label(text: str, width: int = 13) -> str:
    lines = text.split("\n")
    wrapped: list[str] = []
    for line in lines:
        parts = textwrap.wrap(line, width=width, break_long_words=False)
        wrapped.extend(parts if parts else [""])
    return "\n".join(wrapped)


def _baseline_index(methods: list[str], baseline_name: str) -> int:
    if baseline_name in methods:
        return methods.index(baseline_name)
    for i, m in enumerate(methods):
        if "pi" in m.lower() or "baseline" in m.lower():
            return i
    return 0


def _transform_data(
    methods: list[str],
    categories: list[str],
    data: np.ndarray,
    mode: str,
    baseline_name: str,
) -> tuple[list[str], np.ndarray, dict[str, str], tuple[float, float], list[float], str]:
    dim_types = {c: _dimension_type(c) for c in categories}

    if mode == "absolute":
        lower = max(0.0, np.floor((float(np.min(data)) - 2.0) / 5.0) * 5.0)
        upper = 100.0
        yticks = list(np.arange(max(0.0, np.ceil(lower / 5.0) * 5.0), upper + 0.1, 5.0))
        return methods, data, dim_types, (lower, upper), yticks, "Absolute Success (%)"

    # delta mode: improvement over baseline in percentage points (pp)
    b_idx = _baseline_index(methods, baseline_name)
    base = data[b_idx : b_idx + 1, :]
    delta = data - base

    keep_idx = [i for i in range(len(methods)) if i != b_idx]
    delta_methods = [methods[i] for i in keep_idx]
    delta_data = delta[keep_idx, :]

    max_abs = float(np.max(np.abs(delta_data))) if delta_data.size else 1.0
    max_abs = max(1.0, np.ceil((max_abs + 0.3) / 1.0) * 1.0)
    lower, upper = -max_abs, max_abs
    yticks = list(np.arange(lower, upper + 0.01, max(1.0, np.ceil(max_abs / 4.0))))
    return delta_methods, delta_data, dim_types, (lower, upper), yticks, "Improvement over Baseline (pp)"


def _draw_axis_labels(
    ax,
    angles: np.ndarray,
    categories: list[str],
    dim_types: dict[str, str],
    radial_label_pos: float,
):
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([""] * len(categories))

    for i, c in enumerate(categories):
        dtype = dim_types[c]
        base = _display_name(c)
        if dtype == "few-shot":
            text = f"{base}\n(Few-shot)"
            color = "#1d4ed8"
            weight = "semibold"
        elif dtype == "zero-shot":
            text = f"{base}\n(Zero-shot)"
            color = "#334155"
            weight = "normal"
        else:
            text = f"{base}\n(Aggregate)"
            color = "#374151"
            weight = "semibold"

        ax.text(
            angles[i],
            radial_label_pos,
            _wrap_label(text, width=14),
            fontsize=11.2,
            color=color,
            fontweight="bold" if weight == "normal" else "heavy",
            ha="center",
            va="center",
            clip_on=False,
            zorder=20,
        )


def plot_radar(
    methods: list[str],
    categories: list[str],
    data: np.ndarray,
    output_path: Path,
    title: str,
    mode: str,
    baseline_name: str,
):
    n_cat = len(categories)
    angles = np.linspace(0, 2 * np.pi, n_cat, endpoint=False)
    angles = _close_polygon(angles)

    plot_methods, plot_data, dim_types, (lower, upper), yticks, y_label = _transform_data(
        methods, categories, data, mode, baseline_name
    )

    fig = plt.figure(figsize=(8.4, 7.2), facecolor="white")
    ax = fig.add_subplot(111, polar=True)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Soft sector background to distinguish dimension types.
    sector_half_width = np.pi / n_cat * 0.9
    for i, theta in enumerate(angles[:-1]):
        dtype = dim_types[categories[i]]
        if dtype == "few-shot":
            color = "#dbeafe"
            alpha = 0.32
        elif dtype == "zero-shot":
            color = "#f8fafc"
            alpha = 0.20
        else:
            color = "#e5e7eb"
            alpha = 0.35

        ax.bar(
            x=theta,
            height=upper - lower,
            width=2 * sector_half_width,
            bottom=lower,
            color=color,
            edgecolor="none",
            alpha=alpha,
            zorder=0,
            align="center",
        )

    radial_label_pos = upper + (upper - lower) * 0.17
    _draw_axis_labels(ax, angles, categories, dim_types, radial_label_pos)

    ax.set_ylim(lower, upper)
    ax.set_yticks(yticks)
    if mode == "delta":
        ax.set_yticklabels([f"{v:+.0f}" for v in yticks], fontsize=10.0, color="#6b7280", fontweight="bold")
        ax.plot(angles, np.full_like(angles, 0.0), color="#6b7280", lw=1.2, ls="--", alpha=0.85, zorder=1)
    else:
        ax.set_yticklabels([f"{int(v)}" for v in yticks], fontsize=10.0, color="#6b7280", fontweight="bold")
    ax.set_rlabel_position(182)
    ax.grid(color="#d1d5db", alpha=0.8, linewidth=0.8)
    ax.spines["polar"].set_color("#9ca3af")

    # method styles
    style_map = {
        "ReinFlow": {"c": "#3b82f6", "lw": 2.1, "ls": "-.", "alpha": 0.95, "z": 3},
        "SA-VLA": {"c": "#dc2626", "lw": 3.1, "ls": "-", "alpha": 1.0, "z": 6},
    }

    method_colors: dict[str, str] = {}
    for i, method in enumerate(plot_methods):
        vals = _close_polygon(plot_data[i])
        s = style_map.get(method, {"c": f"C{i}", "lw": 2.0, "ls": "-.", "alpha": 0.95, "z": 2})
        method_colors[method] = s["c"]

        ax.plot(
            angles,
            vals,
            color=s["c"],
            linewidth=s["lw"],
            linestyle=s["ls"],
            alpha=s["alpha"],
            label=method,
            zorder=s["z"],
        )

        if method == "SA-VLA":
            ax.fill(angles, vals, color=s["c"], alpha=0.14, zorder=s["z"] - 1)
            ax.scatter(angles[:-1], vals[:-1], color=s["c"], s=20, zorder=s["z"] + 1)
        elif mode == "delta" and method == "ReinFlow":
            # emphasize poor RL gain of ReinFlow against baseline
            ax.fill(angles, vals, color=s["c"], alpha=0.07, zorder=s["z"] - 1)

    # concise annotation for contrast with baseline
    if mode == "delta" and "Total (%)" in categories:
        total_idx = categories.index("Total (%)")
        for m_name in ["SA-VLA", "ReinFlow", "Flow-GRPO"]:
            if m_name in plot_methods:
                m_i = plot_methods.index(m_name)
                v = plot_data[m_i, total_idx]
                color = method_colors.get(m_name, "#111827")
                if m_name == "Flow-GRPO":
                    offset = (0.0, -1.0)
                elif m_name == "ReinFlow":
                    offset = (0.3, 1.0)
                else:
                    offset = (-0.1, 0.0)
                ax.annotate(
                    f"{m_name}: {v:+.2f} pp",
                    xy=(angles[total_idx], v),
                    xytext=(
                        angles[total_idx] + 0.26 + offset[0],
                        v + (1.8 if v >= 0 else 0.8) + offset[1],
                    ),
                    textcoords="data",
                    fontsize=10.2,
                    fontweight="bold",
                    color=color,
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.0),
                    zorder=15,
                )

    if title.strip():
        ax.set_title(title, fontsize=15, fontweight="bold", pad=10, color="#111827")

    handles, labels = ax.get_legend_handles_labels()
    handles.extend(
        [
            Patch(facecolor="#dbeafe", edgecolor="none", alpha=0.6, label="Few-shot dimensions"),
            Patch(facecolor="#f8fafc", edgecolor="#e5e7eb", alpha=1.0, label="Zero-shot dimensions"),
            Patch(facecolor="#e5e7eb", edgecolor="none", alpha=0.9, label="Aggregate metric (Total)"),
        ]
    )
    labels.extend(["Few-shot dimensions", "Zero-shot dimensions", "Aggregate metric (Total)"])
    if mode == "delta":
        handles.append(
            plt.Line2D([0], [0], color="#6b7280", lw=1.4, ls="--", label=f"{baseline_name} (baseline)")
        )
        labels.append(f"{baseline_name} (baseline)")
    legend_title = y_label + (f" | baseline: {baseline_name}" if mode == "delta" else "")
    ax.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
        fontsize=9.6,
        title=y_label,
        title_fontsize=10.2,
        ncol=1,
        labelspacing=0.55,
        handlelength=1.9,
        handletextpad=0.6,
        borderaxespad=0.15,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=1.3)
    fig.savefig(output_path, dpi=500, bbox_inches="tight")
    # PDF for publication
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved: {output_path}")
    print(f"Saved: {output_path.with_suffix('.pdf')}")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description="Plot publication-style radar chart from eval_variants_radar.csv")
    parser.add_argument(
        "--input",
        type=Path,
        default=repo_root / "outputs" / "eval_variants_radar.csv",
        help="Input radar CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "outputs" / "eval_variants_radar.png",
        help="Output figure path (.png); .pdf is also exported",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["absolute", "delta"],
        default="delta",
        help="Plot absolute values or improvement over baseline (delta).",
    )
    parser.add_argument(
        "--baseline-name",
        type=str,
        default="$\\pi_{0.5}^*$",
        help="Baseline method name used in delta mode.",
    )
    args = parser.parse_args()

    methods, categories, data = read_radar_csv(args.input)
    plot_radar(methods, categories, data, args.output, args.title, args.mode, args.baseline_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
