import argparse
import os
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.family"] = "Libre Baskerville"
plt.rcParams["font.weight"] = "medium"

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except Exception as e:
    print("tensorboard is required: pip install tensorboard", file=sys.stderr)
    raise

def read_scalar_series(event_path: str, tag: str):
    """Return list of (step, value) for a given tag."""
    if not os.path.exists(event_path):
        raise FileNotFoundError(f"Not found: {event_path}")
    acc = EventAccumulator(event_path, size_guidance={"scalars": 0})
    acc.Reload()
    try:
        scalars = acc.Scalars(tag)
    except KeyError:
        available = list(acc.Tags().get("scalars", []))
        raise KeyError(f"Tag '{tag}' not found in {event_path}. Available: {available}")
    # Sort by step
    scalars = sorted(scalars, key=lambda x: x.step)
    return [(s.step, float(s.value)) for s in scalars]

def exp_smooth(values: list, alpha: float, adjust: bool = False):
    """EWMA via pandas; returns list."""
    if not values:
        return []
    return pd.Series(values).ewm(alpha=alpha, adjust=adjust).mean().tolist()

def smooth_step_series(step_vals: list, alpha: float, adjust: bool = False):
    """Apply EWMA to (step, value) series and return dict step->smoothed_value."""
    steps = [s for s, _ in step_vals]
    vals = [v for _, v in step_vals]
    smoothed = exp_smooth(vals, alpha, adjust=adjust)
    return {s: sv for s, sv in zip(steps, smoothed)}

def aggregate_mean_std(seed_dicts: list):
    """Align by intersection of steps and compute mean/std arrays."""
    if not seed_dicts:
        raise ValueError("No seed data provided.")
    common_steps = set(seed_dicts[0].keys())
    for d in seed_dicts[1:]:
        common_steps &= set(d.keys())
    if not common_steps:
        raise ValueError("No common steps across seeds.")
    steps = sorted(common_steps)
    stacked = np.stack([[d[s] for s in steps] for d in seed_dicts], axis=0)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0, ddof=0)
    return steps, mean, std

def collect_method_event_files(methods):
    """Return dict: method -> list of latest event files per seed directory."""
    method_files = {}
    for m in methods:
        pattern = os.path.join("logs", f"{m}_seed*", "tensorboard", "events.out*")
        candidates = sorted(glob.glob(pattern))
        if not candidates:
            print(f"Warning: no event files found for method '{m}' (pattern: {pattern})", file=sys.stderr)
            continue
        seed_map = {}
        for f in candidates:
            seed_dir = os.path.dirname(f)
            seed_map.setdefault(seed_dir, []).append(f)
        method_files[m] = [sorted(fs)[-1] for fs in seed_map.values()]
    return method_files

def plot_curve(steps, mean, std, tag: str, alpha: float, out_path: str):
    plt.figure(figsize=(6, 4.5))
    plt.plot(steps, mean, label="mean", color="C0")
    plt.fill_between(steps, mean - std, mean + std, color="C0", alpha=0.2, label="±1 std")
    plt.xlabel("step")
    plt.ylabel(tag)
    # plt.title(f"{tag} (EWMA α={alpha})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=400, bbox_inches="tight")
    print(f"Saved plot: {out_path}")

def plot_methods_curves(method_curves, tag: str, alpha: float, out_path: str):
    fig, ax_main = plt.subplots(figsize=(6, 4.5))
    color_map = {"noise_d": "C2", "noise_s": "C1"}
    for idx, (method, (steps, mean, std)) in enumerate(method_curves.items()):
        color = color_map.get(method, f"C{idx % 10}")
        ax_main.plot(steps, mean, label=f"{method} mean", color=color)
        ax_main.fill_between(steps, mean - std, mean + std, color=color, alpha=0.15, label=f"{method} ±1 std")
    ax_main.set_xlabel("Step")
    ax_main.set_ylabel("Success Rate (%)")
    ax_main.legend()
    ax_main.grid(True, alpha=0.3)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=400, bbox_inches="tight")
    print(f"Saved plot: {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Plot mean/std of smoothed TensorBoard scalars across seeds.")
    parser.add_argument("--paths", nargs="+", default=None, help="Explicit TensorBoard event file paths for a single group.")
    parser.add_argument("--methods", nargs="+", default=["scan_d", "noise_d", "noise_s"],
                        help="Method names; auto-discover logs/<method>_seed*/tensorboard/events.out*.")
    parser.add_argument("--tag", default="env/success_once", help="Scalar tag to read.")
    parser.add_argument("--alpha", type=float, default=0.1, help="Exponential smoothing alpha.")
    parser.add_argument("--out", default="outputs/combined_success_curve.png", help="Output PNG path.")
    parser.add_argument("--adjust", action="store_true", help="Use bias-corrected EWMA (pandas ewm adjust=True).")
    parser.add_argument("--max-steps", type=int, default=100, help="Max step to include in plotting.")
    args = parser.parse_args()

    if args.paths:
        method_to_files = {"custom": args.paths}
    else:
        method_to_files = collect_method_event_files(args.methods)
        if not method_to_files:
            raise ValueError("No event files found for given methods.")

    method_curves = {}
    for method, files in method_to_files.items():
        seed_dicts = []
        for p in files:
            series = read_scalar_series(p, args.tag)
            seed_dicts.append(smooth_step_series(series, args.alpha, adjust=args.adjust))
        if args.max_steps is not None:
            seed_dicts = [{k: v for k, v in d.items() if k <= args.max_steps} for d in seed_dicts]
        steps, mean, std = aggregate_mean_std(seed_dicts)
        method_curves[method] = (steps, mean, std)

    plot_methods_curves(method_curves, args.tag, args.alpha, args.out)

if __name__ == "__main__":
    main()
