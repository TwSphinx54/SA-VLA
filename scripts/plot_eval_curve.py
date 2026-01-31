from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Libre Baskerville"
plt.rcParams["font.weight"] = "medium"


def _try_float(s: str) -> float | None:
    try:
        return float(s)
    except ValueError:
        return None


def read_series(csv_path: Path) -> Tuple[List[str], List[List[float]]]:
    labels: List[str] = []
    series: List[List[float]] = []

    with csv_path.open("r", newline="") as f:
        for i, row in enumerate(csv.reader(f), start=1):
            row = [c.strip() for c in row if c.strip()]
            if not row:
                continue

            first_as_float = _try_float(row[0])

            # New format: first column is label, followed by values
            if first_as_float is None:
                label = row[0]
                vals = [float(x) for x in row[1:]]
            # Old format: entire row is values
            else:
                label = f"row {i}"
                vals = [float(x) for x in row]

            if vals:
                labels.append(label)
                series.append(vals)

    return labels, series


def smooth_ma(y: List[float], window: int) -> List[float]:
    if window <= 1:
        return list(y)
    n = len(y)
    half = window // 2
    out: List[float] = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out.append(sum(y[lo:hi]) / (hi - lo))
    return out


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]

    p = argparse.ArgumentParser(description="Plot curves from CSV (first column = legend label).")
    p.add_argument("-i", "--input", type=Path, default=repo_root / "outputs" / "noise.csv")
    p.add_argument("-o", "--output", type=Path, default=repo_root / "outputs" / "eval.png")
    p.add_argument("--title", type=str, default=None)
    p.add_argument("--xlabel", type=str, default="Step")
    p.add_argument("--ylabel", type=str, default="Success Rate (%)")
    p.add_argument("--smooth-window", type=int, default=3, help="Moving-average window size (odd recommended).")
    p.add_argument("--raw-alpha", type=float, default=0.15, help="Alpha for raw (unsmoothed) lines.")
    p.add_argument("--methods", type=str, nargs="+", default=None, help="Method names; read output/<method>.csv.")
    args = p.parse_args()

    method_entries: List[Tuple[str | None, str, List[float]]] = []
    method_styles: dict[str | None, str] = {}
    style_cycle = ["solid", "dashed", "dashdot", "dotted"]

    if args.methods:
        out_dir = repo_root / "outputs"
        for i, method in enumerate(args.methods):
            csv_path = out_dir / f"{method}.csv"
            if not csv_path.exists():
                print(f"File not found: {csv_path}")
                continue
            labels, series = read_series(csv_path)
            for label, y in zip(labels, series):
                method_entries.append((method, label, y))
            method_styles[method] = style_cycle[i % len(style_cycle)]
    else:
        labels, series = read_series(args.input)
        method_entries = [(None, label, y) for label, y in zip(labels, series)]
        method_styles[None] = "solid"

    if not method_entries:
        print(f"No data loaded.")
        return 2

    plt.figure(figsize=(6, 4.5))

    # Fixed color scheme: use only C2 and C1, cycle as needed
    palette = ["C2", "C1"]

    all_x_ticks: List[int] | None = None

    for idx, (method, label, y) in enumerate(method_entries):
        x = [(i + 1) * 10 for i in range(len(y))]
        if all_x_ticks is None:
            all_x_ticks = x
        elif all_x_ticks != x:
            all_x_ticks = None  # 不强制刻度以避免长度不一致

        c = palette[idx % len(palette)]
        ls = method_styles.get(method, "solid")
        legend_label = f"{method}: {label}" if method else label

        plt.scatter(x, y, color=c, alpha=args.raw_alpha, linestyle=ls)

        ys = smooth_ma(y, args.smooth_window)
        plt.plot(x, ys, color=c, linewidth=2.5, alpha=0.95, linestyle=ls, label=legend_label)

    if all_x_ticks is not None:
        plt.xticks(all_x_ticks)

    if args.title:
        plt.title(args.title)
    plt.xlabel(args.xlabel)
    plt.ylabel(args.ylabel)
    plt.grid(True, alpha=0.3)
    if len(method_entries) > 1:
        plt.legend()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.output, dpi=400)
    print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
