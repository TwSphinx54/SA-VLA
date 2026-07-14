import argparse
import ast
import csv
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


def _now() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _natural_key(text: str):
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", text)]


def _resolve_method_dir(repo_root: Path, method: str) -> Path:
    method_path = Path(method)
    if method_path.is_absolute():
        return method_path

    # 1) direct repo-relative path
    direct = (repo_root / method_path).resolve()
    if direct.exists():
        return direct

    # 2) shorthand: <method_name> => weights/SAVLA/<method_name>
    fallback = (repo_root / "weights" / "SAVLA" / method).resolve()
    return fallback


def _discover_variants(method_dir: Path) -> list[Path]:
    if not method_dir.is_dir():
        return []

    variants = []
    for child in method_dir.iterdir():
        if child.name.startswith("."):
            continue
        if child.is_dir():
            variants.append(child)

    variants.sort(key=lambda p: _natural_key(p.name))
    return variants


def _parse_eval_metrics(raw_text: str) -> dict[str, float]:
    # Try to parse the last dict-like line containing eval metrics.
    lines = raw_text.splitlines()
    for line in reversed(lines):
        if "eval/" not in line or "{" not in line or "}" not in line:
            continue
        dict_str = line[line.find("{") : line.rfind("}") + 1]
        try:
            obj = ast.literal_eval(dict_str)
            if isinstance(obj, dict):
                return {str(k): float(v) for k, v in obj.items() if isinstance(k, str)}
        except Exception:
            pass

        # fallback regex parser for scalar numbers and numpy-style array(...) values
        metrics: dict[str, float] = {}

        # e.g. 'eval/success_once': array(0.88856906, dtype=float32)
        array_pairs = re.findall(
            r"'([^']*eval/[^']*)'\s*:\s*array\(\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
            dict_str,
        )
        for k, v in array_pairs:
            metrics[k] = float(v)

        # e.g. 'eval/xxx': 0.1234
        scalar_pairs = re.findall(
            r"'([^']*eval/[^']*)'\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
            dict_str,
        )
        for k, v in scalar_pairs:
            metrics[k] = float(v)

        if metrics:
            return metrics

    return {}


def _ensure_csv(csv_path: Path, fieldnames: list[str]):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists():
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()


def _append_csv(csv_path: Path, fieldnames: list[str], row: dict):
    if not csv_path.exists():
        _ensure_csv(csv_path, fieldnames)

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        existing_fieldnames = reader.fieldnames or []

    merged_fieldnames = list(existing_fieldnames)
    for key in fieldnames:
        if key not in merged_fieldnames:
            merged_fieldnames.append(key)

    if merged_fieldnames != existing_fieldnames:
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            old_rows = list(reader)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=merged_fieldnames)
            writer.writeheader()
            for old in old_rows:
                writer.writerow({k: old.get(k, "") for k in merged_fieldnames})

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=merged_fieldnames)
        writer.writerow({k: row.get(k, "") for k in merged_fieldnames})


def _metric_col_name(metric_key: str) -> str:
    normalized = re.sub(r"[^0-9a-zA-Z]+", "_", metric_key).strip("_").lower()
    return f"metric__{normalized}"


def _metric_col_to_metric_name(metric_col: str) -> str | None:
    if not metric_col.startswith("metric__"):
        return None

    key = metric_col[len("metric__") :]
    if key.startswith("eval_success_once_by_perturbation_"):
        suffix = key[len("eval_success_once_by_perturbation_") :]
        return f"eval/success_once_by_perturbation/{suffix}"
    if key == "eval_success_once":
        return "eval/success_once"
    if key == "eval_return":
        return "eval/return"
    if key == "eval_episode_len":
        return "eval/episode_len"
    if key == "eval_reward_sparse":
        return "eval/reward_sparse"
    if key == "eval_reward_dense":
        return "eval/reward_dense"
    return key.replace("_", "/", 1) if key.startswith("eval_") else key


def _bootstrap_ci_mean(values: np.ndarray, num_samples: int = 2000, seed: int = 0) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1:
        return float(values[0]), float(values[0])

    rng = np.random.default_rng(seed)
    sample_means = np.empty(num_samples, dtype=np.float64)
    n = values.size
    for i in range(num_samples):
        sample = rng.choice(values, size=n, replace=True)
        sample_means[i] = float(np.mean(sample))
    low, high = np.quantile(sample_means, [0.025, 0.975])
    return float(low), float(high)


def _pct(x: float) -> float:
    return float(x * 100.0)


def _r2(x: float) -> float:
    return float(np.round(x, 2))


def _ci_text(low: float, high: float) -> str:
    return f"[{_r2(low):.2f}, {_r2(high):.2f}]"


def _perturbation_display_name(metric_name: str) -> str:
    prefix = "eval/success_once_by_perturbation/"
    if not metric_name.startswith(prefix):
        return metric_name
    raw = metric_name[len(prefix):]
    return raw.replace("_", " ").title()


def _extract_metric_values_from_row(row: dict[str, str]) -> dict[str, float]:
    metrics: dict[str, float] = {}

    metric_cols = [key for key in row.keys() if key.startswith("metric__")]
    if metric_cols:
        for key, value in row.items():
            metric_name = _metric_col_to_metric_name(key)
            if metric_name is None:
                continue
            if value is None:
                continue
            value_str = str(value).strip()
            if value_str == "":
                continue
            try:
                metrics[metric_name] = float(value_str)
            except Exception:
                continue
    else:
        metrics_json = (row.get("metrics_json") or "").strip()
        if metrics_json and metrics_json != "{}":
            try:
                parsed = json.loads(metrics_json)
                if isinstance(parsed, dict):
                    for k, v in parsed.items():
                        try:
                            metrics[str(k)] = float(v)
                        except Exception:
                            continue
            except Exception:
                pass

    for key, value in row.items():
        if key.startswith("metric__"):
            continue
        if value is None:
            continue

    if "eval_success_once" in row and str(row.get("eval_success_once", "")).strip() != "":
        try:
            metrics.setdefault("eval/success_once", float(row["eval_success_once"]))
        except Exception:
            pass

    return metrics


def analyze_success_metrics(
    input_csv: Path,
    output_csv: Path,
    chance_level: float = 0.5,
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 0,
) -> int:
    if not input_csv.exists():
        print(f"[ERROR] Input CSV not found: {input_csv}", file=sys.stderr)
        return 2

    with input_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print(f"[ERROR] No rows found in input CSV: {input_csv}", file=sys.stderr)
        return 2

    method_to_metric_values: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        method = (row.get("method") or "").strip()
        if not method:
            continue

        metrics = _extract_metric_values_from_row(row)
        method_metrics = method_to_metric_values.setdefault(method, {})
        if "eval/success_once" in metrics:
            method_metrics.setdefault("eval/success_once", []).append(_pct(float(metrics["eval/success_once"])))

        for k, v in metrics.items():
            if k.startswith("eval/success_once_by_perturbation/"):
                method_metrics.setdefault(k, []).append(_pct(float(v)))

    if not method_to_metric_values:
        print(f"[ERROR] No success metrics found in CSV: {input_csv}", file=sys.stderr)
        return 2

    # Collect all perturbation categories for stable columns.
    all_perturbation_metrics = sorted(
        {
            metric_name
            for metric_map in method_to_metric_values.values()
            for metric_name in metric_map.keys()
            if metric_name.startswith("eval/success_once_by_perturbation/")
        }
    )

    chance_level_pct = _pct(chance_level)
    output_rows: list[dict[str, object]] = []
    for method, metric_map in sorted(method_to_metric_values.items()):
        row: dict[str, object] = {
            "method": method,
            "Chance Level (%)": _r2(chance_level_pct),
        }

        total_values = metric_map.get("eval/success_once", [])
        if total_values:
            arr = np.asarray(total_values, dtype=np.float64)
            mean = float(arr.mean())
            std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
            ci_low, ci_high = _bootstrap_ci_mean(arr, num_samples=bootstrap_samples, seed=bootstrap_seed)
            row["Total Mean (%)"] = _r2(mean)
            row["Total Std (%)"] = _r2(std)
            row["Total CI95 (%)"] = _ci_text(ci_low, ci_high)
            row["Total Sig > Chance"] = bool(ci_low > chance_level_pct)
            row["Total Sig < Chance"] = bool(ci_high < chance_level_pct)

        for metric_name in all_perturbation_metrics:
            display = _perturbation_display_name(metric_name)
            values = metric_map.get(metric_name, [])
            if not values:
                row[f"{display} Mean (%)"] = ""
                row[f"{display} Std (%)"] = ""
                row[f"{display} CI95 (%)"] = ""
                row[f"{display} Sig > Chance"] = ""
                row[f"{display} Sig < Chance"] = ""
                continue

            arr = np.asarray(values, dtype=np.float64)
            mean = float(arr.mean())
            std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
            ci_low, ci_high = _bootstrap_ci_mean(arr, num_samples=bootstrap_samples, seed=bootstrap_seed)
            row[f"{display} Mean (%)"] = _r2(mean)
            row[f"{display} Std (%)"] = _r2(std)
            row[f"{display} CI95 (%)"] = _ci_text(ci_low, ci_high)
            row[f"{display} Sig > Chance"] = bool(ci_low > chance_level_pct)
            row[f"{display} Sig < Chance"] = bool(ci_high < chance_level_pct)

        output_rows.append(row)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "Chance Level (%)",
        "Total Mean (%)",
        "Total Std (%)",
        "Total CI95 (%)",
        "Total Sig > Chance",
        "Total Sig < Chance",
    ]
    for metric_name in all_perturbation_metrics:
        display = _perturbation_display_name(metric_name)
        fieldnames.extend(
            [
                f"{display} Mean (%)",
                f"{display} Std (%)",
                f"{display} CI95 (%)",
                f"{display} Sig > Chance",
                f"{display} Sig < Chance",
            ]
        )

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in output_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    print(f"[INFO] Wrote analysis CSV to: {output_csv}")
    return 0


def _build_env(repo_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    env.setdefault("CUDA_LAUNCH_BLOCKING", "1")
    env.setdefault("HYDRA_FULL_ERROR", "1")

    old_pythonpath = env.get("PYTHONPATH", "")
    if old_pythonpath:
        env["PYTHONPATH"] = f"{repo_root}:{old_pythonpath}"
    else:
        env["PYTHONPATH"] = str(repo_root)
    return env


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Automatically evaluate all model variants under one method directory and append results to CSV."
    )
    parser.add_argument(
        "--method",
        default='sde_s',
        help="Method directory path (e.g. weights/SAVLA/scan_d) or shorthand method name (e.g. scan_d).",
    )
    parser.add_argument(
        "--config-name",
        default="libero_spatial_ppo_openpi_pi05_eval",
        help="Hydra config name used by examples/embodiment/eval_embodiment.sh",
    )
    parser.add_argument(
        "--eval-sh",
        default="examples/embodiment/eval_embodiment.sh",
        help="Path to eval bash script",
    )
    parser.add_argument(
        "--output-csv",
        default="outputs/ep120/eval_variants_results.csv",
        help="CSV file to append eval results",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=0,
        help="Timeout for each eval run in seconds (0 means no timeout)",
    )
    parser.add_argument(
        "--extra-override",
        action="append",
        default=[],
        help="Extra Hydra override, can be passed multiple times",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print commands, do not execute",
    )
    parser.add_argument(
        "--analyze-input-csv",
        type=str,
        default="",
        help="Read an existing eval CSV and generate statistical analysis instead of launching eval.",
    )
    parser.add_argument(
        "--analysis-output-csv",
        type=str,
        default="outputs/ep120/eval_variants_analysis.csv",
        help="Output CSV for statistical analysis results.",
    )
    parser.add_argument(
        "--chance-level",
        type=float,
        default=0.5,
        help="Chance level used for significance comparison.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2000,
        help="Number of bootstrap samples used for confidence intervals.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=0,
        help="Random seed for bootstrap confidence intervals.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    if args.analyze_input_csv:
        input_csv = (repo_root / args.analyze_input_csv).resolve()
        output_csv = (repo_root / args.analysis_output_csv).resolve()
        return analyze_success_metrics(
            input_csv=input_csv,
            output_csv=output_csv,
            chance_level=args.chance_level,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        )

    method_dir = _resolve_method_dir(repo_root, args.method)
    variants = _discover_variants(method_dir)
    if not variants:
        print(f"[ERROR] No variant directories found under: {method_dir}", file=sys.stderr)
        return 2

    csv_path = (repo_root / args.output_csv).resolve()
    base_fieldnames = [
        "timestamp",
        "method",
        "variant",
        "model_path",
        "config_name",
        "log_dir",
        "return_code",
        "eval_success_once",
        "metrics_json",
    ]
    _ensure_csv(csv_path, base_fieldnames)

    eval_sh = (repo_root / args.eval_sh).resolve()
    if not eval_sh.exists():
        print(f"[ERROR] Eval script not found: {eval_sh}", file=sys.stderr)
        return 2

    env = _build_env(repo_root)

    print(f"[INFO] method_dir={method_dir}")
    print(f"[INFO] discovered {len(variants)} variants: {[v.name for v in variants]}")

    for idx, variant_dir in enumerate(variants, start=1):
        ts = _now()
        method_name = method_dir.name
        variant_name = variant_dir.name
        log_dir = repo_root / "logs" / "eval_variants" / method_name / variant_name / ts
        log_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "bash",
            str(eval_sh),
            args.config_name,
            f"rollout.model.model_path={str(variant_dir)}",
            f"actor.model.model_path={str(variant_dir)}",
        ]
        cmd.extend(args.extra_override)

        print(f"\n[{idx}/{len(variants)}] Evaluating variant={variant_name}")
        print("[CMD] " + " ".join(cmd))

        if args.dry_run:
            continue

        try:
            run_env = env.copy()
            run_env["EMBODIED_LOG_DIR"] = str(log_dir)
            proc = subprocess.run(
                cmd,
                cwd=str(repo_root),
                env=run_env,
                text=True,
                timeout=None if args.timeout <= 0 else args.timeout,
            )
            # The eval bash script writes full logs into eval_embodiment.log under EMBODIED_LOG_DIR.
            # Keep a local copy as run.log for downstream parsing.
            eval_log_path = log_dir / "eval_embodiment.log"
            if eval_log_path.exists():
                raw_text = eval_log_path.read_text(encoding="utf-8", errors="ignore")
            else:
                raw_text = ""
            (log_dir / "run.log").write_text(raw_text, encoding="utf-8")

            metrics = _parse_eval_metrics(raw_text)
            row = {
                "timestamp": ts,
                "method": method_name,
                "variant": variant_name,
                "model_path": str(variant_dir),
                "config_name": args.config_name,
                "log_dir": str(log_dir),
                "return_code": proc.returncode,
                "eval_success_once": metrics.get("eval/success_once", ""),
                "metrics_json": json.dumps(metrics, ensure_ascii=False, sort_keys=True),
            }
            for key, value in metrics.items():
                row[_metric_col_name(key)] = value
            _append_csv(csv_path, list(row.keys()), row)

            if proc.returncode != 0:
                print(f"[WARN] Eval failed for variant={variant_name}, return_code={proc.returncode}", file=sys.stderr)

        except subprocess.TimeoutExpired:
            eval_log_path = log_dir / "eval_embodiment.log"
            if eval_log_path.exists():
                timeout_text = eval_log_path.read_text(encoding="utf-8", errors="ignore")
            else:
                timeout_text = ""
            (log_dir / "run.log").write_text(timeout_text, encoding="utf-8")
            row = {
                "timestamp": ts,
                "method": method_name,
                "variant": variant_name,
                "model_path": str(variant_dir),
                "config_name": args.config_name,
                "log_dir": str(log_dir),
                "return_code": "timeout",
                "eval_success_once": "",
                "metrics_json": "{}",
            }
            _append_csv(csv_path, list(row.keys()), row)
            print(f"[WARN] Timeout for variant={variant_name}", file=sys.stderr)

    print(f"\n[INFO] Done. CSV appended at: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
