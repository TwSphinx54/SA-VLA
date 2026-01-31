import argparse
import re
import subprocess
import sys
from pathlib import Path

DEFAULT_CFG_NAME = "libero_spatial_ppo_openpi_pi05_eval"
DEFAULT_CFG_PATH = Path("examples/embodiment/config/libero_spatial_ppo_openpi_pi05_eval.yaml")
DEFAULT_EVAL_SH = Path("examples/embodiment/eval_embodiment.sh")

PATTERN = re.compile(r"(checkpoints/global_step_)\d+(/actor/model)")
MODEL_PATH_METHOD_PATTERN = re.compile(r'(model_path:\s*")logs/[^/]+/')

def render_config(text: str, step: int, method: str | None) -> tuple[str, int, int]:
    n_method = 0
    if method:
        text, n_method = MODEL_PATH_METHOD_PATTERN.subn(rf'\1logs/{method}/', text)
    text, n_step = PATTERN.subn(rf"\g<1>{step}\2", text)
    return text, n_method, n_step

def patch_config(cfg_path: Path, step: int) -> int:
    text = cfg_path.read_text(encoding="utf-8")
    new_text, n = PATTERN.subn(rf"\g<1>{step}\2", text)
    cfg_path.write_text(new_text, encoding="utf-8")
    return n

def _normalize_methods(methods: list[str] | None) -> list[str | None]:
    if not methods:
        return [None]
    if len(methods) == 1 and "," in methods[0]:
        methods = [m.strip() for m in methods[0].split(",") if m.strip()]
    return methods

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", default=DEFAULT_CFG_NAME,
                    help="Config name passed to eval_embodiment.sh (without .yaml)")
    ap.add_argument("--config-path", type=Path, default=DEFAULT_CFG_PATH,
                    help="Path to yaml file to modify")
    ap.add_argument("--eval-sh", type=Path, default=DEFAULT_EVAL_SH,
                    help="Path to eval script")
    ap.add_argument("--methods", nargs="*", default=['scan_d_seed42', 'noise_d_seed42', 'noise_s_seed42'],
                    help='List of logs/<method_name>/ to evaluate; supports space-separated or single comma-separated string, e.g.: --methods cps_s foo or --methods cps_s,foo')
    ap.add_argument("--start", type=int, default=10)
    ap.add_argument("--end", type=int, default=100)
    ap.add_argument("--step", type=int, default=10)
    args = ap.parse_args()

    if not args.config_path.exists():
        print(f"Config not found: {args.config_path}", file=sys.stderr)
        return 2
    if not args.eval_sh.exists():
        print(f"Eval script not found: {args.eval_sh}", file=sys.stderr)
        return 2

    base_text = args.config_path.read_text(encoding="utf-8")
    methods = _normalize_methods(args.methods)

    for method in methods:
        method_tag = method if method else "<keep>"
        for gs in range(args.start, args.end + 1, args.step):
            new_text, n_method, n_step = render_config(base_text, gs, method)
            args.config_path.write_text(new_text, encoding="utf-8")

            if method and n_method != 2:
                print(f"[WARN] expected to patch 2 model_path method occurrences, but patched {n_method} (method={method_tag}).", file=sys.stderr)
            if n_step != 2:
                print(f"[WARN] expected to patch 2 global_step occurrences, but patched {n_step} (global_step={gs}).", file=sys.stderr)

            print(f"=== Running eval for method={method_tag} global_step_{gs} ===")
            cmd = ["bash", str(args.eval_sh), args.config_name]
            subprocess.run(cmd, check=True)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())