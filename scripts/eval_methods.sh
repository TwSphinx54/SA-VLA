#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_CFG="$REPO_ROOT/examples/embodiment/config/model/pi0_5.yaml"

# 输出文件（仅包含本次三种方法的结果）
RESULT_CSV="outputs/eval_variants_results_methods.csv"
ANALYSIS_CSV="outputs/eval_variants_analysis_methods.csv"

cd "$REPO_ROOT"

set_noise_method() {
	local method="$1"
	sed -i -E 's/^([[:space:]]*noise_method:[[:space:]]*).*/\1"'"$method"'"/' "$MODEL_CFG"
}

# 清理旧结果，确保分析针对本次运行
rm -f "$RESULT_CSV" "$ANALYSIS_CSV"

# 1) scan_d -> scan
echo "[INFO] Evaluating method=scan_d with noise_method=scan"
set_noise_method "scan"
python scripts/eval_all_variants.py \
	--method scan_d \
	--output-csv "$RESULT_CSV"

# 2) noise_s -> flow_noise
echo "[INFO] Evaluating method=noise_s with noise_method=flow_noise"
set_noise_method "flow_noise"
python scripts/eval_all_variants.py \
	--method noise_s \
	--output-csv "$RESULT_CSV"

# # 3) noise_d -> flow_noise
# echo "[INFO] Evaluating method=noise_d with noise_method=flow_noise"
# set_noise_method "flow_noise"
# python scripts/eval_all_variants.py \
# 	--method noise_d \
# 	--output-csv "$RESULT_CSV"

# 分析汇总
echo "[INFO] Analyzing CSV: $RESULT_CSV"
python scripts/eval_all_variants.py \
	--analyze-input-csv "$RESULT_CSV" \
	--analysis-output-csv "$ANALYSIS_CSV"

echo "[INFO] Done."
echo "[INFO] Results:  $RESULT_CSV"
echo "[INFO] Analysis: $ANALYSIS_CSV"