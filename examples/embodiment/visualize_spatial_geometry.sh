#!/usr/bin/env bash
set -euo pipefail

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname "$(dirname "$EMBODIED_PATH")")
export PYTHONPATH=${REPO_PATH}:$PYTHONPATH
export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1

CONFIG_NAME="${1:-libero_spatial_ppo_openpi_pi05_eval}"
EXP_LABEL="${2:-default}"
shift || true
shift || true
EXTRA_OVERRIDES=("$@")

# Fixed reset state for the legacy target task:
# "pick up the black bowl on the stove and place it on the plate view 318 15 100 0 0 initstate 0"
FIXED_RESET_ID="44600"

OUT_DIR="${REPO_PATH}/outputs/spatial_geometry/${EXP_LABEL}/$(date +'%Y%m%d-%H:%M:%S')"
mkdir -p "${OUT_DIR}"

CMD=(
  python "${REPO_PATH}/scripts/visualize_spatial_geometry.py"
  --mode direct
  --config-dir "${EMBODIED_PATH}/config"
  --config-name "${CONFIG_NAME}"
  --output-dir "${OUT_DIR}"
  --rollout-steps 48
  --noise-mode train
  --specific-reset-id "${FIXED_RESET_ID}"
  --override actor.model.openpi.noise_method=scan
)

if [ ${#EXTRA_OVERRIDES[@]} -gt 0 ]; then
  for OVERRIDE in "${EXTRA_OVERRIDES[@]}"; do
    CMD+=(--override "$OVERRIDE")
  done
fi

echo "${CMD[*]}"
"${CMD[@]}"
