#!/usr/bin/env bash
set -euo pipefail

# switch_libero.sh
# Usage:
#   ./switch_libero.sh libero        [/opt/venv/openpi]
#   ./switch_libero.sh libero_plus   [/opt/venv/openpi]
#
# Notes:
# - Requires: uv (uv pip), mv, python
# - You may need sudo if /opt/venv/openpi is not writable.

MODE="${1:-}"
ROOT="${2:-/opt/venv/openpi}"

if [[ -z "$MODE" || ( "$MODE" != "libero" && "$MODE" != "libero_plus" ) ]]; then
  echo "Usage: $0 {libero|libero_plus} [ROOT_DIR]"
  echo "Example: $0 libero /opt/venv/openpi"
  exit 2
fi

LIBERO_DIR="${ROOT}/libero"
BASE_DIR="${ROOT}/libero_base"
PLUS_DIR="${ROOT}/libero_plus"

require_dir() {
  local d="$1"
  if [[ ! -d "$d" ]]; then
    echo "Error: directory not found: $d" >&2
    exit 1
  fi
}

ensure_not_exists() {
  local d="$1"
  if [[ -e "$d" ]]; then
    echo "Error: destination already exists (won't overwrite): $d" >&2
    exit 1
  fi
}

echo "[1/4] ROOT=${ROOT}, MODE=${MODE}"

# Uninstall current editable (or any) libero first (ignore if not installed)
echo "[2/4] Uninstalling current 'libero' (if installed) via uv pip..."
uv pip uninstall -y libero >/dev/null 2>&1 || true

# Sanity checks
require_dir "$LIBERO_DIR"

if [[ "$MODE" == "libero" ]]; then
  require_dir "$BASE_DIR"
  ensure_not_exists "$PLUS_DIR"

  echo "[3/4] Renaming directories:"
  echo "  ${LIBERO_DIR} -> ${PLUS_DIR}"
  echo "  ${BASE_DIR}   -> ${LIBERO_DIR}"
  mv "$LIBERO_DIR" "$PLUS_DIR"
  mv "$BASE_DIR" "$LIBERO_DIR"

elif [[ "$MODE" == "libero_plus" ]]; then
  require_dir "$PLUS_DIR"
  ensure_not_exists "$BASE_DIR"

  echo "[3/4] Renaming directories:"
  echo "  ${LIBERO_DIR} -> ${BASE_DIR}"
  echo "  ${PLUS_DIR}   -> ${LIBERO_DIR}"
  mv "$LIBERO_DIR" "$BASE_DIR"
  mv "$PLUS_DIR" "$LIBERO_DIR"
fi

# Reinstall editable from the (new) libero directory
echo "[4/4] Installing editable 'libero' from: ${LIBERO_DIR}"
uv pip install -e "$LIBERO_DIR"
