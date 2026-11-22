#!/usr/bin/env bash
# Unpack full V2X-Radar-V.zip, verify integrity, convert all official splits, and run Stage 1 audit.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

ZIP_PATH="${1:-.downloads/V2X-Radar-V/V2X-Radar-V.zip}"
DATA_ROOT="${2:-data/V2X-Radar-V}"
EXPECTED_SHA256="3d7506ccd8716e70cccdb3ec034de42c439b2a01399da04bef3f6f6fafbd93df"

if [[ ! -f "${ZIP_PATH}" ]]; then
  echo "ERROR: ${ZIP_PATH} not found. Wait for download to finish." >&2
  exit 1
fi

echo "[1/4] Verifying SHA256 of ${ZIP_PATH}..."
ACTUAL_SHA256="$(sha256sum "${ZIP_PATH}" | awk '{print $1}')"
echo "  actual:   ${ACTUAL_SHA256}"
echo "  expected: ${EXPECTED_SHA256}"
if [[ "${ACTUAL_SHA256}" != "${EXPECTED_SHA256}" ]]; then
  echo "ERROR: SHA256 mismatch (archive may still be downloading)." >&2
  exit 1
fi

echo "[2/4] Extracting ${ZIP_PATH} into $(dirname "${DATA_ROOT}")..."
mkdir -p "$(dirname "${DATA_ROOT}")"
unzip -q -o "${ZIP_PATH}" -d "$(dirname "${DATA_ROOT}")"

if [[ -f ".downloads/V2X-Radar-V/ImageSets.zip" ]]; then
  unzip -q -o .downloads/V2X-Radar-V/ImageSets.zip -d "${DATA_ROOT}"
fi

echo "[3/4] Converting official splits (train, val, trainval, test) to MMDet3D .pkl..."
.venv/bin/python tools/data_converter/v2x_radar_v_converter.py \
  --data-root "${DATA_ROOT}" \
  --splits train val trainval test \
  --class-scheme tj4d_compat

echo "[4/4] Running full Stage 1 CPU audit on frame 000009..."
.venv/bin/python tools/inspect_v2x_radar_v.py \
  --data-root "${DATA_ROOT}" \
  --frame-id 000009 \
  --out-dir work_dirs/v2x_radar_v/inspection_full

echo "FULL_V2X_RADAR_V_READY"
