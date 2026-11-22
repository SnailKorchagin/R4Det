#!/usr/bin/env bash
# Auto-tuned Colab launcher for R4Det x V2X-Radar-V Stage 4 training (with Drive resume).
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

ENV_DIR="${COLAB_ENV_DIR:-/content/r4det_env}"
WORK_DIR="${WORK_DIR:-/content/drive/MyDrive/R4Det_V2X/stage4_full_single_frame}"
INIT_CKPT="${INIT_CKPT:-checkpoints/final_tj4d.pth}"
MAX_EPOCHS="${MAX_EPOCHS:-12}"
LR="${LR:-2e-4}"
SEED="${SEED:-42}"
LOG_INTERVAL="${LOG_INTERVAL:-50}"

export PATH="${ENV_DIR}/bin:/usr/bin:/bin"
export CUDA_HOME="${ENV_DIR}"
export CC="${ENV_DIR}/bin/x86_64-conda-linux-gnu-cc"
export CXX="${ENV_DIR}/bin/x86_64-conda-linux-gnu-c++"
export TORCH_CUDA_ARCH_LIST="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d '[:space:]')"
export TORCH_EXTENSIONS_DIR="${REPO_DIR}/.cache/torch_extensions"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
export MPLCONFIGDIR="/tmp/r4det-colab-mpl"

GPU_MEM_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | tr -d '[:space:]')"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"

# Select VRAM-safe preset automatically (override via BATCH_SIZE / GRAD_ACCUM / FREEZE_MODULES)
if (( GPU_MEM_MIB >= 35000 )); then
  # A100 40GB/80GB
  BATCH_SIZE="${BATCH_SIZE:-4}"
  GRAD_ACCUM="${GRAD_ACCUM:-1}"
  FREEZE_ARGS=()
elif (( GPU_MEM_MIB >= 20000 )); then
  # L4 24GB
  BATCH_SIZE="${BATCH_SIZE:-2}"
  GRAD_ACCUM="${GRAD_ACCUM:-2}"
  FREEZE_ARGS=()
else
  # T4 16GB (or smaller)
  BATCH_SIZE="${BATCH_SIZE:-2}"
  GRAD_ACCUM="${GRAD_ACCUM:-2}"
  FREEZE_ARGS=(--freeze-modules depth_net --cfg-options model.freeze_depths=True model.use_depth_supervision=False)
fi

if [[ -n "${FREEZE_MODULES:-}" ]]; then
  read -r -a _mods <<< "${FREEZE_MODULES}"
  FREEZE_ARGS=(--freeze-modules "${_mods[@]}")
fi

mkdir -p "${WORK_DIR}"
if [[ -f "${WORK_DIR}/latest.pth" ]]; then
  echo "[Colab Train] Resuming from ${WORK_DIR}/latest.pth"
  CKPT_ARGS=(--resume-from "${WORK_DIR}/latest.pth")
elif [[ -f "${INIT_CKPT}" ]]; then
  echo "[Colab Train] Warm-starting from ${INIT_CKPT}"
  CKPT_ARGS=(--load-from "${INIT_CKPT}")
else
  echo "[Colab Train] Starting from scratch (no INIT_CKPT found at ${INIT_CKPT})"
  CKPT_ARGS=()
fi

echo "[Colab Train] GPU=${GPU_NAME} (${GPU_MEM_MIB} MiB) | batch=${BATCH_SIZE} x accum=${GRAD_ACCUM} | epochs=${MAX_EPOCHS} | work_dir=${WORK_DIR}"

"${ENV_DIR}/bin/python" tools/train_v2x_radar_v.py \
  --config configs/r4det/V2X-Radar-V-R4Det_single_frame.py \
  --work-dir "${WORK_DIR}" \
  "${CKPT_ARGS[@]}" \
  --max-epochs "${MAX_EPOCHS}" \
  --samples-per-gpu "${BATCH_SIZE}" \
  --grad-accum-steps "${GRAD_ACCUM}" \
  --workers-per-gpu 2 \
  --lr "${LR}" \
  --filter-empty-gt \
  --amp --amp-dtype fp16 \
  --seed "${SEED}" \
  --save-interval 1 \
  --log-interval "${LOG_INTERVAL}" \
  "${FREEZE_ARGS[@]}" \
  "$@"
