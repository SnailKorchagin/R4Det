#!/usr/bin/env bash
# Set up isolated Python 3.8 + PyTorch 2.0.1+cu118 + MMCV 1.7.2 + mmdet3d ops on Google Colab.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${1:-/content/r4det_env}"
MAMBA_BIN="${REPO_DIR}/.tools/micromamba"

cd "${REPO_DIR}"
mkdir -p "${REPO_DIR}/.tools" "${REPO_DIR}/.cache/torch_extensions"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: No NVIDIA GPU detected. Switch Colab runtime to GPU (T4/L4/A100)." >&2
  exit 1
fi

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
GPU_ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d '[:space:]')"
echo "[Colab Setup] GPU: ${GPU_NAME} (sm_${GPU_ARCH//./})"

if [[ ! -x "${MAMBA_BIN}" ]]; then
  echo "[1/4] Downloading micromamba..."
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C "${REPO_DIR}/.tools" --strip-components=1 bin/micromamba
fi

if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
  echo "[2/4] Creating Python 3.8 + CUDA 11.8 toolchain env at ${ENV_DIR}..."
  "${MAMBA_BIN}" create -y -p "${ENV_DIR}" \
    -c nvidia/label/cuda-11.8.0 -c conda-forge \
    python=3.8 pip ninja gcc_linux-64=11 gxx_linux-64=11 \
    cuda-nvcc=11.8 cuda-cudart-dev=11.8 cuda-libraries-dev=11.8
else
  echo "[2/4] Reusing existing env at ${ENV_DIR}."
fi

export PATH="${ENV_DIR}/bin:/usr/bin:/bin"
export CUDA_HOME="${ENV_DIR}"
export CC="${ENV_DIR}/bin/x86_64-conda-linux-gnu-cc"
export CXX="${ENV_DIR}/bin/x86_64-conda-linux-gnu-c++"
export TORCH_CUDA_ARCH_LIST="${GPU_ARCH}"
export TORCH_EXTENSIONS_DIR="${REPO_DIR}/.cache/torch_extensions"

PY="${ENV_DIR}/bin/python"
PIP="${ENV_DIR}/bin/pip"

echo "[3/4] Installing PyTorch 2.0.1+cu118, MMCV 1.7.2, MMDet 2.28.2, and repo requirements..."
"${PIP}" install -q --upgrade "pip<24.1" "setuptools==60.2.0" wheel
"${PIP}" install -q torch==2.0.1+cu118 torchvision==0.15.2+cu118 \
  --index-url https://download.pytorch.org/whl/cu118
"${PIP}" install -q mmcv-full==1.7.2 \
  -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.0/index.html
"${PIP}" install -q mmdet==2.28.2 mmsegmentation==0.30.0 Pyro4 \
  lyft-dataset-sdk==0.0.8 terminaltables pycocotools descartes fire networkx pandas \
  fvcore iopath yacs termcolor cloudpickle addict hydra-core omegaconf
"${PIP}" install -q natten==0.14.6 --no-build-isolation
FORCE_CUDA=1 "${PIP}" install -q -e "${REPO_DIR}/detr2" --no-build-isolation --no-deps
"${PIP}" install -q -r "${REPO_DIR}/requirements.txt"

echo "[4/4] Building mmdet3d CUDA extensions for sm_${GPU_ARCH//./}..."
FORCE_CUDA=1 "${PIP}" install -q -e "${REPO_DIR}" --no-build-isolation --no-deps

"${PY}" -c "import torch, mmcv, mmdet, mmdet3d; from mmdet3d.models import build_model; print('COLAB_ENV_OK', torch.__version__, 'cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0), 'mmcv:', mmcv.__version__)"
