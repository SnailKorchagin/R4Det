#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${repo_dir}"
export PATH="${repo_dir}/.venv/bin:/usr/bin:/bin"
export CUDA_HOME="${repo_dir}/.venv"
export CC="${repo_dir}/.venv/bin/x86_64-conda-linux-gnu-cc"
export CXX="${repo_dir}/.venv/bin/x86_64-conda-linux-gnu-c++"
export TORCH_EXTENSIONS_DIR="${repo_dir}/.cache/torch_extensions"
export MPLCONFIGDIR="/tmp/r4det-walkthrough-mpl"
export TORCH_CUDA_ARCH_LIST="8.9"

.venv/bin/python tools/prepare_r4det_public_smoke.py
.venv/bin/python code_walkthrough/generate_r4det_storyboard.py
