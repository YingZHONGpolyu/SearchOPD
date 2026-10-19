#!/usr/bin/env bash
# Isolate FAISS/CUDA 12 inside opd_env without replacing trainer dependencies.
set -euo pipefail
conda_root=${OPD_CONDA_ROOT:-/data/work/YingZhong/miniconda3}
source "$conda_root/etc/profile.d/conda.sh"
conda activate "${OPD_CONDA_ENV:-opd_env}"
"$CONDA_PREFIX/bin/uv" pip install \
    --python "$CONDA_PREFIX/bin/python" \
    --target "$CONDA_PREFIX/retriever_gpu" --no-deps \
    --index-url "${OPD_PYPI_INDEX:-https://mirrors.aliyun.com/pypi/simple}" \
    faiss-gpu-cu12==1.14.1.post1 \
    nvidia-cuda-runtime-cu12==12.1.105 \
    nvidia-cublas-cu12==12.1.3.1
