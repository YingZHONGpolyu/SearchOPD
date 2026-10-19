#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
search_r1=${SEARCH_R1_ROOT:-/data/work/YingZhong/code/Search-R1}
data_dir=${SEARCH_DATA_DIR:-/data/work/YingZhong/data/search_data}
conda_root=${OPD_CONDA_ROOT:-/data/work/YingZhong/miniconda3}
source "$conda_root/etc/profile.d/conda.sh"
conda activate "${OPD_CONDA_ENV:-opd_env}"
python=${RETRIEVER_PYTHON:-$CONDA_PREFIX/bin/python}
model_root=${OPD_MODEL_ROOT:-/data/work/YingZhong/models}
retriever_model=${RETRIEVER_MODEL:-intfloat/e5-base-v2}
if [[ -z "${RETRIEVER_MODEL:-}" && -f "$model_root/e5-base-v2/model.safetensors" ]]; then
    retriever_model="$model_root/e5-base-v2"
fi
export CUDA_VISIBLE_DEVICES=${RETRIEVER_GPU:-7}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-16}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-16}
export TOKENIZERS_PARALLELISM=false
faiss_device=${RETRIEVER_FAISS_DEVICE:-auto}
gpu_args=()
gpu_packages="$CONDA_PREFIX/retriever_gpu"
if [[ "$faiss_device" == gpu || ( "$faiss_device" == auto && -d "$gpu_packages/faiss" ) ]]; then
    if [[ ! -d "$gpu_packages/faiss" ]]; then
        echo "Run examples/opd_search/install_retriever_gpu.sh before selecting GPU FAISS." >&2
        exit 1
    fi
    # Only this service sees CUDA 12 FAISS. The trainer keeps its CUDA 13 stack.
    export PYTHONPATH="$gpu_packages${PYTHONPATH:+:$PYTHONPATH}"
    gpu_args=(--faiss_gpu)
elif [[ "$faiss_device" != auto && "$faiss_device" != cpu ]]; then
    echo "RETRIEVER_FAISS_DEVICE must be auto, cpu or gpu." >&2
    exit 2
fi
cd "$search_r1"
# The GPU path preserves the original FP32 index and shares GPU 7 with E5.
exec "$python" "$repo_root/examples/opd_search/batched_retriever.py" \
    --search-r1-root "$search_r1" \
    --index_path "$data_dir/e5_Flat.index" \
    --corpus_path "$data_dir/wiki-18.jsonl" \
    --retriever_name e5 \
    --retriever_model "$retriever_model" \
    --topk 3 \
    --batch-queries "${RETRIEVER_BATCH_QUERIES:-32}" \
    --threads "$OMP_NUM_THREADS" "${gpu_args[@]}" "$@"
