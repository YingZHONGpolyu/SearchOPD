#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"
export OPD_REPO_ROOT="$repo_root"
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HUB_DISABLE_XET=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0

# Use the requested Conda environment, including its environment-local CUDA
# compatibility hook. Do not let uv recreate/synchronize a different .venv.
conda_root=${OPD_CONDA_ROOT:-/data/work/YingZhong/miniconda3}
source "$conda_root/etc/profile.d/conda.sh"
conda activate "${OPD_CONDA_ENV:-opd_env}"
python="$CONDA_PREFIX/bin/python"
"$python" examples/opd_search/preflight.py --check-models --check-retriever "$@"
# Export only non-secret W&B settings. The SDK reads credentials from the
# existing login or WANDB_API_KEY; never place them in Hydra overrides.
wandb_env_file=$(mktemp)
trap 'rm -f -- "$wandb_env_file"' EXIT
"$python" examples/opd_search/wandb_setup.py --env-file "$wandb_env_file" "$@"
source "$wandb_env_file"
"$python" -m verl.trainer.main_ppo \
    --config-path "$repo_root/examples/opd_search/config" \
    --config-name opd_search \
    "ray_kwargs.ray_init.runtime_env.py_executable=$python" \
    "$@"
