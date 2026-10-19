#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"
# The requested environment is already installed, including host-specific
# CUDA/glibc compatibility and the documented lockfile dependency corrections.
# Reuse and verify it; a raw uv sync would undo those corrections.
conda_root=${OPD_CONDA_ROOT:-/data/work/YingZhong/miniconda3}
source "$conda_root/etc/profile.d/conda.sh"
conda activate "${OPD_CONDA_ENV:-opd_env}"
python="$CONDA_PREFIX/bin/python"
echo "Verifying existing Conda environment: $CONDA_PREFIX"
"$CONDA_PREFIX/bin/uv" pip check --python "$python"
"$python" examples/opd_search/preflight.py --runtime-only
