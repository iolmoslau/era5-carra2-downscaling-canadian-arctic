#!/bin/bash
# One-time environment setup for CorrDiff-Mini training on Fir (run on a LOGIN node,
# which has internet). Creates a venv with torch>=2.10 + physicsnemo + our deps.
#
#     bash training_mini/slurm/setup_env.sh
#
# KEY CONSTRAINT: physicsnemo requires **torch >= 2.10** (that's why the Kaggle P100 failed:
# torch 2.10 dropped older GPUs). So the venv's torch must be >= 2.10 AND built for Fir's CUDA.
#
# If the Alliance wheelhouse does NOT have torch>=2.10, the cleaner route on Fir is an
# Apptainer container from NGC instead of a venv, e.g.:
#     module load apptainer
#     apptainer pull physicsnemo.sif docker://nvcr.io/nvidia/physicsnemo/physicsnemo:<tag>
# and change the sbatch scripts to `apptainer exec --nv physicsnemo.sif torchrun ...`.
# Tell me which route Fir supports and I'll wire whichever you pick.

set -euo pipefail

ENV_DIR="${ENV_DIR:-$HOME/corrdiff-env}"

# ---- modules (CONFIRM against `module avail` on Fir) ----
module load python/3.11 cuda cudnn

# ---- venv ----
if [[ ! -d "$ENV_DIR" ]]; then
  virtualenv --no-download "$ENV_DIR" 2>/dev/null || python -m venv "$ENV_DIR"
fi
source "$ENV_DIR/bin/activate"
pip install --upgrade pip

# ---- torch >= 2.10 (physicsnemo requirement) ----
# Try the Alliance wheelhouse first; fall back to the PyPI CUDA wheel (login node has internet).
# If the wheelhouse torch is < 2.10, delete this and pin a cu wheel matching Fir's CUDA, e.g.:
#   pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install --no-index torch torchvision 2>/dev/null || pip install torch torchvision

python - <<'PY'
import torch
v = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
assert v >= (2, 10), f"torch {torch.__version__} < 2.10 -- physicsnemo needs >=2.10 (see notes above)"
print("torch OK:", torch.__version__)
PY

# ---- physicsnemo + the rest of our training deps ----
pip install nvidia-physicsnemo
pip install "zarr>=3" dask "netCDF4>=1.7" xarray pandas numpy scipy numba \
            "hydra-core>=1.2" "omegaconf>=2.3" nvtx "cftime>=1.6" wandb tensorboard

# ---- verify ----
python - <<'PY'
import physicsnemo, hydra, zarr, xarray, netCDF4, wandb
print("physicsnemo", getattr(physicsnemo, "__version__", "?"), "| zarr", zarr.__version__)
print("ENV OK")
PY
echo "Environment ready at: $ENV_DIR"
echo "Point the sbatch scripts' 'source ...' line at $ENV_DIR/bin/activate"
