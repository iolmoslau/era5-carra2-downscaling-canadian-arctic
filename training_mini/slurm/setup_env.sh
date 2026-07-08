#!/bin/bash
# One-time environment setup for CorrDiff-Mini training on Fir. Run on a LOGIN node
# (internet access needed for physicsnemo, which is not in the Alliance wheelhouse).
#
#     bash training_mini/slurm/setup_env.sh
#     # to reuse your existing data-pipeline venv instead of a separate one:
#     ENV_DIR=$HOME/ENV bash training_mini/slurm/setup_env.sh
#
# Notes on two Alliance quirks handled below:
#  * mpi4py: the Alliance netCDF4 wheel is MPI-built and imports mpi4py, which is provided as
#    a MODULE (not pip) and must be loaded BEFORE activating the venv -- hence the module line.
#  * torch: Fir's wheelhouse has torch 2.12.1 (>= physicsnemo's 2.10). Installing physicsnemo
#    tends to replace that optimized build with a generic PyPI one, so we re-install the
#    Alliance torch/torchvision LAST to keep the optimized build (physicsnemo only needs >=2.10).

set -euo pipefail

ENV_DIR="${ENV_DIR:-$HOME/corrdiff-env}"

module load python/3.11 mpi4py/4.1.0    # mpi4py BEFORE activating (Alliance netCDF4 needs it)

if [[ ! -d "$ENV_DIR" ]]; then
  virtualenv --no-download "$ENV_DIR"
fi
source "$ENV_DIR/bin/activate"
pip install --no-index --upgrade pip

# Cluster-optimized torch first (so warp/physicsnemo build against a present torch).
pip install --no-index torch torchvision

# physicsnemo (+ warp-lang) from PyPI. This may swap torch for a PyPI build -- we fix it below.
pip install nvidia-physicsnemo

# Our deps + psutil (imported by train.py). cv2/opencv is intentionally NOT required: we dropped
# the cwb/gefs_hrrr readers from datasets/dataset.py (they were the only cv2 importers).
pip install "zarr>=3" dask "netCDF4>=1.7" xarray pandas numpy scipy numba psutil matplotlib \
            "hydra-core>=1.2" "omegaconf>=2.3" nvtx "cftime>=1.6" wandb tensorboard

# Restore the Alliance-optimized torch so it's the one that sticks (>=2.10 keeps physicsnemo happy).
pip install --no-index --force-reinstall torch torchvision

# ---- verify ----
python - <<'PY'
import torch, physicsnemo, hydra, zarr, xarray, netCDF4, wandb, psutil
print("torch", torch.__version__, "| cuda build:", torch.version.cuda)
v = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
assert v >= (2, 10), f"torch {torch.__version__} < 2.10 (physicsnemo needs >=2.10)"
assert torch.version.cuda, "torch has no CUDA build (CPU-only wheel) -- reinstall a CUDA torch"
print("physicsnemo", getattr(physicsnemo, "__version__", "?"), "| zarr", zarr.__version__)
print("ENV OK")
PY
echo "Environment ready: $ENV_DIR"
echo "Remember: jobs must 'module load python/3.11 mpi4py/4.1.0' before activating this venv."
