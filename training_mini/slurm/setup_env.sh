#!/bin/bash
# One-time environment setup for CorrDiff-Mini training on Fir. Run on a LOGIN node
# (internet access needed for physicsnemo, which is not in the Alliance wheelhouse).
#
#     bash training_mini/slurm/setup_env.sh
#     # to reuse your existing data-pipeline venv instead of a separate one:
#     ENV_DIR=$HOME/ENV bash training_mini/slurm/setup_env.sh
#
# Fir wheelhouse has torch 2.12.1 (>= physicsnemo's 2.10 requirement), so we install the
# cluster-optimized torch with --no-index and only reach out to PyPI for physicsnemo + friends.

set -euo pipefail

ENV_DIR="${ENV_DIR:-$HOME/corrdiff-env}"

module load python/3.11            # Fir: python/3.11.5

if [[ ! -d "$ENV_DIR" ]]; then
  virtualenv --no-download "$ENV_DIR"
fi
source "$ENV_DIR/bin/activate"
pip install --no-index --upgrade pip

# Cluster-optimized torch/torchvision from the Alliance wheelhouse (torch 2.12.1).
pip install --no-index torch torchvision
python - <<'PY'
import torch
v = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
assert v >= (2, 10), f"torch {torch.__version__} < 2.10 -- physicsnemo needs >=2.10"
print("torch OK:", torch.__version__)
PY

# physicsnemo (+ warp-lang) are not in the wheelhouse -> from PyPI. The rest are our light deps.
pip install nvidia-physicsnemo
pip install "zarr>=3" dask "netCDF4>=1.7" xarray pandas numpy scipy numba \
            "hydra-core>=1.2" "omegaconf>=2.3" nvtx "cftime>=1.6" wandb tensorboard

python - <<'PY'
import physicsnemo, hydra, zarr, xarray, netCDF4, wandb
print("physicsnemo", getattr(physicsnemo, "__version__", "?"), "| zarr", zarr.__version__)
print("ENV OK")
PY
echo "Environment ready: $ENV_DIR"
echo "The sbatch scripts default to \$HOME/corrdiff-env; set ENV_DIR to override."
