#!/bin/bash
# Small test build (2 days) on the Alliance "Fir" cluster (SLURM).
# Submit from the repo root:   sbatch scripts/test_build_2day.sh
#
# ============================ FILL THESE IN ============================
#SBATCH --account=def-stockie         # <-- your Alliance allocation (def-<pi> / rrg-<pi>)
#SBATCH --mail-user=ioa4@sfu.ca  # <-- for notifications (or delete the mail lines)
# ======================================================================
#SBATCH --job-name=carra2_build_test
#SBATCH --time=03:00:00                  # generous: CDS queue latency dominates
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --output=logs/build_test_%j.out
#SBATCH --error=logs/build_test_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# ---- paths (FILL/CHECK) ----------------------------------------------------
REPO="$HOME/thesis/era5-carra2-downscaling-canadian-arctic"   # where you cloned the repo
STORE="$PROJECT/data/test_2day.zarr"               # output store (persistent)
WORKDIR="$SLURM_TMPDIR/work"                            # transient downloads (node-local, auto-cleaned)

# ---- environment (FILL: your module + venv setup) --------------------------
module load python/3.11 mpi4py/4.1.0
source ~/ENV/bin/activate   

# If Fir's compute nodes reach the internet only via a proxy, set it here:
# export http_proxy="http://PROXY:PORT"; export https_proxy="$http_proxy"

cd "$REPO"
mkdir -p logs "$(dirname "$STORE")" "$WORKDIR"

# ---- preflight: fail fast if there is no internet on the compute node ------
echo "Checking CDS connectivity..."
if ! curl -sSf -m 20 -o /dev/null https://cds.climate.copernicus.eu/api ; then
  echo "ERROR: cannot reach the CDS from this node (no internet on compute nodes?)." >&2
  echo "       Run the build on a login node instead, or set the proxy env vars above." >&2
  exit 1
fi
echo "OK."

# ---- the test build: 2 days -----------------------------------------------
python scripts/build_split.py \
  --store "$STORE" \
  --start 2013-01-21 --end 2013-01-22 \
  --work-dir "$WORKDIR" \
  --chunk-days 1

echo "Build finished. Inspect with:  python -c \"import xarray as xr; print(xr.open_zarr('$STORE'))\""
