#!/bin/bash
# Parallel per-year build on Fir: each array task builds ONE year into its own zarr shard.
# Parallel CDS connections multiply throughput (CDS throttles per-connection).
# Submit:   sbatch scripts/build_years_array.sh
#
# ====================== FILL THESE IN ======================
#SBATCH --account=def-stockie          # <-- your allocation
#SBATCH --mail-user=ioa4@sfu.ca        # <-- or delete the mail lines
# ===========================================================
# Array: one task per year. 0-7 => 8 years. %4 = at most 4 running at once.
# Keep the concurrency (%N) at or below your CDS active-request limit.
#SBATCH --array=0-8%4
#SBATCH --job-name=carra2_year
#SBATCH --time=24:00:00                 # per year; resumable, so a timeout just requeues
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=logs/build_year_%A_%a.out
#SBATCH --error=logs/build_year_%A_%a.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# ---- config (FILL/CHECK) ---------------------------------------------------
REPO="$HOME/thesis/era5-carra2-downscaling-canadian-arctic"
OUT="$PROJECT/data"              # one shard_YYYY.zarr per year goes here
START_YEAR=2011                        # year for array index 0 (index N => START_YEAR+N)
WORKDIR="$SLURM_TMPDIR/work"            # node-local transient downloads (auto-cleaned)

# ---- environment -----------------------------------------------------------
module load python/3.11 mpi4py/4.1.0   # <-- use the mpi4py version from `module avail mpi4py`
source ~/ENV/bin/activate

cd "$REPO"
mkdir -p logs "$OUT" "$WORKDIR"

YEAR=$(( START_YEAR + SLURM_ARRAY_TASK_ID ))
STORE="$OUT/shard_${YEAR}.zarr"
echo "Task ${SLURM_ARRAY_TASK_ID}: building year ${YEAR} -> ${STORE}"

# preflight: need internet for the CDS
curl -sSf -m 20 -o /dev/null https://cds.climate.copernicus.eu/api \
  || { echo "ERROR: no CDS access from this node" >&2; exit 1; }

# whole-month writes (fewer zarr appends; one month ~580 MB in RAM). Resumable: re-running
# the same array index appends only the months still missing from the shard.
python scripts/build_split.py \
  --store "$STORE" \
  --start "${YEAR}-01-01" --end "${YEAR}-12-31" \
  --work-dir "$WORKDIR" \
  --chunk-days 31

echo "DONE year ${YEAR}"
