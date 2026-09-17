#!/bin/bash
# Archive shards to single-file zarr ZipStores, one array task per year. CPU/IO only, no GPU.
#
# WHY: a loose shard_YYYY.zarr is ~5,900 tiny chunk files. Staging nine of them to
# $SLURM_TMPDIR measured ~6 min EACH -- ~54 min per training submission, at ~13 MB/s, which is
# metadata cost rather than bandwidth. Nine single-file archives turn that into nine large
# sequential reads. Training is re-submitted repeatedly on the opportunistic queue, so this is
# paid back within the first couple of resubmissions and then keeps paying. (Audit P1-3.)
#
# Safe by default: the loose stores are KEPT. Archives are verified against them (time axis,
# mask, attrs, plus random timesteps), and only once you have run a training job off the
# archives is it worth coming back with REMOVE_SRC=1 to reclaim the inodes.
#
# Submit:
#     sbatch scripts/zip_shards.sh
# Different data dir / years:
#     DATA_DIR=$PROJECT/data/derot YEARS="2011 2012 2013" sbatch scripts/zip_shards.sh
# Once verified in a real training run, reclaim the inodes:
#     REMOVE_SRC=1 sbatch scripts/zip_shards.sh
#
# Resumable: a year whose .zarr.zip already exists is skipped, so a timed-out array just requeues.

#SBATCH --account=def-stockie_cpu
#SBATCH --job-name=zip_shards
#SBATCH --array=0-8%3
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH --output=logs/zip_shards_%A_%a.out
#SBATCH --error=logs/zip_shards_%A_%a.err

set -euo pipefail
export PYTHONUNBUFFERED=1

REPO="${REPO:-$HOME/thesis/era5-carra2-downscaling-canadian-arctic}"
ENV_DIR="${ENV_DIR:-$HOME/corrdiff-env}"
DATA_DIR="${DATA_DIR:-$PROJECT/data/derot}"
YEARS="${YEARS:-2011 2012 2013 2014 2015 2016 2017 2018 2019}"
REMOVE_SRC="${REMOVE_SRC:-0}"
VERIFY_SAMPLES="${VERIFY_SAMPLES:-3}"

# %3 caps concurrency: the tasks all hammer the same filesystem, and this is metadata-bound
# rather than CPU-bound, so more parallelism mostly buys contention.
read -r -a _years <<< "$YEARS"
idx="${SLURM_ARRAY_TASK_ID:-0}"
if (( idx >= ${#_years[@]} )); then
  echo "array index $idx beyond the ${#_years[@]} requested year(s) -- nothing to do"
  exit 0
fi
YEAR="${_years[$idx]}"
SRC="$DATA_DIR/shard_${YEAR}.zarr"

echo "[paths] DATA_DIR=$DATA_DIR  YEAR=$YEAR  REMOVE_SRC=$REMOVE_SRC"
case "$DATA_DIR" in
  /scratch/*|/project/*|/home/*) : ;;
  *) echo "ERROR: '$DATA_DIR' is not a valid absolute location -- is \$PROJECT set?" >&2; exit 1 ;;
esac

if [[ ! -d "$SRC" ]]; then
  if [[ -f "$SRC.zip" ]]; then
    echo "== $YEAR: already archived and the loose store is gone -- nothing to do"
    exit 0
  fi
  echo "== $YEAR: no loose store at $SRC, skipping" >&2
  exit 0
fi

module load python/3.11 mpi4py/4.1.0
source "$ENV_DIR/bin/activate"
cd "$REPO"
mkdir -p logs

CMD=(python scripts/zip_shard.py --src "$SRC" --skip-existing
     --verify --verify-samples "$VERIFY_SAMPLES")
# Deletion is gated on verification inside zip_shard.py; keep the loose store until a training
# job has actually read the archives.
[[ "$REMOVE_SRC" == "1" ]] && CMD+=(--remove-src)

echo "  ${CMD[*]}"
"${CMD[@]}"

echo "DONE year $YEAR"
