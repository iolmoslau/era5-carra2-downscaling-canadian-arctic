#!/bin/bash
# De-rotate CARRA2 HR winds (grid-relative -> earth-relative) for all shards, then recompute
# train-only normalization stats on the corrected winds. CPU/IO-only (no GPU).
#
# Submit:   sbatch scripts/derotate_shards.sh
# Override via env, e.g.:  DST_DIR=$PROJECT/data/derot sbatch scripts/derotate_shards.sh
#
# De-rotate extra TEST years to scratch without touching the fixed train stats (SKIP_STATS=1):
#   DST_DIR=$SCRATCH/data/derot YEARS="2020 2021 2022" SKIP_STATS=1 sbatch scripts/derotate_shards.sh
#
# Resumable: re-submitting skips shards already written to $DST_DIR.

#SBATCH --account=def-stockie_cpu
#SBATCH --job-name=derotate_winds
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --output=logs/derotate_%j.out
#SBATCH --error=logs/derotate_%j.err
#SBATCH --mail-user=ioa4@sfu.ca          # <-- or delete these two lines
#SBATCH --mail-type=END,FAIL

set -euo pipefail
export PYTHONUNBUFFERED=1

REPO="${REPO:-$HOME/thesis/era5-carra2-downscaling-canadian-arctic}"
ENV_DIR="${ENV_DIR:-$HOME/corrdiff-env}"
SRC_DIR="${SRC_DIR:-$PROJECT/data}"                 # original shard_YYYY.zarr
DST_DIR="${DST_DIR:-$PROJECT/data/derot}"           # corrected shards go here (originals untouched)
YEARS="${YEARS:-2011 2012 2013 2014 2015 2016 2017 2018 2019}"
TRAIN_YEARS="${TRAIN_YEARS:-2011 2012 2013 2014 2015 2016 2017 2018}"
STATS="${STATS:-$DST_DIR/stats_train_2011_2018.json}"

module load python/3.11 mpi4py/4.1.0
source "$ENV_DIR/bin/activate"

cd "$REPO"
mkdir -p logs "$DST_DIR"

for y in $YEARS; do
  dst="$DST_DIR/shard_${y}.zarr"
  # Done if the loose store OR its archived form is present: zipping a de-rotated shard to
  # reclaim project inodes (see evaluate/README.md) must not make it look un-de-rotated and
  # trigger a costly rebuild on the next submission.
  if [[ -d "$dst" || -f "$dst.zip" ]]; then
    echo "== $y: already de-rotated ($(basename "$dst")[.zip]), skipping"
    continue
  fi
  # Source: prefer an archived shard_YYYY.zarr.zip, matching how the dataset loader resolves
  # shards -- derotate_winds.py reads either form.
  src="$SRC_DIR/shard_${y}.zarr"
  if [[ -f "$src.zip" ]]; then
    src="$src.zip"
  elif [[ ! -d "$src" ]]; then
    echo "== $y: source missing ($src[.zip]), skipping" >&2
    continue
  fi
  echo "== de-rotating $y ($(basename "$src")) -> $dst"
  python scripts/derotate_winds.py --src "$src" --dst "$dst"
done

# Recompute train-only stats -- SKIP when de-rotating extra *test* years into a different DST_DIR
# that doesn't hold the 2011-2018 train shards (e.g. writing eval years to $SCRATCH). The train
# stats are fixed and already exist; the new years must NOT enter them anyway.
if [[ "${SKIP_STATS:-0}" == "1" ]]; then
  echo "== SKIP_STATS=1: leaving train stats untouched ($STATS)"
else
  echo "== recomputing train stats -> $STATS"
  python training_mini/tools/make_stats.py --data-dir "$DST_DIR" \
    --years $TRAIN_YEARS --out "$STATS"
fi

echo "DONE. Corrected (earth-relative wind) shards in $DST_DIR"
echo "Retrain with: DATA_DIR=$DST_DIR STATS=$STATS OUTPUT_DIR=\$SCRATCH/corrdiff_mini_derot \\"
echo "              sbatch --gpus=h100:2 training_mini/slurm/train_regression.sh"
