#!/bin/bash
# De-rotate CARRA2 HR winds (grid-relative -> earth-relative) for all shards, then recompute
# train-only normalization stats on the corrected winds. CPU/IO-only (no GPU).
#
# Submit:   sbatch scripts/derotate_shards.sh
# Override via env, e.g.:  DST_DIR=$PROJECT/data/derot sbatch scripts/derotate_shards.sh
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
  src="$SRC_DIR/shard_${y}.zarr"
  dst="$DST_DIR/shard_${y}.zarr"
  if [[ -d "$dst" ]]; then
    echo "== $y: already de-rotated ($dst), skipping"
    continue
  fi
  if [[ ! -d "$src" ]]; then
    echo "== $y: source missing ($src), skipping" >&2
    continue
  fi
  echo "== de-rotating $y -> $dst"
  python scripts/derotate_winds.py --src "$src" --dst "$dst"
done

echo "== recomputing train stats -> $STATS"
python training_mini/tools/make_stats.py --data-dir "$DST_DIR" \
  --years $TRAIN_YEARS --out "$STATS"

echo "DONE. Corrected (earth-relative wind) shards + stats in $DST_DIR"
echo "Retrain with: DATA_DIR=$DST_DIR STATS=$STATS OUTPUT_DIR=\$SCRATCH/corrdiff_mini_derot \\"
echo "              sbatch --gpus=h100:2 training_mini/slurm/train_regression.sh"
