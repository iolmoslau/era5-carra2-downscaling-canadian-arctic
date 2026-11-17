#!/bin/bash
# BCSD benchmark on fir: fit on the train years, downscale the test times, score.  CPU only.
#
# Takes its times from a model's eval NetCDF, so BCSD is scored on EXACTLY the same times as the
# model it is compared with -- and checks that its truth matches that file's truth group, which is
# what catches a raw-vs-de-rotated store mix-up before it becomes a wrong table in the thesis:
#
#   NAME=bcsd TIMES_FROM=$SCRATCH/corrdiff_runs/diffusion_2/eval/full.nc \
#     bash training_mini/slurm/submit.sh benchmarks/run_bcsd.sh
#
# If that NetCDF is gone, re-draw the times the way run_eval.sh does (same YEARS, N, SEED):
#
#   NAME=bcsd YEARS="2020 2021 2022" N=400 SEED=0 \
#     bash training_mini/slurm/submit.sh benchmarks/run_bcsd.sh
#
# The fit (~1 h, 8 years) is reused if $FIT exists; delete it or set REFIT=1 to refit.
# Metrics -> training_mini/results/<NAME>/eval/metrics_crps_mae.json (commit these).
# NetCDF  -> $NC_DIR/bcsd.nc (several GB, on scratch).
#
# To score BCSD alongside the models in one table afterwards (login node is fine):
#   python evaluate/eval_crps_mae.py --nc full=.../full.nc reg=.../reg.nc bcsd=$NC_DIR/bcsd.nc
#
#SBATCH --account=def-stockie_cpu
#SBATCH --job-name=bcsd
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/bcsd_%j.out
#SBATCH --error=logs/bcsd_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE

REPO="${REPO:-$HOME/thesis/era5-carra2-downscaling-canadian-arctic}"
ENV_DIR="${ENV_DIR:-$HOME/corrdiff-env}"
DATA_DIR="${DATA_DIR:-$PROJECT/data/derot}"
TRAIN_YEARS="${TRAIN_YEARS:-2011 2012 2013 2014 2015 2016 2017 2018}"
NAME="${NAME:-}"
WORK="${WORK:-$SCRATCH/benchmarks/bcsd}"
FIT="${FIT:-$WORK/fit_$(echo "$TRAIN_YEARS" | awk '{print $1"_"$NF}').npz}"
NC_DIR="${NC_DIR:-$WORK}"
RESULT_DIR="${RESULT_DIR:-$REPO/training_mini/results/$NAME/eval}"
TIMES_FROM="${TIMES_FROM:-}"
YEARS="${YEARS:-}"
N="${N:-400}"
SEED="${SEED:-0}"

if [[ -z "$NAME" ]]; then
  echo "ERROR: set NAME=<results name> (e.g. NAME=bcsd) -- metrics go to results/<NAME>/eval." >&2
  exit 1
fi
if [[ -n "$TIMES_FROM" && -n "$YEARS" ]] || [[ -z "$TIMES_FROM" && -z "$YEARS" ]]; then
  echo "ERROR: set exactly one of TIMES_FROM=<model eval .nc> or YEARS=\"2020 2021 2022\"." >&2
  exit 1
fi
if [[ -n "$TIMES_FROM" && ! -f "$TIMES_FROM" ]]; then
  echo "ERROR: TIMES_FROM=$TIMES_FROM does not exist (scratch purge?). Use YEARS/N/SEED." >&2
  exit 1
fi

echo "[paths] DATA_DIR=$DATA_DIR  FIT=$FIT"
echo "        NC_DIR=$NC_DIR  RESULT_DIR=$RESULT_DIR"
echo "        TRAIN_YEARS=$TRAIN_YEARS  times: ${TIMES_FROM:-YEARS=$YEARS N=$N SEED=$SEED}"

module load python/3.11 mpi4py/4.1.0
source "$ENV_DIR/bin/activate"
cd "$REPO"
mkdir -p "$WORK" "$NC_DIR" "$RESULT_DIR"

if [[ -f "$FIT" && "${REFIT:-0}" != "1" ]]; then
  echo "== reusing fit $FIT =="
else
  echo "== FIT on $TRAIN_YEARS =="
  python benchmarks/bcsd.py fit --data-dir "$DATA_DIR" --years $TRAIN_YEARS --out "$FIT"
fi

echo "== PREDICT -> $NC_DIR/bcsd.nc =="
if [[ -n "$TIMES_FROM" ]]; then
  python benchmarks/bcsd.py predict --fit "$FIT" --data-dir "$DATA_DIR" \
    --times-from "$TIMES_FROM" --out "$NC_DIR/bcsd.nc"
else
  python benchmarks/bcsd.py predict --fit "$FIT" --data-dir "$DATA_DIR" \
    --years $YEARS --n "$N" --seed "$SEED" --out "$NC_DIR/bcsd.nc"
fi

echo "== SCORE =="
python evaluate/eval_crps_mae.py --nc bcsd="$NC_DIR/bcsd.nc" --out "$RESULT_DIR/metrics_crps_mae.json"
echo "DONE. metrics -> $RESULT_DIR/metrics_crps_mae.json   |   NetCDF -> $NC_DIR/bcsd.nc"
