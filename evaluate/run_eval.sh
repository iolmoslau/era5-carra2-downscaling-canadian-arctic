#!/bin/bash
# Evaluate CorrDiff-Mini on N 2019 validation times: per-channel CRPS + MAE for the FULL model
# (regression mean + diffusion residual ensemble) vs the REGRESSION-ONLY net.
#
# Runs two generate.py passes (MODE=all, MODE=regression) over the same N times, then
# evaluate/eval_crps_mae.py. Submit via slurm/submit.sh so logs land in $REPO/logs:
#
#   N=100 NUM_ENS=15 OUTPUT_DIR=$SCRATCH/corrdiff_runs/diffusion_2 DATA_DIR=$PROJECT/data/derot \
#     bash training_mini/slurm/submit.sh evaluate/run_eval.sh
#
# Checkpoints: auto-picked as the highest-step .mdlus in $OUTPUT_DIR/checkpoints_{regression,
# diffusion}; override with REG_CKPT=/path RES_CKPT=/path (e.g. to pair a diffusion run with a
# regression from a DIFFERENT run dir). Cost/disk scale with N*NUM_ENS.
#
#SBATCH --account=def-stockie_gpu
#SBATCH --gpus=h100:1
#SBATCH --job-name=corrdiff_eval
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/corrdiff_eval_%j.out
#SBATCH --error=logs/corrdiff_eval_%j.err

set -euo pipefail

REPO="${REPO:-$HOME/thesis/era5-carra2-downscaling-canadian-arctic}"
TRAIN_DIR="$REPO/training_mini"
EVAL_DIR="$REPO/evaluate"
ENV_DIR="${ENV_DIR:-$HOME/corrdiff-env}"
DATA_DIR="${DATA_DIR:-$PROJECT/data/derot}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRATCH/corrdiff_mini}"     # holds checkpoints_regression / _diffusion
STATS="${STATS:-$DATA_DIR/stats_train_2011_2018.json}"
CONFIG="${CONFIG:-config_generate_era5_carra2_eval}"
NUM_ENS="${NUM_ENS:-15}"                               # ensemble members for the FULL model
N="${N:-${1:-50}}"                                     # number of RANDOM eval times to draw
SEED="${SEED:-0}"                                      # RNG seed: reproducible; both passes share it
YEARS="${YEARS:-2019}"                                 # space-separated year(s) to sample from
NAME="${NAME:-$(basename "$OUTPUT_DIR")}"              # run name -> results/<name>/eval
# Small, kept artifacts (metrics JSON) go with the run's tracked results in the repo.
RESULT_DIR="${RESULT_DIR:-$TRAIN_DIR/results/$NAME/eval}"
# Bulky NetCDFs (full.nc/reg.nc, several GB) stay on scratch -- too big for the $HOME repo quota,
# and .nc is gitignored anyway. Set NC_DIR=$RESULT_DIR if you really want them alongside metrics.
NC_DIR="${NC_DIR:-$OUTPUT_DIR/eval}"
NPROC="${SLURM_GPUS_ON_NODE:-1}"
(( N < 1 )) && N=1
YEARS_CSV=$(echo "$YEARS" | tr ' ' ',')                # "2018 2019" -> "2018,2019" for Hydra

# ---- sanity: log resolved paths, fail fast if $SCRATCH/$PROJECT were unset at submit -------
echo "[paths] OUTPUT_DIR=$OUTPUT_DIR  DATA_DIR=$DATA_DIR"
echo "        RESULT_DIR=$RESULT_DIR (metrics)  NC_DIR=$NC_DIR (NetCDFs)"
echo "        (SCRATCH=${SCRATCH:-<unset>} PROJECT=${PROJECT:-<unset>})"
for p in "$OUTPUT_DIR" "$DATA_DIR"; do
  case "$p" in
    /scratch/*|/project/*|/home/*) : ;;
    *) echo "ERROR: path '$p' is not absolute -- is \$SCRATCH/\$PROJECT set when you run sbatch?" >&2; exit 1 ;;
  esac
done

# ---- resolve checkpoints by highest step (nimg in filename), robust to copy mtimes ----------
if [[ -z "${REG_CKPT:-}" ]]; then
  REG_CKPT=$(ls "$OUTPUT_DIR"/checkpoints_regression/*.mdlus 2>/dev/null | sort -t. -k3 -n | tail -1 || true)
fi
if [[ -z "${RES_CKPT:-}" ]]; then
  RES_CKPT=$(ls "$OUTPUT_DIR"/checkpoints_diffusion/*.mdlus 2>/dev/null | sort -t. -k3 -n | tail -1 || true)
fi
if [[ -z "${REG_CKPT:-}" || ! -f "$REG_CKPT" ]]; then
  echo "ERROR: no regression checkpoint (set REG_CKPT or OUTPUT_DIR)" >&2; exit 1
fi
if [[ -z "${RES_CKPT:-}" || ! -f "$RES_CKPT" ]]; then
  echo "ERROR: no diffusion checkpoint (set RES_CKPT or OUTPUT_DIR) -- needed for the full model" >&2; exit 1
fi

module load python/3.11 mpi4py/4.1.0
source "$ENV_DIR/bin/activate"
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE
export CORRDIFF_LOG_DIR="$REPO/logs"
mkdir -p "$CORRDIFF_LOG_DIR" "$RESULT_DIR" "$NC_DIR"

cd "$TRAIN_DIR"
ln -sfn "$DATA_DIR" ./data

# Draw N random times from the requested year(s). Computed ONCE and reused for both the full
# and regression passes, so the two predictions are scored on identical times.
echo "[eval] sampling $N random times from year(s) {$YEARS}, seed $SEED"
TIMES=$(python "$EVAL_DIR/sample_times.py" --data-dir "$DATA_DIR" --years $YEARS --n "$N" --seed "$SEED")
echo "[eval] NUM_ENS=$NUM_ENS"
echo "[eval] times=$TIMES"
echo "[eval] REG_CKPT=$REG_CKPT"
echo "[eval] RES_CKPT=$RES_CKPT"

# shared generate.py args (torchrun so physicsnemo's DistributedManager sees RANK/WORLD_SIZE).
# Explicit random `times` list; null out times_range so generate.py uses the list.
COMMON=(torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC"
        generate.py --config-name="$CONFIG"
        hydra.run.dir="$CORRDIFF_LOG_DIR/hydra/${SLURM_JOB_ID:-manual}"
        ++dataset.data_path="$DATA_DIR"
        ++dataset.stats_path="$STATS"
        ++dataset.years="[$YEARS_CSV]"
        ++generation.times="$TIMES"
        ++generation.times_range=null)

echo "== FULL model (regression + diffusion, $NUM_ENS members) -> $NC_DIR/full.nc =="
"${COMMON[@]}" \
  ++generation.inference_mode=all \
  ++generation.num_ensembles="$NUM_ENS" \
  ++generation.io.reg_ckpt_filename="$REG_CKPT" \
  ++generation.io.res_ckpt_filename="$RES_CKPT" \
  ++generation.io.output_filename="$NC_DIR/full.nc"

echo "== REGRESSION only (deterministic mean) -> $NC_DIR/reg.nc =="
"${COMMON[@]}" \
  ++generation.inference_mode=regression \
  ++generation.num_ensembles=1 \
  ++generation.io.reg_ckpt_filename="$REG_CKPT" \
  ++generation.io.output_filename="$NC_DIR/reg.nc"

echo "== CRPS / MAE per channel: full vs reg =="
python "$EVAL_DIR/eval_crps_mae.py" \
  --nc full="$NC_DIR/full.nc" reg="$NC_DIR/reg.nc" \
  --out "$RESULT_DIR/metrics_crps_mae.json"

echo "DONE. metrics -> $RESULT_DIR/metrics_crps_mae.json   |   NetCDFs -> $NC_DIR"
