#!/bin/bash
# Evaluate CorrDiff-Mini on N 2019 validation times: per-channel CRPS + MAE for the FULL model
# (regression mean + diffusion residual ensemble) vs the REGRESSION-ONLY net.
#
# Runs two generate.py passes (MODE=all, MODE=regression) over the same N times, then
# evaluate/eval_crps_mae.py. Submit via slurm/submit.sh so logs land in $REPO/logs:
#
#   NAME=diffusion_2 N=100 NUM_ENS=15 \
#     REG_RUN=$SCRATCH/corrdiff_runs/regression_2 \
#     RES_RUN=$SCRATCH/corrdiff_runs/diffusion_2 \
#     DATA_DIR=$PROJECT/data/derot bash training_mini/slurm/submit.sh evaluate/run_eval.sh
#
# TWO run dirs, because generation is `regression mean + diffusion residual` and WORKFLOW.md B
# trains a diffusion run against an EARLIER regression run -- so a diffusion run dir holds only
# checkpoints_diffusion. Each run's highest-step .mdlus is picked (by nimg, not mtime):
#   REG_RUN -> $REG_RUN/checkpoints_regression      RES_RUN -> $RES_RUN/checkpoints_diffusion
# Name a specific checkpoint with REG_CKPT=/path / RES_CKPT=/path instead. OUTPUT_DIR still
# works and sets both, for a directory that holds both nets.
#
# NAME is REQUIRED: metrics go to training_mini/results/<NAME>/eval/. It is NOT guessed from the
# run dirs, so an oddly-named scratch dir can't silently write to the wrong results folder.
# Cost/disk scale with N*NUM_ENS.
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
source "$REPO/training_mini/slurm/common.sh"   # resolve_eval_paths, latest_ckpt, seed_batch_for
TRAIN_DIR="$REPO/training_mini"
EVAL_DIR="$REPO/evaluate"
ENV_DIR="${ENV_DIR:-$HOME/corrdiff-env}"
DATA_DIR="${DATA_DIR:-$PROJECT/data/derot}"
# Run dirs: REG_RUN holds checkpoints_regression/, RES_RUN holds checkpoints_diffusion/. They
# are normally different runs. OUTPUT_DIR is the legacy fallback that sets both.
REG_RUN="${REG_RUN:-}"
RES_RUN="${RES_RUN:-}"
STATS="${STATS:-$DATA_DIR/stats_train_2011_2018.json}"
CONFIG="${CONFIG:-config_generate_era5_carra2_eval}"
NUM_ENS="${NUM_ENS:-15}"                               # ensemble members for the FULL model
# Members denoised per sampler call (the FULL pass only; the regression pass is 1 member, so it
# is pinned to 1). Defaults to the largest divisor of NUM_ENS that is <= 8, and must divide
# NUM_ENS exactly -- see seed_batch_for/require_divisor in training_mini/slurm/common.sh.
SEED_BATCH="${SEED_BATCH:-$(seed_batch_for "$NUM_ENS")}"
N="${N:-${1:-50}}"                                     # number of RANDOM eval times to draw
SEED="${SEED:-0}"                                      # RNG seed: reproducible; both passes share it
YEARS="${YEARS:-2019}"                                 # space-separated year(s) to sample from
NAME="${NAME:-}"                                       # REQUIRED run name -> results/<name>/eval
# Small, kept artifacts (metrics JSON) go with the run's tracked results in the repo.
RESULT_DIR="${RESULT_DIR:-$TRAIN_DIR/results/$NAME/eval}"
# Bulky NetCDFs (full.nc/reg.nc, several GB) stay on scratch -- too big for the $HOME repo quota,
# and .nc is gitignored anyway. Defaults to $RES_RUN/eval (resolve_eval_paths, below); set
# NC_DIR=$RESULT_DIR if you really want them alongside the metrics.
NC_DIR="${NC_DIR:-}"
NPROC="${SLURM_GPUS_ON_NODE:-1}"
(( N < 1 )) && N=1
YEARS_CSV=$(echo "$YEARS" | tr ' ' ',')                # "2018 2019" -> "2018,2019" for Hydra
require_divisor "$NUM_ENS" "$SEED_BATCH" SEED_BATCH || exit 1

# ---- sanity: NAME is required so metrics land in the intended results/<NAME>/eval ----------
if [[ -z "$NAME" ]]; then
  echo "ERROR: set NAME=<run name> (e.g. NAME=diffusion_2) -- metrics go to results/<NAME>/eval." >&2
  echo "       (Not derived from the run dirs, to avoid silently writing to the wrong folder.)" >&2
  exit 1
fi

# ---- resolve the two nets + NC_DIR (see resolve_eval_paths in slurm/common.sh) --------------
resolve_eval_paths || exit 1

# ---- sanity: log resolved paths, fail fast if $SCRATCH/$PROJECT were unset at submit -------
echo "[paths] REG_RUN=${REG_RUN:-<from REG_CKPT>}  RES_RUN=${RES_RUN:-<from RES_CKPT>}"
echo "        DATA_DIR=$DATA_DIR"
echo "        RESULT_DIR=$RESULT_DIR (metrics)  NC_DIR=$NC_DIR (NetCDFs)"
echo "        (SCRATCH=${SCRATCH:-<unset>} PROJECT=${PROJECT:-<unset>})"
for p in ${REG_RUN:+"$REG_RUN"} ${RES_RUN:+"$RES_RUN"} "$DATA_DIR"; do
  case "$p" in
    /scratch/*|/project/*|/home/*) : ;;
    *) echo "ERROR: path '$p' is not absolute -- is \$SCRATCH/\$PROJECT set when you run sbatch?" >&2; exit 1 ;;
  esac
  # A checkpoint path pasted in place of a run dir otherwise gets 'checkpoints_regression'
  # appended to a FILE, and the failure only shows up as a confusing missing-checkpoint error.
  if [[ -e "$p" && ! -d "$p" ]]; then
    echo "ERROR: '$p' is a file, not a directory." >&2
    echo "       REG_RUN/RES_RUN are RUN directories (e.g. \$SCRATCH/corrdiff_runs/diffusion_2)," >&2
    echo "       which hold checkpoints_regression/ or checkpoints_diffusion/." >&2
    echo "       To name a specific checkpoint file, use REG_CKPT= / RES_CKPT= instead." >&2
    exit 1
  fi
done

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

echo "== FULL model (regression + diffusion, $NUM_ENS members, $SEED_BATCH per sampler call) \
-> $NC_DIR/full.nc =="
"${COMMON[@]}" \
  ++generation.inference_mode=all \
  ++generation.num_ensembles="$NUM_ENS" \
  ++generation.seed_batch_size="$SEED_BATCH" \
  ++generation.io.reg_ckpt_filename="$REG_CKPT" \
  ++generation.io.res_ckpt_filename="$RES_CKPT" \
  ++generation.io.output_filename="$NC_DIR/full.nc"

echo "== REGRESSION only (deterministic mean) -> $NC_DIR/reg.nc =="
"${COMMON[@]}" \
  ++generation.inference_mode=regression \
  ++generation.num_ensembles=1 \
  ++generation.seed_batch_size=1 \
  ++generation.io.reg_ckpt_filename="$REG_CKPT" \
  ++generation.io.output_filename="$NC_DIR/reg.nc"

echo "== CRPS / MAE per channel: full vs reg =="
python "$EVAL_DIR/eval_crps_mae.py" \
  --nc full="$NC_DIR/full.nc" reg="$NC_DIR/reg.nc" \
  --out "$RESULT_DIR/metrics_crps_mae.json"

echo "DONE. metrics -> $RESULT_DIR/metrics_crps_mae.json   |   NetCDFs -> $NC_DIR"
