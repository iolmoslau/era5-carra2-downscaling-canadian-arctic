#!/bin/bash
# CorrDiff-Mini STAGE 2 (diffusion / residual predictor) on Fir (H100).
# Requires a trained regression checkpoint from stage 1.
#
# Submit (auto-finds newest regression ckpt in $OUTPUT_DIR/checkpoints_regression):
#     bash training_mini/slurm/submit.sh --gpus-per-node=h100:2 training_mini/slurm/train_diffusion.sh
# Or point it explicitly:
#     bash training_mini/slurm/submit.sh training_mini/slurm/train_diffusion.sh $OUTPUT_DIR/checkpoints_regression/CorrDiffRegressionUNet.0.NNN.mdlus
# No-sea-ice variant:
#     CONFIG=config_training_era5_carra2_mini_diffusion_noice \
#         bash training_mini/slurm/submit.sh training_mini/slurm/train_diffusion.sh <regression_noice.mdlus>
#
# Fewer/leaner checkpoints:
#     CKPT_FREQ=50000 KEEP_CKPTS=3 bash training_mini/slurm/submit.sh --gpus-per-node=h100:2 \
#         training_mini/slurm/train_diffusion.sh
#
# Env passthroughs: TRAIN_DURATION, TOTAL_BATCH, BATCH_PER_GPU, CKPT_FREQ, KEEP_CKPTS, CONFIG,
# DATA_DIR, OUTPUT_DIR, STATS, STAGE, YEARS, REG_CKPT, ENV_DIR, REPO.
#
# STAGE=1 copies the shards the CONFIG needs (dataset.years + validation.years) to node-local
# $SLURM_TMPDIR; YEARS="2011 2012" overrides that for a subset. Archived shard_YYYY.zarr.zip
# stages as readily as a loose store, and a year that is absent warns instead of killing the job.
#
# Always submit via slurm/submit.sh so job logs land in $REPO/logs regardless of your CWD.
# Resumable: re-submitting continues from the last diffusion checkpoint in $OUTPUT_DIR.

#SBATCH --account=def-stockie_gpu
# Multi-GPU speedup: override on submit, e.g. `--gpus-per-node=h100:4`. Use --gpus-per-node,
# NOT --gpus: the latter is a job-level count SLURM may spread across nodes (one GPU each), and
# torchrun --standalone --nnodes=1 can only reach the first node. 1 GPU backfills faster on the
# opportunistic queue. Valid counts are constrained by total_batch_size -- see require_world_size.
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --job-name=corrdiff_diff
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/corrdiff_diff_%j.out
#SBATCH --error=logs/corrdiff_diff_%j.err
#SBATCH --mail-user=ioa4@sfu.ca          # <-- or delete these two lines
#SBATCH --mail-type=END,FAIL

set -euo pipefail

REPO="${REPO:-$HOME/thesis/era5-carra2-downscaling-canadian-arctic}"   # respects an existing $REPO
source "$REPO/training_mini/slurm/common.sh"   # latest_ckpt, config_years, stage_shards
TRAIN_DIR="$REPO/training_mini"
ENV_DIR="${ENV_DIR:-$HOME/corrdiff-env}"
DATA_DIR="${DATA_DIR:-$PROJECT/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRATCH/corrdiff_mini}"
CONFIG="${CONFIG:-config_training_era5_carra2_mini_diffusion}"
STATS="${STATS:-$DATA_DIR/stats_train_2011_2018.json}"
STAGE="${STAGE:-1}"
# Devices actually visible first: $SLURM_GPUS_ON_NODE is a claim about the allocation, and an
# absent one silently inherits whatever the submitting shell had (sbatch --export=ALL), which is
# how a 3-GPU job once trained on 1 for 12 hours. CUDA_VISIBLE_DEVICES/nvidia-smi cannot be stale.
# A count of 0 (or none at all) is not a rank count -- fall back rather than launch zero ranks.
NPROC="${NPROC:-}"
if [[ ! "$NPROC" =~ ^[1-9][0-9]*$ ]]; then
  NPROC=$(visible_gpus)
  [[ "$NPROC" =~ ^[1-9][0-9]*$ ]] || NPROC="${SLURM_GPUS_ON_NODE:-1}"
fi

# ---- sanity: log resolved paths, fail fast on a bad OUTPUT_DIR/DATA_DIR --------------------
echo "[paths] OUTPUT_DIR=$OUTPUT_DIR"
echo "[paths] DATA_DIR=$DATA_DIR  STATS=$STATS  (SCRATCH=${SCRATCH:-<unset>} PROJECT=${PROJECT:-<unset>})"
for p in "$OUTPUT_DIR" "$DATA_DIR"; do
  case "$p" in
    /scratch/*|/project/*|/home/*) : ;;
    *) echo "ERROR: path '$p' is not a valid absolute location -- is \$SCRATCH/\$PROJECT set when you run sbatch?" >&2; exit 1 ;;
  esac
done

# regression checkpoint: arg 1, else $REG_CKPT, else the furthest-trained one in
# $OUTPUT_DIR/checkpoints_regression -- selected by the nimg in the filename, NOT by mtime,
# which a copy/restore reorders (see latest_ckpt in slurm/common.sh).
if [[ -n "${1:-}" ]]; then
  REG_CKPT="$1"
elif [[ -z "${REG_CKPT:-}" ]]; then
  REG_CKPT=$(latest_ckpt "$OUTPUT_DIR/checkpoints_regression" || true)
fi
if [[ -z "${REG_CKPT:-}" || ! -f "$REG_CKPT" ]]; then
  echo "ERROR: no regression checkpoint found. Pass it as arg 1 or set REG_CKPT." >&2
  exit 1
fi

module load python/3.11 mpi4py/4.1.0   # mpi4py BEFORE activating (Alliance netCDF4 needs it);
source "$ENV_DIR/bin/activate"         # add cuda/12.6 above only if physicsnemo/warp errors on CUDA

cd "$TRAIN_DIR"
export CORRDIFF_LOG_DIR="$REPO/logs"   # Hydra run dir, wandb offline, generate.log all go here
mkdir -p "$CORRDIFF_LOG_DIR" "$OUTPUT_DIR"

# ---- preflight: do the ranks match the GPUs, and can that count carry this batch? ----------
# Before staging, because staging is the slow part and the allocation is already burning.
require_single_node || exit 1
check_gpu_alloc "$NPROC" || exit 1
# Read into private names: TOTAL_BATCH/BATCH_PER_GPU stay purely user overrides, so filling
# them here would silently start appending ++training.hp.* to every launch line.
_tb="${TOTAL_BATCH:-$(config_hp "$TRAIN_DIR/conf/$CONFIG.yaml" total_batch_size)}"
_bpg="${BATCH_PER_GPU:-$(config_hp "$TRAIN_DIR/conf/$CONFIG.yaml" batch_size_per_gpu)}"
require_world_size "$NPROC" "$_tb" "$_bpg" || exit 1

# Years come from the CONFIG (dataset.years + validation.years) so staging can never drift from
# what the run reads; override with YEARS="2011 2012" for a quick subset.
YEARS="${YEARS:-$(config_years "$TRAIN_DIR/conf/$CONFIG.yaml")}"
if [[ "$STAGE" == "1" && -z "${SLURM_TMPDIR:-}" ]]; then
  echo "WARNING: STAGE=1 but \$SLURM_TMPDIR is unset (not in a job?) -- reading from $DATA_DIR" >&2
  STAGE=0
fi
if [[ "$STAGE" == "1" ]]; then
  echo "Staging years [$YEARS] -> $SLURM_TMPDIR/data"
  stage_shards "$DATA_DIR" "$SLURM_TMPDIR/data" $YEARS
  RUN_DATA="$SLURM_TMPDIR/data"
else
  RUN_DATA="$DATA_DIR"
fi
ln -sfn "$RUN_DATA" ./data

# optional overrides for a quick env-test or tuning, e.g. TRAIN_DURATION=2000
CMD=(torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC"
     train.py --config-name="$CONFIG"
     ++dataset.stats_path="$STATS"
     ++training.io.checkpoint_dir="$OUTPUT_DIR"
     ++training.io.regression_checkpoint_path="$REG_CKPT"
     hydra.run.dir="$CORRDIFF_LOG_DIR/hydra/${SLURM_JOB_ID:-manual}"
     ++wandb.results_dir="$CORRDIFF_LOG_DIR/wandb")
[[ -n "${TRAIN_DURATION:-}" ]] && CMD+=("++training.hp.training_duration=$TRAIN_DURATION")
[[ -n "${TOTAL_BATCH:-}"    ]] && CMD+=("++training.hp.total_batch_size=$TOTAL_BATCH")
[[ -n "${BATCH_PER_GPU:-}"  ]] && CMD+=("++training.hp.batch_size_per_gpu=$BATCH_PER_GPU")
# Checkpointing dials (config defaults: every 5000 samples, keep everything). Writes are
# synchronous behind a barrier, so they stall BOTH GPUs -- raising CKPT_FREQ buys throughput at
# the cost of losing more progress to preemption. KEEP_CKPTS prunes on the NEXT save, so setting
# it partway through a run retroactively deletes that run's history; prefer setting it at the
# start. e.g. CKPT_FREQ=50000 KEEP_CKPTS=3
[[ -n "${CKPT_FREQ:-}"      ]] && CMD+=("++training.io.save_checkpoint_freq=$CKPT_FREQ")
[[ -n "${KEEP_CKPTS:-}"     ]] && CMD+=("++training.io.save_n_recent_checkpoints=$KEEP_CKPTS")

echo "Launching diffusion ($CONFIG) on $NPROC H100; reg ckpt: $REG_CKPT"
echo "  ${CMD[*]}"
"${CMD[@]}"

echo "DONE. Diffusion checkpoints in $OUTPUT_DIR/checkpoints_diffusion/ (EDMPrecondSuperResolution.*.mdlus)"
