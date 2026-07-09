#!/bin/bash
# CorrDiff-Mini STAGE 2 (diffusion / residual predictor) on Fir (H100).
# Requires a trained regression checkpoint from stage 1.
#
# Submit (auto-finds newest regression ckpt in $OUTPUT_DIR/checkpoints_regression):
#     sbatch training_mini/slurm/train_diffusion.sh
# Or point it explicitly:
#     sbatch training_mini/slurm/train_diffusion.sh $SCRATCH/corrdiff_mini/checkpoints_regression/CorrDiffRegressionUNet.0.NNN.mdlus
# No-sea-ice variant:
#     CONFIG=config_training_era5_carra2_mini_diffusion_noice \
#         sbatch training_mini/slurm/train_diffusion.sh <regression_noice.mdlus>
#
# Resumable: re-submitting continues from the last diffusion checkpoint in $OUTPUT_DIR.

#SBATCH --account=def-stockie_gpu
# Multi-GPU speedup: override on submit, e.g. `sbatch --gpus=h100:4 <script>` (torchrun scales
# automatically via SLURM_GPUS_ON_NODE). 1 GPU backfills faster on the opportunistic queue.
#SBATCH --gpus=h100:1
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
TRAIN_DIR="$REPO/training_mini"
ENV_DIR="${ENV_DIR:-$HOME/corrdiff-env}"
DATA_DIR="${DATA_DIR:-$PROJECT/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRATCH/corrdiff_mini}"
CONFIG="${CONFIG:-config_training_era5_carra2_mini_diffusion}"
STATS="${STATS:-$DATA_DIR/stats_train_2011_2018.json}"
STAGE="${STAGE:-1}"
NPROC="${SLURM_GPUS_ON_NODE:-1}"

# regression checkpoint: arg 1, else $REG_CKPT, else newest in $OUTPUT_DIR/checkpoints_regression.
# Avoid `ls | head` under `set -o pipefail`: with hundreds of .mdlus files, head closing the pipe
# makes ls die on SIGPIPE and silently aborts the job.
if [[ -n "${1:-}" ]]; then
  REG_CKPT="$1"
elif [[ -z "${REG_CKPT:-}" ]]; then
  mapfile -t _ckpts < <(ls -t "$OUTPUT_DIR"/checkpoints_regression/*.mdlus 2>/dev/null || true)
  REG_CKPT="${_ckpts[0]:-}"
fi
if [[ -z "${REG_CKPT:-}" || ! -f "$REG_CKPT" ]]; then
  echo "ERROR: no regression checkpoint found. Pass it as arg 1 or set REG_CKPT." >&2
  exit 1
fi

module load python/3.11 mpi4py/4.1.0   # mpi4py BEFORE activating (Alliance netCDF4 needs it);
source "$ENV_DIR/bin/activate"         # add cuda/12.6 above only if physicsnemo/warp errors on CUDA

cd "$TRAIN_DIR"
mkdir -p logs "$OUTPUT_DIR"

if [[ "$STAGE" == "1" ]]; then
  echo "Staging shards -> $SLURM_TMPDIR/data"
  mkdir -p "$SLURM_TMPDIR/data"
  cp -r "$DATA_DIR"/shard_20{11,12,13,14,15,16,17,18,19}.zarr "$SLURM_TMPDIR/data/"
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
     ++training.io.regression_checkpoint_path="$REG_CKPT")
[[ -n "${TRAIN_DURATION:-}" ]] && CMD+=("++training.hp.training_duration=$TRAIN_DURATION")
[[ -n "${TOTAL_BATCH:-}"    ]] && CMD+=("++training.hp.total_batch_size=$TOTAL_BATCH")
[[ -n "${BATCH_PER_GPU:-}"  ]] && CMD+=("++training.hp.batch_size_per_gpu=$BATCH_PER_GPU")

echo "Launching diffusion ($CONFIG) on $NPROC H100; reg ckpt: $REG_CKPT"
echo "  ${CMD[*]}"
"${CMD[@]}"

echo "DONE. Diffusion checkpoints in $OUTPUT_DIR/checkpoints_diffusion/ (EDMPrecondSuperResolution.*.mdlus)"
