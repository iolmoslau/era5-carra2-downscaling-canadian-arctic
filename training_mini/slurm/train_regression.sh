#!/bin/bash
# CorrDiff-Mini STAGE 1 (regression / mean predictor) on Fir (H100).
#
# One-time env setup first:  bash training_mini/slurm/setup_env.sh
# Submit:                    sbatch training_mini/slurm/train_regression.sh
# Quick env test on Fir:     TRAIN_DURATION=2000 STAGE=0 sbatch training_mini/slurm/train_regression.sh
# No-sea-ice variant:        CONFIG=config_training_era5_carra2_mini_regression_noice \
#                                sbatch training_mini/slurm/train_regression.sh
#
# Opportunistic (no allocation) => keep --time modest for backfill priority. Training is
# resumable: if the job hits the time limit, just re-submit and it continues from the last
# checkpoint in $OUTPUT_DIR (train.py loads cur_nimg automatically).

#SBATCH --account=def-stockie_gpu
# Multi-GPU speedup: override on submit, e.g. `sbatch --gpus=h100:4 <script>` (torchrun scales
# automatically via SLURM_GPUS_ON_NODE). 1 GPU backfills faster on the opportunistic queue.
#SBATCH --gpus=h100:1
#SBATCH --job-name=corrdiff_reg
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/corrdiff_reg_%j.out
#SBATCH --error=logs/corrdiff_reg_%j.err
#SBATCH --mail-user=ioa4@sfu.ca          # <-- or delete these two lines
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# ---- config ----------------------------------------------------------------
REPO="${REPO:-$HOME/thesis/era5-carra2-downscaling-canadian-arctic}"   # respects an existing $REPO
TRAIN_DIR="$REPO/training_mini"
ENV_DIR="${ENV_DIR:-$HOME/corrdiff-env}"
DATA_DIR="${DATA_DIR:-$PROJECT/data}"                 # holds shard_YYYY.zarr (2011-2019)
OUTPUT_DIR="${OUTPUT_DIR:-$SCRATCH/corrdiff_mini}"    # checkpoints (persistent; NOT $SLURM_TMPDIR)
CONFIG="${CONFIG:-config_training_era5_carra2_mini_regression}"
STATS="${STATS:-$DATA_DIR/stats_train_2011_2018.json}"
STAGE="${STAGE:-1}"                                   # 1 = copy shards to fast node-local $SLURM_TMPDIR
NPROC="${SLURM_GPUS_ON_NODE:-1}"

# ---- environment -----------------------------------------------------------
module load python/3.11        # add `cuda/12.6` here only if physicsnemo/warp errors on CUDA
source "$ENV_DIR/bin/activate"

cd "$TRAIN_DIR"
mkdir -p logs "$OUTPUT_DIR"

# ---- stage zarr shards to node-local storage (many tiny files -> avoid /project thrash) ----
if [[ "$STAGE" == "1" ]]; then
  echo "Staging shards -> $SLURM_TMPDIR/data"
  mkdir -p "$SLURM_TMPDIR/data"
  cp -r "$DATA_DIR"/shard_20{11,12,13,14,15,16,17,18,19}.zarr "$SLURM_TMPDIR/data/"
  RUN_DATA="$SLURM_TMPDIR/data"
else
  RUN_DATA="$DATA_DIR"
fi
ln -sfn "$RUN_DATA" ./data                            # configs reference ./data

# ---- train-only normalization stats (2011-2018), computed once ----
if [[ ! -f "$STATS" ]]; then
  echo "Computing train stats -> $STATS"
  python tools/make_stats.py --data-dir "$RUN_DATA" \
     --years 2011 2012 2013 2014 2015 2016 2017 2018 --out "$STATS"
fi

echo "Launching regression ($CONFIG) on $NPROC H100; checkpoints -> $OUTPUT_DIR"
torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC" \
  train.py --config-name="$CONFIG" \
  ++dataset.stats_path="$STATS" \
  ++training.io.checkpoint_dir="$OUTPUT_DIR"

echo "DONE. Regression checkpoints in $OUTPUT_DIR/checkpoints_regression/ (CorrDiffRegressionUNet.*.mdlus)"
