#!/bin/bash
# CorrDiff-Mini STAGE 1 (regression / mean predictor) on the Fir GPU cluster.
#
# One-time setup first:  bash training_mini/slurm/setup_env.sh   (see that file)
# Submit:                sbatch training_mini/slurm/train_regression.sh
# No-sea-ice variant:    CONFIG=config_training_era5_carra2_mini_regression_noice \
#                            sbatch training_mini/slurm/train_regression.sh
#
# ============ FILL THESE IN (confirm against `module avail` / the Fir docs) ============
#SBATCH --account=def-stockie            # <-- your GPU allocation (may differ from CPU)
#SBATCH --mail-user=ioa4@sfu.ca          # <-- or delete the mail lines
#SBATCH --gpus-per-node=1                # <-- may need a type, e.g. h100:1 / a100:1 on Fir
# ======================================================================================
#SBATCH --job-name=corrdiff_reg
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/corrdiff_reg_%j.out
#SBATCH --error=logs/corrdiff_reg_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# ---- paths (FILL/CHECK) ----------------------------------------------------
REPO="$HOME/thesis/era5-carra2-downscaling-canadian-arctic"
TRAIN_DIR="$REPO/training_mini"
DATA_DIR="${DATA_DIR:-$PROJECT/data}"                 # holds shard_YYYY.zarr (built earlier)
OUTPUT_DIR="${OUTPUT_DIR:-$SCRATCH/corrdiff_mini}"    # checkpoints + hydra outputs (large; not $HOME)
CONFIG="${CONFIG:-config_training_era5_carra2_mini_regression}"
STATS="${STATS:-$DATA_DIR/stats_train_2011_2018.json}"
NPROC="${SLURM_GPUS_ON_NODE:-1}"

# ---- environment -----------------------------------------------------------
module load python/3.11 cuda cudnn        # <-- match setup_env.sh / `module avail`
source ~/corrdiff-env/bin/activate        # venv built by setup_env.sh (torch>=2.10 + physicsnemo)

cd "$TRAIN_DIR"
mkdir -p logs "$OUTPUT_DIR"
ln -sfn "$DATA_DIR" ./data                # configs reference ./data

# train-only normalization stats (2011-2018), computed once
if [[ ! -f "$STATS" ]]; then
  echo "Computing train stats -> $STATS"
  python tools/make_stats.py --data-dir "$DATA_DIR" \
     --years 2011 2012 2013 2014 2015 2016 2017 2018 --out "$STATS"
fi

echo "Launching regression ($CONFIG) on $NPROC GPU(s); checkpoints -> $OUTPUT_DIR"
torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC" \
  train.py --config-name="$CONFIG" \
  ++dataset.stats_path="$STATS" \
  ++training.io.checkpoint_dir="$OUTPUT_DIR"

echo "DONE. Regression checkpoints in $OUTPUT_DIR/checkpoints_regression/ (CorrDiffRegressionUNet.*.mdlus)"
