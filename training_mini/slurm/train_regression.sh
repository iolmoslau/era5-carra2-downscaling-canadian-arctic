#!/bin/bash
# CorrDiff-Mini STAGE 1 (regression / mean predictor) on the Fir GPU cluster.
# Submit:   sbatch training_mini/slurm/train_regression.sh
# Override config/data via env, e.g.:
#   CONFIG=config_training_era5_carra2_mini_regression_noice sbatch training_mini/slurm/train_regression.sh
#
# ====================== FILL THESE IN ======================
#SBATCH --account=def-stockie          # <-- your allocation
#SBATCH --mail-user=ioa4@sfu.ca        # <-- or delete the mail lines
# ===========================================================
#SBATCH --job-name=corrdiff_reg
#SBATCH --gpus-per-node=1              # bump for multi-GPU (torchrun scales automatically)
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/corrdiff_reg_%j.out
#SBATCH --error=logs/corrdiff_reg_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# ---- config (FILL/CHECK) ---------------------------------------------------
REPO="$HOME/thesis/era5-carra2-downscaling-canadian-arctic"
TRAIN_DIR="$REPO/training_mini"
DATA_DIR="${DATA_DIR:-$PROJECT/data}"                        # holds shard_YYYY.zarr
CONFIG="${CONFIG:-config_training_era5_carra2_mini_regression}"
STATS="${STATS:-$DATA_DIR/stats_train_2011_2018.json}"
NPROC="${SLURM_GPUS_ON_NODE:-1}"

# ---- environment -----------------------------------------------------------
module load python/3.11 cuda cudnn     # <-- adjust to `module avail` on Fir
source ~/ENV/bin/activate              # venv with `pip install -r requirements-train.txt`

cd "$TRAIN_DIR"
mkdir -p logs
# configs reference ./data ; point it at the real shard directory
ln -sfn "$DATA_DIR" ./data

# train-only normalization stats (2011-2018), computed once
if [[ ! -f "$STATS" ]]; then
  echo "Computing train stats -> $STATS"
  python tools/make_stats.py --data-dir "$DATA_DIR" \
     --years 2011 2012 2013 2014 2015 2016 2017 2018 --out "$STATS"
fi

echo "Launching regression training ($CONFIG) on $NPROC GPU(s)"
torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC" \
  train.py --config-name="$CONFIG"

echo "DONE. Regression checkpoints under $TRAIN_DIR (see checkpoints_regression/ *.mdlus)"
