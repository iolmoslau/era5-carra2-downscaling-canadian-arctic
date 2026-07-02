#!/bin/bash
# CorrDiff-Mini STAGE 2 (diffusion / residual predictor) on the Fir GPU cluster.
# Requires a trained regression checkpoint from stage 1.
# Submit:   sbatch training_mini/slurm/train_diffusion.sh /path/to/regression.mdlus
#   (or set REG_CKPT=... in the environment)
# No-ice variant:
#   CONFIG=config_training_era5_carra2_mini_diffusion_noice \
#     sbatch training_mini/slurm/train_diffusion.sh /path/to/regression_noice.mdlus
#
# ====================== FILL THESE IN ======================
#SBATCH --account=def-stockie          # <-- your allocation
#SBATCH --mail-user=ioa4@sfu.ca        # <-- or delete the mail lines
# ===========================================================
#SBATCH --job-name=corrdiff_diff
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/corrdiff_diff_%j.out
#SBATCH --error=logs/corrdiff_diff_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# ---- config (FILL/CHECK) ---------------------------------------------------
REPO="$HOME/thesis/era5-carra2-downscaling-canadian-arctic"
TRAIN_DIR="$REPO/training_mini"
DATA_DIR="${DATA_DIR:-$PROJECT/data}"
CONFIG="${CONFIG:-config_training_era5_carra2_mini_diffusion}"
STATS="${STATS:-$DATA_DIR/stats_train_2011_2018.json}"
REG_CKPT="${1:-${REG_CKPT:?ERROR: pass the regression .mdlus as arg 1 or set REG_CKPT}}"
NPROC="${SLURM_GPUS_ON_NODE:-1}"

# ---- environment -----------------------------------------------------------
module load python/3.11 cuda cudnn     # <-- adjust to `module avail` on Fir
source ~/ENV/bin/activate

cd "$TRAIN_DIR"
mkdir -p logs
ln -sfn "$DATA_DIR" ./data

if [[ ! -f "$STATS" ]]; then
  echo "Computing train stats -> $STATS"
  python tools/make_stats.py --data-dir "$DATA_DIR" \
     --years 2011 2012 2013 2014 2015 2016 2017 2018 --out "$STATS"
fi

echo "Launching diffusion training ($CONFIG) on $NPROC GPU(s); reg ckpt: $REG_CKPT"
torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC" \
  train.py --config-name="$CONFIG" \
  ++training.io.regression_checkpoint_path="$REG_CKPT"

echo "DONE. Diffusion checkpoints under $TRAIN_DIR (see checkpoints_diffusion/ *.mdlus)"
