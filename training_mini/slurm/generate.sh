#!/bin/bash
# Generate samples with the trained CorrDiff-Mini model on Fir (short GPU job).
#
# Regression only (deterministic mean; works before diffusion is trained -- the default):
#     sbatch training_mini/slurm/generate.sh
# Full pipeline once diffusion is trained (regression mean + diffusion residual, ensemble):
#     MODE=all NUM_ENS=4 RES_CKPT=$SCRATCH/corrdiff_mini/checkpoints_diffusion/EDMPrecondSuperResolution.0.NNN.mdlus \
#         sbatch training_mini/slurm/generate.sh
#
# Output NetCDF (truth/prediction/input groups) lands in training_mini/ as corrdiff_output.nc.

#SBATCH --account=def-stockie_gpu
#SBATCH --gpus=h100:1
#SBATCH --job-name=corrdiff_gen
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=logs/corrdiff_gen_%j.out
#SBATCH --error=logs/corrdiff_gen_%j.err

set -euo pipefail

REPO="${REPO:-$HOME/thesis/era5-carra2-downscaling-canadian-arctic}"
TRAIN_DIR="$REPO/training_mini"
ENV_DIR="${ENV_DIR:-$HOME/corrdiff-env}"
DATA_DIR="${DATA_DIR:-$PROJECT/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRATCH/corrdiff_mini}"
STATS="${STATS:-$DATA_DIR/stats_train_2011_2018.json}"
MODE="${MODE:-regression}"                 # regression | diffusion | all
NUM_ENS="${NUM_ENS:-1}"                     # ensemble members (use >1 for diffusion/all)
CONFIG="${CONFIG:-config_generate_era5_carra2_mini}"

REG_CKPT="${REG_CKPT:-$(ls -t "$OUTPUT_DIR"/checkpoints_regression/*.mdlus 2>/dev/null | head -1)}"
RES_CKPT="${RES_CKPT:-}"                    # required only for MODE=diffusion|all
if [[ -z "${REG_CKPT:-}" || ! -f "$REG_CKPT" ]]; then
  echo "ERROR: no regression checkpoint found in $OUTPUT_DIR/checkpoints_regression" >&2
  exit 1
fi

module load python/3.11 mpi4py/4.1.0
source "$ENV_DIR/bin/activate"

cd "$TRAIN_DIR"
mkdir -p logs
ln -sfn "$DATA_DIR" ./data

CMD=(python generate.py --config-name="$CONFIG"
     ++generation.inference_mode="$MODE"
     ++generation.num_ensembles="$NUM_ENS"
     ++dataset.stats_path="$STATS"
     ++generation.io.reg_ckpt_filename="$REG_CKPT")
[[ -n "$RES_CKPT" ]] && CMD+=("++generation.io.res_ckpt_filename=$RES_CKPT")

echo "Generating ($MODE) with reg ckpt: $REG_CKPT"
echo "  ${CMD[*]}"
"${CMD[@]}"

echo "DONE. Output NetCDF: $TRAIN_DIR/corrdiff_output.nc"
echo "Plot it with:  python tools/plot_sample.py --nc corrdiff_output.nc --out sample.png"
