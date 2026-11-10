#!/bin/bash
# Generate samples with the trained CorrDiff-Mini model on Fir (short GPU job).
#
# Always submit via slurm/submit.sh so job logs land in $REPO/logs regardless of your CWD.
# Regression only (deterministic mean; works before diffusion is trained -- the default):
#     bash training_mini/slurm/submit.sh training_mini/slurm/generate.sh
# Full pipeline once diffusion is trained (regression mean + diffusion residual):
#     MODE=all NUM_ENS=4 RES_CKPT=$OUTPUT_DIR/checkpoints_diffusion/EDMPrecondSuperResolution.0.NNN.mdlus \
#         bash training_mini/slurm/submit.sh training_mini/slurm/generate.sh
# Ensemble spread: NUM_ENS>1 draws that many stochastic diffusion samples per input time, so the
# prediction group gets an `ensemble` dim -> plot_sample_native/collect_run compute the per-pixel
# and per-channel-mean variance. e.g. for a 15-member spread estimate:
#     MODE=all NUM_ENS=15 REG_CKPT=... RES_CKPT=... bash training_mini/slurm/submit.sh training_mini/slurm/generate.sh
#
# SEED_BATCH = members denoised per sampler call. Defaults to the largest divisor of NUM_ENS that
# is <= 8, so a 15-member ensemble is 3 sampler calls instead of 15. It MUST divide NUM_ENS (the
# job refuses to start otherwise -- see require_divisor in slurm/common.sh). Raise it for more
# GPU throughput, lower it if you hit OOM:
#     MODE=all NUM_ENS=16 SEED_BATCH=16 bash training_mini/slurm/submit.sh training_mini/slurm/generate.sh
#
# Output NetCDF (truth/prediction/input groups) lands in training_mini/ as corrdiff_output.nc.

#SBATCH --account=def-stockie_gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --job-name=corrdiff_gen
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=logs/corrdiff_gen_%j.out
#SBATCH --error=logs/corrdiff_gen_%j.err

set -euo pipefail

REPO="${REPO:-$HOME/thesis/era5-carra2-downscaling-canadian-arctic}"
source "$REPO/training_mini/slurm/common.sh"   # latest_ckpt (selects by nimg, not mtime)
TRAIN_DIR="$REPO/training_mini"
ENV_DIR="${ENV_DIR:-$HOME/corrdiff-env}"
DATA_DIR="${DATA_DIR:-$PROJECT/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRATCH/corrdiff_mini}"
STATS="${STATS:-$DATA_DIR/stats_train_2011_2018.json}"
MODE="${MODE:-regression}"                 # regression | diffusion | all
NUM_ENS="${NUM_ENS:-1}"                     # ensemble members (use >1 for diffusion/all)
# Members denoised per sampler call. Defaults to the largest divisor of NUM_ENS that is <= 8;
# must divide NUM_ENS exactly (see seed_batch_for/require_divisor in slurm/common.sh).
SEED_BATCH="${SEED_BATCH:-$(seed_batch_for "$NUM_ENS")}"
CONFIG="${CONFIG:-config_generate_era5_carra2_mini}"
# Visible devices before $SLURM_GPUS_ON_NODE: see the note on require_single_node in common.sh.
NPROC="${NPROC:-}"
if [[ ! "$NPROC" =~ ^[1-9][0-9]*$ ]]; then
  NPROC=$(visible_gpus)
  [[ "$NPROC" =~ ^[1-9][0-9]*$ ]] || NPROC="${SLURM_GPUS_ON_NODE:-1}"
fi
# Channels to withhold from the LR stack. NOICE=1 is the no-sea-ice model; DROP is the general
# form, and the hook for the channel-ablation sweep. The list itself is read from the data store
# by tools/lr_channels.py and verified against the checkpoint -- never typed out here.
DROP="${DROP:-}"
[[ "${NOICE:-0}" == "1" ]] && DROP="${DROP:+$DROP,}siconc"

require_divisor "$NUM_ENS" "$SEED_BATCH" SEED_BATCH || exit 1

# ---- sanity: log resolved paths, fail fast if $SCRATCH/$PROJECT were unset at submit -------
echo "[paths] OUTPUT_DIR=$OUTPUT_DIR  DATA_DIR=$DATA_DIR  (SCRATCH=${SCRATCH:-<unset>} PROJECT=${PROJECT:-<unset>})"
for p in "$OUTPUT_DIR" "$DATA_DIR"; do
  case "$p" in
    /scratch/*|/project/*|/home/*) : ;;
    *) echo "ERROR: path '$p' is not a valid absolute location -- is \$SCRATCH/\$PROJECT set when you run sbatch?" >&2; exit 1 ;;
  esac
done

# Furthest-trained regression checkpoint, selected by the nimg in the filename -- NOT by mtime,
# which a copy/restore reorders (see latest_ckpt in slurm/common.sh).
if [[ -z "${REG_CKPT:-}" ]]; then
  REG_CKPT=$(latest_ckpt "$OUTPUT_DIR/checkpoints_regression" || true)
fi
RES_CKPT="${RES_CKPT:-}"                    # required only for MODE=diffusion|all
if [[ -z "${REG_CKPT:-}" || ! -f "$REG_CKPT" ]]; then
  echo "ERROR: no regression checkpoint found in $OUTPUT_DIR/checkpoints_regression" >&2
  exit 1
fi

module load python/3.11 mpi4py/4.1.0
source "$ENV_DIR/bin/activate"
export PYTHONUNBUFFERED=1              # flush logs promptly so SLURM output isn't lost on exit
export HDF5_USE_FILE_LOCKING=FALSE    # network FS: avoid NetCDF/HDF5 file-locking errors

cd "$TRAIN_DIR"
export CORRDIFF_LOG_DIR="$REPO/logs"   # Hydra run dir + generate.log go here (not the CWD)
mkdir -p "$CORRDIFF_LOG_DIR"
ln -sfn "$DATA_DIR" ./data

# generate.py (like train.py) needs torchrun so physicsnemo's DistributedManager can read
# RANK/WORLD_SIZE; plain `python generate.py` crashes in DistributedManager.initialize().
# Resolve the LR channel list from the store and verify it against the net that will consume it.
# BEFORE torchrun: a mismatch otherwise surfaces as a conv shape error minutes in, or -- for a
# wrong-but-same-length subset -- not at all. The regression net is checked because it reads the
# conditioning first; a paired diffusion net shares its LR stack.
# SKIP_CHANNEL_CHECK=1 still resolves the list from the store but skips reading the
# checkpoint, which needs physicsnemo + torch. Use it to take the preflight out of the picture
# when diagnosing an unrelated import/MPI failure -- the list is still correct, just unverified.
CHECK=(--check-against "$REG_CKPT" --stage regression)
[[ "${SKIP_CHANNEL_CHECK:-0}" == "1" ]] && CHECK=()
LR_ARG=$(python tools/lr_channels.py --data-dir "$DATA_DIR" --drop "$DROP" "${CHECK[@]}") || exit 1
echo "LR channels: $LR_ARG"

CMD=(torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC"
     generate.py --config-name="$CONFIG"
     hydra.run.dir="$CORRDIFF_LOG_DIR/hydra/${SLURM_JOB_ID:-manual}"
     ++generation.inference_mode="$MODE"
     ++generation.num_ensembles="$NUM_ENS"
     ++generation.seed_batch_size="$SEED_BATCH"
     ++dataset.data_path="$DATA_DIR"
     ++dataset.stats_path="$STATS"
     ++dataset.lr_channels="$LR_ARG"
     ++generation.io.reg_ckpt_filename="$REG_CKPT")
[[ -n "$RES_CKPT" ]] && CMD+=("++generation.io.res_ckpt_filename=$RES_CKPT")

echo "Generating ($MODE) with reg ckpt: $REG_CKPT"
echo "  ${CMD[*]}"
"${CMD[@]}"

echo "DONE. Output NetCDF: $TRAIN_DIR/corrdiff_output.nc"
echo "Plot it with:  python tools/plot_sample.py --nc corrdiff_output.nc --out sample.png"
