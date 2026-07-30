#!/bin/bash
# Submit any training_mini SLURM job so its logs ALWAYS land in $REPO/logs, no matter which
# directory you run this from.
#
# WHY THIS EXISTS: SLURM does not expand shell/env vars in `#SBATCH` lines, so a relative
# `#SBATCH --output=logs/...` directive is resolved against the *submission* directory. That is
# why job logs scattered to ~/logs, $REPO/logs, and $REPO/training_mini/logs depending on where
# `sbatch` happened to be run. Passing --output/--error on the command line (the shell expands
# $REPO here) overrides the in-script directive and pins the location once and for all.
#
# Usage -- put any sbatch options and the job script after it, exactly as you would with sbatch:
#     bash training_mini/slurm/submit.sh <sbatch opts> <script.sh> [script args]
#   e.g.
#     DATA_DIR=$DATA OUTPUT_DIR=$OUT TRAIN_DURATION=800000 \
#       bash training_mini/slurm/submit.sh --gpus=h100:2 training_mini/slurm/train_regression.sh
#
# Env-var prefixes (DATA_DIR=, OUTPUT_DIR=, ...) and sbatch flags (--gpus=, ...) both pass through:
# the prefixes are exported into this wrapper's environment and sbatch forwards them to the job
# (--export=ALL is the default); the flags are forwarded verbatim via "$@".
set -euo pipefail

REPO="${REPO:-$HOME/thesis/era5-carra2-downscaling-canadian-arctic}"
mkdir -p "$REPO/logs"
exec sbatch --output="$REPO/logs/%x_%j.out" --error="$REPO/logs/%x_%j.err" "$@"
