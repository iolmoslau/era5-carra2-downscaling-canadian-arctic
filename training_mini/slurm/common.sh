#!/bin/bash
# Shared shell helpers for the training / generation / evaluation jobs.
#
# Source it after $REPO is set:
#     source "$REPO/training_mini/slurm/common.sh"
#
# Also useful interactively on a login node, e.g. when following WORKFLOW.md:
#     source ~/thesis/era5-carra2-downscaling-canadian-arctic/training_mini/slurm/common.sh
#     REG=$(latest_ckpt $OUT/checkpoints_regression)

# ---------------------------------------------------------------------------------------
# Checkpoint selection
#
# WHY THIS EXISTS: checkpoints are named <Model>.0.<nimg>.mdlus, where <nimg> is the number
# of processed samples -- i.e. training progress. Picking the "latest" by MODIFICATION TIME
# (`ls -t | head -1`) agrees with that only until the files are copied: WORKFLOW.md section C
# has you archive checkpoints to $PROJECT and restore them later, and a copy resets mtimes to
# the copy order. After that, `ls -t` returns an arbitrary checkpoint and the failure is
# SILENT -- you generate, collect and log metrics against a model that is not the one you
# think it is. Always select on <nimg>.
#
# Implementation notes:
#  * The nimg field is taken from the BASENAME, so dots anywhere in the parent path (e.g.
#    /scratch/user/run_v1.2/...) cannot shift the field position -- which is the flaw in the
#    `sort -t. -k3 -n` idiom this replaces.
#  * No `ls | head`: with hundreds of .mdlus files and `set -o pipefail`, head closing the
#    pipe makes ls die on SIGPIPE and silently aborts the job.
# ---------------------------------------------------------------------------------------

# latest_ckpt <checkpoint-dir> -- print the .mdlus with the highest nimg; return 1 if none.
latest_ckpt() {
  local dir="${1:-}" f base n best="" best_n=-1
  [[ -d "$dir" ]] || return 1
  for f in "$dir"/*.mdlus; do
    [[ -f "$f" ]] || continue           # unmatched glob expands to the pattern itself
    base="${f##*/}"                     # drop the directory (and any dots it contains)
    base="${base%.mdlus}"
    n="${base##*.}"                     # trailing dot-field = nimg
    [[ "$n" =~ ^[0-9]+$ ]] || continue  # skip anything not following the naming scheme
    if (( 10#$n > best_n )); then best_n=$(( 10#$n )); best="$f"; fi
  done
  [[ -n "$best" ]] || return 1
  printf '%s\n' "$best"
}

# ckpt_nimg <checkpoint-path> -- print the nimg (processed-sample count) encoded in the name.
ckpt_nimg() {
  local base="${1:-}"
  base="${base##*/}"
  base="${base%.mdlus}"
  base="${base%.pt}"
  printf '%s\n' "${base##*.}"
}
