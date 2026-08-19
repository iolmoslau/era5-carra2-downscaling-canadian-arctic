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

# ---------------------------------------------------------------------------------------
# Ensemble batching for generation
#
# generation.seed_batch_size is how many ensemble members are denoised in ONE sampler call.
# At 1 -- the config default -- a NUM_ENS-member ensemble is NUM_ENS sequential batch-of-one
# runs of an 18-step sampler, which leaves an H100 mostly idle. (The regression net is already
# batched: generate.py sizes its latents from sum(map(len, rank_batches)).)
#
# CONSTRAINT: seed_batch_size must DIVIDE num_ensembles. physicsnemo's diffusion_step sizes its
# latents from `img_lr.shape[0]` -- which generate.py expands to seed_batch_size -- rather than
# from len(batch_seeds). So an uneven split makes the short final batch still emit
# seed_batch_size samples, and the total no longer matches the regression mean it is added to:
#     latents_shape = [img_lr.shape[0], img_out_channels, img_shape[0], img_shape[1]]
# The failure is a tensor-shape error deep inside physicsnemo, so we check it up front instead.
# ---------------------------------------------------------------------------------------

# seed_batch_for <num_ensembles> [cap] -- largest divisor of N that is <= cap (default 8).
seed_batch_for() {
  local n="${1:?num_ensembles required}" cap="${2:-8}" b
  (( cap > n )) && cap=$n
  for (( b = cap; b > 1; b-- )); do
    (( n % b == 0 )) && { printf '%s\n' "$b"; return 0; }
  done
  printf '1\n'
}

# require_divisor <num_ensembles> <seed_batch> [label] -- return 1 with an explanation unless
# seed_batch is a positive divisor of num_ensembles.
require_divisor() {
  local n="${1:-}" b="${2:-}" label="${3:-SEED_BATCH}" d divisors=""
  if [[ ! "$b" =~ ^[0-9]+$ ]] || (( b < 1 )) || (( n % b != 0 )); then
    for (( d = 1; d <= n; d++ )); do (( n % d == 0 )) && divisors+="$d "; done
    echo "ERROR: $label=$b must be a positive divisor of NUM_ENS=$n." >&2
    echo "       CorrDiff sizes the diffusion latents from the conditioning batch, not from the" >&2
    echo "       seed count, so an uneven split generates more members than requested and then" >&2
    echo "       fails when they are added to the regression mean." >&2
    echo "       Divisors of $n: ${divisors% }" >&2
    return 1
  fi
}
