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
# Staging shards to node-local storage
#
# WHY THIS EXISTS: the staging copy used to be
#     cp -r "$DATA_DIR"/shard_20{11,...,19}.zarr "$SLURM_TMPDIR/data/"
# which has three problems. The year list was hardcoded and could drift from the config's
# dataset.years; it could not see a shard archived to shard_YYYY.zarr.zip; and under
# `set -euo pipefail` a single missing year made cp return non-zero and killed the job at
# staging -- AFTER the GPUs had been allocated.
#
# A missing year is not fatal here: ERA5CARRA2Dataset raises FileNotFoundError for any store it
# actually needs, so that check stays authoritative and we only warn.
# ---------------------------------------------------------------------------------------

# shard_src <dir> <year> -- that year's shard, preferring the 1-inode .zarr.zip archive over the
# loose directory store. Prints nothing and returns 1 if neither form is present.
shard_src() {
  local d="${1:-}" y="${2:-}"
  if [[ -f "$d/shard_${y}.zarr.zip" ]]; then printf '%s\n' "$d/shard_${y}.zarr.zip"; return 0; fi
  if [[ -d "$d/shard_${y}.zarr" ]]; then printf '%s\n' "$d/shard_${y}.zarr"; return 0; fi
  return 1
}

# config_years <config.yaml> [all|dataset|validation] -- the years that config reads, space
# separated. Keeps staging (and the stats build) tied to what the run actually uses, instead of
# a hardcoded list that can drift. `dataset` is the TRAIN years, which is what normalization
# stats must be computed over -- never the validation year. Needs the venv active (PyYAML).
config_years() {
  python - "${1:?config path required}" "${2:-all}" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1])) or {}
which = sys.argv[2]
years = []
if which in ("all", "dataset"):
    years += list((cfg.get("dataset") or {}).get("years") or [])
if which in ("all", "validation"):
    years += list((cfg.get("validation") or {}).get("years") or [])
print(" ".join(str(int(y)) for y in sorted(set(years))))
PY
}

# stage_shards <src-dir> <dest-dir> <year>... -- copy each year's shard (either form) to dest.
# Warns on a year that is absent, fails only if nothing at all could be staged.
stage_shards() {
  local src_dir="${1:?}" dest="${2:?}"; shift 2
  local y src staged=0 missing=""
  mkdir -p "$dest"
  for y in "$@"; do
    if src=$(shard_src "$src_dir" "$y"); then
      cp -r "$src" "$dest/"
      staged=$(( staged + 1 ))
    else
      missing+="$y "
    fi
  done
  if [[ -n "$missing" ]]; then
    echo "WARNING: no shard_YYYY.zarr[.zip] for year(s) ${missing% } in $src_dir." >&2
    echo "         Staging continues; the dataset fails loudly if the run needs them." >&2
  fi
  if (( staged == 0 )); then
    echo "ERROR: staged 0 shards from $src_dir -- nothing to train on." >&2
    return 1
  fi
  echo "staged $staged shard(s) -> $dest"
}

# ---------------------------------------------------------------------------------------
# Which two nets to evaluate
#
# Generation is `regression mean + diffusion residual`, so evaluation needs BOTH nets -- and
# WORKFLOW.md section B trains a diffusion run against an EARLIER regression run, so they
# normally live in different directories (regression_2/, diffusion_2/). There is therefore no
# single "the run" to point at: a diffusion run dir holds only checkpoints_diffusion, so
# auto-discovery from one directory can never find the regression net for a real run.
#
# Hence two run dirs. OUTPUT_DIR is kept as a fallback that sets both, for the case where one
# directory happens to hold both nets (and so older commands keep working).
# ---------------------------------------------------------------------------------------

# resolve_eval_paths -- decide which checkpoints to evaluate and where the NetCDFs go.
#   reads/sets: REG_RUN RES_RUN REG_CKPT RES_CKPT NC_DIR   (OUTPUT_DIR read as a fallback)
# Returns 1 with an explanation if either net cannot be resolved.
resolve_eval_paths() {
  OUTPUT_DIR="${OUTPUT_DIR:-}"
  REG_RUN="${REG_RUN:-$OUTPUT_DIR}"
  RES_RUN="${RES_RUN:-$OUTPUT_DIR}"
  REG_CKPT="${REG_CKPT:-}"
  RES_CKPT="${RES_CKPT:-}"

  if [[ -z "$REG_CKPT" && -n "$REG_RUN" ]]; then
    REG_CKPT=$(latest_ckpt "$REG_RUN/checkpoints_regression" || true)
  fi
  if [[ -z "$RES_CKPT" && -n "$RES_RUN" ]]; then
    RES_CKPT=$(latest_ckpt "$RES_RUN/checkpoints_diffusion" || true)
  fi

  if [[ -z "$REG_CKPT" || ! -f "$REG_CKPT" ]]; then
    echo "ERROR: no regression checkpoint." >&2
    echo "       Set REG_RUN=<dir containing checkpoints_regression/> or REG_CKPT=<file>." >&2
    echo "       A diffusion run dir holds only checkpoints_diffusion -- the regression net" >&2
    echo "       lives in the run it was trained against (WORKFLOW.md section B). e.g." >&2
    echo "         REG_RUN=\$SCRATCH/corrdiff_runs/regression_2 \\" >&2
    echo "         RES_RUN=\$SCRATCH/corrdiff_runs/diffusion_2" >&2
    return 1
  fi
  if [[ -z "$RES_CKPT" || ! -f "$RES_CKPT" ]]; then
    echo "ERROR: no diffusion checkpoint -- needed for the full model." >&2
    echo "       Set RES_RUN=<dir containing checkpoints_diffusion/> or RES_CKPT=<file>." >&2
    return 1
  fi

  # NetCDFs land with the run being evaluated (the diffusion one). When only RES_CKPT was
  # given, its run dir is two levels up from the checkpoint file.
  if [[ -z "${NC_DIR:-}" ]]; then
    local base="$RES_RUN"
    [[ -n "$base" ]] || base=$(dirname "$(dirname "$RES_CKPT")")
    NC_DIR="$base/eval"
  fi
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
