#!/bin/bash
# Archive a run's final checkpoint from $SCRATCH to $PROJECT, verified.
#
# WHY: $SCRATCH is purged by file age (~60 days) and is not backed up. The results/ diagnostics
# are safe in git, but the weights are not -- and a diffusion run is only reproducible together
# with the regression net it was trained against, so losing either loses both.
#
#     bash scripts/archive_run.sh regression_2 regression_2_extended diffusion_2
#
# Lands at  $PROJECT/trained_models/<run>/checkpoint/  and writes a manifest.json beside it
# recording which step was kept, where it came from, and the run's channel provenance.
#
# Copies the FINAL checkpoint only -- highest nimg, selected by latest_ckpt, never by mtime.
# Both files come across: the .mdlus is the weights, the .pt the optimizer state, which you
# need only to resume or extend training later (regression_2_extended is exactly that, so it
# is worth keeping). Drop the .pt by hand afterwards if a run is inference-only.
#
# Idempotent: a run already archived is skipped unless FORCE=1.

set -euo pipefail

REPO="${REPO:-$HOME/thesis/era5-carra2-downscaling-canadian-arctic}"
source "$REPO/training_mini/slurm/common.sh"   # latest_ckpt, ckpt_nimg

# ${SCRATCH:-} rather than $SCRATCH: under `set -u` a bare reference aborts with "unbound
# variable" before the check below can say something useful about it.
RUNS_DIR="${RUNS_DIR:-${SCRATCH:-}/corrdiff_runs}"
DEST_ROOT="${DEST_ROOT:-${PROJECT:-}/trained_models}"
FORCE="${FORCE:-0}"

(( $# )) || { echo "usage: bash scripts/archive_run.sh <run-name> [run-name ...]" >&2; exit 1; }

# An unset $SCRATCH/$PROJECT collapses these to "/corrdiff_runs" and "/trained_models", which
# is what this catches. Checking the source EXISTS is both stricter and more general than
# whitelisting cluster roots -- and it keeps the script testable off-cluster.
for p in "$RUNS_DIR" "$DEST_ROOT"; do
  [[ "$p" == /* ]] || { echo "ERROR: '$p' is not an absolute path" >&2; exit 1; }
done
if [[ ! -d "$RUNS_DIR" ]]; then
  echo "ERROR: run directory '$RUNS_DIR' does not exist." >&2
  echo "       \$SCRATCH=${SCRATCH:-<unset>} -- set it, or pass RUNS_DIR=." >&2
  exit 1
fi

archived=0 skipped=0 failed=0

for NAME in "$@"; do
  echo "== $NAME"
  SRC=""
  for kind in regression diffusion; do
    cand="$RUNS_DIR/$NAME/checkpoints_$kind"
    [[ -d "$cand" ]] && SRC="$cand" && STAGE="$kind"
  done
  if [[ -z "$SRC" ]]; then
    echo "   no checkpoints_{regression,diffusion} under $RUNS_DIR/$NAME -- skipping" >&2
    failed=$(( failed + 1 )); continue
  fi

  LAST=$(latest_ckpt "$SRC" || true)
  if [[ -z "$LAST" ]]; then
    echo "   no .mdlus in $SRC -- skipping" >&2
    failed=$(( failed + 1 )); continue
  fi
  NIMG=$(ckpt_nimg "$LAST")
  DST="$DEST_ROOT/$NAME/checkpoint"

  if [[ -f "$DST/$(basename "$LAST")" && "$FORCE" != "1" ]]; then
    echo "   already archived at step $NIMG -> $DST (FORCE=1 to redo)"
    skipped=$(( skipped + 1 )); continue
  fi

  mkdir -p "$DST"
  echo "   stage=$STAGE  step=$NIMG"
  shopt -s nullglob
  files=("$SRC"/*."$NIMG".mdlus "$SRC"/*."$NIMG".pt)
  shopt -u nullglob
  for f in "${files[@]}"; do
    cp -p "$f" "$DST/"
    # A silently truncated copy is the failure that matters here, because the source is the
    # thing that disappears. Byte-compare rather than trust the exit status.
    if cmp -s "$f" "$DST/$(basename "$f")"; then
      printf '   copied  %-58s %s\n' "$(basename "$f")" "$(du -h "$f" | cut -f1)"
    else
      echo "   ERROR: copy of $(basename "$f") does not match the source" >&2
      failed=$(( failed + 1 )); continue 2
    fi
  done

  # Manifest: what was kept, and enough provenance to pair it back up later.
  python - "$NAME" "$STAGE" "$NIMG" "$SRC" "$DST" "$REPO" <<'PY'
import json, sys, datetime, pathlib
name, stage, nimg, src, dst, repo = sys.argv[1:7]
info = {"run": name, "stage": stage, "nimg": int(nimg), "source": src,
        "archived": datetime.date.today().isoformat(),
        "files": sorted(p.name for p in pathlib.Path(dst).iterdir() if p.is_file()
                        and p.suffix in (".mdlus", ".pt"))}
# carry the channel provenance across from the collected run, so the archive stands alone
ri = pathlib.Path(repo) / "training_mini/results" / name / "run_info.json"
if ri.is_file():
    try:
        prov = (json.loads(ri.read_text()).get("provenance") or {})
        info["provenance"] = {k: prov.get(k) for k in
                              ("config", "sea_ice", "lr_n", "in_channels", "data", "stats")}
    except Exception as e:
        info["provenance_error"] = f"{type(e).__name__}: {e}"
if stage == "diffusion":
    info["note"] = ("generation is regression mean + diffusion residual: this checkpoint is "
                    "only reproducible together with the regression run it was trained against")
pathlib.Path(dst, "manifest.json").write_text(json.dumps(info, indent=2) + "\n")
print(f"   manifest {dst}/manifest.json")
PY
  archived=$(( archived + 1 ))
done

echo
echo "archived $archived, skipped $skipped, failed $failed  ->  $DEST_ROOT"
(( failed == 0 )) || exit 1
