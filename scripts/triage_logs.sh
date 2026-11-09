#!/bin/bash
# What happened to the last N training jobs? One compact block per job.
#
# WHY: $REPO/logs is gitignored and lives only on the cluster, so a run that failed weeks ago
# leaves no trace in git, in results/, or on a laptop. Reconstructing it means reading several
# hundred lines of torchrun boilerplate per job. This prints only the lines that decide what
# happened, few enough to paste into a conversation.
#
#     bash scripts/triage_logs.sh              # newest 12 jobs
#     N=30 bash scripts/triage_logs.sh         # more history
#     PAT=corrdiff_reg bash scripts/triage_logs.sh    # regression jobs only
#     LOG_DIR=/path/to/logs bash scripts/triage_logs.sh
#
# The four lines worth knowing per job:
#   "Launching ... on N H100"          -- the GPU count ACTUALLY used (not what you requested)
#   "Using N gradient accumulation"    -- proof it reached the training loop at all
#   ERROR:/WARNING: from our scripts   -- the preflight guards in slurm/common.sh
#   the exception, or the SLURM kill   -- why it stopped

# NOT -e: grep exiting 1 on "no match" is the normal case throughout.
set -uo pipefail

REPO="${REPO:-$HOME/thesis/era5-carra2-downscaling-canadian-arctic}"
LOG_DIR="${LOG_DIR:-$REPO/logs}"
N="${N:-12}"
PAT="${PAT:-corrdiff_}"

[[ -d "$LOG_DIR" ]] || { echo "ERROR: no log directory at $LOG_DIR" >&2; exit 1; }

# Lines that carry signal. Deliberately narrow: the point is a block short enough to read.
SIGNALS='Launching (regression|diffusion)|gradient accumulation rounds|^ERROR|^WARNING|Error executing job|^[A-Za-z][A-Za-z.]*(Error|Exception):|raise [A-Z]|CANCELLED|TIME LIMIT|out of memory|No such file|FileNotFoundError'

# while-read rather than mapfile: macOS ships bash 3.2, and this script has to be runnable
# (and testable) off-cluster as well as on Fir.
files=()
while IFS= read -r line; do files+=("$line"); done < <(
  ls -t "$LOG_DIR"/${PAT}*.out 2>/dev/null | head -n "$N")
if (( ${#files[@]} == 0 )); then
  echo "no ${PAT}*.out logs under $LOG_DIR" >&2
  exit 1
fi

echo "== $LOG_DIR -- newest ${#files[@]} job(s) matching ${PAT}*"
echo

for out in "${files[@]}"; do
  base=$(basename "$out" .out)
  id="${base##*_}"
  err="${out%.out}.err"

  # GNU stat -c vs BSD stat -f: try each, so the timestamp shows on Fir and on a laptop.
  mtime=$(stat -c '%y' "$out" 2>/dev/null | cut -d. -f1)
  [[ -n "$mtime" ]] || mtime=$(stat -f '%Sm' -t '%Y-%m-%d %H:%M' "$out" 2>/dev/null)
  printf '===== %s   (%s)\n' "$base" "$mtime"

  # Scheduler's own verdict: state, runtime, and what was actually allocated. -X = job level only.
  if command -v sacct >/dev/null 2>&1 && [[ "$id" =~ ^[0-9]+$ ]]; then
    sacct -j "$id" -X -n -P --format=State,Elapsed,AllocTRES 2>/dev/null |
      head -1 | awk -F'|' 'NF{printf "  slurm   %s, ran %s\n          alloc %s\n", $1, $2, $3}'
  fi

  # Our own echoes plus whatever went wrong, in file order so cause precedes effect.
  grep -hE "$SIGNALS" "$out" "$err" 2>/dev/null | head -8 |
    sed 's/[[:space:]]*$//; s/^/  | /'

  # Last sign of life: the most recent progress line, which says how far it actually got.
  grep -hoE 'cur_nimg[ =:]+[0-9]+|samples[ =:]+[0-9]+' "$out" 2>/dev/null |
    tail -1 | sed 's/^/  last    /'

  echo
done

echo "Paste this whole block. If a job shows no 'gradient accumulation' line it never started"
echo "training; if it shows one, look at 'last' for how far it got before stopping."
