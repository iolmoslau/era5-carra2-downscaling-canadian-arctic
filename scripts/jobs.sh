#!/bin/bash
# Recent SLURM jobs, and where each one's logs actually went.
#
# WHY: "the job ran but there are no logs" is almost always a path question, not a missing file.
# SLURM cannot expand variables in #SBATCH lines, so a relative --output=logs/... resolves
# against the SUBMISSION directory -- launch the same script from $HOME instead of through
# slurm/submit.sh and the logs land in ~/logs, silently. sacct's WorkDir column is the answer,
# and scontrol knows the exact StdOut path for jobs recent enough to still be in its memory
# (MinJobAge, typically 5 minutes after completion).
#
#     bash scripts/jobs.sh                 # today
#     DAYS=3 bash scripts/jobs.sh          # last 3 days
#     bash scripts/jobs.sh 61298029        # one job, in detail
#
# A job listed here with no file at its StdOut path either never started (check State) or wrote
# somewhere else entirely -- the LOG CANDIDATES block says which paths were checked.

set -uo pipefail

command -v sacct >/dev/null 2>&1 || {
  echo "ERROR: no sacct on PATH -- run this on the cluster, not locally." >&2; exit 1; }

REPO="${REPO:-$HOME/thesis/era5-carra2-downscaling-canadian-arctic}"
DAYS="${DAYS:-1}"
JOB="${1:-}"

if [[ -n "$JOB" ]]; then
  echo "===== job $JOB"
  sacct -j "$JOB" -X -o JobID%14,JobName%20,State%14,Elapsed,Submit,End | sed 's/^/  /'
  echo
  echo "  WorkDir (where relative --output paths resolve):"
  sacct -j "$JOB" -X -n -P -o WorkDir | head -1 | sed 's/^/    /'
  echo
  # scontrol holds the resolved StdOut/StdErr, but only while the job is still in its memory.
  if scontrol show job "$JOB" >/dev/null 2>&1; then
    echo "  scontrol (authoritative, while still cached):"
    scontrol show job "$JOB" | grep -E 'StdOut|StdErr|WorkDir|JobState' | sed 's/^ */    /'
  else
    echo "  (scontrol has forgotten this job -- past MinJobAge; use the candidates below)"
  fi
  echo
  echo "  LOG CANDIDATES:"
  wd=$(sacct -j "$JOB" -X -n -P -o WorkDir | head -1)
  name=$(sacct -j "$JOB" -X -n -P -o JobName | head -1)
  for p in "$REPO/logs/${name}_${JOB}.out" "$wd/logs/${name}_${JOB}.out" \
           "$wd/slurm-${JOB}.out" "$HOME/logs/${name}_${JOB}.out" "$HOME/slurm-${JOB}.out"; do
    [[ -f "$p" ]] && printf '    FOUND  %s  (%s)\n' "$p" "$(du -h "$p" | cut -f1)" \
                  || printf '    --     %s\n' "$p"
  done
  exit 0
fi

echo "===== jobs since $DAYS day(s) ago"
sacct -u "$USER" -X -S "now-${DAYS}days" \
  -o JobID%14,JobName%20,State%14,Elapsed%10,Submit%20,WorkDir%55 | sed 's/^/  /'

echo
echo "===== log files present under \$REPO/logs (newest 10)"
if [[ -d "$REPO/logs" ]]; then
  ls -t "$REPO/logs"/*.out 2>/dev/null | head -10 | while read -r f; do
    printf '  %-10s %s\n' "$(du -h "$f" | cut -f1)" "$f"
  done
  [[ -n "$(ls "$REPO/logs"/*.out 2>/dev/null)" ]] || echo "  (none)"
else
  echo "  (no $REPO/logs directory)"
fi

echo
echo "===== stray logs elsewhere (default slurm-<id>.out, or a logs/ in the wrong place)"
# -newermt rather than -mtime: a job from 20 minutes ago should show up.
find "$HOME" "${SCRATCH:-/nonexistent}" -maxdepth 3 \
     \( -name 'slurm-*.out' -o -name 'corrdiff_*.out' -o -name 'corrdiff_*.err' \) \
     -newermt "-${DAYS} days" 2>/dev/null | grep -v "^$REPO/logs/" | head -20 |
  sed 's/^/  /'
echo
echo "Tip: a job whose WorkDir is not \$REPO was not submitted through slurm/submit.sh."
echo "     Run 'bash scripts/jobs.sh <jobid>' for one job's resolved log paths."
