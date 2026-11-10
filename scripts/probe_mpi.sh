#!/bin/bash
# Why does an import abort in MPI_Init_thread inside a job but not on a login node?
#
# THE SYMPTOM (generate job 3213027, 2026-11-09):
#   OPAL ERROR: Unreachable in file ext3x_client.c at line 111
#   The application appears to have been direct launched using "srun", but OMPI was not built
#   with SLURM's PMI support ... An error occurred in MPI_Init_thread on a NULL communicator
#
# THE MECHANISM, and why the login-node test is useless: Alliance's netCDF4 is built WITH
# parallel support, so importing it pulls in mpi4py, and importing mpi4py.MPI calls
# MPI_Init_thread. OpenMPI then looks at the environment, sees SLURM_* variables, concludes it
# was launched by srun, and tries to reach a PMIx server. Under torchrun there is no such
# server, so the client reports "Unreachable" and MPI aborts. On a login node there are no
# SLURM_* variables, OpenMPI initialises as a singleton, and the identical import succeeds --
# which is why "it imports fine" proves nothing.
#
# WHY NOW: nothing in the venv has changed since 2026-07-30 and the module line is original;
# the same scripts generated successfully in August. But SLURM's job counter restarted
# (61298029 in September -> 3213027 in November), which means SLURM was reinstalled or
# upgraded -- and SLURM's PMI/PMIx is exactly what the error is about.
#
# THE CANDIDATE FIX: mpi4py reads MPI4PY_RC_* from the environment (>=3.1), so
# MPI4PY_RC_INITIALIZE=0 makes `import mpi4py.MPI` skip MPI_Init entirely. generate.py writes
# its NetCDF serially, so nothing here needs a live MPI communicator.
#
# MUST RUN INSIDE AN ALLOCATION -- that is the whole point:
#     salloc --time=20:00 --mem=8G --cpus-per-task=2 --account=def-stockie_cpu
#     bash scripts/probe_mpi.sh

set -uo pipefail

REPO="${REPO:-$HOME/thesis/era5-carra2-downscaling-canadian-arctic}"
ENV_DIR="${ENV_DIR:-$HOME/corrdiff-env}"

echo "=============================================================="
echo " MPI init probe"
echo "=============================================================="
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "WARNING: \$SLURM_JOB_ID is unset -- you are NOT in an allocation." >&2
  echo "         The failure only reproduces where SLURM_* variables exist, so this run" >&2
  echo "         will almost certainly pass and tell you nothing. Use salloc first." >&2
else
  echo "allocation: job $SLURM_JOB_ID on $(hostname -s)"
fi
echo "slurm vars: $(env | grep -c '^SLURM_') set"

module load python/3.11 mpi4py/4.1.0 >/dev/null 2>&1
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate" 2>/dev/null || { echo "ERROR: no venv at $ENV_DIR" >&2; exit 1; }
echo "python:     $(command -v python)"
# Record the MPI stack versions. The error names ext3x -- OpenMPI's EXTERNAL PMIx 3.x client --
# while a pmix/4.x module may be what is loaded, and that mismatch is a candidate cause. These
# are also the numbers to compare against if this ever works again.
echo "openmpi:    ${EBVERSIONOPENMPI:-<unset>}   pmix: ${EBVERSIONPMIX:-<unset>}" \
     "  mpi4py: ${EBVERSIONMPI4PY:-<unset>}"
echo "node:       $(hostname -s)   (the September failures were fc10514 / fc10512, November fc10609)"
echo

cat > "${TMPDIR:-/tmp}/_probe_nc.py" <<'PY'
import os, sys
print(f"  MPI4PY_RC_INITIALIZE={os.environ.get('MPI4PY_RC_INITIALIZE', '<unset>')}")
import netCDF4
print(f"  netCDF4 {netCDF4.__version__}")
print(f"  parallel4 support = {getattr(netCDF4, '__has_parallel4_support__', '?')}")
print(f"  pnetcdf  support = {getattr(netCDF4, '__has_pnetcdf_support__', '?')}")
print(f"  mpi4py.MPI imported as a side effect = {'mpi4py.MPI' in sys.modules}")
if "mpi4py.MPI" in sys.modules:
    from mpi4py import MPI
    print(f"  MPI initialised = {MPI.Is_initialized()}")
print("  IMPORT OK")
PY

echo "---- 1. plain python, MPI init allowed (expect the abort) ----"
python "${TMPDIR:-/tmp}/_probe_nc.py" 2>&1 | tail -20
echo "   exit=$?"
echo

echo "---- 2. plain python, MPI4PY_RC_INITIALIZE=0 ------------------"
MPI4PY_RC_INITIALIZE=0 python "${TMPDIR:-/tmp}/_probe_nc.py" 2>&1 | tail -20
echo "   exit=$?"
echo

echo "---- 3. under torchrun, as the job does ----------------------"
torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  "${TMPDIR:-/tmp}/_probe_nc.py" 2>&1 | tail -20
echo "   exit=$?"
echo

echo "---- 4. under torchrun, MPI4PY_RC_INITIALIZE=0 ---------------"
MPI4PY_RC_INITIALIZE=0 torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  "${TMPDIR:-/tmp}/_probe_nc.py" 2>&1 | tail -20
echo "   exit=$?"
echo

echo "=============================================================="
echo " How to read this:"
echo "   3 fails, 4 passes  -> MPI4PY_RC_INITIALIZE=0 is the fix; it goes in the job scripts."
echo "   1 and 3 both fail  -> any MPI init inside an allocation is broken, not a torchrun"
echo "                         issue. Worth an Alliance support ticket citing the SLURM"
echo "                         upgrade, quoting ext3x_client.c:111."
echo "   all four pass      -> netCDF4 is not the importer; something else in generate.py's"
echo "                         chain calls MPI_Init. Re-run the job and read the .err again."
echo "   parallel4 = False  -> netCDF4 never imports mpi4py, so look elsewhere entirely."
echo "=============================================================="
