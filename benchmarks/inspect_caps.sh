#!/bin/bash
# Interrogate the CAPS dynamical-downscaling archive BEFORE writing an adapter for it.
#
# WHY: four facts decide how benchmarks/adapt_caps.py must be written, and all four are cheap to
# read off one file and expensive to guess wrong:
#
#   1. VARIABLES -- is t2m / u10 / v10 actually in here, and under what names?
#   2. UNITS     -- K or degC; m/s. A unit error is a silent, plausible-looking bias.
#   3. GRID      -- projection, spacing, extent. Does it even cover our 448x448 patch?
#   4. WIND FRAME -- grid-relative or earth-relative. THIS IS THE ONE THAT MATTERS MOST: grid
#      convergence is ~100 degrees over this domain, it cost us a full retrain once already, and
#      it is invisible in RMSE until you plot direction. wgrib2 says "winds(grid)" or
#      "winds(earth)"; eccodes exposes it as uvRelativeToGrid (1 = grid, 0 = earth).
#
# It also reports which GRIB readers the Python env actually has, since that decides whether the
# adapter can use xarray+cfgrib or has to shell out to wgrib2.
#
#     bash benchmarks/inspect_caps.sh                     # one sample member, fast
#     SAMPLE=caps/2021071512_006.grib2 bash benchmarks/inspect_caps.sh
#     MANIFEST=1 bash benchmarks/inspect_caps.sh          # + summarise all ~8,756 names (slow)
#     EXTRACTED=$SCRATCH/dynamical bash benchmarks/inspect_caps.sh   # already untarred
#
# Reads ONE member out of the archive by default and stops there -- it does not extract 12 GB.

set -uo pipefail

ARCHIVE="${ARCHIVE:-${PROJECT:-}/data/dynamical/caps.tar.gz}"
EXTRACTED="${EXTRACTED:-}"
WORK="${WORK:-${SLURM_TMPDIR:-${SCRATCH:-/tmp}}/caps_inspect}"
SAMPLE="${SAMPLE:-}"

echo "=============================================================="
echo " CAPS archive inspection"
echo "=============================================================="

# ---------------------------------------------------------------- 1. find one member to look at
FILE=""
if [[ -n "$EXTRACTED" && -d "$EXTRACTED" ]]; then
  FILE=$(find "$EXTRACTED" -name '*.grib2' -type f 2>/dev/null | sort | head -1)
  [[ -n "$FILE" ]] && echo "sample (already extracted): $FILE"
fi

if [[ -z "$FILE" ]]; then
  if [[ ! -f "$ARCHIVE" ]]; then
    echo "ERROR: no archive at $ARCHIVE" >&2
    echo "       set ARCHIVE=/path/to/caps.tar.gz, or EXTRACTED=/dir/with/grib2" >&2
    exit 1
  fi
  echo "archive: $ARCHIVE  ($(du -h "$ARCHIVE" | cut -f1))"
  mkdir -p "$WORK" || exit 1
  # --occurrence=1 stops tar after the first match, so this decompresses only as far as it must
  # rather than streaming all 12 GB. A .tar.gz is not seekable, hence "as far as it must".
  # --occurrence/--wildcards are GNU tar only (Fir has it; macOS bsdtar does not), so detect
  # rather than assume -- otherwise this is untestable off-cluster and fails confusingly on it.
  if tar --version 2>/dev/null | grep -qi 'gnu tar'; then
    echo "extracting one member -> $WORK (stops at the first match)"
    if [[ -n "$SAMPLE" ]]; then
      tar -xzf "$ARCHIVE" -C "$WORK" --occurrence=1 "$SAMPLE" 2>/dev/null
    else
      tar -xzf "$ARCHIVE" -C "$WORK" --occurrence=1 --wildcards '*.grib2' 2>/dev/null
    fi
  else
    # bsdtar cannot stop early; name one member so it at least writes only that one.
    if [[ -z "$SAMPLE" ]]; then
      echo "ERROR: this tar is not GNU tar, so it cannot stop at the first match." >&2
      echo "       Pass SAMPLE=caps/<one file>.grib2, or extract the archive and use" >&2
      echo "       EXTRACTED=<dir>. (On Fir, GNU tar is present and no flag is needed.)" >&2
      exit 1
    fi
    echo "extracting $SAMPLE -> $WORK (bsdtar: scans the whole archive)"
    tar -xzf "$ARCHIVE" -C "$WORK" "$SAMPLE" 2>/dev/null
  fi
  FILE=$(find "$WORK" -name '*.grib2' -type f | sort | head -1)
fi

if [[ -z "$FILE" || ! -f "$FILE" ]]; then
  echo "ERROR: could not obtain a .grib2 member to inspect" >&2
  exit 1
fi
echo
echo "---- sample member -------------------------------------------"
ls -lh "$FILE" | awk '{print "  "$5"  "$NF}'

# ---------------------------------------------------------------- 2. inventory + grid + winds
found_tool=0

if command -v wgrib2 >/dev/null 2>&1; then
  found_tool=1
  echo
  echo "---- wgrib2 inventory (one line per record) ------------------"
  wgrib2 "$FILE" 2>&1 | head -40
  echo
  echo "---- wgrib2 -V on record 1 (grid, units, wind frame) --------"
  wgrib2 -d 1 -V "$FILE" 2>&1 | head -40
  echo
  echo "---- WIND FRAME (the decisive line) -------------------------"
  wgrib2 -V "$FILE" 2>&1 | grep -oE 'winds\((grid|earth)\)' | sort -u | sed 's/^/  /' \
    || echo "  (no winds(...) line -- check whether u/v are present at all)"
fi

if command -v grib_ls >/dev/null 2>&1; then
  found_tool=1
  echo
  echo "---- eccodes grib_ls ----------------------------------------"
  grib_ls -p shortName,name,units,level,typeOfLevel,stepRange "$FILE" 2>&1 | head -40
  echo
  echo "---- eccodes grid geometry ----------------------------------"
  # Dx/Dy decide whether the adapter interpolates DOWN to the 2.5 km patch or UP to it.
  grib_ls -p gridType,Ni,Nj,DxInMetres,DyInMetres,LaDInDegrees,orientationOfTheGridInDegrees,iScansNegatively,jScansPositively \
    "$FILE" 2>&1 | head -12
  echo
  echo "---- eccodes corner coordinates -----------------------------"
  # grib_get_data prints lat/lon/value per point; the first and last lines are opposite corners.
  # 1.2M lines is a second or two and gives the extent exactly, rather than deriving it from the
  # projection parameters and hoping the scanning flags were read the right way round.
  grib_get_data -w count=1 "$FILE" 2>/dev/null |
    awk 'NR==2{print "  first point  lat="$1"  lon="$2} END{print "  last point   lat="$1"  lon="$2}'

  echo
  echo "---- WIND FRAME (the decisive fact) -------------------------"
  # uvRelativeToGrid is undefined for some GRIB2 templates -- it came back "not_found" for CAPS.
  # The authority is GRIB2 Flag Table 3.3, bit 5 of resolutionAndComponentFlags (counting from
  # the MSB as bit 1): SET = u/v resolved along increasing x/y of the DEFINED GRID; CLEAR =
  # resolved along easterly/northerly. So bit 5 set means a de-rotation is required.
  rcf=$(grib_get -w count=1 -p resolutionAndComponentFlags "$FILE" 2>/dev/null | tr -d ' ')
  uvr=$(grib_get -w count=1 -p uvRelativeToGrid "$FILE" 2>/dev/null | tr -d ' ')
  echo "  resolutionAndComponentFlags = ${rcf:-<unreadable>}"
  echo "  uvRelativeToGrid            = ${uvr:-<not defined for this template>}"
  if [[ "$rcf" =~ ^[0-9]+$ ]]; then
    # bit 5 from the MSB of an 8-bit field == numeric value 8
    if (( (rcf & 8) != 0 )); then
      echo "  -> bit 5 SET: winds are GRID-RELATIVE. adapt_caps.py MUST de-rotate."
    else
      echo "  -> bit 5 clear: winds are EARTH-RELATIVE. No de-rotation needed."
    fi
  else
    echo "  -> could not read the flag; cross-check with 'wgrib2 -V' (winds(grid)/winds(earth))"
  fi
fi

if (( found_tool == 0 )); then
  echo
  echo "---- no wgrib2 / grib_ls on PATH ----------------------------"
  echo "  try:  module load wgrib2     or     module load eccodes"
  echo "  module spider wgrib2  /  module spider eccodes   to find them"
fi

# ---------------------------------------------------------------- 3. Python readers
echo
echo "---- GRIB readers in the Python env -------------------------"
# Activate corrdiff-env first. Probing bare `python` tests the SYSTEM interpreter and reported
# "no xarray" on a cluster that trains with xarray daily -- a measurement of the wrong thing.
ENV_DIR="${ENV_DIR:-$HOME/corrdiff-env}"
if [[ -f "$ENV_DIR/bin/activate" ]]; then
  module load python/3.11 mpi4py/4.1.0 >/dev/null 2>&1
  # shellcheck disable=SC1091
  source "$ENV_DIR/bin/activate"
  echo "  (env: $ENV_DIR)"
else
  echo "  (no venv at $ENV_DIR -- probing '$(command -v python || echo python)')"
fi
python - "$FILE" <<'PY' 2>&1 | sed 's/^/  /'
import importlib, sys

path = sys.argv[1]
for mod in ("cfgrib", "eccodes", "pygrib", "xarray"):
    try:
        m = importlib.import_module(mod)
        print(f"{mod:10s} OK   {getattr(m, '__version__', '?')}")
    except Exception as e:
        print(f"{mod:10s} --   {type(e).__name__}: {e}")

# If a reader exists, the variable names and shapes it reports are what the adapter will see --
# which can differ from wgrib2's naming, so report them too rather than assuming.
try:
    import xarray as xr
    ds = xr.open_dataset(path, engine="cfgrib")
    print("\nxarray+cfgrib opened it:")
    print(f"  dims   {dict(ds.sizes)}")
    print(f"  vars   {list(ds.data_vars)}")
    for v in ds.data_vars:
        a = ds[v].attrs
        print(f"    {v:12s} units={a.get('units','?'):8s} {a.get('long_name','')}")
    print(f"  coords {list(ds.coords)}")
except Exception as e:
    print(f"\nxarray+cfgrib could not open it: {type(e).__name__}: {e}")
PY

# ---------------------------------------------------------------- 4. optional: whole manifest
if [[ "${MANIFEST:-0}" == "1" && -f "$ARCHIVE" ]]; then
  echo
  echo "---- full manifest (decompresses the whole stream, minutes) --"
  tar -tzf "$ARCHIVE" | grep '\.grib2$' | sed 's|.*/||' > "$WORK/names.txt"
  echo "  members:    $(wc -l < "$WORK/names.txt")"
  echo "  init hours: $(sed 's/^[0-9]\{8\}\([0-9]\{2\}\)_.*/\1/' "$WORK/names.txt" | sort -u | tr '\n' ' ')"
  echo "  lead times: $(sed 's/.*_\([0-9]\{3\}\)\.grib2/\1/' "$WORK/names.txt" | sort -u | tr '\n' ' ')"
  echo "  years:      $(cut -c1-4 "$WORK/names.txt" | sort -u | tr '\n' ' ')"
  echo "  first/last: $(sort "$WORK/names.txt" | head -1) .. $(sort "$WORK/names.txt" | tail -1)"
fi

echo
echo "=============================================================="
echo " Paste this whole block. The four things I need from it:"
echo "   variables & names | units | grid/extent | WIND FRAME"
echo "=============================================================="
