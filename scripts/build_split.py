"""CLI entry point for building one split's sample zarr (run from the repo root).

    python scripts/build_split.py --store PATH --start YYYY-MM-DD --end YYYY-MM-DD \
        --work-dir PATH [--center-lat 67.2 --center-lon -130.23 --patch 448 --chunk-days 1]

Downloads CARRA2 (full domain, per month) + ERA5 (CDS area-subset), crops the fixed patch,
and writes/append the samples to the zarr at --store, discarding the transient downloads.
Resumable: re-running with the same --store appends (so a timed-out job can be requeued).
Requires a configured ~/.cdsapirc and accepted CARRA2 + ERA5 licences, and INTERNET ACCESS.
"""

import argparse
import os
import sys

# Make the repo root importable regardless of how this script is invoked.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_acquisition.dataset_builder import build_dataset  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store", required=True, help="output zarr path (persistent storage)")
    p.add_argument("--start", required=True, help="inclusive start date, e.g. 2013-01-21")
    p.add_argument("--end", required=True, help="inclusive end date, e.g. 2013-01-22")
    p.add_argument("--work-dir", required=True, help="scratch dir for transient downloads")
    p.add_argument("--center-lat", type=float, default=67.2)
    p.add_argument("--center-lon", type=float, default=-130.23)
    p.add_argument("--patch", type=int, default=448)
    p.add_argument("--chunk-days", type=int, default=1,
                   help="days assembled+written per write (1=day .. >=31=whole month)")
    a = p.parse_args()

    build_dataset(
        a.store,
        center=(a.center_lat, a.center_lon),
        patch_size=a.patch,
        start=a.start,
        end=a.end,
        work_dir=a.work_dir,
        chunk_days=a.chunk_days,
        attrs={"center_lat": a.center_lat, "center_lon": a.center_lon, "patch": a.patch,
               "levels": "500,850", "source": "CARRA2+CDS-ERA5"},
    )
    print(f"DONE: {a.store} ({a.start}..{a.end})")


if __name__ == "__main__":
    main()
