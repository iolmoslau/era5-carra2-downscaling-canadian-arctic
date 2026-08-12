#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Draw N random timestamps from the ERA5/CARRA2 shards for a set of years.

Reads the `time` coordinate straight from the zarr shards -- the exact times the dataset
serves -- filters to the requested year(s), and samples N of them uniformly at random WITHOUT
replacement (seeded, so a run is reproducible and the full-model and regression passes get the
*same* times). Sampling from the real time index guarantees every pick exists in the shard, and
random draws avoid the diurnal bias a fixed stride would have (a multiple-of-24h stride only
ever hits one time of day).

Emits a Hydra list literal by default, for `run_eval.sh` to pass as
`++generation.times=<...>`; use --plain for a newline-separated list.

    python sample_times.py --data-dir $DATA --years 2019 --n 100 --seed 0
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr


def load_times(data_dir: str, years: list[int]) -> np.ndarray:
    """All shard timestamps for the requested years, sorted & de-duplicated (datetime64[ns])."""
    stores = []
    for y in years:
        cand = os.path.join(data_dir, f"shard_{y}.zarr")          # dir-of-shards convention
        if os.path.exists(cand):
            stores.append(cand)
    if not stores:                                                # fall back to every store
        stores = sorted(glob.glob(os.path.join(data_dir, "*.zarr")))
    if not stores:
        sys.exit(f"no .zarr shards found under {data_dir}")

    chunks = [xr.open_zarr(s)["time"].values for s in stores]
    t = np.unique(np.concatenate(chunks))
    keep = np.isin(pd.DatetimeIndex(t).year, years)
    return np.sort(t[keep])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--years", type=int, nargs="+", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plain", action="store_true", help="newline list instead of a Hydra list")
    args = ap.parse_args()

    t = load_times(args.data_dir, args.years)
    if t.size == 0:
        sys.exit(f"no times found for years {args.years} under {args.data_dir}")

    n = min(args.n, t.size)
    if n < args.n:
        print(f"WARNING: only {t.size} times available; sampling {n} of {args.n}", file=sys.stderr)
    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(t.size, size=n, replace=False))
    iso = [pd.Timestamp(x).strftime("%Y-%m-%dT%H:%M:%S") for x in t[idx]]

    if args.plain:
        print("\n".join(iso))
    else:
        # Hydra list literal with single-quoted elements (timestamps contain ':').
        print("[" + ",".join(f"'{s}'" for s in iso) + "]")


if __name__ == "__main__":
    main()
