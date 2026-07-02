#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Trim a built sample shard to its first N timesteps -> a small zarr for Kaggle upload.

The full year shards are ~4 GB; for a GPU smoke test we only need a short slice. This keeps
the first ``--steps`` timesteps (default 240 == ~1 month at 3-hourly cadence) plus all the
static fields (grids, land-sea mask) and store attributes, so ``ERA5CARRA2Dataset`` reads it
identically to a full shard.

Example
-------
    python tools/trim_shard.py \
        --src ../testing_data/shard_2011.zarr \
        --dst ../testing_data/shard_2011_smoke.zarr --steps 240
"""

from __future__ import annotations

import argparse
from pathlib import Path

import xarray as xr


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, help="source shard_YYYY.zarr")
    ap.add_argument("--dst", required=True, help="destination (small) zarr store")
    ap.add_argument("--steps", type=int, default=240,
                    help="number of leading timesteps to keep (default 240 = ~1 month)")
    args = ap.parse_args()

    ds = xr.open_zarr(args.src)
    n = min(args.steps, ds.sizes["time"])
    sub = ds.isel(time=slice(0, n))
    # one sample per chunk (matches the full shard) so per-item reads stay cheap
    sub = sub.chunk({"time": 1})

    dst = Path(args.dst)
    if dst.exists():
        raise SystemExit(f"destination already exists: {dst}")
    sub.to_zarr(dst, mode="w-")
    print(f"[trim_shard] wrote {dst} with {n} timesteps (from {ds.sizes['time']})")


if __name__ == "__main__":
    main()
