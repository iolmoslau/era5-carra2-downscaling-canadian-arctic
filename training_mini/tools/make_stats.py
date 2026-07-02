#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Compute train-only normalization stats for the CorrDiff ERA5->CARRA2 adapter.

Reuses ``dataloading.stats.compute_norm_stats`` (LR + HR per-channel mean/std over the
TRAIN shards only) and appends static land-sea-mask stats, writing one JSON that
``datasets/era5_carra2.py`` reads. The no-sea-ice variant reuses the same file and just
selects a subset of the LR channels.

Example
-------
    python tools/make_stats.py --data-dir $PROJECT/data \
        --years 2011 2012 2013 2014 2015 2016 2017 2018 \
        --out $PROJECT/data/stats_train_2011_2018.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dataloading.stats import compute_norm_stats  # noqa: E402


def resolve_stores(data_dir, years, stores):
    if stores:
        return [str(Path(s)) for s in stores]
    if not (data_dir and years):
        raise SystemExit("provide either --stores, or both --data-dir and --years")
    return [str(Path(data_dir) / f"shard_{int(y)}.zarr") for y in years]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", help="directory holding shard_YYYY.zarr")
    ap.add_argument("--years", nargs="+", type=int, help="TRAIN years, e.g. 2011..2018")
    ap.add_argument("--stores", nargs="+", help="explicit shard store paths (overrides --data-dir/--years)")
    ap.add_argument("--out", required=True, help="output stats JSON path")
    args = ap.parse_args()

    stores = resolve_stores(args.data_dir, args.years, args.stores)
    for s in stores:
        if not Path(s).exists():
            raise SystemExit(f"store not found: {s}")
    print(f"[make_stats] computing LR/HR stats over {len(stores)} shard(s):")
    for s in stores:
        print(f"  - {s}")

    stats = compute_norm_stats(stores)  # LR + HR per-channel mean/std + channel-name lists

    # Static land-sea mask stats (shared geometry across shards -> use the first store).
    with xr.open_zarr(stores[0]) as z:
        mask = np.asarray(z["land_sea_mask"].values, dtype="float64")
    stats["lsm_mean"] = float(mask.mean())
    stats["lsm_std"] = float(mask.std())

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[make_stats] wrote {args.out}")
    print(f"  lr_channels = {stats['lr_channels']}")
    print(f"  hr_channels = {stats['hr_channels']}")
    print(f"  lsm_mean={stats['lsm_mean']:.4f} lsm_std={stats['lsm_std']:.4f}")


if __name__ == "__main__":
    main()
