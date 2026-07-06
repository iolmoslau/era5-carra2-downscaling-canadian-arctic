"""Verify built sample shards: completeness, integrity, and cross-shard consistency.

    python scripts/verify_shards.py <dir-with-shards | glob>

Checks per shard: full set of 3-hourly timestamps present (no gaps/dups), no NaNs in hr/lr,
and physically plausible per-channel ranges. Across shards: identical LR/HR grids, channel
order, land-sea mask, and patch centre -- i.e. that they can be concatenated as one dataset.
"""

import glob
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "."
    stores = (sorted(glob.glob(os.path.join(arg, "shard_*.zarr")))
              if os.path.isdir(arg) else sorted(glob.glob(arg)))
    if not stores:
        print(f"no shard_*.zarr found at: {arg}")
        sys.exit(1)

    ref = None
    total, total_exp, all_ok = 0, 0, True
    print(f"Verifying {len(stores)} shard(s)\n" + "=" * 78)

    for s in stores:
        name = os.path.basename(s)
        z = xr.open_zarr(s)
        try:
            yr = int(name.split("_")[1][:4])
        except ValueError:
            yr = pd.to_datetime(z["time"].values[0]).year

        t = pd.to_datetime(z["time"].values)
        exp = pd.date_range(f"{yr}-01-01", f"{yr}-12-31 21:00", freq="3h")
        missing = sorted(set(exp) - set(t))
        dups = len(t) - len(set(t))
        hr_nan = int(np.isnan(z["hr"]).sum())
        lr_nan = int(np.isnan(z["lr"]).sum())
        ok = not missing and dups == 0 and hr_nan == 0 and lr_nan == 0
        all_ok &= ok
        total += len(t)
        total_exp += len(exp)

        flag = "OK " if ok else "!! "
        print(f"{flag}{name:20s} {len(t):5d}/{len(exp)} steps  "
              f"missing={len(missing)} dups={dups} hr_nan={hr_nan} lr_nan={lr_nan}"
              + (f"  first_missing={str(missing[0])[:13]}" if missing else ""))

        # cross-shard consistency vs the first shard
        if ref is None:
            ref = dict(lat=z["lat"].values, lon=z["lon"].values,
                       hr_lat=z["hr_lat"].values, hr_lon=z["hr_lon"].values,
                       lsm=z["land_sea_mask"].values,
                       hc=list(z.attrs["hr_channels"]), lc=list(z.attrs["lr_channels"]),
                       center=(z.attrs.get("center_lat"), z.attrs.get("center_lon")))
        else:
            geo_ok = (np.allclose(ref["lat"], z["lat"].values)
                      and np.allclose(ref["lon"], z["lon"].values)
                      and np.allclose(ref["hr_lat"], z["hr_lat"].values)
                      and np.allclose(ref["hr_lon"], z["hr_lon"].values)
                      and np.array_equal(ref["lsm"], z["land_sea_mask"].values)
                      and ref["hc"] == list(z.attrs["hr_channels"])
                      and ref["lc"] == list(z.attrs["lr_channels"])
                      and ref["center"] == (z.attrs.get("center_lat"), z.attrs.get("center_lon")))
            if not geo_ok:
                all_ok = False
                print(f"   !! geometry/channels/mask DIFFER from {os.path.basename(stores[0])}")

    print("=" * 78)
    print(f"channels: HR={ref['hc']}  LR={ref['lc']}")
    print(f"grids: LR {ref['lat'].shape[0]}x{ref['lon'].shape[0]}  HR {ref['hr_lat'].shape}  "
          f"centre={ref['center']}")
    print(f"TOTAL samples: {total} / {total_exp} expected")
    print("RESULT:", "ALL GOOD ✅" if all_ok else "PROBLEMS FOUND — see !! lines above")

    # quick physical-range spot check on the last shard opened
    print("\nper-channel range (last shard):")
    for var, cdim, names in (("hr", "hr_channel", ref["hc"]), ("lr", "lr_channel", ref["lc"])):
        mn = z[var].min(dim=[d for d in z[var].dims if d != cdim]).values
        mx = z[var].max(dim=[d for d in z[var].dims if d != cdim]).values
        for nm, a, b in zip(names, mn, mx):
            print(f"  {nm:8s} [{a:10.2f}, {b:10.2f}]")


if __name__ == "__main__":
    main()
