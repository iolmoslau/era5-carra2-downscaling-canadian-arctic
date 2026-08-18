#!/usr/bin/env python
"""De-rotate CARRA2 (HR) 10 m winds from grid-relative to earth-relative (true E/N).

CARRA2 is HARMONIE-AROME on a Lambert grid, and its stored ``u10``/``v10`` are along the grid
axes. At this domain (~-130 deg lon, far from the projection's central meridian) the grid axes
are rotated ~100 deg from geographic E/N, so the stored HR winds are in a different frame than
ERA5's earth-relative LR winds. This rewrites the HR ``u10``/``v10`` to earth-relative using the
per-cell grid-convergence angle computed from the store's own 2-D ``hr_lat``/``hr_lon`` -- no
re-download, and ``data_acquisition`` is untouched.

Only the HR ``u10``/``v10`` channels change; ``t2m`` (a scalar) and every LR channel are copied
through unchanged. Writes a NEW store (safe); replace the original yourself once happy.

    python scripts/derotate_winds.py --src shard_2011.zarr --dst shard_2011_derot.zarr
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dataloading.dataset import open_store, store_exists  # noqa: E402


def grid_convergence(lat2d: np.ndarray, lon2d: np.ndarray) -> np.ndarray:
    """Per-cell angle (radians) of the grid +x axis relative to true east (CCW positive).

    Derived purely from the 2-D coordinate fields: step +1 in the grid's x (column) direction
    and measure the bearing of that displacement in local east/north space.
    """
    lat = np.asarray(lat2d, dtype=float)
    lon = np.asarray(lon2d, dtype=float)
    dlon_dx = np.gradient(lon, axis=1)                 # deg per +x step
    dlat_dx = np.gradient(lat, axis=1)                 # deg per +x step
    east = dlon_dx * np.cos(np.deg2rad(lat))           # local eastward component
    north = dlat_dx                                    # local northward component
    return np.arctan2(north, east)


def derotate_uv(u_grid, v_grid, alpha):
    """Rotate grid-relative (u,v) to earth-relative, with ``alpha`` = grid-x bearing from east.

    ``u_e = u_g cos a - v_g sin a`` ; ``v_e = u_g sin a + v_g cos a``. Works on numpy or xarray.
    """
    ca, sa = np.cos(alpha), np.sin(alpha)
    return u_grid * ca - v_grid * sa, u_grid * sa + v_grid * ca


def derotate_store(src: str, dst: str, overwrite: bool = False) -> dict:
    """De-rotate the HR u10/v10 of a built shard store; write the corrected store to ``dst``.

    Returns a small summary dict (mean convergence, channel indices).
    """
    if store_exists(dst) and not overwrite:
        raise SystemExit(f"destination exists: {dst} (use --overwrite)")

    # Source may be a loose store or an archived `*.zarr.zip`; the destination is always
    # written as a loose directory store (zip it afterwards with scripts/zip_shard.py).
    ds = xr.open_zarr(open_store(src))
    hr_channels = list(ds.attrs["hr_channels"])
    if "u10" not in hr_channels or "v10" not in hr_channels:
        raise SystemExit(f"store has no HR u10/v10 (channels={hr_channels})")
    iu, iv = hr_channels.index("u10"), hr_channels.index("v10")

    alpha_np = grid_convergence(ds["hr_lat"].values, ds["hr_lon"].values)
    alpha = xr.DataArray(alpha_np, dims=("y", "x"))

    hr = ds["hr"]                                       # (time, hr_channel, y, x), lazy
    u_e, v_e = derotate_uv(hr.isel(hr_channel=iu), hr.isel(hr_channel=iv), alpha)

    chans = [hr.isel(hr_channel=k) for k in range(len(hr_channels))]
    chans[iu], chans[iv] = u_e, v_e
    new_hr = xr.concat(chans, dim="hr_channel").transpose(*hr.dims)
    # concat splits hr_channel into one dask chunk per channel; restore a single channel chunk
    # so it aligns with the store's (time, hr_channel, y, x) zarr chunking on write.
    new_hr = new_hr.chunk({"hr_channel": len(hr_channels)})
    new_hr.encoding = {}

    new_ds = ds.copy()
    new_ds["hr"] = new_hr
    new_ds.attrs = dict(ds.attrs)
    new_ds.attrs["winds_frame"] = "earth-relative (de-rotated from CARRA2 grid via scripts/derotate_winds.py)"

    new_ds.to_zarr(dst, mode="w")
    return {
        "u10_index": iu, "v10_index": iv,
        "mean_convergence_deg": float(np.rad2deg(np.nanmean(alpha_np))),
        "n_time": int(ds.sizes["time"]),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="source shard_YYYY.zarr")
    ap.add_argument("--dst", required=True, help="destination (corrected) zarr store")
    ap.add_argument("--overwrite", action="store_true", help="overwrite dst if it exists")
    args = ap.parse_args()

    info = derotate_store(args.src, args.dst, overwrite=args.overwrite)
    print(f"[derotate] wrote {args.dst}")
    print(f"  HR u10/v10 at channel idx {info['u10_index']}/{info['v10_index']}, "
          f"{info['n_time']} timesteps")
    print(f"  mean grid convergence = {info['mean_convergence_deg']:.1f} deg")


if __name__ == "__main__":
    main()
