#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""ERA5 input beside the CARRA2 target, for one timestamp, in the native grid frame.

The training pair as the model sees it: three rows (t2m, u10, v10), ERA5 on the left and
CARRA2 on the right, sharing a colour scale per row so the two are directly comparable. Same
map conventions as the rest of the repo via ``visualization.plotting``.

Both come straight out of a built shard, so no downloads are needed. The stored LR is on
ERA5's COARSE 0.25 deg grid, so it is bilinearly regridded onto the patch grid first -- with
``data_utils.bilinear_weights``/``apply_bilinear``, the same geometry ``BilinearUpsampler``
uses at train time, so what you see is bit-identical to the model's input.

WINDS
-----
CARRA2's stored u10/v10 are GRID-relative unless the shard has been through
``scripts/derotate_winds.py``. Plotting those beside ERA5's earth-relative winds would show a
~100 deg rotation and look like a model failure rather than a frame mismatch. So a shard
without the ``winds_frame`` attribute gets de-rotated on the fly for display, and the figure
says so. ``--no-derotate`` shows the raw stored values instead.

Signed fields get a diverging colour map centred on zero; a sequential map on u10/v10 would
hide the sign, which is the only thing that matters for a wind component.

    python -m visualization.plot_era5_carra2 --store testing_data/shard_2011.zarr \
        --out figures/era5_vs_carra2.png
"""
from __future__ import annotations

import os

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

# Alliance's pyproj can leave PROJ_DATA unset -> cartopy CRS init raises DataDirError.
# Must run before cartopy loads pyproj. Mirrors tools/plot_sample_native.py.
if not (os.environ.get("PROJ_DATA") or os.environ.get("PROJ_LIB")):
    try:
        import pyproj

        _cand = os.path.join(os.path.dirname(pyproj.__file__), "proj_dir", "share", "proj")
        if os.path.exists(os.path.join(_cand, "proj.db")):
            os.environ["PROJ_DATA"] = _cand
    except Exception:
        pass

import argparse  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import data_acquisition.data_utils as du  # noqa: E402
from dataloading.dataset import open_store, shard_path, store_exists  # noqa: E402
from scripts.derotate_winds import derotate_uv, grid_convergence  # noqa: E402
from visualization.plotting import (  # noqa: E402
    DEFAULT_STATIONS,
    apply_orientation,
    grid_spacing_km,
    inverse_index_map,
    orient_north_up,
    plot_native_panel,
)

CHANNELS = ("t2m", "u10", "v10")
UNITS = {"t2m": "K", "u10": "m s$^{-1}$", "v10": "m s$^{-1}$"}
# t2m is sequential; wind components are signed, so they get a diverging map centred on 0.
CMAPS = {"t2m": "turbo", "u10": "RdBu_r", "v10": "RdBu_r"}


def lr_on_patch_grid(lr: np.ndarray, lat: np.ndarray, lon: np.ndarray,
                     hr_lat: np.ndarray, hr_lon: np.ndarray) -> np.ndarray:
    """Bilinearly regrid one coarse LR field onto the HR patch grid.

    Same weights the train-time ``BilinearUpsampler`` builds, so this is the model's actual
    input, not an approximation of it. ``bilinear_weights`` needs a strictly increasing source
    latitude and a matching longitude convention.
    """
    asc = lat[0] < lat[-1]
    src_lat = lat if asc else lat[::-1]
    values = lr if asc else lr[::-1, :]
    idx, w = du.bilinear_weights(src_lat, lon, hr_lat, du._to_360(hr_lon))
    return du.apply_bilinear(values, idx, w).reshape(hr_lat.shape)


def load_pair(store: str, when, derotate: bool):
    """Read one timestamp: HR fields, LR regridded onto the patch grid, and the geometry."""
    with xr.open_zarr(open_store(store)) as z:
        times = pd.to_datetime(np.asarray(z["time"].values))
        ti = int(when) if isinstance(when, int) else int(np.argmin(np.abs(times - pd.Timestamp(when))))
        hr_names = list(z.attrs["hr_channels"])
        lr_names = list(z.attrs["lr_channels"])
        hr = np.asarray(z["hr"].isel(time=ti).values, dtype=float)
        lr = np.asarray(z["lr"].isel(time=ti).values, dtype=float)
        hr_lat = np.asarray(z["hr_lat"].values, dtype=float)
        hr_lon = np.asarray(z["hr_lon"].values, dtype=float)
        lat = np.asarray(z["lat"].values, dtype=float)
        lon = np.asarray(z["lon"].values, dtype=float)
        attrs = dict(z.attrs)

    stamp = times[ti]
    already = "winds_frame" in attrs
    if derotate and not already:
        # Grid-relative -> earth-relative, so the winds are in ERA5's frame for the comparison.
        alpha = grid_convergence(hr_lat, hr_lon)
        iu, iv = hr_names.index("u10"), hr_names.index("v10")
        hr[iu], hr[iv] = derotate_uv(hr[iu], hr[iv], alpha)

    fields = {}
    for c in CHANNELS:
        fields[c] = {
            "hr": hr[hr_names.index(c)],
            "lr": lr_on_patch_grid(lr[lr_names.index(c)], lat, lon, hr_lat, hr_lon),
        }
    return fields, hr_lat, hr_lon, stamp, attrs, (derotate and not already), already


def plot_pair(store, when=0, *, derotate=True, stations=DEFAULT_STATIONS, clip_pct=0.5,
              out_path=None):
    fields, hr_lat, hr_lon, stamp, attrs, applied, already = load_pair(store, when, derotate)

    lon_pm = du._to_pm180(hr_lon)
    o = orient_north_up(hr_lat, lon_pm)
    lat_o, lon_o = apply_orientation(hr_lat, o), apply_orientation(lon_pm, o)
    fx, fy = inverse_index_map(lat_o, lon_o)
    extent_ll = [lon_o.min(), lon_o.max(), lat_o.min(), lat_o.max()]
    spacing_km = grid_spacing_km(hr_lat, lon_pm)

    fig, axes = plt.subplots(len(CHANNELS), 2, figsize=(12.4, 5.9 * len(CHANNELS)))
    for r, c in enumerate(CHANNELS):
        lo_f = apply_orientation(fields[c]["lr"], o)
        hi_f = apply_orientation(fields[c]["hr"], o)
        # One colour scale per row, so left and right are directly comparable. Limits come
        # from percentiles, not min/max: a single genuine extreme (a 30 m/s coastal jet in one
        # corner is real, not an artefact) otherwise flattens the contrast everywhere else.
        both = np.concatenate([lo_f.ravel(), hi_f.ravel()])
        vmin, vmax = (float(x) for x in np.nanpercentile(both, [clip_pct, 100 - clip_pct]))
        if c != "t2m":                       # centre the diverging map on zero
            lim = max(abs(vmin), abs(vmax))
            vmin, vmax = -lim, lim
        common = dict(vmin=vmin, vmax=vmax, cmap=CMAPS[c], spacing_km=spacing_km,
                      stations=stations)
        plot_native_panel(axes[r, 0], lo_f, lat_o, lon_o, fx, fy, extent_ll,
                          title=f"ERA5 {c}  (0.25° source, bilinear → patch grid)",
                          scalebar_km=100 if r == 0 else None, **common)
        m = plot_native_panel(axes[r, 1], hi_f, lat_o, lon_o, fx, fy, extent_ll,
                              title=f"CARRA2 {c}  (native ~{spacing_km:.2f} km)",
                              scalebar_km=None, **common)
        cb = fig.colorbar(m, ax=axes[r, :], fraction=0.035, pad=0.02)
        cb.set_label(f"{c}  [{UNITS[c]}]"
                     + (f"   (colour clipped at {clip_pct}/{100 - clip_pct} pct)"
                        if clip_pct else ""))

    note = ("winds de-rotated for display (store is grid-relative)" if applied else
            "winds earth-relative in store" if already else
            "RAW grid-relative winds — not comparable to ERA5")
    fig.suptitle(f"ERA5 → CARRA2 training pair  |  {stamp:%Y-%m-%d %H:%M} UTC  |  {note}",
                 y=0.995, fontsize=12)

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        print("saved figure:", out_path)
    return fig, axes


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--store", help="a built shard (.zarr or .zarr.zip)")
    src.add_argument("--data-dir", help="shard directory; use with --year")
    ap.add_argument("--year", type=int)
    ap.add_argument("--index", type=int, default=0, help="time index (default 0)")
    ap.add_argument("--date", help="ISO timestamp; nearest stored time is used")
    ap.add_argument("--no-derotate", action="store_true",
                    help="plot the raw stored winds even if the shard is grid-relative")
    ap.add_argument("--clip-pct", type=float, default=0.5,
                    help="percentile clip for the colour limits (default 0.5; 0 = min/max)")
    ap.add_argument("--no-stations", action="store_true")
    ap.add_argument("--out", default="figures/era5_vs_carra2.png")
    args = ap.parse_args()

    if args.store:
        store = args.store
    else:
        if args.year is None:
            ap.error("--data-dir requires --year")
        store = shard_path(args.data_dir, args.year)
    if not store_exists(store):
        ap.error(f"store not found: {store}")

    plot_pair(store, args.date or args.index, derotate=not args.no_derotate,
              stations=None if args.no_stations else DEFAULT_STATIONS,
              clip_pct=args.clip_pct, out_path=args.out)


if __name__ == "__main__":
    main()
