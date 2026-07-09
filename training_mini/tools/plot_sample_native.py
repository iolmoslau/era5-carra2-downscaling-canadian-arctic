#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Native-grid plots of generate.py output: truth / prediction / difference per channel.

Reuses visualization/plotting.py so the panels match the data-pipeline figures -- north up,
lat/lon graticule, coastlines + borders, a km scale bar, and community markers -- rendered in
the CARRA2 patch's own (oriented) index space from the 2-D lat/lon fields in the NetCDF.

Needs cartopy (+ scipy), which the training env (corrdiff-env) does not have -- run it in the
data-pipeline venv (~/ENV) or locally. First run downloads Natural Earth shapefiles (internet).

    python tools/plot_sample_native.py --nc corrdiff_output.nc --out sample_native.png --time 0
"""

from __future__ import annotations

import os
# Network-FS NetCDF reads: disable HDF5 file locking before xarray/netCDF4 load.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import data_acquisition.data_utils as du  # noqa: E402
from visualization.plotting import (  # noqa: E402
    orient_north_up, apply_orientation, inverse_index_map, grid_spacing_km,
    plot_native_panel, DEFAULT_STATIONS,
)


def _pred_field(pred, v, t, ens):
    da = pred[v]
    if "ensemble" in da.dims:
        return np.asarray(da.isel(ensemble=ens, time=t).values)
    return np.asarray(da.isel(time=t).values)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nc", required=True)
    ap.add_argument("--out", default="sample_native.png")
    ap.add_argument("--time", type=int, default=0)
    ap.add_argument("--ensemble", type=int, default=0)
    ap.add_argument("--cmap", default="turbo")
    ap.add_argument("--no-stations", action="store_true")
    args = ap.parse_args()

    root = xr.open_dataset(args.nc)                 # 2-D lat/lon live at the root
    truth = xr.open_dataset(args.nc, group="truth")
    pred = xr.open_dataset(args.nc, group="prediction")

    lat2d = np.asarray(root["lat"].values)
    lon2d = du._to_pm180(np.asarray(root["lon"].values))
    o = orient_north_up(lat2d, lon2d)
    lat_o, lon_o = apply_orientation(lat2d, o), apply_orientation(lon2d, o)
    fx, fy = inverse_index_map(lat_o, lon_o)
    extent_ll = [lon_o.min(), lon_o.max(), lat_o.min(), lat_o.max()]
    spacing_km = grid_spacing_km(lat2d, lon2d)
    stations = None if args.no_stations else DEFAULT_STATIONS
    common = dict(lat_o=lat_o, lon_o=lon_o, fx=fx, fy=fy, extent_ll=extent_ll,
                  spacing_km=spacing_km, stations=stations)

    channels = list(truth.data_vars)
    n = len(channels)
    fig, axes = plt.subplots(n, 3, figsize=(15, 5.4 * n), squeeze=False)
    for i, v in enumerate(channels):
        t = apply_orientation(np.asarray(truth[v].isel(time=args.time).values), o)
        p = apply_orientation(_pred_field(pred, v, args.time, args.ensemble), o)
        d = p - t
        vmin, vmax = float(np.nanmin(t)), float(np.nanmax(t))
        dm = float(np.nanmax(np.abs(d))) or 1.0
        rmse = float(np.sqrt(np.nanmean(d ** 2)))
        print(f"{v}: RMSE={rmse:.4g}  truth[{vmin:.3g},{vmax:.3g}]")

        mt = plot_native_panel(axes[i, 0], t, **common, vmin=vmin, vmax=vmax,
                               cmap=args.cmap, title=f"truth  {v}")
        plot_native_panel(axes[i, 1], p, **common, vmin=vmin, vmax=vmax,
                          cmap=args.cmap, title=f"prediction  {v}")
        md = plot_native_panel(axes[i, 2], d, **common, vmin=-dm, vmax=dm,
                               cmap="RdBu_r", title=f"pred - truth  {v}  (RMSE {rmse:.3g})")
        fig.colorbar(mt, ax=axes[i, :2].tolist(), orientation="vertical", fraction=0.025, pad=0.02)
        fig.colorbar(md, ax=axes[i, 2], fraction=0.046, pad=0.04)

    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
