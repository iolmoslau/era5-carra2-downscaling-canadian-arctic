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


def error_field(p, t, mode):
    """Return (error_map, colorbar_label, symmetric_color_limit) for the chosen error mode."""
    if mode == "abs":
        e, label = p - t, "pred - truth"
    elif mode == "sigma":
        s = float(np.nanstd(t)) or 1.0
        e, label = (p - t) / s, "error / σ(truth)"
    else:  # percent: mask where |truth| is tiny (winds cross zero) to avoid blow-ups
        scale = float(np.nanmax(np.abs(t))) or 1.0
        t_safe = np.where(np.abs(t) < 0.02 * scale, np.nan, t)
        e, label = 100.0 * (p - t) / t_safe, "relative error (%)"
    lim = float(np.nanpercentile(np.abs(e), 98)) or 1.0   # robust symmetric limits
    return e, label, lim


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nc", required=True)
    ap.add_argument("--out", default="sample_native.png")
    ap.add_argument("--time", type=int, default=0)
    ap.add_argument("--ensemble", type=int, default=0)
    ap.add_argument("--cmap", default="turbo")
    ap.add_argument("--no-stations", action="store_true")
    ap.add_argument("--no-input", action="store_true", help="omit the LR input column")
    ap.add_argument("--error", choices=["percent", "sigma", "abs"], default="percent",
                    help="error panel: percent=100*(p-t)/t (masked near 0); "
                         "sigma=(p-t)/std(truth); abs=p-t")
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

    # The "input" group holds the conditioned LR (ERA5 upsampled to the patch grid), named the
    # same as the output channels -- show it as the first column (coarse in -> sharp out).
    inp = None
    if not args.no_input:
        try:
            inp = xr.open_dataset(args.nc, group="input")
        except (OSError, KeyError):
            inp = None

    channels = list(truth.data_vars)
    n = len(channels)
    ncols = 4 if inp is not None else 3
    fig, axes = plt.subplots(n, ncols, figsize=(5.0 * ncols, 5.4 * n), squeeze=False)
    for i, v in enumerate(channels):
        t = apply_orientation(np.asarray(truth[v].isel(time=args.time).values), o)
        p = apply_orientation(_pred_field(pred, v, args.time, args.ensemble), o)
        vmin, vmax = float(np.nanmin(t)), float(np.nanmax(t))
        rmse = float(np.sqrt(np.nanmean((p - t) ** 2)))          # absolute RMSE (physical units)
        nrmse = 100.0 * rmse / (float(np.nanstd(t)) or 1.0)      # normalized by field variability
        err, elabel, elim = error_field(p, t, args.error)
        print(f"{v}: RMSE={rmse:.4g}  RMSE/σ={nrmse:.1f}%  truth[{vmin:.3g},{vmax:.3g}]")

        c = 0
        if inp is not None:
            if v in inp.data_vars:
                lr = apply_orientation(np.asarray(inp[v].isel(time=args.time).values), o)
                plot_native_panel(axes[i, c], lr, **common, vmin=vmin, vmax=vmax,
                                  cmap=args.cmap, title=f"input LR  {v}  (ERA5 → grid)")
            else:
                axes[i, c].axis("off")
            c += 1

        mt = plot_native_panel(axes[i, c], t, **common, vmin=vmin, vmax=vmax,
                               cmap=args.cmap, title=f"truth  {v}")
        plot_native_panel(axes[i, c + 1], p, **common, vmin=vmin, vmax=vmax,
                          cmap=args.cmap, title=f"prediction  {v}")
        md = plot_native_panel(axes[i, c + 2], err, **common, vmin=-elim, vmax=elim,
                               cmap="RdBu_r", title=f"{elabel}  {v}  (RMSE {rmse:.3g})")
        # shared colorbar for the physical-unit panels (input/truth/prediction), separate for error
        fig.colorbar(mt, ax=axes[i, :c + 2].tolist(), orientation="vertical",
                     fraction=0.025, pad=0.02)
        cbe = fig.colorbar(md, ax=axes[i, c + 2], fraction=0.046, pad=0.04)
        cbe.set_label(elabel)

    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
