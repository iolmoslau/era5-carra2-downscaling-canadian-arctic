#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Plot truth / prediction / (prediction-truth) per output channel from a generate.py NetCDF.

The NetCDF (from generate.py) has `truth`, `prediction`, and `input` groups. This makes a
per-channel row: truth, prediction, and the signed difference, and prints per-channel RMSE.

    python tools/plot_sample.py --nc corrdiff_output.nc --out sample.png --time 0
"""

from __future__ import annotations

import argparse

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nc", required=True, help="generate.py output NetCDF")
    ap.add_argument("--out", default="sample.png")
    ap.add_argument("--time", type=int, default=0, help="time index to plot")
    ap.add_argument("--ensemble", type=int, default=0, help="ensemble member (diffusion)")
    args = ap.parse_args()

    truth = xr.open_dataset(args.nc, group="truth")
    pred = xr.open_dataset(args.nc, group="prediction")
    channels = list(truth.data_vars)
    n = len(channels)

    fig, axes = plt.subplots(n, 3, figsize=(11, 3.3 * n), squeeze=False)
    for i, v in enumerate(channels):
        t = np.asarray(truth[v].isel(time=args.time))
        pv = pred[v]
        p = np.asarray(pv.isel(ensemble=args.ensemble, time=args.time)
                       if "ensemble" in pv.dims else pv.isel(time=args.time))
        vmin, vmax = float(np.nanmin(t)), float(np.nanmax(t))
        d = p - t
        m = float(np.nanmax(np.abs(d))) or 1.0
        rmse = float(np.sqrt(np.nanmean(d ** 2)))
        print(f"{v}: RMSE={rmse:.4g}  truth[{vmin:.3g},{vmax:.3g}]")

        axes[i, 0].imshow(t, origin="lower", vmin=vmin, vmax=vmax)
        axes[i, 0].set_title(f"truth {v}")
        axes[i, 1].imshow(p, origin="lower", vmin=vmin, vmax=vmax)
        axes[i, 1].set_title(f"prediction {v}")
        im = axes[i, 2].imshow(d, origin="lower", cmap="RdBu_r", vmin=-m, vmax=m)
        axes[i, 2].set_title(f"pred - truth {v}  (RMSE {rmse:.3g})")
        fig.colorbar(im, ax=axes[i, 2], fraction=0.046, pad=0.04)
        for a in axes[i]:
            a.axis("off")

    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
