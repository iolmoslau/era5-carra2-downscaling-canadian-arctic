#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Per-channel CRPS and MAE for CorrDiff generate.py output (truth vs prediction).

Reads one or more NetCDFs written by `generate.py` (root lat/lon + `truth` and `prediction`
groups) and reports, per output channel, metrics averaged over all pixels and all times:

  crps  -- ensemble CRPS (NRG estimator, physical units). For a single-member (deterministic)
           prediction it reduces exactly to the MAE, so the regression-only net and the full
           diffusion model land on the *same* axis.
  mae   -- MAE of the ensemble mean (the point forecast). == |pred - truth| for 1 member.
  rmse  -- RMSE of the ensemble mean (handy alongside; not the primary score).

Ensemble CRPS estimator (Gneiting & Raftery form, a.k.a. NRG):

    CRPS = (1/M) Σ_i |x_i - y|  -  1/(2 M²) Σ_i Σ_j |x_i - x_j|

The first term rewards members being close to the observation; the second credits the
ensemble for being spread out. Computed per pixel/time via the sorted-order identity
Σ_{i<j}(x_(j) - x_(i)) = Σ_k (2k - M - 1) x_(k)  (x sorted ascending), which is O(M log M).

Usage (label=path pairs; run it on the full-model and regression-only NetCDFs):

    python eval_crps_mae.py --nc full=full.nc reg=reg.nc --out metrics_crps_mae.json

Pure numpy/xarray -- no GPU, no cartopy; runs on a login node or locally.
"""
from __future__ import annotations

import os
# Network-FS NetCDF reads: disable HDF5 file locking before xarray/netCDF4 load.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import argparse
import json
import sys

import numpy as np
import xarray as xr


def crps_ensemble_map(members: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """NRG ensemble CRPS per pixel.  members: (M, H, W)  truth: (H, W)  ->  (H, W).

    For M == 1 the spread term vanishes and this returns |member - truth| (i.e. CRPS == MAE).
    """
    M = members.shape[0]
    ae = np.nanmean(np.abs(members - truth[None]), axis=0)          # (H, W) accuracy term
    if M == 1:
        return ae
    xs = np.sort(members, axis=0)                                   # ascending along members
    k = np.arange(1, M + 1).reshape(-1, 1, 1)
    spread = np.sum((2 * k - M - 1) * xs, axis=0) / (M * M)         # 1/M² Σ_k(2k-M-1)x_(k)
    return ae - spread


def eval_nc(path: str) -> dict:
    """CRPS/MAE/RMSE per channel for one generate.py NetCDF, averaged over pixels & times."""
    truth = xr.open_dataset(path, group="truth")
    pred = xr.open_dataset(path, group="prediction")
    out = {}
    for v in list(truth.data_vars):
        t_da, p_da = truth[v], pred[v]
        has_ens = "ensemble" in p_da.dims
        n_time = int(t_da.sizes["time"])
        s_crps = s_mae = s_se = 0.0
        n_pix = 0
        m_seen = 1
        for ti in range(n_time):
            truth_map = np.asarray(t_da.isel(time=ti).values, dtype=np.float64)   # (H, W)
            members = np.asarray(p_da.isel(time=ti).values, dtype=np.float64)
            if not has_ens:
                members = members[None]                                            # (1, H, W)
            m_seen = members.shape[0]
            crps_map = crps_ensemble_map(members, truth_map)
            err = np.nanmean(members, axis=0) - truth_map                          # ens-mean error
            mask = np.isfinite(crps_map) & np.isfinite(err)
            s_crps += float(np.nansum(np.where(mask, crps_map, 0.0)))
            s_mae += float(np.nansum(np.where(mask, np.abs(err), 0.0)))
            s_se += float(np.nansum(np.where(mask, err * err, 0.0)))
            n_pix += int(mask.sum())
        out[v] = {
            "crps": s_crps / n_pix,
            "mae": s_mae / n_pix,
            "rmse": (s_se / n_pix) ** 0.5,
            "ens_members": int(m_seen),
            "n_times": n_time,
        }
    truth.close()
    pred.close()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nc", nargs="+", required=True, metavar="label=path.nc",
                    help="one or more NetCDFs, each tagged with a label (e.g. full=full.nc)")
    ap.add_argument("--out", help="write the metrics to this JSON file")
    args = ap.parse_args()

    runs = {}
    for item in args.nc:
        if "=" not in item:
            ap.error(f"--nc entries must be label=path, got '{item}'")
        label, path = item.split("=", 1)
        if not os.path.exists(path):
            ap.error(f"no such NetCDF: {path}")
        runs[label] = eval_nc(path)

    channels = sorted({c for r in runs.values() for c in r})
    labels = list(runs)
    print(f"\nPer-channel CRPS / MAE / RMSE  (physical units; averaged over pixels & times)\n")
    for ch in channels:
        print(ch)
        for label in labels:
            m = runs[label].get(ch)
            if m is None:
                continue
            print(f"  {label:8s}  CRPS={m['crps']:8.4g}  MAE={m['mae']:8.4g}  "
                  f"RMSE={m['rmse']:8.4g}   (M={m['ens_members']}, {m['n_times']} times)")
        # if both a full-model and a regression baseline are present, show the CRPS gain
        if "full" in runs and "reg" in runs and ch in runs["full"] and ch in runs["reg"]:
            cf, cr = runs["full"][ch]["crps"], runs["reg"][ch]["crps"]
            print(f"  {'Δcrps':8s}  full vs reg: {cf - cr:+.4g}  ({100 * (cf - cr) / cr:+.1f}%  "
                  f"{'better' if cf < cr else 'worse'})")
        print()

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"runs": runs}, f, indent=2)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
