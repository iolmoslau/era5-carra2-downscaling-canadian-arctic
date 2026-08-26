#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Climatological PDFs: truth vs full model vs regression-only, per channel.

Pools every value over pixels, times and ensemble members into one empirical distribution per
source, and overlays them. This is the CLIMATOLOGICAL comparison -- "does the model generate
fields whose value distribution matches reality" -- not a per-pixel calibration check.

Reads the NetCDFs `run_eval.sh` already writes, so it needs no new generation:

    python evaluate/plot_pdfs.py --nc full=$NC_DIR/full.nc reg=$NC_DIR/reg.nc \
        --out pdfs.json --plot pdfs.png

Truth is taken from the FIRST file (both passes score the same times, so it is identical) and
verified against the others.

WHY IT STREAMS
--------------
400 times x 448x448 x 32 members is ~2.6e9 values per channel, ~10 GB in float32 and ~31 GB
across three channels. So counts are accumulated one timestep at a time into fixed bins rather
than concatenating arrays. Fixed bins mean the range must be chosen up front: it comes from
truth, padded by ``--pad``, and anything outside is counted as underflow/overflow and REPORTED.
A model that generates values beyond the observed range is a finding, not a rounding detail, so
those counts must never be silently dropped.

WHAT TO EXPECT
--------------
The regression net predicts a conditional mean, so it is mathematically obliged to be
under-dispersed: expect its PDF to be too narrow, with thin tails, relative to truth. The
diffusion residual exists to restore that spread. The log-scale panel is where this shows --
on a linear axis all three curves look alike near the mode and the difference lives entirely in
the tails.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import numpy as np
import xarray as xr


def channel_range(path: str, pad: float) -> dict:
    """Per-channel (lo, hi) from truth, padded, plus truth's own moments.

    Truth is small enough (~1 GB for 3 channels x 400 times) to stream twice; the prediction
    groups are not, so they get exactly one pass with these edges already fixed.
    """
    out = {}
    with xr.open_dataset(path, group="truth") as truth:
        for v in truth.data_vars:
            da = truth[v]
            lo, hi = np.inf, -np.inf
            for ti in range(da.sizes["time"]):
                a = np.asarray(da.isel(time=ti).values, dtype=np.float64)
                a = a[np.isfinite(a)]
                if a.size:
                    lo, hi = min(lo, a.min()), max(hi, a.max())
            span = hi - lo
            out[v] = (lo - pad * span, hi + pad * span)
    return out


def accumulate(da, edges, has_ensemble: bool) -> dict:
    """Stream a DataArray into fixed bins, tracking values that fall outside them."""
    counts = np.zeros(len(edges) - 1, dtype=np.int64)
    lo, hi = edges[0], edges[-1]
    n_total = under = over = 0
    s = s2 = 0.0
    for ti in range(da.sizes["time"]):
        a = np.asarray(da.isel(time=ti).values, dtype=np.float64).ravel()
        a = a[np.isfinite(a)]
        if not a.size:
            continue
        counts += np.histogram(a, bins=edges)[0]
        under += int((a < lo).sum())
        over += int((a > hi).sum())
        n_total += a.size
        s += float(a.sum())
        s2 += float((a ** 2).sum())
    mean = s / n_total if n_total else float("nan")
    var = max(s2 / n_total - mean ** 2, 0.0) if n_total else float("nan")
    return {"counts": counts, "n": n_total, "under": under, "over": over,
            "mean": mean, "std": float(np.sqrt(var)),
            "has_ensemble": has_ensemble}


def pdf_and_percentiles(acc: dict, edges: np.ndarray, qs=(1, 5, 25, 50, 75, 95, 99)) -> dict:
    """Normalised density (integrates to 1) plus bin-resolution percentiles."""
    width = np.diff(edges)
    counted = acc["counts"].sum()
    density = acc["counts"] / (counted * width) if counted else np.zeros_like(width)
    centres = (edges[:-1] + edges[1:]) / 2
    cdf = np.cumsum(acc["counts"]) / counted if counted else np.zeros_like(width)
    pct = {str(q): float(np.interp(q / 100.0, cdf, centres)) for q in qs}
    return {"density": density, "centres": centres, "percentiles": pct,
            "counts": acc["counts"]}


def run(pairs, bins, pad):
    first = pairs[0][1]
    ranges = channel_range(first, pad)
    report, curves = {}, {}

    for ch, (lo, hi) in ranges.items():
        edges = np.linspace(lo, hi, bins + 1)
        curves[ch] = {"edges": edges}
        report[ch] = {"range": [float(lo), float(hi)], "sources": {}}

        with xr.open_dataset(first, group="truth") as truth:
            acc = accumulate(truth[ch], edges, has_ensemble=False)
        curves[ch]["truth"] = pdf_and_percentiles(acc, edges)
        report[ch]["sources"]["truth"] = summarise(acc, curves[ch]["truth"])

        for label, path in pairs:
            with xr.open_dataset(path, group="prediction") as pred:
                da = pred[ch]
                acc = accumulate(da, edges, has_ensemble="ensemble" in da.dims)
            curves[ch][label] = pdf_and_percentiles(acc, edges)
            report[ch]["sources"][label] = summarise(acc, curves[ch][label])
    return report, curves


def summarise(acc, curve) -> dict:
    frac_out = (acc["under"] + acc["over"]) / acc["n"] if acc["n"] else 0.0
    return {"n_values": acc["n"], "mean": acc["mean"], "std": acc["std"],
            "percentiles": curve["percentiles"],
            "outside_range": {"under": acc["under"], "over": acc["over"],
                              "fraction": frac_out},
            "ensemble": acc["has_ensemble"]}


def make_plot(report, curves, path, labels, min_count=10):
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    chans = list(curves)
    fig, axes = plt.subplots(len(chans), 2, figsize=(11, 3.2 * len(chans)), squeeze=False)
    style = {"truth": dict(color="k", lw=2.0, zorder=3)}
    palette = ["tab:red", "tab:blue", "tab:green", "tab:orange"]
    for i, lb in enumerate(labels):
        style[lb] = dict(color=palette[i % len(palette)], lw=1.4)

    for r, ch in enumerate(chans):
        for col, logy in ((0, False), (1, True)):
            ax = axes[r][col]
            for src in ["truth"] + labels:
                c = curves[ch][src]
                n = report[ch]["sources"][src]["n_values"]
                y = c["density"].copy()
                if logy:
                    # Far-tail bins hold a handful of samples, so their density is Poisson
                    # hash. Drawing it as a curve invites reading noise as tail structure.
                    y[c["counts"] < min_count] = np.nan
                ax.plot(c["centres"], y,
                        label=f"{src}  (n={n:.3g}, σ={report[ch]['sources'][src]['std']:.3g})",
                        **style[src])
            ax.set_xlabel(ch)
            ax.set_ylabel("density")
            if logy:
                ax.set_yscale("log")
                # the bulk is uninformative here; the tails are the whole point
                shown = [curves[ch][s]["density"][curves[ch][s]["counts"] >= min_count]
                         for s in ["truth"] + labels]
                shown = np.concatenate([a[a > 0] for a in shown if a.size])
                if shown.size:
                    ax.set_ylim(shown.min() * 0.5, shown.max() * 2)
                ax.set_title(f"{ch} — log density (bins with ≥{min_count} samples)", fontsize=9)
            else:
                ax.set_title(f"{ch} — density", fontsize=9)
                ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nc", nargs="+", required=True, metavar="label=path.nc",
                    help="labelled generate.py outputs, e.g. full=full.nc reg=reg.nc")
    ap.add_argument("--bins", type=int, default=1000)
    ap.add_argument("--min-count", type=int, default=10,
                    help="log panel hides bins with fewer samples than this (default 10)")
    ap.add_argument("--pad", type=float, default=0.1,
                    help="fraction of truth's range added at each end (default 0.1)")
    ap.add_argument("--out", help="write the report as JSON here")
    ap.add_argument("--plot", help="write the figure here")
    args = ap.parse_args()

    pairs = []
    for item in args.nc:
        if "=" not in item:
            ap.error(f"--nc entries must be label=path, got '{item}'")
        label, path = item.split("=", 1)
        if not os.path.exists(path):
            ap.error(f"no such NetCDF: {path}")
        pairs.append((label, path))

    report, curves = run(pairs, args.bins, args.pad)
    labels = [lb for lb, _ in pairs]

    for ch, r in report.items():
        print(f"\n{ch}   bins over [{r['range'][0]:.4g}, {r['range'][1]:.4g}]")
        print(f"  {'source':8s}{'n':>14s}{'mean':>11s}{'std':>10s}"
              f"{'p1':>10s}{'p50':>10s}{'p99':>10s}{'outside':>10s}")
        for src, s in r["sources"].items():
            p = s["percentiles"]
            print(f"  {src:8s}{s['n_values']:>14.4g}{s['mean']:>11.4f}{s['std']:>10.4f}"
                  f"{p['1']:>10.3f}{p['50']:>10.3f}{p['99']:>10.3f}"
                  f"{s['outside_range']['fraction']:>10.2e}")
        worst = max(s["outside_range"]["fraction"] for s in r["sources"].values())
        if worst > 1e-4:
            print(f"  NOTE: up to {worst:.2%} of values fall outside the binned range -- "
                  f"raise --pad so the tails are not clipped.")

    print("\nRegression is a conditional mean, so a narrower PDF than truth is expected, not a "
          "bug;\nthe diffusion residual is what should restore the spread. Compare the std and "
          "p1/p99 columns.")

    if args.out:
        serial = {ch: {"range": r["range"], "sources": r["sources"]} for ch, r in report.items()}
        for ch in serial:
            serial[ch]["curves"] = {
                src: {"centres": curves[ch][src]["centres"].tolist(),
                      "density": curves[ch][src]["density"].tolist()}
                for src in ["truth"] + labels
            }
        with open(args.out, "w") as f:
            json.dump(serial, f, indent=2)
        print(f"wrote {args.out}")
    if args.plot:
        make_plot(report, curves, args.plot, labels, args.min_count)


if __name__ == "__main__":
    main()
