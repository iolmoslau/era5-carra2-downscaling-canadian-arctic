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
diffusion residual exists to restore that spread.

Density is plotted on a LOG axis only. On a linear axis all three curves look alike near the
mode and the entire difference lives in the tails, so a linear panel shows nothing the summary
table does not already give you.
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


WIND_SPEED, WIND_DIR = "wind_speed", "wind_direction"


def wind_from_uv(u: np.ndarray, v: np.ndarray, calm: float):
    """Speed, and METEOROLOGICAL direction: degrees the wind blows FROM, clockwise from north.

        met = (270 - atan2(v, u)) mod 360
        blowing toward the east  (u>0) -> 270  (a westerly)
        blowing toward the north (v>0) -> 180  (a southerly)

    Direction is dropped below `calm`: the angle of a near-zero vector is numerical noise, and
    keeping it drags the distribution toward uniform for reasons that have nothing to do with
    the weather.
    """
    speed = np.hypot(u, v)
    direction = np.mod(270.0 - np.degrees(np.arctan2(v, u)), 360.0)
    return speed, np.where(speed >= calm, direction, np.nan)


def accumulate_wind(u_da, v_da, edges, kind: str, calm: float, has_ensemble: bool) -> dict:
    """Stream speed or direction, derived from u and v together one timestep at a time.

    Direction is summarised with CIRCULAR statistics -- the mean of 350 deg and 10 deg is 0 deg,
    not 180 deg -- via the resultant vector: circular mean = atan2(<sin>, <cos>), and circular
    sd = sqrt(-2 ln R) with R the resultant length. R near 1 means a tightly-clustered
    direction; R near 0 means no preferred direction at all.
    """
    counts = np.zeros(len(edges) - 1, dtype=np.int64)
    lo, hi = edges[0], edges[-1]
    n_total = under = over = n_calm = 0
    s = s2 = s_cos = s_sin = 0.0
    circular = kind == "direction"

    for ti in range(u_da.sizes["time"]):
        u = np.asarray(u_da.isel(time=ti).values, dtype=np.float64).ravel()
        v = np.asarray(v_da.isel(time=ti).values, dtype=np.float64).ravel()
        ok = np.isfinite(u) & np.isfinite(v)
        speed, direction = wind_from_uv(u[ok], v[ok], calm)
        a = direction if circular else speed
        if circular:
            n_calm += int((~np.isfinite(a)).sum())
        a = a[np.isfinite(a)]
        if not a.size:
            continue
        counts += np.histogram(a, bins=edges)[0]
        under += int((a < lo).sum())
        over += int((a > hi).sum())
        n_total += a.size
        if circular:
            th = np.radians(a)
            s_cos += float(np.cos(th).sum())
            s_sin += float(np.sin(th).sum())
        else:
            s += float(a.sum())
            s2 += float((a ** 2).sum())

    if circular and n_total:
        C, S = s_cos / n_total, s_sin / n_total
        R = float(np.hypot(C, S))
        mean = float(np.mod(np.degrees(np.arctan2(S, C)), 360.0))
        std = float(np.degrees(np.sqrt(-2.0 * np.log(R)))) if R > 0 else float("nan")
    elif n_total:
        mean = s / n_total
        R = float("nan")
        std = float(np.sqrt(max(s2 / n_total - mean ** 2, 0.0)))
    else:
        mean = std = R = float("nan")

    return {"counts": counts, "n": n_total, "under": under, "over": over,
            "mean": mean, "std": std, "resultant_length": R, "n_calm": n_calm,
            "circular": circular, "has_ensemble": has_ensemble}


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


def run(pairs, bins, pad, wind=True, calm=0.5):
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

    if wind and {"u10", "v10"} <= set(ranges):
        add_wind_channels(pairs, bins, pad, calm, report, curves)
    return report, curves


def add_wind_channels(pairs, bins, pad, calm, report, curves):
    """Speed and direction, derived from u10/v10 rather than stored as channels."""
    first = pairs[0][1]
    with xr.open_dataset(first, group="truth") as truth:
        hi = 0.0
        for ti in range(truth["u10"].sizes["time"]):
            sp = np.hypot(np.asarray(truth["u10"].isel(time=ti).values, dtype=np.float64),
                          np.asarray(truth["v10"].isel(time=ti).values, dtype=np.float64))
            hi = max(hi, float(np.nanmax(sp)))
    specs = {
        WIND_SPEED: (np.linspace(0.0, hi * (1.0 + pad), bins + 1), "speed"),
        WIND_DIR: (np.linspace(0.0, 360.0, 361), "direction"),   # 1-degree sectors
    }
    for ch, (edges, kind) in specs.items():
        curves[ch] = {"edges": edges}
        report[ch] = {"range": [float(edges[0]), float(edges[-1])], "sources": {},
                      "circular": kind == "direction"}
        with xr.open_dataset(first, group="truth") as truth:
            acc = accumulate_wind(truth["u10"], truth["v10"], edges, kind, calm, False)
        curves[ch]["truth"] = pdf_and_percentiles(acc, edges)
        report[ch]["sources"]["truth"] = summarise(acc, curves[ch]["truth"])
        for label, path in pairs:
            with xr.open_dataset(path, group="prediction") as pred:
                acc = accumulate_wind(pred["u10"], pred["v10"], edges, kind, calm,
                                      "ensemble" in pred["u10"].dims)
            curves[ch][label] = pdf_and_percentiles(acc, edges)
            report[ch]["sources"][label] = summarise(acc, curves[ch][label])


def summarise(acc, curve) -> dict:
    frac_out = (acc["under"] + acc["over"]) / acc["n"] if acc["n"] else 0.0
    out = {"n_values": acc["n"], "mean": acc["mean"], "std": acc["std"],
           "percentiles": curve["percentiles"],
           "outside_range": {"under": acc["under"], "over": acc["over"],
                             "fraction": frac_out},
           "ensemble": acc["has_ensemble"]}
    if acc.get("circular"):
        # Percentiles of an angle depend on where you cut the circle, so they are not reported.
        out.update({"circular": True, "percentiles": {},
                    "resultant_length": acc["resultant_length"], "n_calm": acc["n_calm"]})
    return out


def make_plot(report, curves, path, labels, min_count=10):
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    chans = list(curves)
    fig, axes = plt.subplots(len(chans), 1, figsize=(8, 3.4 * len(chans)), squeeze=False)
    style = {"truth": dict(color="k", lw=2.0, zorder=3)}
    palette = ["tab:red", "tab:blue", "tab:green", "tab:orange"]
    for i, lb in enumerate(labels):
        style[lb] = dict(color=palette[i % len(palette)], lw=1.4)

    for r, ch in enumerate(chans):
        ax = axes[r][0]
        for src in ["truth"] + labels:
            c = curves[ch][src]
            st = report[ch]["sources"][src]
            # Far-tail bins hold a handful of samples, so their density is Poisson hash.
            # Drawing it as a curve invites reading noise as tail structure.
            y = np.where(c["counts"] >= min_count, c["density"], np.nan)
            lab = (f"{src}  (circ-mean={st['mean']:.1f}°, R={st['resultant_length']:.3f})"
                   if st.get("circular") else
                   f"{src}  (σ={st['std']:.3g}, p1={st['percentiles']['1']:.3g}, "
                   f"p99={st['percentiles']['99']:.3g})")
            ax.plot(c["centres"], y, label=lab, **style[src])
        if report[ch].get("circular"):
            # Direction spans a factor of a few, not orders of magnitude: a log axis would
            # flatten exactly the sector-to-sector contrast you are trying to read.
            ax.set_xlim(0, 360)
            ax.set_xticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
            ax.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW", "N"])
            ax.set_xlabel("direction wind blows FROM")
            ax.set_ylabel("density")
            ax.set_title(f"{ch} — climatological PDF (1° sectors)", fontsize=9)
        else:
            ax.set_yscale("log")
            shown = [curves[ch][s]["density"][curves[ch][s]["counts"] >= min_count]
                     for s in ["truth"] + labels]
            shown = np.concatenate([a[a > 0] for a in shown if a.size])
            if shown.size:
                ax.set_ylim(shown.min() * 0.5, shown.max() * 2)
            ax.set_xlabel(ch)
            ax.set_ylabel("density (log)")
            ax.set_title(f"{ch} — climatological PDF (bins with ≥{min_count} samples)",
                         fontsize=9)
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
    ap.add_argument("--no-wind", action="store_true",
                    help="skip the derived wind_speed / wind_direction panels")
    ap.add_argument("--calm", type=float, default=0.5,
                    help="drop direction below this speed, m/s (default 0.5); the angle of a "
                         "near-zero vector is noise")
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

    report, curves = run(pairs, args.bins, args.pad, not args.no_wind, args.calm)
    labels = [lb for lb, _ in pairs]

    for ch, r in report.items():
        print(f"\n{ch}   bins over [{r['range'][0]:.4g}, {r['range'][1]:.4g}]")
        print(f"  {'source':8s}{'n':>14s}{'mean':>11s}{'std':>10s}"
              f"{'p1':>10s}{'p50':>10s}{'p99':>10s}{'outside':>10s}")
        for src, s in r["sources"].items():
            p = s["percentiles"]
            cells = (f"{p['1']:>10.3f}{p['50']:>10.3f}{p['99']:>10.3f}" if p
                     else f"{'-':>10s}{'-':>10s}{'-':>10s}")
            print(f"  {src:8s}{s['n_values']:>14.4g}{s['mean']:>11.4f}{s['std']:>10.4f}"
                  f"{cells}{s['outside_range']['fraction']:>10.2e}")
        if r.get("circular"):
            print("  mean/std are CIRCULAR (deg from north, wind blowing FROM); percentiles of "
                  "an angle\n  depend on where you cut the circle, so they are omitted.")
            for src, s in r["sources"].items():
                print(f"    {src:8s}resultant R={s['resultant_length']:.4f}  "
                      f"(1 = one fixed direction, 0 = no preferred direction);  "
                      f"calms dropped: {s['n_calm']:.4g}")
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
