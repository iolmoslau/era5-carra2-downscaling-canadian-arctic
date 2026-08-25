#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Is the LR input physically consistent with the HR target, beyond the known wind rotation?

The CARRA2 grid-relative wind bug (see ``scripts/derotate_winds.py``) was a *silent* mismatch:
the model could partly absorb it during training, so it never showed up as a failure -- only as
a worse model. This script looks for other mismatches of that kind by coarsening the HR target
onto the LR grid and comparing the two directly.

WHAT A DIFFERENCE MEANS
-----------------------
ERA5 and CARRA2 are different models, not two views of one truth: CARRA2 is HARMONIE-AROME at
2.5 km with its own physics, assimilation and orography. A nonzero difference is EXPECTED --
it is the signal the downscaling model exists to learn. What matters is its *character*:

    constant offset, no seasonality  ->  units / convention bug (K vs degC, z vs z/g)
    OLS slope != 1                   ->  scaling error
    spatial correlation ~ 0          ->  grid orientation / flip / transpose
    correlation peaks at lag != 0    ->  LR and HR taken at different valid times
    SEASONALLY VARYING offset        ->  physical (orography, snow, boundary layer)

The monthly bias table is the discriminator between the first and last rows.

METHOD NOTES
------------
* HR is AREA-AVERAGED onto each LR cell (bin by cell footprint, then mean), not bilinearly
  sampled. Sampling a point keeps 2.5 km variance and biases toward whatever sits at the sample
  location; the LR field is a grid-box quantity, so a footprint mean is the honest comparison.
* LR cells only partly covered by the HR patch are dropped (``--coverage``), or the patch edge
  contaminates every statistic.
* Everything is split LAND / SEA. Over land, t2m differences are dominated by orography --
  CARRA2 resolves real terrain, ERA5 smooths it, and the lapse rate is ~6.5 K/km, so a few
  hundred metres is a couple of K of entirely legitimate bias. Over ocean that confound is
  absent, so an offset there is far more suspicious.
* Winds are compared as SPEED and DIRECTION, not u/v separately. A residual mean direction
  offset means the de-rotation is still not exact; a speed ratio > 1 is just resolution (2.5 km
  resolves terrain-driven gusts that 0.25 deg smooths away). Keeping them apart stops one from
  masquerading as the other.
* Only ``t2m``/``u10``/``v10`` are comparable -- the other LR channels are pressure-level or sea
  ice and have no HR counterpart.

    python scripts/lr_hr_consistency.py --data-dir $PROJECT/data/derot --years 2011 2019 \
        --n 200 --out lr_hr_consistency.json --plot lr_hr_consistency.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import numpy as np
import pandas as pd
import xarray as xr

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dataloading.dataset import open_store, shard_path, store_exists  # noqa: E402

SURFACE = ("t2m", "u10", "v10")


# ----------------------------------------------------------------------------- grid binning
def build_binning(z: xr.Dataset, coverage: float = 0.5) -> dict:
    """Map every HR cell to the LR cell containing it (area-average weights).

    Returns the flat LR index per HR cell, the per-cell counts, and a validity mask over LR
    cells that are sufficiently covered by the HR patch.
    """
    lat = np.asarray(z["lat"].values, dtype=float)          # 1-D, stored descending
    lon = np.asarray(z["lon"].values, dtype=float)          # 1-D, [0, 360)
    lat_asc = lat[::-1] if lat[0] > lat[-1] else lat
    flipped = lat[0] > lat[-1]

    def edges(c: np.ndarray) -> np.ndarray:
        step = np.abs(np.diff(c)).mean()
        return np.concatenate([[c[0] - step / 2], (c[:-1] + c[1:]) / 2, [c[-1] + step / 2]])

    lat_edges, lon_edges = edges(lat_asc), edges(lon)
    hr_lat = np.asarray(z["hr_lat"].values, dtype=float).ravel()
    hr_lon = np.mod(np.asarray(z["hr_lon"].values, dtype=float), 360.0).ravel()  # match LR

    iy = np.digitize(hr_lat, lat_edges) - 1
    ix = np.digitize(hr_lon, lon_edges) - 1
    ny, nx = lat_asc.size, lon.size
    ok = (iy >= 0) & (iy < ny) & (ix >= 0) & (ix < nx)
    flat = np.where(ok, iy * nx + ix, 0)

    counts = np.bincount(flat[ok], minlength=ny * nx).astype(float)
    # Full cells all hold about the same number of HR cells; partial ones sit at the patch edge.
    full = np.median(counts[counts > 0]) if np.any(counts > 0) else 0.0
    valid = counts >= coverage * full
    return {"flat": flat, "ok": ok, "counts": counts, "valid": valid,
            "shape": (ny, nx), "flipped": flipped, "full": float(full)}


def coarsen(field_hr: np.ndarray, b: dict) -> np.ndarray:
    """Area-average one HR field (y, x) onto the LR grid, oriented like the stored LR."""
    ny, nx = b["shape"]
    flat_vals = np.asarray(field_hr, dtype=float).ravel()[b["ok"]]
    sums = np.bincount(b["flat"][b["ok"]], weights=flat_vals, minlength=ny * nx)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(b["counts"] > 0, sums / b["counts"], np.nan)
    out = mean.reshape(ny, nx)
    return out[::-1] if b["flipped"] else out     # back to the store's lat order


# ----------------------------------------------------------------------------- statistics
def pair_stats(hr: np.ndarray, lr: np.ndarray) -> dict:
    """Compare coarsened HR against LR. Both arrays are (time, cell).

    Two correlations are reported on purpose, because the obvious one flatters the data:

    ``corr_pooled``  -- over every (time, cell) pair at once. For t2m this is dominated by the
                        SEASONAL CYCLE (220-280 K over a year), so any two fields that merely
                        follow the seasons score ~0.98 whatever their spatial skill.
    ``corr_spatial`` -- computed per timestep across cells, then averaged. The seasonal signal
                        is constant within a timestep, so this measures actual spatial
                        agreement. It is the honest number, and it is always the lower one.

    The OLS slope is fitted on ANOMALIES (each field minus its own mean), so it answers "are the
    variations the same size" rather than being pinned by the distance from absolute zero -- a
    raw fit on Kelvin gives a meaningless slope with a huge compensating intercept. Any constant
    offset is already reported as ``bias_hr_minus_lr``.
    """
    m = np.isfinite(hr) & np.isfinite(lr)
    if m.sum() < 10:
        return {"n": int(m.sum())}
    x, y = lr[m], hr[m]
    d = y - x
    slope = float(np.polyfit(x - x.mean(), y - y.mean(), 1)[0])

    spatial = []
    for t in range(hr.shape[0]):
        mt = np.isfinite(hr[t]) & np.isfinite(lr[t])
        if mt.sum() > 10 and np.std(lr[t][mt]) > 0 and np.std(hr[t][mt]) > 0:
            spatial.append(float(np.corrcoef(lr[t][mt], hr[t][mt])[0, 1]))

    return {
        "n": int(m.sum()),
        "bias_hr_minus_lr": float(d.mean()),
        "mae": float(np.abs(d).mean()),
        "rmse": float(np.sqrt((d ** 2).mean())),
        "corr_pooled": float(np.corrcoef(x, y)[0, 1]),
        "corr_spatial": float(np.mean(spatial)) if spatial else float("nan"),
        "corr_spatial_std": float(np.std(spatial)) if spatial else float("nan"),
        "anomaly_slope": slope,       # 1.0 if HR and LR variations have the same amplitude
        "lr_mean": float(x.mean()),
        "hr_mean": float(y.mean()),
        "lr_std": float(x.std()),
        "hr_std": float(y.std()),
    }


def wind_stats(u_hr, v_hr, u_lr, v_lr) -> dict:
    """Direction offset and speed ratio, kept separate so neither hides the other.

    The mean rotation from LR to HR is the argument of sum(z_hr * conj(z_lr)) with z = u + iv --
    a magnitude-weighted circular mean, so calm cells cannot dominate. |.| normalised gives the
    vector correlation. A residual offset here means de-rotation is still imperfect; a speed
    ratio > 1 is just 2.5 km resolving gusts that 0.25 deg cannot.
    """
    m = np.isfinite(u_hr) & np.isfinite(v_hr) & np.isfinite(u_lr) & np.isfinite(v_lr)
    if m.sum() < 10:
        return {"n": int(m.sum())}
    z_hr = u_hr[m] + 1j * v_hr[m]
    z_lr = u_lr[m] + 1j * v_lr[m]
    cross = np.sum(z_hr * np.conj(z_lr))
    denom = np.sqrt(np.sum(np.abs(z_hr) ** 2) * np.sum(np.abs(z_lr) ** 2))
    sp_hr, sp_lr = np.abs(z_hr), np.abs(z_lr)
    return {
        "n": int(m.sum()),
        "mean_direction_offset_deg": float(np.rad2deg(np.angle(cross))),
        "vector_correlation": float(np.abs(cross) / denom) if denom > 0 else float("nan"),
        "speed_ratio_hr_over_lr": float(sp_hr.mean() / sp_lr.mean()) if sp_lr.mean() else float("nan"),
        "speed_bias_hr_minus_lr": float(sp_hr.mean() - sp_lr.mean()),
    }


# ----------------------------------------------------------------------------- sampling
def collect_samples(data_dir: str, years, n: int, seed: int):
    """(store, time-index, timestamp) triples drawn uniformly across the requested years."""
    pool = []
    for y in years:
        s = shard_path(data_dir, y)
        if not store_exists(s):
            print(f"WARNING: no shard for {y} under {data_dir}", file=sys.stderr)
            continue
        with xr.open_zarr(open_store(s)) as z:
            times = pd.to_datetime(np.asarray(z["time"].values))
        pool.extend((s, i, t) for i, t in enumerate(times))
    if not pool:
        sys.exit(f"no shards found for years {years} under {data_dir}")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
    return [pool[i] for i in sorted(idx)]


# ----------------------------------------------------------------------------- driver
def run(data_dir, years, n, seed, coverage, lags, want_residual):
    samples = collect_samples(data_dir, years, n, seed)
    by_store: dict[str, list] = {}
    for store, i, t in samples:
        by_store.setdefault(store, []).append((i, t))

    binning = None
    lsm_frac = None
    lr_names = None
    acc = {c: {"hr": [], "lr": []} for c in SURFACE}
    months, lag_corr = [], {L: [] for L in lags}

    for store, items in by_store.items():
        with xr.open_zarr(open_store(store)) as z:
            if binning is None:
                binning = build_binning(z, coverage)
                lsm_frac = coarsen(np.asarray(z["land_sea_mask"].values), binning)
                lr_names = list(z.attrs["lr_channels"])
                hr_names = list(z.attrs["hr_channels"])
                missing = [c for c in SURFACE if c not in lr_names or c not in hr_names]
                if missing:
                    sys.exit(f"store lacks comparable surface channel(s): {missing}")
            nt = z.sizes["time"]
            for i, t in items:
                hr = np.asarray(z["hr"].isel(time=i).values, dtype=float)
                lr = np.asarray(z["lr"].isel(time=i).values, dtype=float)
                for c in SURFACE:
                    acc[c]["hr"].append(coarsen(hr[hr_names.index(c)], binning))
                    acc[c]["lr"].append(lr[lr_names.index(c)])
                months.append(t.month)
                # time alignment: coarsened HR(t) vs LR(t+lag); a peak away from 0 means the
                # two sources were pulled at different valid times.
                hr_t2m = coarsen(hr[hr_names.index("t2m")], binning)
                for L in lags:
                    j = i + L
                    if 0 <= j < nt:
                        lr_j = np.asarray(z["lr"].isel(time=j).values, dtype=float)[lr_names.index("t2m")]
                        m = np.isfinite(hr_t2m) & np.isfinite(lr_j) & binning["valid"].reshape(hr_t2m.shape)
                        if m.sum() > 10:
                            lag_corr[L].append(float(np.corrcoef(hr_t2m[m], lr_j[m])[0, 1]))

    valid = binning["valid"].reshape(binning["shape"])
    if binning["flipped"]:
        valid = valid[::-1]
    land = valid & (lsm_frac > 0.8)      # coastal cells excluded from both masks on purpose
    sea = valid & (lsm_frac < 0.2)

    out = {
        "config": {"data_dir": str(data_dir), "years": list(years), "n_times": len(samples),
                   "seed": seed, "coverage": coverage,
                   "hr_cells_per_lr_cell": binning["full"],
                   "lr_cells": {"valid": int(valid.sum()), "land": int(land.sum()),
                                "sea": int(sea.sum())}},
        "channels": {}, "winds": {}, "monthly_bias": {}, "lag_correlation_t2m": {},
    }

    stacks = {c: (np.stack(acc[c]["hr"]), np.stack(acc[c]["lr"])) for c in SURFACE}
    for c in SURFACE:
        hr_s, lr_s = stacks[c]
        out["channels"][c] = {
            "all": pair_stats(hr_s[:, valid], lr_s[:, valid]),
            "land": pair_stats(hr_s[:, land], lr_s[:, land]),
            "sea": pair_stats(hr_s[:, sea], lr_s[:, sea]),
        }
        mo = np.asarray(months)
        # Sample counts travel with the biases: with few times per month these bars are noisy,
        # and a "seasonal cycle" read off 3 samples a month is not a seasonal cycle.
        out["monthly_bias"][c] = {
            int(m): float(np.nanmean((hr_s[mo == m][:, valid] - lr_s[mo == m][:, valid])))
            for m in sorted(set(months))
        }
    out["monthly_n_times"] = {int(m): int((np.asarray(months) == m).sum())
                              for m in sorted(set(months))}

    for label, mask in (("all", valid), ("land", land), ("sea", sea)):
        out["winds"][label] = wind_stats(stacks["u10"][0][:, mask], stacks["v10"][0][:, mask],
                                         stacks["u10"][1][:, mask], stacks["v10"][1][:, mask])

    for L, vals in lag_corr.items():
        if vals:
            out["lag_correlation_t2m"][str(L)] = float(np.mean(vals))

    if want_residual:
        out["residual_hr_minus_upsampled_lr"] = residual_stats(by_store, SURFACE)
    return out, stacks, np.asarray(months), (valid, land, sea)


def residual_stats(by_store, channels) -> dict:
    """Summarise HR - upsample(LR): literally what the model has to learn.

    Structure that should not be there -- a constant, a land/sea step -- is the most direct
    answer to "what is the model secretly adjusting for". Needs torch, so it is opt-in.
    """
    from dataloading.dataset import PatchDataset  # noqa: PLC0415

    stats: dict = {}
    store = next(iter(by_store))
    items = by_store[store][:64]                    # a residual is cheap to characterise
    ds = PatchDataset(store)
    up = ds.make_upsampler()
    import torch  # noqa: PLC0415

    diffs = {c: [] for c in channels}
    for i, _ in items:
        s = ds[i]
        with torch.no_grad():
            lr_up = up(s["lr"].unsqueeze(0))[0].numpy()
        hr = s["hr"].numpy()
        for c in channels:
            if c in ds.hr_channels and c in ds.lr_channels:
                diffs[c].append(hr[ds.hr_channels.index(c)] - lr_up[ds.lr_channels.index(c)])
    for c, d in diffs.items():
        if d:
            a = np.stack(d)
            stats[c] = {"mean": float(a.mean()), "std": float(a.std()),
                        "abs_mean": float(np.abs(a).mean()), "n_times": len(d)}
    return stats


def make_plot(out, stacks, months, masks, path):
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    valid, land, sea = masks
    fig, axes = plt.subplots(len(SURFACE), 2, figsize=(11, 3.1 * len(SURFACE)))
    rng = np.random.default_rng(0)
    for r, c in enumerate(SURFACE):
        hr_s, lr_s = stacks[c]
        x, y = lr_s[:, valid].ravel(), hr_s[:, valid].ravel()
        k = rng.choice(x.size, size=min(20000, x.size), replace=False)
        ax = axes[r, 0]
        ax.plot(x[k], y[k], ".", ms=1, alpha=0.25)
        lo, hi = np.nanpercentile(np.concatenate([x[k], y[k]]), [0.5, 99.5])
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="1:1")
        s = out["channels"][c]["all"]
        xm, ym = s["lr_mean"], s["hr_mean"]
        ax.plot([lo, hi], [ym + s["anomaly_slope"] * (lo - xm),
                           ym + s["anomaly_slope"] * (hi - xm)], "r-", lw=1,
                label=f"anomaly slope {s['anomaly_slope']:.3f}")
        ax.set(xlabel=f"LR {c}", ylabel=f"coarsened HR {c}")
        ax.legend(fontsize=7, loc="upper left")
        ax.set_title(f"{c}: bias {s['bias_hr_minus_lr']:+.3f}, "
                     f"r_spatial={s['corr_spatial']:.4f}", fontsize=9)

        ax = axes[r, 1]
        mb = out["monthly_bias"][c]
        ax.bar(list(mb), [mb[m] for m in mb], color="tab:blue")
        ax.axhline(0, color="k", lw=0.8)
        ax.set(xlabel="month", ylabel=f"bias (HR-LR) {c}")
        ax.set_title("flat = units/convention; seasonal = physical", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, help="directory holding shard_YYYY.zarr[.zip]")
    ap.add_argument("--years", type=int, nargs="+", required=True)
    ap.add_argument("--n", type=int, default=200, help="random times to sample (default 200)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--coverage", type=float, default=0.5,
                    help="min fraction of a full LR cell's HR count to keep it (default 0.5)")
    ap.add_argument("--lags", type=int, nargs="*", default=[-2, -1, 0, 1, 2],
                    help="time-step lags for the alignment check (0 = aligned)")
    ap.add_argument("--residual", action="store_true",
                    help="also summarise HR - upsample(LR) (needs torch)")
    ap.add_argument("--out", help="write the report as JSON here")
    ap.add_argument("--plot", help="write a diagnostic figure here")
    args = ap.parse_args()

    out, stacks, months, masks = run(args.data_dir, args.years, args.n, args.seed,
                                     args.coverage, args.lags, args.residual)

    cfg = out["config"]
    print(f"\n{cfg['n_times']} times, years {cfg['years']}, "
          f"{cfg['hr_cells_per_lr_cell']:.0f} HR cells per LR cell")
    print(f"LR cells: {cfg['lr_cells']['valid']} valid "
          f"({cfg['lr_cells']['land']} land, {cfg['lr_cells']['sea']} sea)\n")
    hdr = (f"{'channel':8s}{'domain':7s}{'bias':>10s}{'rmse':>9s}"
           f"{'r_pool':>9s}{'r_spat':>9s}{'slope':>8s}")
    print(hdr + "\n" + "-" * len(hdr))
    for c in SURFACE:
        for dom in ("all", "land", "sea"):
            s = out["channels"][c][dom]
            if "bias_hr_minus_lr" not in s:
                continue
            print(f"{c:8s}{dom:7s}{s['bias_hr_minus_lr']:>10.4f}{s['rmse']:>9.4f}"
                  f"{s['corr_pooled']:>9.4f}{s['corr_spatial']:>9.4f}{s['anomaly_slope']:>8.4f}")
    print("  r_pool is inflated by the seasonal cycle; r_spat (per-timestep, across cells) is "
          "the honest one.")
    print("\nwinds (direction offset should be ~0 after de-rotation; speed ratio >1 is resolution)")
    for dom, w in out["winds"].items():
        if "mean_direction_offset_deg" in w:
            print(f"  {dom:5s} dir_offset={w['mean_direction_offset_deg']:+7.2f} deg   "
                  f"speed_ratio={w['speed_ratio_hr_over_lr']:.4f}   "
                  f"vec_corr={w['vector_correlation']:.4f}")
    thin = [m for m, k in out["monthly_n_times"].items() if k < 10]
    if thin:
        print(f"\nNOTE: months {thin} have <10 sampled times -- their monthly bias is noisy. "
              f"Use --n 500+ if you need the seasonal shape.")
    if out["lag_correlation_t2m"]:
        print("\nt2m corr vs LR lag (peak must be at 0):")
        for L, v in sorted(out["lag_correlation_t2m"].items(), key=lambda kv: int(kv[0])):
            print(f"  lag {int(L):+d}: {v:.5f}" + ("   <-- peak" if v == max(
                out["lag_correlation_t2m"].values()) else ""))
    if "residual_hr_minus_upsampled_lr" in out:
        print("\nHR - upsample(LR)  (what the model must learn):")
        for c, s in out["residual_hr_minus_upsampled_lr"].items():
            print(f"  {c:5s} mean={s['mean']:+.4f}  std={s['std']:.4f}  |mean|={s['abs_mean']:.4f}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote {args.out}")
    if args.plot:
        make_plot(out, stacks, months, masks, args.plot)


if __name__ == "__main__":
    main()
