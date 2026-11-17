#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""BCSD -- Bias-Correction Spatial Disaggregation (Wood et al. 2004), ERA5 -> CARRA2 t2m/u10/v10.

The classic statistical-downscaling benchmark: no learning beyond per-cell distributions and
climatologies, CPU-only, deterministic. CorrDiff has to beat this for the deep model to be worth
its cost.

TWO STEPS, BOTH FITTED ON THE TRAIN YEARS (2011-2018) ONLY
----------------------------------------------------------
**BC -- bias correction, on the coarse grid.** CARRA2 is coarsened onto the ERA5 LR grid by
footprint mean (``benchmarks/coarsen.py``, the same operator as the consistency check), so at
every LR cell there are two time series of the same quantity: ERA5's, and CARRA2's view of that
grid box. ERA5 is quantile-mapped onto CARRA2's distribution, per cell and per calendar month
(pooling a +-``window``-month neighbourhood, so each map sees ~3 months x 8 years of samples).

**SD -- spatial disaggregation, onto the patch.** The bias-corrected coarse field is expressed
as an anomaly from the coarse CARRA2 climatology, that anomaly is interpolated to 2.5 km with
the model's own bilinear weights, and the HR CARRA2 climatology is put back:

    t2m    HR = clim_HR(s)      +  bilinear( BC - clim_coarse(s) )          (additive)
    speed  HR = clim_HR(s)      *  bilinear( BC / clim_coarse(s) )          (multiplicative)

All the fine-scale structure -- orography, coastlines, lakes -- comes from the climatology. The
day-to-day weather comes from ERA5. That is the method's whole bet, and its known limitation:
spatial detail that is not climatological (a front, a cold pool on one night) cannot appear.

Climatologies are stratified by month x hour of day (``s``: 12 x 8 strata at CARRA2's 3-hourly
cadence) because the diurnal cycle of the fine-scale structure is real -- a valley inversion is a
night-time feature. Quantile maps pool the hours: ~700 samples per cell is enough for a
distribution, ~90 is not.

WINDS -- SPEED AND DIRECTION, NOT u AND v
-----------------------------------------
Quantile-mapping u and v independently treats two components of one vector as unrelated, and
bends the wind direction in ways that depend on the sign of each component. So:

* **speed** is quantile-mapped (BC) and disaggregated multiplicatively, as above;
* **direction** is bias-corrected by a per-cell, per-month mean rotation -- the magnitude-weighted
  circular mean of CARRA2-minus-ERA5 direction, arg sum(z_obs * conj(z_era)) with z = u + iv,
  the same estimator ``scripts/lr_hr_consistency.py`` uses -- and disaggregated by interpolating
  the BC vector bilinearly and adding a per-pixel climatological rotation of HR wind relative to
  the interpolated coarse wind (month x hour). That last term is where terrain channelling lives.

Then ``u, v = speed * (cos, sin)(direction)``, in the store's frame (earth-relative after the
de-rotation). The fit records the frame and ``predict`` refuses a store in a different one.

USAGE
-----
    # fit once (CPU job, ~1 h for 8 years; --n-times subsamples for a quick look)
    python benchmarks/bcsd.py fit --data-dir $PROJECT/data/derot \\
        --years 2011 2012 2013 2014 2015 2016 2017 2018 --out $SCRATCH/benchmarks/bcsd/fit.npz

    # predict on EXACTLY the model's test times, taken from its NetCDF (preferred) ...
    python benchmarks/bcsd.py predict --fit fit.npz --data-dir $PROJECT/data/derot \\
        --times-from $SCRATCH/corrdiff_runs/diffusion_2/eval/full.nc --out bcsd.nc
    # ... or by re-drawing them the way run_eval.sh does (same years, N, seed)
    python benchmarks/bcsd.py predict --fit fit.npz --data-dir $PROJECT/data/derot \\
        --years 2020 2021 2022 --n 400 --seed 0 --out bcsd.nc

    python evaluate/eval_crps_mae.py --nc full=full.nc reg=reg.nc bcsd=bcsd.nc

The output is the ``NetCDFWriter`` schema with a 1-member ensemble, so CRPS == MAE for BCSD and
it lands on the same axis as the diffusion ensemble.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time as _time
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import numpy as np
import pandas as pd
import xarray as xr

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (_REPO_ROOT, _REPO_ROOT / "benchmarks"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from coarsen import (  # noqa: E402
    LRtoHR,
    build_binning_from_store,
    coarsen,
    nearest_fill_index,
    store_order,
)
from dataloading.dataset import open_store, shard_path, store_exists  # noqa: E402

CHANNELS = ("t2m", "u10", "v10")
STEPS_PER_DAY = 8                      # CARRA2 is 3-hourly
N_STRATA = 12 * STEPS_PER_DAY
SPEED_FLOOR = 0.1                      # m/s; guards the multiplicative anomaly against calm climatologies
FIT_VERSION = 1


# ----------------------------------------------------------------------------- strata
def strata(times) -> tuple[np.ndarray, np.ndarray]:
    """(month 0..11, stratum 0..95 = month * 8 + hour // 3) for each time.

    Refuses times off the 3-hourly grid: a 01Z time would silently borrow the 00Z climatology.
    """
    t = pd.DatetimeIndex(times)
    if np.any(t.hour % 3) or np.any(t.minute) or np.any(t.second):
        bad = t[(t.hour % 3 != 0) | (t.minute != 0) | (t.second != 0)][:3]
        raise ValueError(f"times must be on CARRA2's 3-hourly grid, got e.g. {list(bad)}")
    month = np.asarray(t.month - 1)
    return month, month * STEPS_PER_DAY + np.asarray(t.hour // 3)


def month_window(m: int, window: int) -> list[int]:
    """Calendar months within +-window of ``m`` (0-based), wrapping December -> January."""
    return sorted({(m + k) % 12 for k in range(-window, window + 1)})


# ----------------------------------------------------------------------------- quantile mapping
def qmap(x: np.ndarray, q_model: np.ndarray, q_obs: np.ndarray) -> np.ndarray:
    """Empirical quantile mapping, one map per cell.  x: (n,)  q_*: (nq, n)  ->  (n,).

    Piecewise-linear between the fitted quantiles; beyond them the end offset is held constant
    (``x + q_obs[end] - q_model[end]``), the usual choice -- a test-year extreme outside the
    training range gets the tail's bias correction rather than being clipped to the training max.
    """
    x = np.asarray(x, dtype=np.float64)
    qm = np.asarray(q_model, dtype=np.float64)
    qo = np.asarray(q_obs, dtype=np.float64)
    nq, n = qm.shape
    cols = np.arange(n)
    k = (qm < x[None]).sum(axis=0)                 # qm[k-1] < x <= qm[k]
    lo = np.clip(k - 1, 0, nq - 1)
    hi = np.clip(k, 0, nq - 1)
    qm_lo, qm_hi = qm[lo, cols], qm[hi, cols]
    qo_lo, qo_hi = qo[lo, cols], qo[hi, cols]
    denom = qm_hi - qm_lo
    frac = np.divide(x - qm_lo, denom, out=np.zeros_like(x), where=denom > 0)
    y = qo_lo + frac * (qo_hi - qo_lo)
    y = np.where(k == 0, x + (qo[0] - qm[0]), y)
    y = np.where(k == nq, x + (qo[-1] - qm[-1]), y)
    return y


# ----------------------------------------------------------------------------- store access
def _open(data_dir, year) -> xr.Dataset:
    s = shard_path(data_dir, year)
    if not store_exists(s):
        sys.exit(f"no shard for {year} under {data_dir}")
    return xr.open_zarr(open_store(s), chunks=None)


def _channel_index(z: xr.Dataset) -> tuple[list[int], list[int]]:
    hr_names, lr_names = list(z.attrs["hr_channels"]), list(z.attrs["lr_channels"])
    missing = [c for c in CHANNELS if c not in hr_names or c not in lr_names]
    if missing:
        sys.exit(f"store lacks channel(s) {missing}: hr={hr_names} lr={lr_names}")
    return [hr_names.index(c) for c in CHANNELS], [lr_names.index(c) for c in CHANNELS]


def winds_frame(z: xr.Dataset) -> str:
    """The store's recorded wind frame, or "unrecorded" (raw, pre-de-rotation shards)."""
    return str(z.attrs.get("winds_frame", "unrecorded"))


class Geometry:
    """Everything grid-related, built once from a store: coarsening, filling, refining."""

    def __init__(self, z: xr.Dataset):
        self.lat = np.asarray(z["lat"].values, dtype=float)
        self.lon = np.asarray(z["lon"].values, dtype=float)
        self.hr_lat = np.asarray(z["hr_lat"].values, dtype=float)
        self.hr_lon = np.asarray(z["hr_lon"].values, dtype=float)
        self.binning = build_binning_from_store(z)
        self.refine = LRtoHR(self.lat, self.lon, self.hr_lat, self.hr_lon)
        self.active = self.refine.active
        # Cells the stencil touches but the patch does not cover borrow the nearest covered cell.
        has = store_order(self.binning["counts"] > 0, self.binning)
        ii, jj = nearest_fill_index(has)
        nx = self.lon.size
        self.fill_active = (ii * nx + jj).ravel()[self.active]

    def coarse_obs(self, field_hr: np.ndarray) -> np.ndarray:
        """Footprint-mean an HR field onto the active LR cells (nearest-filled off the patch)."""
        return coarsen(field_hr, self.binning).ravel()[self.fill_active]

    def coarse_era(self, field_lr: np.ndarray) -> np.ndarray:
        return np.asarray(field_lr, dtype=np.float64).ravel()[self.active]

    def to_hr(self, *active_fields: np.ndarray) -> np.ndarray:
        """Bilinear (active-cell vectors) -> HR; one field gives (H, W), k fields (H, W, k)."""
        stack = np.stack([self.refine.scatter(f) for f in active_fields], axis=-1)
        out = self.refine(stack)
        return out[..., 0] if len(active_fields) == 1 else out

    def check_same(self, other: dict) -> None:
        for k in ("lat", "lon", "hr_lat", "hr_lon"):
            if not np.allclose(getattr(self, k), other[k]):
                raise ValueError(f"grid mismatch between the fit and this store: {k}")


# ----------------------------------------------------------------------------- fit
def fit(data_dir, years, *, window: int = 1, n_quantiles: int = 101, n_times: int | None = None,
        seed: int = 0, block: int = 32, log=print) -> dict:
    """Fit BCSD on the given (train) years. Streams the HR fields; holds only coarse samples."""
    pool = []
    for y in years:
        with _open(data_dir, y) as z:
            pool.extend((y, i, t) for i, t in enumerate(pd.to_datetime(z["time"].values)))
    if n_times and n_times < len(pool):
        rng = np.random.default_rng(seed)
        pool = [pool[i] for i in np.sort(rng.choice(len(pool), n_times, replace=False))]
    by_year: dict[int, list] = {}
    for y, i, t in pool:
        by_year.setdefault(y, []).append((i, t))

    first = years[0]
    with _open(data_dir, first) as z:
        geo = Geometry(z)
        frame = winds_frame(z)
    if frame == "unrecorded":
        log("WARNING: the store records no winds_frame -- if these are raw CARRA2 shards the HR "
            "winds are GRID-relative, ~100 deg from ERA5's. Fit on the de-rotated shards.")
    H, W = geo.hr_lat.shape
    na = geo.active.size
    T = len(pool)
    log(f"[fit] {T} times from years {list(years)}; {na} active LR cells; HR {H}x{W}")

    # coarse samples for the quantile maps (float32: ~1.5 GB for 8 full years)
    era_T = np.empty((T, na), np.float32)
    obs_T = np.empty((T, na), np.float32)
    era_S = np.empty((T, na), np.float32)
    obs_S = np.empty((T, na), np.float32)
    months = np.empty(T, np.int16)
    # streaming sums per (month, hour) stratum
    n_s = np.zeros(N_STRATA, np.int64)
    sum_T_hr = np.zeros((N_STRATA, H, W))
    sum_S_hr = np.zeros((N_STRATA, H, W))
    cross_hr = np.zeros((N_STRATA, H, W), np.complex128)   # sum z_hr * conj(interp z_obs)
    sum_T_c = np.zeros((N_STRATA, na))
    sum_S_c = np.zeros((N_STRATA, na))
    cross_c = np.zeros((12, na), np.complex128)            # sum z_obs * conj(z_era), per month

    k = 0
    t0 = _time.time()
    for y in sorted(by_year):
        items = by_year[y]
        with _open(data_dir, y) as z:
            geo.check_same({"lat": z["lat"].values, "lon": z["lon"].values,
                            "hr_lat": z["hr_lat"].values, "hr_lon": z["hr_lon"].values})
            if winds_frame(z) != frame:
                sys.exit(f"shard {y} winds_frame {winds_frame(z)!r} != {frame!r} of {first}")
            hr_i, lr_i = _channel_index(z)
            for b0 in range(0, len(items), block):
                chunk = items[b0:b0 + block]
                idx = [i for i, _ in chunk]
                hr = np.asarray(z["hr"].isel(time=idx).values, dtype=np.float64)[:, hr_i]
                lr = np.asarray(z["lr"].isel(time=idx).values, dtype=np.float64)[:, lr_i]
                mo, st = strata([t for _, t in chunk])
                for j in range(len(chunk)):
                    s = st[j]
                    t2m, u, v = hr[j]
                    oT, ou, ov = (geo.coarse_obs(f) for f in (t2m, u, v))
                    eT, eu, ev = (geo.coarse_era(f) for f in lr[j])
                    z_obs, z_era = ou + 1j * ov, eu + 1j * ev
                    era_T[k], obs_T[k] = eT, oT
                    era_S[k], obs_S[k] = np.abs(z_era), np.abs(z_obs)
                    months[k] = mo[j]

                    n_s[s] += 1
                    sum_T_hr[s] += t2m
                    sum_S_hr[s] += np.hypot(u, v)
                    up = geo.to_hr(ou, ov)
                    cross_hr[s] += (u + 1j * v) * np.conj(up[..., 0] + 1j * up[..., 1])
                    sum_T_c[s] += oT
                    sum_S_c[s] += np.abs(z_obs)
                    cross_c[mo[j]] += z_obs * np.conj(z_era)
                    k += 1
                if (b0 // block) % 25 == 0:
                    rate = k / max(_time.time() - t0, 1e-9)
                    log(f"[fit] {k}/{T} times  ({rate:.1f}/s, ~{(T - k) / max(rate, 1e-9) / 60:.0f}"
                        f" min left)")

    empty = [int(s) for s in np.flatnonzero(n_s == 0)]
    if empty:
        raise ValueError(f"strata {empty} (month*8 + hour//3) have no samples -- too few times "
                         f"for a month x hour climatology; raise --n-times")

    levels = np.linspace(0.0, 1.0, n_quantiles)
    q = {name: np.empty((12, n_quantiles, na), np.float32)
         for name in ("era_T", "obs_T", "era_S", "obs_S")}
    theta = np.empty((12, na), np.float32)
    n_month = np.zeros(12, np.int64)
    for m in range(12):
        win = month_window(m, window)
        rows = np.isin(months, win)
        n_month[m] = rows.sum()
        for name, arr in (("era_T", era_T), ("obs_T", obs_T), ("era_S", era_S), ("obs_S", obs_S)):
            q[name][m] = np.quantile(arr[rows], levels, axis=0)
        theta[m] = np.angle(cross_c[win].sum(axis=0))

    # windowed (month x hour) climatologies: stratum (m, h) pools months m-window..m+window at h
    clim = {n: np.empty((N_STRATA,) + shp, np.float32) for n, shp in
            (("T_hr", (H, W)), ("S_hr", (H, W)), ("delta_hr", (H, W)), ("T_c", (na,)), ("S_c", (na,)))}
    for m in range(12):
        for h in range(STEPS_PER_DAY):
            src = [mm * STEPS_PER_DAY + h for mm in month_window(m, window)]
            n = n_s[src].sum()
            s = m * STEPS_PER_DAY + h
            clim["T_hr"][s] = sum_T_hr[src].sum(0) / n
            clim["S_hr"][s] = sum_S_hr[src].sum(0) / n
            clim["delta_hr"][s] = np.angle(cross_hr[src].sum(0))
            clim["T_c"][s] = sum_T_c[src].sum(0) / n
            clim["S_c"][s] = sum_S_c[src].sum(0) / n

    config = {"version": FIT_VERSION, "data_dir": str(data_dir), "years": [int(y) for y in years],
              "n_times": T, "window_months": window, "n_quantiles": n_quantiles, "seed": seed,
              "winds_frame": frame, "channels": list(CHANNELS),
              "samples_per_month_map": [int(x) for x in n_month],
              "samples_per_stratum": [int(x) for x in n_s]}
    log(f"[fit] done in {(_time.time() - t0) / 60:.1f} min; per-month QM samples "
        f"{int(n_month.min())}..{int(n_month.max())}")
    return {"config": config, "levels": levels, "active": geo.active,
            "lat": geo.lat, "lon": geo.lon, "hr_lat": geo.hr_lat, "hr_lon": geo.hr_lon,
            "theta": theta, **{f"q_{k}": v for k, v in q.items()},
            **{f"clim_{k}": v for k, v in clim.items()}}


def save_fit(model: dict, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: v for k, v in model.items() if k != "config"}
    np.savez_compressed(path, config=json.dumps(model["config"]), **arrays)


def load_fit(path) -> dict:
    with np.load(path) as f:
        out = {k: f[k] for k in f.files if k != "config"}
        out["config"] = json.loads(str(f["config"]))
    if out["config"].get("version") != FIT_VERSION:
        raise ValueError(f"{path}: fit version {out['config'].get('version')}, "
                         f"this code reads {FIT_VERSION}; refit")
    return out


# ----------------------------------------------------------------------------- predict
class BCSD:
    """A fitted model, applied one time at a time."""

    def __init__(self, model: dict, geo: Geometry):
        geo.check_same(model)
        if not np.array_equal(geo.active, model["active"]):
            raise ValueError("active LR cell set differs between the fit and this store")
        self.m, self.geo = model, geo

    def predict(self, lr_t2m, lr_u, lr_v, time) -> dict:
        """ERA5 t2m/u10/v10 on the stored LR grid at ``time`` -> HR fields (H, W) per channel."""
        m, geo = self.m, self.geo
        (month,), (s,) = strata([time])

        # t2m: BC by quantile map, SD by additive anomaly
        eT = geo.coarse_era(lr_t2m)
        T_bc = qmap(eT, m["q_era_T"][month], m["q_obs_T"][month])
        t2m = m["clim_T_hr"][s] + geo.to_hr(T_bc - m["clim_T_c"][s])

        # wind: speed by quantile map + multiplicative anomaly; direction by mean rotation
        z_era = geo.coarse_era(lr_u) + 1j * geo.coarse_era(lr_v)
        S_bc = np.maximum(qmap(np.abs(z_era), m["q_era_S"][month], m["q_obs_S"][month]), 0.0)
        z_bc = S_bc * np.exp(1j * (np.angle(z_era) + m["theta"][month]))
        ratio = S_bc / np.maximum(m["clim_S_c"][s], SPEED_FLOOR)
        speed = m["clim_S_hr"][s] * np.maximum(geo.to_hr(ratio), 0.0)
        up = geo.to_hr(z_bc.real, z_bc.imag)
        direction = np.arctan2(up[..., 1], up[..., 0]) + m["clim_delta_hr"][s]
        return {"t2m": t2m, "u10": speed * np.cos(direction), "v10": speed * np.sin(direction)}


def resolve_times(args) -> pd.DatetimeIndex:
    if args.times_from:
        from write_nc import read_times  # noqa: PLC0415
        return read_times(args.times_from)
    # Re-draw exactly as run_eval.sh does (evaluate/sample_times.py): same pool, same RNG calls.
    sys.path.insert(0, str(_REPO_ROOT / "evaluate"))
    from sample_times import load_times  # noqa: PLC0415
    t = load_times(args.data_dir, args.years)
    n = min(args.n, t.size)
    idx = np.sort(np.random.default_rng(args.seed).choice(t.size, size=n, replace=False))
    return pd.DatetimeIndex(t[idx])


def run_predict(args) -> None:
    from write_nc import SchemaWriter  # noqa: PLC0415

    model = load_fit(args.fit)
    cfg = model["config"]
    times = resolve_times(args)
    train_overlap = sorted(set(times.year) & set(cfg["years"]))
    if train_overlap:
        print(f"WARNING: predicting on year(s) {train_overlap}, which the fit was TRAINED on -- "
              f"these scores are in-sample and say nothing about skill.", file=sys.stderr)
    print(f"[predict] {len(times)} times, {times.min()} .. {times.max()}")

    ref_truth = None
    if args.times_from and not args.no_check_truth:
        ref_truth = xr.open_dataset(args.times_from, group="truth")

    years = sorted(set(times.year))
    first = True
    worst = {c: 0.0 for c in CHANNELS}
    with _open(args.data_dir, years[0]) as z:
        geo = Geometry(z)
        frame = winds_frame(z)
    if frame != cfg["winds_frame"]:
        sys.exit(f"wind frame mismatch: fit {cfg['winds_frame']!r}, store {frame!r}")
    bcsd = BCSD(model, geo)

    attrs = {"source": "benchmarks/bcsd.py", "method": "BCSD (Wood et al. 2004)",
             "fit_config": json.dumps(cfg), "fit_path": str(args.fit)}
    out_i = 0
    with SchemaWriter(args.out, geo.hr_lat, geo.hr_lon, CHANNELS, attrs=attrs) as w:
        for y in years:
            sel = times[times.year == y]
            with _open(args.data_dir, y) as z:
                if winds_frame(z) != frame:
                    sys.exit(f"shard {y} winds_frame {winds_frame(z)!r} != {frame!r}")
                geo.check_same({"lat": z["lat"].values, "lon": z["lon"].values,
                                "hr_lat": z["hr_lat"].values, "hr_lon": z["hr_lon"].values})
                hr_i, lr_i = _channel_index(z)
                store_t = pd.DatetimeIndex(pd.to_datetime(z["time"].values))
                pos = store_t.get_indexer(sel)
                if (pos < 0).any():
                    sys.exit(f"times not in shard {y}: {list(sel[pos < 0][:3])}")
                for t, i in zip(sel, pos):
                    hr = np.asarray(z["hr"].isel(time=int(i)).values, dtype=np.float64)[hr_i]
                    lr = np.asarray(z["lr"].isel(time=int(i)).values, dtype=np.float64)[lr_i]
                    pred = bcsd.predict(*lr, t)
                    truth = dict(zip(CHANNELS, hr))
                    if ref_truth is not None:
                        # out_i indexes the reference file because times were read from it in order
                        for c in CHANNELS:
                            d = np.nanmax(np.abs(ref_truth[c].isel(time=out_i).values - truth[c]))
                            worst[c] = max(worst[c], float(d))
                    w.write(t, truth, pred)
                    out_i += 1
                    if first or out_i % 50 == 0:
                        print(f"[predict] {out_i}/{len(times)}")
                        first = False
    if ref_truth is not None:
        ref_truth.close()
        print("[predict] max |truth - reference truth|: "
              + "  ".join(f"{c}={worst[c]:.2e}" for c in CHANNELS))
        bad = {c: v for c, v in worst.items() if v > args.truth_tol}
        if bad:
            sys.exit(f"truth disagrees with {args.times_from} beyond {args.truth_tol}: {bad} -- "
                     f"BCSD was scored against a different store (raw vs de-rotated?). The NetCDF "
                     f"at {args.out} must not be compared with that model.")
    print(f"[predict] wrote {args.out}")


# ----------------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fit", help="fit on the train years")
    f.add_argument("--data-dir", required=True)
    f.add_argument("--years", type=int, nargs="+", required=True)
    f.add_argument("--out", required=True, help="fitted model (.npz)")
    f.add_argument("--window", type=int, default=1, help="+- months pooled per map (default 1)")
    f.add_argument("--n-quantiles", type=int, default=101)
    f.add_argument("--n-times", type=int, default=None,
                   help="random subsample of training times (default: all)")
    f.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("predict", help="downscale the test times to a schema NetCDF")
    p.add_argument("--fit", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out", required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--times-from", help="schema NetCDF (e.g. full.nc) to take the times from")
    g.add_argument("--years", type=int, nargs="+", help="re-draw times as run_eval.sh does")
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-check-truth", action="store_true",
                   help="skip comparing our truth with --times-from's truth group")
    p.add_argument("--truth-tol", type=float, default=1e-2,
                   help="max allowed |truth - reference truth| (default 1e-2; float32 round trip "
                        "through normalisation is ~1e-5)")
    args = ap.parse_args(argv)

    if args.cmd == "fit":
        model = fit(args.data_dir, args.years, window=args.window, n_quantiles=args.n_quantiles,
                    n_times=args.n_times, seed=args.seed)
        save_fit(model, args.out)
        print(f"[fit] wrote {args.out}")
    else:
        run_predict(args)


if __name__ == "__main__":
    main()
