# SPDX-License-Identifier: Apache-2.0
"""BCSD benchmark: the pieces, the schema it emits, and an end-to-end recovery test.

The end-to-end test builds two synthetic yearly shards where the answer is known: ERA5 is the
large-scale CARRA2 field plus a +2 K bias, with winds 20% too slow and rotated 30 deg; CARRA2 adds
a fixed fine-scale pattern (stand-in for orography) and per-pixel wind channelling. BCSD fitted
on one year must remove the bias, the rotation and recover the pattern on the OTHER year -- which
bilinear interpolation of ERA5 cannot.

    pytest training_mini/tests/test_bcsd.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO), str(REPO / "benchmarks"), str(REPO / "evaluate")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

pytest.importorskip("scipy")
pytest.importorskip("netCDF4")
xr = pytest.importorskip("xarray")
bcsd = pytest.importorskip("bcsd")
coarsen = pytest.importorskip("coarsen")
write_nc = pytest.importorskip("write_nc")


# ----------------------------------------------------------------------------- quantile mapping
def test_qmap_maps_model_quantiles_onto_obs_quantiles():
    rng = np.random.default_rng(0)
    qm = np.sort(rng.normal(size=(21, 5)), axis=0)
    qo = np.sort(rng.normal(2.0, 3.0, size=(21, 5)), axis=0)
    for k in (0, 7, 20):
        np.testing.assert_allclose(bcsd.qmap(qm[k], qm, qo), qo[k])


def test_qmap_holds_the_tail_offset_constant_beyond_the_fit():
    qm = np.linspace(0, 10, 11)[:, None]
    qo = qm + 3.0
    qo[-1] = 15.0                                         # top offset +5, bottom +3
    np.testing.assert_allclose(bcsd.qmap(np.array([-4.0]), qm, qo), [-1.0])
    np.testing.assert_allclose(bcsd.qmap(np.array([20.0]), qm, qo), [25.0])


def test_qmap_is_identity_when_distributions_match_and_survives_ties():
    qm = np.array([[0.0], [0.0], [0.0], [1.0], [2.0]])      # calm winds: many zero quantiles
    for xi in (0.0, 0.5, 1.5):
        np.testing.assert_allclose(bcsd.qmap(np.array([xi]), qm, qm), [xi])


# ----------------------------------------------------------------------------- strata
def test_strata_and_month_window():
    month, s = bcsd.strata(pd.to_datetime(["2020-01-01T00", "2020-01-01T21", "2020-12-31T03"]))
    assert list(month) == [0, 0, 11] and list(s) == [0, 7, 89]
    assert bcsd.month_window(0, 1) == [0, 1, 11]
    assert bcsd.month_window(5, 0) == [5]
    with pytest.raises(ValueError, match="3-hourly"):
        bcsd.strata(pd.to_datetime(["2020-01-01T01"]))


# ----------------------------------------------------------------------------- geometry
LR_LAT = np.arange(70.0, 68.0, -0.25)            # 8, descending like the store
LR_LON = np.arange(230.0, 232.0, 0.25)           # 8
HR_LON, HR_LAT = np.meshgrid(np.linspace(230.15, 231.6, 24), np.linspace(68.4, 69.85, 24))


def test_bilinear_refine_reproduces_a_linear_field_exactly():
    r = coarsen.LRtoHR(LR_LAT, LR_LON, HR_LAT, HR_LON)
    lon2, lat2 = np.meshgrid(LR_LON, LR_LAT)
    out = r(3.0 * lat2 - 2.0 * lon2)
    np.testing.assert_allclose(out, 3.0 * HR_LAT - 2.0 * HR_LON, atol=1e-9)


def test_refine_on_active_cells_matches_refine_on_the_full_grid():
    r = coarsen.LRtoHR(LR_LAT, LR_LON, HR_LAT, HR_LON)
    f = np.random.default_rng(1).normal(size=r.lr_shape)
    np.testing.assert_allclose(r(r.scatter(f.ravel()[r.active])), r(f))


def test_coarsen_footprint_mean_of_a_constant_is_that_constant():
    b = coarsen.build_binning(LR_LAT, LR_LON, HR_LAT, HR_LON)
    c = coarsen.coarsen(np.full(HR_LAT.shape, 7.5), b)
    assert np.allclose(c[np.isfinite(c)], 7.5)
    assert np.isnan(c[0, 0])                         # (70.0, 230.0) holds no HR point


# ----------------------------------------------------------------------------- schema
def test_schema_writer_is_what_the_evaluator_reads(tmp_path):
    from eval_crps_mae import eval_nc

    times = pd.to_datetime(["2020-03-01T06", "2021-07-04T21"])
    path = tmp_path / "b.nc"
    with write_nc.SchemaWriter(path, HR_LAT, HR_LON, ["t2m"]) as w:
        for t in times:
            w.write(t, {"t2m": np.zeros(HR_LAT.shape)}, {"t2m": np.full(HR_LAT.shape, 2.0)})
    m = eval_nc(str(path))["t2m"]
    assert m["crps"] == pytest.approx(2.0) and m["mae"] == pytest.approx(2.0)
    assert m["ens_members"] == 1 and m["n_times"] == 2
    assert list(write_nc.read_times(path)) == list(times)
    with xr.open_dataset(path) as root:
        assert root["time"].dtype.kind == "M"        # decodes as datetimes, like generate.py's
        assert root["lat"].dims == ("y", "x")
    with xr.open_dataset(path, group="prediction") as p:
        assert p["t2m"].dims == ("ensemble", "time", "y", "x")


# ----------------------------------------------------------------------------- end to end
FINE = 2.0 * np.sin(HR_LAT * 9.0) * np.cos(HR_LON * 7.0)      # static fine-scale t2m pattern
CHANNEL = 0.25 * np.cos(HR_LAT * 5.0)                         # static per-pixel wind rotation (rad)
GUST = 1.0 + 0.3 * np.sin(HR_LON * 6.0)                       # static per-pixel speed-up


def _large_scale(lat, lon, t):
    doy = t.dayofyear + t.hour / 24.0
    T = 255.0 + 15.0 * np.sin(2 * np.pi * (doy - 100) / 365) + 2.0 * np.sin(2 * np.pi * t.hour / 24)
    T = T + 3.0 * np.sin(0.9 * doy + 2.0 * (lat - 69.0)) + 1.5 * (lon - 231.0)
    speed = 6.0 + 3.0 * np.sin(0.37 * doy)
    ang = 0.53 * doy + 0.2 * (lat - 69.0)
    return T, speed * np.exp(1j * ang)


def _write_shard(path, year):
    times = pd.date_range(f"{year}-01-01", f"{year}-12-31T21", freq="3h")
    lon2, lat2 = np.meshgrid(LR_LON, LR_LAT)
    hr = np.empty((len(times), 3) + HR_LAT.shape, np.float32)
    lr = np.empty((len(times), 3) + lat2.shape, np.float32)
    for i, t in enumerate(times):
        T_hr, z_hr = _large_scale(HR_LAT, HR_LON, t)
        z_hr = z_hr * GUST * np.exp(1j * CHANNEL)
        hr[i] = np.stack([T_hr + FINE, z_hr.real, z_hr.imag])
        T_lr, z_lr = _large_scale(lat2, lon2, t)
        z_lr = 0.8 * z_lr * np.exp(-1j * np.deg2rad(30.0))          # slow and rotated
        lr[i] = np.stack([T_lr + 2.0, z_lr.real, z_lr.imag])        # +2 K warm bias
    ds = xr.Dataset(
        {"hr": (("time", "hr_channel", "y", "x"), hr),
         "lr": (("time", "lr_channel", "lat", "lon"), lr),
         "land_sea_mask": (("y", "x"), np.ones(HR_LAT.shape, np.float32))},
        coords={"time": times, "lat": LR_LAT, "lon": LR_LON,
                "hr_lat": (("y", "x"), HR_LAT), "hr_lon": (("y", "x"), HR_LON)},
        attrs={"hr_channels": ["t2m", "u10", "v10"], "lr_channels": ["t2m", "u10", "v10"],
               "winds_frame": "earth-relative (synthetic)"})
    ds.to_zarr(path, mode="w")


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    d = tmp_path_factory.mktemp("shards")
    for y in (2001, 2002):
        _write_shard(d / f"shard_{y}.zarr", y)
    model = bcsd.fit(d, [2001], log=lambda *_: None)
    return d, model


def test_fit_round_trips_through_npz(synthetic, tmp_path):
    _, model = synthetic
    bcsd.save_fit(model, tmp_path / "fit.npz")
    back = bcsd.load_fit(tmp_path / "fit.npz")
    assert back["config"] == model["config"]
    np.testing.assert_array_equal(back["clim_T_hr"], model["clim_T_hr"])


def test_bcsd_recovers_bias_rotation_and_fine_scale_out_of_sample(synthetic, tmp_path):
    d, model = synthetic
    out = tmp_path / "bcsd.nc"
    bcsd.main(["predict", "--fit", str(_saved(model, tmp_path)), "--data-dir", str(d),
               "--years", "2002", "--n", "40", "--out", str(out)])

    truth = xr.open_dataset(out, group="truth")
    pred = xr.open_dataset(out, group="prediction").isel(ensemble=0)
    times = write_nc.read_times(out)

    # bilinear ERA5 on the same times -- what BCSD has to beat
    r = coarsen.LRtoHR(LR_LAT, LR_LON, HR_LAT, HR_LON)
    with xr.open_zarr(d / "shard_2002.zarr") as z:
        idx = pd.DatetimeIndex(z["time"].values).get_indexer(times)
        lr = z["lr"].values[idx]
    bil = np.stack([[r(lr[i, c]) for c in range(3)] for i in range(len(times))])

    # The synthetic world is exactly what BCSD assumes (static fine scale, distributional bias),
    # so recovery should be near-exact, not merely better than bilinear. Bounds are set against
    # measured errors with each term disabled: no coarse rotation -> 48% vector error; no
    # per-pixel channelling -> 2.7%; a flat HR climatology -> t2m MAE ~1.3 K.
    tt = truth["t2m"].values
    assert np.abs(bil[:, 0] - tt).mean() > 1.5                       # the problem is real
    assert np.abs(pred["t2m"].values - tt).mean() < 0.15             # bias + fine scale recovered
    assert abs((pred["t2m"].values - tt).mean()) < 0.1               # the +2 K is gone

    z_t = truth["u10"].values + 1j * truth["v10"].values
    z_p = pred["u10"].values + 1j * pred["v10"].values
    z_b = bil[:, 1] + 1j * bil[:, 2]
    offset = np.rad2deg(np.angle(np.sum(z_p * np.conj(z_t))))
    assert abs(offset) < 1.0                                         # the 30 deg rotation is gone
    assert np.abs(z_b - z_t).mean() / np.abs(z_t).mean() > 0.3       # bilinear is far off
    assert np.abs(z_p - z_t).mean() / np.abs(z_t).mean() < 0.01      # incl. per-pixel channelling
    assert np.abs(z_p).mean() / np.abs(z_t).mean() == pytest.approx(1.0, abs=0.05)
    truth.close()
    pred.close()


def test_predict_refuses_a_store_in_another_wind_frame(synthetic, tmp_path):
    d, model = synthetic
    model = {**model, "config": {**model["config"], "winds_frame": "grid-relative"}}
    with pytest.raises(SystemExit, match="wind frame"):
        bcsd.main(["predict", "--fit", str(_saved(model, tmp_path)), "--data-dir", str(d),
                   "--years", "2002", "--n", "2", "--out", str(tmp_path / "x.nc")])


def test_predict_refuses_truth_that_disagrees_with_the_reference(synthetic, tmp_path):
    d, model = synthetic
    fit = _saved(model, tmp_path)
    ref = tmp_path / "ref.nc"
    bcsd.main(["predict", "--fit", str(fit), "--data-dir", str(d), "--years", "2002",
               "--n", "3", "--out", str(ref)])
    import netCDF4 as nc
    with nc.Dataset(ref, "a") as f:                 # pretend the model saw different truth
        f["truth"]["t2m"][1] = f["truth"]["t2m"][1] + 1.0
    with pytest.raises(SystemExit, match="truth disagrees"):
        bcsd.main(["predict", "--fit", str(fit), "--data-dir", str(d),
                   "--times-from", str(ref), "--out", str(tmp_path / "y.nc")])


def _saved(model, tmp_path):
    p = tmp_path / "fit.npz"
    bcsd.save_fit(model, p)
    return p
