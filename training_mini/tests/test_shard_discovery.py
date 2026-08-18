# SPDX-License-Identifier: Apache-2.0
"""Shard *location* is shared, so archiving a shard never hides it from a tool.

A shard exists in two interchangeable forms -- a loose ``shard_YYYY.zarr`` directory and a
1-inode ``shard_YYYY.zarr.zip`` archive (``scripts/zip_shard.py``). The dataset loader has
always understood both; the tools around it (stats, eval-time sampling, verification) used to
hardcode ``.zarr`` and would silently skip an archived shard -- which broke the archive-then-eval
recipe in ``evaluate/README.md``. These tests pin the shared behaviour down.

Runs anywhere: the fixtures build a tiny synthetic shard in the store schema rather than
depending on ``testing_data/``.

    pytest training_mini/tests/test_shard_discovery.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_DIR = REPO_ROOT / "training_mini"
for _p in (str(TRAIN_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dataloading.dataset import (  # noqa: E402
    PatchDataset,
    discover_shards,
    open_store,
    resolve_stores,
    shard_path,
    store_exists,
)
from dataloading.stats import compute_norm_stats  # noqa: E402

HR_CHANNELS = ["t2m", "u10", "v10"]
LR_CHANNELS = ["t2m", "u10", "v10", "siconc"]
NT, NY, NX, NLAT, NLON = 6, 8, 8, 4, 5


def make_shard(path: Path, year: int, seed: int = 0) -> Path:
    """Write a tiny store in the real schema (see data_acquisition/dataset_builder.py)."""
    rng = np.random.default_rng(seed)
    lat = np.linspace(68.0, 66.0, NLAT)           # descending, as ERA5 is stored
    lon = np.linspace(228.0, 232.0, NLON)         # [0, 360)
    hr_lon, hr_lat = np.meshgrid(
        np.linspace(229.0, 231.0, NX), np.linspace(67.5, 66.5, NY)
    )
    ds = xr.Dataset(
        {
            "hr": (("time", "hr_channel", "y", "x"),
                   rng.normal(260.0, 5.0, (NT, len(HR_CHANNELS), NY, NX)).astype("float32")),
            "lr": (("time", "lr_channel", "lat", "lon"),
                   rng.normal(260.0, 5.0, (NT, len(LR_CHANNELS), NLAT, NLON)).astype("float32")),
            "land_sea_mask": (("y", "x"), rng.integers(0, 2, (NY, NX)).astype("float32")),
        },
        coords={
            "time": np.array([f"{year}-01-01T{3 * i:02d}:00:00" for i in range(NT)],
                             dtype="datetime64[ns]"),
            "hr_lat": (("y", "x"), hr_lat), "hr_lon": (("y", "x"), hr_lon),
            "lat": lat, "lon": lon,
        },
    )
    ds.attrs["hr_channels"] = list(HR_CHANNELS)
    ds.attrs["lr_channels"] = list(LR_CHANNELS)
    ds.to_zarr(path, mode="w", encoding={
        "hr": {"chunks": (1, len(HR_CHANNELS), NY, NX)},
        "lr": {"chunks": (1, len(LR_CHANNELS), NLAT, NLON)},
    })
    return path


def zip_of(loose: Path, remove_src: bool = True) -> Path:
    """Archive `loose` to `<loose>.zip`, optionally dropping the directory store."""
    import shutil

    import scripts.zip_shard as zs

    dst = loose.with_suffix(".zarr.zip")
    zs.zip_store(str(loose), str(dst))
    if remove_src:
        shutil.rmtree(loose)
    return dst


@pytest.fixture(scope="module")
def mixed_dir(tmp_path_factory) -> Path:
    """The layout evaluate/README.md produces: loose train years + archived test years."""
    d = tmp_path_factory.mktemp("shards")
    make_shard(d / "shard_2018.zarr", 2018, seed=1)          # loose (train)
    make_shard(d / "shard_2019.zarr", 2019, seed=2)          # loose (train)
    zip_of(make_shard(d / "shard_2020.zarr", 2020, seed=3))  # archived (test)
    zip_of(make_shard(d / "shard_2021.zarr", 2021, seed=4))  # archived (test)
    (d / "shard_2022.zarr.zip.partial").write_bytes(b"")     # interrupted archive run
    return d


# ------------------------------------------------------------------ discovery primitives
def test_shard_path_prefers_archive(mixed_dir):
    assert shard_path(mixed_dir, 2018).endswith("shard_2018.zarr")
    assert shard_path(mixed_dir, 2020).endswith("shard_2020.zarr.zip")


def test_shard_path_falls_back_to_loose_when_absent(mixed_dir):
    """A year that isn't there yields the loose path, so callers raise their own error."""
    assert shard_path(mixed_dir, 1999).endswith("shard_1999.zarr")
    assert not store_exists(shard_path(mixed_dir, 1999))


def test_discover_shards_spans_both_forms_and_ignores_partials(mixed_dir):
    found = [Path(s).name for s in discover_shards(mixed_dir)]
    assert found == ["shard_2018.zarr", "shard_2019.zarr",
                     "shard_2020.zarr.zip", "shard_2021.zarr.zip"]


def test_discover_shards_dedupes_to_the_archive(tmp_path):
    """Both forms of one year present -> exactly one store, the archive."""
    make_shard(tmp_path / "shard_2011.zarr", 2011)
    zip_of(tmp_path / "shard_2011.zarr", remove_src=False)
    found = discover_shards(tmp_path)
    assert found == [str(tmp_path / "shard_2011.zarr.zip")]


def test_store_exists_discriminates_file_and_dir(mixed_dir):
    assert store_exists(mixed_dir / "shard_2018.zarr")
    assert store_exists(mixed_dir / "shard_2020.zarr.zip")
    assert not store_exists(mixed_dir / "shard_2020.zarr")      # archived, dir is gone


def test_resolve_stores_single_store_and_directory(mixed_dir):
    assert resolve_stores(mixed_dir / "shard_2020.zarr.zip") == [
        str(mixed_dir / "shard_2020.zarr.zip")]
    assert resolve_stores(mixed_dir / "shard_2018.zarr") == [
        str(mixed_dir / "shard_2018.zarr")]
    assert [Path(s).name for s in resolve_stores(mixed_dir, [2019, 2020])] == [
        "shard_2019.zarr", "shard_2020.zarr.zip"]
    with pytest.raises(ValueError, match="years"):
        resolve_stores(mixed_dir)


# ------------------------------------------------------------------ the tools that were blind
def test_compute_norm_stats_reads_an_archived_shard(tmp_path):
    """P0-1: stats.py opened stores bare, so it could not see a .zarr.zip."""
    loose = make_shard(tmp_path / "shard_2011.zarr", 2011, seed=7)
    from_loose = compute_norm_stats([str(loose)])
    archived = zip_of(loose)
    from_zip = compute_norm_stats([str(archived)])

    assert from_zip["hr_channels"] == HR_CHANNELS
    for key in ("hr_mean", "hr_std", "lr_mean", "lr_std"):
        assert np.allclose(from_loose[key], from_zip[key])


def test_make_stats_resolves_and_writes_over_archived_shards(mixed_dir, tmp_path, monkeypatch):
    """P0-1: tools/make_stats.py built shard_{y}.zarr paths only."""
    import json

    import tools.make_stats as ms

    stores = ms.resolve_stores(str(mixed_dir), [2020, 2021], None)
    assert [Path(s).name for s in stores] == ["shard_2020.zarr.zip", "shard_2021.zarr.zip"]

    out = tmp_path / "stats.json"
    monkeypatch.setattr(sys, "argv", ["make_stats.py", "--data-dir", str(mixed_dir),
                                      "--years", "2020", "2021", "--out", str(out)])
    ms.main()
    stats = json.loads(out.read_text())
    assert stats["lr_channels"] == LR_CHANNELS
    assert "lsm_mean" in stats and "lsm_std" in stats   # read from the archived store


def test_sample_times_finds_archived_years_in_a_mixed_dir(mixed_dir):
    """P0-1, the headline regression: run_eval.sh died here on the README's own recipe."""
    import evaluate.sample_times as st

    times = st.load_times(str(mixed_dir), [2020, 2021])
    assert len(times) == 2 * NT
    assert set(pd_years(times)) == {2020, 2021}


def test_sample_times_spans_loose_and_archived_together(mixed_dir):
    """A mixed dir must serve both forms in one draw -- the case that failed most confusingly."""
    import evaluate.sample_times as st

    times = st.load_times(str(mixed_dir), [2019, 2020])
    assert len(times) == 2 * NT
    assert set(pd_years(times)) == {2019, 2020}


def pd_years(times):
    import pandas as pd

    return pd.DatetimeIndex(times).year.tolist()


def test_sample_times_still_errors_when_nothing_matches(tmp_path):
    import evaluate.sample_times as st

    with pytest.raises(SystemExit):
        st.load_times(str(tmp_path), [2020])


def test_sample_times_warns_about_a_missing_year(mixed_dir, capsys):
    """Under-sampling must be loud: a year with no shard is the same silent failure as P0-1."""
    import evaluate.sample_times as st

    times = st.load_times(str(mixed_dir), [2020, 1999])
    assert "1999" in capsys.readouterr().err
    assert set(pd_years(times)) == {2020}


def test_verify_shards_discovers_archived_shards(mixed_dir, capsys, monkeypatch):
    """P0-1: verify_shards globbed shard_*.zarr, so archived shards went unverified."""
    import scripts.verify_shards as vs

    monkeypatch.setattr(sys, "argv", ["verify_shards.py", str(mixed_dir)])
    vs.main()
    out = capsys.readouterr().out
    for name in ("shard_2018.zarr", "shard_2020.zarr.zip", "shard_2021.zarr.zip"):
        assert name in out
    assert "hr_nonfinite=0" in out


def test_patch_dataset_reads_both_forms_identically(tmp_path):
    """The loader contract the tools above now share: same bytes either way."""
    loose = make_shard(tmp_path / "shard_2011.zarr", 2011, seed=11)
    a = PatchDataset(str(loose))
    archived = zip_of(loose, remove_src=False)
    b = PatchDataset(str(archived))
    assert len(a) == len(b)
    for i in range(len(a)):
        assert np.array_equal(a[i]["hr"].numpy(), b[i]["hr"].numpy())
        assert np.array_equal(a[i]["lr"].numpy(), b[i]["lr"].numpy())
