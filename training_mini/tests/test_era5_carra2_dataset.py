# SPDX-License-Identifier: Apache-2.0
"""Local (CPU) unit test for the ERA5->CARRA2 CorrDiff adapter.

Runs on the dev machine (no CUDA / physicsnemo needed) against
``testing_data/shard_2011.zarr``. Validates the DownscalingDataset contract, the GPU
LRConditioner shapes (on CPU), channel selection for the no-sea-ice variant, and that
the (de)normalization methods are proper inverses.

    pytest training_mini/tests/test_era5_carra2_dataset.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_DIR = REPO_ROOT / "training_mini"
for _p in (str(TRAIN_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dataloading.stats import compute_norm_stats  # noqa: E402
from datasets.era5_carra2 import ERA5CARRA2Dataset  # noqa: E402

TESTING_DATA = REPO_ROOT / "testing_data"
SHARD = TESTING_DATA / "shard_2011.zarr"

pytestmark = pytest.mark.skipif(
    not SHARD.exists(), reason="testing_data/shard_2011.zarr not present"
)

FULL_LR = ["t2m", "u10", "v10", "t500", "t850", "z500", "z850",
           "u500", "u850", "v500", "v850", "siconc"]
HR = ["t2m", "u10", "v10"]


@pytest.fixture(scope="module")
def stats_path(tmp_path_factory):
    stats = compute_norm_stats([str(SHARD)])
    with xr.open_zarr(str(SHARD)) as z:
        mask = np.asarray(z["land_sea_mask"].values, dtype="float64")
    stats["lsm_mean"] = float(mask.mean())
    stats["lsm_std"] = float(mask.std())
    p = tmp_path_factory.mktemp("stats") / "stats.json"
    p.write_text(json.dumps(stats))
    return str(p)


@pytest.fixture(scope="module")
def ds(stats_path):
    return ERA5CARRA2Dataset(
        data_path=str(TESTING_DATA), stats_path=stats_path, years=[2011]
    )


def test_len(ds):
    assert len(ds) > 0


def test_getitem_shapes(ds):
    hr, lr = ds[0]
    assert isinstance(hr, torch.Tensor) and isinstance(lr, torch.Tensor)
    assert hr.dtype == torch.float32 and lr.dtype == torch.float32
    assert tuple(hr.shape) == (3, 448, 448)
    # LR stays on the coarse ERA5 grid (upsampling happens later on GPU)
    coarse_hw = (ds._ref.lr_lat.shape[0], ds._ref.lr_lon.shape[0])
    assert tuple(lr.shape) == (12,) + coarse_hw
    assert torch.isfinite(hr).all() and torch.isfinite(lr).all()


def test_metadata(ds):
    ic = ds.input_channels()
    oc = ds.output_channels()
    assert [c.name for c in ic] == FULL_LR + ["lsm"]
    assert ic[-1].auxiliary is True
    assert [c.name for c in oc] == HR
    assert ds.image_shape() == (448, 448)
    assert ds.latitude().shape == (448, 448)
    assert ds.longitude().shape == (448, 448)


def test_conditioner_shapes(ds):
    _, lr = ds[0]
    cond = ds.make_conditioner()
    out = cond(lr.unsqueeze(0))  # (1, C+lsm, H, W)
    assert tuple(out.shape) == (1, 13, 448, 448)
    assert torch.isfinite(out).all()


def test_output_normalization_roundtrip(ds):
    hr, _ = ds[0]
    x = hr.numpy()  # already normalized
    back = ds.normalize_output(ds.denormalize_output(x))
    assert np.allclose(back, x, atol=1e-3)


def test_input_normalization_roundtrip(ds):
    _, lr = ds[0]
    y = ds.make_conditioner()(lr.unsqueeze(0))[0].numpy()  # normalized, (13,H,W)
    back = ds.normalize_input(ds.denormalize_input(y))
    assert np.allclose(back, y, atol=1e-3)


def test_noice_channel_selection(stats_path):
    ds2 = ERA5CARRA2Dataset(
        data_path=str(TESTING_DATA), stats_path=stats_path, years=[2011],
        lr_channels=FULL_LR[:-1],  # drop siconc
    )
    _, lr = ds2[0]
    assert lr.shape[0] == 11
    assert [c.name for c in ds2.input_channels()] == FULL_LR[:-1] + ["lsm"]
    out = ds2.make_conditioner()(lr.unsqueeze(0))
    assert tuple(out.shape) == (1, 12, 448, 448)


def test_invalid_channel_raises(stats_path):
    with pytest.raises(ValueError):
        ERA5CARRA2Dataset(
            data_path=str(TESTING_DATA), stats_path=stats_path, years=[2011],
            lr_channels=["t2m", "not_a_channel"],
        )
