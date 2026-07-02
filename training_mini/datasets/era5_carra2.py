# SPDX-License-Identifier: Apache-2.0
"""CorrDiff ``DownscalingDataset`` adapter for the ERA5 -> CARRA2 zarr shards.

Bridges this repo's data pipeline (``dataloading.PatchDataset`` -- which serves the
*coarse* ERA5 LR grid + native-resolution CARRA2 HR patch) to NVIDIA PhysicsNeMo's
CorrDiff ``DownscalingDataset`` interface.

Design note -- GPU upsampling is preserved
------------------------------------------
CorrDiff expects ``img_lr`` already at the HR resolution (its patching + UNet need
``img_clean`` and ``img_lr`` co-located at 448x448). This repo deliberately keeps LR
*coarse* on disk and upsamples on GPU (see ``dataloading.upsample.BilinearUpsampler``).
We keep that: ``__getitem__`` returns the normalized *coarse* LR, and the upsampling +
static land-sea-mask concat happen on GPU via :class:`LRConditioner`, which the vendored
``train.py`` / ``generate.py`` apply as the first step after the batch lands on device.

Because per-channel normalization is affine and bilinear upsampling is linear across
space, ``upsample(normalize(x)) == normalize(upsample(x))`` -- so normalizing the coarse
LR here stays exactly consistent with the stats.

The two model variants (with / without sea ice) differ only in the ``lr_channels`` list:
drop ``"siconc"`` for the no-ice model. ``input_channels()`` reports the post-conditioning
channel count (selected LR + lsm) so ``train.py`` sizes the input conv correctly.
"""

from __future__ import annotations

import datetime
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xarray as xr

# Make the repo root importable so ``dataloading`` / ``data_acquisition`` resolve when this
# file is loaded standalone via CorrDiff's ``register_dataset`` (importlib by file path).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from datasets.base import ChannelMetadata, DownscalingDataset  # vendored CorrDiff base

from dataloading.dataset import concat_split
from dataloading.stats import load_norm_stats
from dataloading.upsample import BilinearUpsampler

EPS = 1e-6  # guard against zero-variance channels when normalizing


class LRConditioner(nn.Module):
    """GPU module: upsample coarse LR to the HR patch grid and append the static LSM.

    ``forward``: ``(B, C, H_src, W_src) -> (B, C[+1], H_patch, W_patch)``. Built once from
    the dataset's store and moved to the training device; runs as the first step of the
    forward pass, before CorrDiff patching / UNet.
    """

    def __init__(self, upsampler: BilinearUpsampler, lsm: Optional[np.ndarray] = None):
        super().__init__()
        self.upsampler = upsampler
        self.include_lsm = lsm is not None
        if self.include_lsm:
            # already normalized, shape (1, H, W)
            self.register_buffer("lsm", torch.as_tensor(lsm, dtype=torch.float32))

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        up = self.upsampler(lr)  # (B, C, H, W)
        if self.include_lsm:
            lsm = self.lsm.to(up.dtype).expand(up.shape[0], *self.lsm.shape)
            up = torch.cat([up, lsm], dim=1)
        return up


class ERA5CARRA2Dataset(DownscalingDataset):
    """ERA5 (coarse LR) -> CARRA2 (HR) paired samples from per-year zarr shards.

    Parameters
    ----------
    data_path : directory holding ``shard_YYYY.zarr`` shards, or a single ``.zarr`` store.
    stats_path : JSON of train-only normalization stats (see ``tools/make_stats.py``).
    years : years to include (selects ``shard_YYYY.zarr``). Ignored if ``data_path`` is a
        single ``.zarr`` store.
    lr_channels : subset/order of LR channels to feed the model. Default: all channels in
        the store. Drop ``"siconc"`` for the no-sea-ice variant.
    include_lsm : append the static land-sea mask as an auxiliary input channel (on GPU).
    hr_channels : subset/order of HR output channels. Default: all channels in the store.
    """

    def __init__(
        self,
        data_path: str,
        stats_path: str,
        years: Optional[Sequence[int]] = None,
        lr_channels: Optional[Sequence[str]] = None,
        include_lsm: bool = True,
        hr_channels: Optional[Sequence[str]] = None,
    ):
        self.data_path = os.fspath(data_path)
        self.stats_path = os.fspath(stats_path)
        self.include_lsm = bool(include_lsm)

        self.stores = self._resolve_stores(self.data_path, years)
        for s in self.stores:
            if not os.path.exists(s):
                raise FileNotFoundError(f"shard store not found: {s}")

        # Concat the per-year PatchDatasets WITHOUT internal normalization (we normalize
        # here, in CorrDiff's convention). concat_split validates shared grid/channels.
        self._ds = concat_split(self.stores, stats=None, return_mask=False)
        self._ref = self._ds.datasets[0]  # a PatchDataset: geometry + channel names

        # ---- channel selection ----
        store_lr = list(self._ref.lr_channels)
        store_hr = list(self._ref.hr_channels)
        self.lr_channel_names = list(lr_channels) if lr_channels else store_lr
        self.hr_channel_names = list(hr_channels) if hr_channels else store_hr
        self._check_subset(self.lr_channel_names, store_lr, "lr_channels")
        self._check_subset(self.hr_channel_names, store_hr, "hr_channels")
        self._lr_sel = [store_lr.index(c) for c in self.lr_channel_names]
        self._hr_sel = [store_hr.index(c) for c in self.hr_channel_names]

        # ---- normalization stats (torch for __getitem__, numpy for de/normalize_*) ----
        stats = load_norm_stats(self.stats_path)
        lr_mean, lr_std = self._select_stats(stats, "lr", self.lr_channel_names)
        hr_mean, hr_std = self._select_stats(stats, "hr", self.hr_channel_names)
        self._lr_mean = torch.tensor(lr_mean).reshape(-1, 1, 1)
        self._lr_std = torch.tensor(lr_std).clamp_min(EPS).reshape(-1, 1, 1)
        self._hr_mean_t = torch.tensor(hr_mean).reshape(-1, 1, 1)
        self._hr_std_t = torch.tensor(hr_std).clamp_min(EPS).reshape(-1, 1, 1)

        # land-sea-mask stats (for the conditioner + input de/normalization)
        self._lsm_mean = float(stats.get("lsm_mean", 0.0))
        self._lsm_std = float(stats.get("lsm_std", 1.0)) or 1.0

        # numpy stats aligned with input_channels()/output_channels() ordering
        in_mean = list(lr_mean) + ([self._lsm_mean] if self.include_lsm else [])
        in_std = list(lr_std) + ([self._lsm_std] if self.include_lsm else [])
        self._in_mean_np = np.asarray(in_mean, dtype=np.float32).reshape(-1, 1, 1)
        self._in_std_np = np.clip(np.asarray(in_std, dtype=np.float32), EPS, None).reshape(-1, 1, 1)
        self._out_mean_np = np.asarray(hr_mean, dtype=np.float32).reshape(-1, 1, 1)
        self._out_std_np = np.clip(np.asarray(hr_std, dtype=np.float32), EPS, None).reshape(-1, 1, 1)

        self._img_shape = tuple(int(x) for x in self._ref.hr_lat.shape)  # (H, W)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _resolve_stores(data_path: str, years: Optional[Sequence[int]]) -> List[str]:
        p = Path(data_path)
        if p.suffix == ".zarr":
            return [str(p)]
        if not years:
            raise ValueError("`years` is required when `data_path` is a shard directory")
        return [str(p / f"shard_{int(y)}.zarr") for y in years]

    @staticmethod
    def _check_subset(names: Sequence[str], available: Sequence[str], label: str) -> None:
        missing = [c for c in names if c not in available]
        if missing:
            raise ValueError(f"{label} not present in store {list(available)}: {missing}")

    @staticmethod
    def _select_stats(stats: dict, key: str, names: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        chan = list(stats[f"{key}_channels"])
        mean_by = dict(zip(chan, stats[f"{key}_mean"]))
        std_by = dict(zip(chan, stats[f"{key}_std"]))
        mean = np.asarray([mean_by[n] for n in names], dtype=np.float32)
        std = np.asarray([std_by[n] for n in names], dtype=np.float32)
        return mean, std

    # ------------------------------------------------------------------ Dataset API
    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(img_clean=hr [C_out,H,W], img_lr=coarse_lr [C_in,H_src,W_src])``.

        Both normalized float32. Upsampling of the coarse LR + LSM concat happen later on
        GPU via :class:`LRConditioner`.
        """
        sample = self._ds[idx]
        lr = sample["lr"][self._lr_sel]
        hr = sample["hr"][self._hr_sel]
        lr = (lr - self._lr_mean) / self._lr_std
        hr = (hr - self._hr_mean_t) / self._hr_std_t
        return hr.float(), lr.float()

    def make_conditioner(self) -> LRConditioner:
        """Build the GPU :class:`LRConditioner` (upsampler + normalized static LSM)."""
        up = BilinearUpsampler(self._ref.lr_lat, self._ref.lr_lon,
                               self._ref.hr_lat, self._ref.hr_lon)
        lsm = None
        if self.include_lsm:
            with xr.open_zarr(self.stores[0]) as z:
                mask = np.asarray(z["land_sea_mask"].values, dtype=np.float32)
            lsm = ((mask - self._lsm_mean) / self._lsm_std)[None]  # (1, H, W)
        return LRConditioner(up, lsm)

    def input_channels(self) -> List[ChannelMetadata]:
        ch = [ChannelMetadata(name=c) for c in self.lr_channel_names]
        if self.include_lsm:
            ch.append(ChannelMetadata(name="lsm", auxiliary=True))
        return ch

    def output_channels(self) -> List[ChannelMetadata]:
        return [ChannelMetadata(name=c) for c in self.hr_channel_names]

    def image_shape(self) -> Tuple[int, int]:
        return self._img_shape

    def longitude(self) -> np.ndarray:
        return np.asarray(self._ref.hr_lon)

    def latitude(self) -> np.ndarray:
        return np.asarray(self._ref.hr_lat)

    def time(self) -> List:
        # Matches helpers.train_helpers._convert_datetime_to_cftime but inlined to avoid
        # pulling in omegaconf (a training-time dep) so time() works on the dev box too.
        import cftime

        out = []
        for s in self.stores:
            with xr.open_zarr(s) as z:
                stamps = pd.to_datetime(np.asarray(z["time"].values))
            out.extend(
                cftime.DatetimeGregorian(t.year, t.month, t.day, t.hour, t.minute, t.second)
                for t in stamps
            )
        return out

    # ------------------------------------------------------------------ (de)normalization
    def normalize_input(self, x: np.ndarray) -> np.ndarray:
        return (x - self._in_mean_np) / self._in_std_np

    def denormalize_input(self, x: np.ndarray) -> np.ndarray:
        return x * self._in_std_np + self._in_mean_np

    def normalize_output(self, x: np.ndarray) -> np.ndarray:
        return (x - self._out_mean_np) / self._out_std_np

    def denormalize_output(self, x: np.ndarray) -> np.ndarray:
        return x * self._out_std_np + self._out_mean_np

    def info(self) -> dict:
        return {
            "stores": self.stores,
            "lr_channels": self.lr_channel_names,
            "hr_channels": self.hr_channel_names,
            "include_lsm": self.include_lsm,
            "image_shape": self._img_shape,
        }
