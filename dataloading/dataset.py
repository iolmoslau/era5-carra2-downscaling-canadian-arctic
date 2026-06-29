"""torch Dataset over a built sample zarr store (one store == one split).

Each split (train / val / test) is its own store, built from a distinct, contiguous time
range -- range-based splits avoid the leakage that random splitting would cause given the
strong autocorrelation of 3-hourly samples. The only things shared across splits are the
normalization stats (computed on TRAIN only; see :mod:`dataloading.stats`) and the grid
geometry (identical when the stores are built with the same center/patch/levels).

``__getitem__`` returns the COARSE LR (small); upsampling to the patch grid happens on GPU
via :class:`dataloading.upsample.BilinearUpsampler` in the forward pass.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import xarray as xr
from torch.utils.data import ConcatDataset, Dataset

EPS = 1e-6  # guard against zero-variance channels when normalizing


class PatchDataset(Dataset):
    """LR/HR sample pairs from one store.

    Parameters
    ----------
    store : path to a zarr store built by ``dataset_builder.build_dataset``.
    stats : optional normalization dict (from ``stats.compute_norm_stats`` on the TRAIN store).
        If given, ``lr``/``hr`` are standardized per channel. The same dict must be used for
        every split.
    time_slice : optional ``slice`` to restrict to a contiguous time window (e.g. to carve a
        validation window out of a store rather than building a separate one).
    return_mask : include the static land-sea mask in each sample.
    """

    def __init__(self, store, stats: dict | None = None, time_slice: slice | None = None,
                 return_mask: bool = True):
        self.store = os.fspath(store)
        self._z: xr.Dataset | None = None  # opened lazily per worker

        meta = xr.open_zarr(self.store)
        n = meta.sizes["time"]
        self.indices = list(range(*time_slice.indices(n))) if time_slice else list(range(n))
        self.lr_channels = list(meta.attrs["lr_channels"])  # names live in attrs (not coords)
        self.hr_channels = list(meta.attrs["hr_channels"])
        # geometry kept for building the upsampler / cross-split consistency checks
        self.lr_lat, self.lr_lon = meta["lat"].values, meta["lon"].values
        self.hr_lat, self.hr_lon = meta["hr_lat"].values, meta["hr_lon"].values
        self.mask = (meta["land_sea_mask"].values.astype("float32") if return_mask else None)
        meta.close()

        self._norm = self._prep_stats(stats) if stats else None

    def _prep_stats(self, stats: dict) -> dict:
        if stats.get("hr_channels", self.hr_channels) != self.hr_channels or \
           stats.get("lr_channels", self.lr_channels) != self.lr_channels:
            raise ValueError("stats channel order does not match the store")
        out = {}
        for key, n_ch in (("hr", len(self.hr_channels)), ("lr", len(self.lr_channels))):
            mean = torch.tensor(stats[f"{key}_mean"], dtype=torch.float32).reshape(n_ch, 1, 1)
            std = torch.tensor(stats[f"{key}_std"], dtype=torch.float32).reshape(n_ch, 1, 1)
            out[f"{key}_mean"], out[f"{key}_std"] = mean, std.clamp_min(EPS)
        return out

    @property
    def z(self) -> xr.Dataset:
        # Open inside the worker process (lazy) rather than forking an open handle.
        if self._z is None:
            self._z = xr.open_zarr(self.store, chunks=None)
        return self._z

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict:
        t = self.indices[i]
        lr = torch.from_numpy(np.asarray(self.z["lr"].isel(time=t).values, dtype="float32"))
        hr = torch.from_numpy(np.asarray(self.z["hr"].isel(time=t).values, dtype="float32"))
        if self._norm is not None:
            lr = (lr - self._norm["lr_mean"]) / self._norm["lr_std"]
            hr = (hr - self._norm["hr_mean"]) / self._norm["hr_std"]
        sample = {"lr": lr, "hr": hr}
        if self.mask is not None:
            sample["mask"] = torch.from_numpy(self.mask)
        return sample

    def make_upsampler(self):
        """Build a :class:`BilinearUpsampler` matched to this store's grid."""
        from dataloading.upsample import BilinearUpsampler
        return BilinearUpsampler(self.lr_lat, self.lr_lon, self.hr_lat, self.hr_lon)


def assert_same_geometry(a: PatchDataset, b: PatchDataset) -> None:
    """Raise if two splits don't share LR/HR grids + channel order (they must, for shared stats/upsampler)."""
    checks = [("lr_lat", a.lr_lat, b.lr_lat), ("lr_lon", a.lr_lon, b.lr_lon),
              ("hr_lat", a.hr_lat, b.hr_lat), ("hr_lon", a.hr_lon, b.hr_lon)]
    for name, x, y in checks:
        if not np.allclose(x, y):
            raise ValueError(f"geometry mismatch between splits: {name}")
    if a.lr_channels != b.lr_channels or a.hr_channels != b.hr_channels:
        raise ValueError("channel order mismatch between splits")


def concat_split(stores, stats: dict | None = None, **kwargs) -> ConcatDataset:
    """Build one split from several shard stores (e.g. per-year), as a torch ``ConcatDataset``.

    Each shard becomes a :class:`PatchDataset`; all are checked to share grid/channel geometry
    (so the same stats + upsampler apply). Use this when a split spans multiple yearly shards
    produced by the parallel build. The first shard's dataset is reachable as ``cd.datasets[0]``
    for ``.make_upsampler()``.
    """
    parts = [PatchDataset(s, stats=stats, **kwargs) for s in stores]
    for p in parts[1:]:
        assert_same_geometry(parts[0], p)
    return ConcatDataset(parts)
