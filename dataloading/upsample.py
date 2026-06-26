"""Train-time bilinear upsampler: coarse ERA5 LR grid -> the HR patch grid, on GPU.

Reuses the offline geometry (``data_utils.bilinear_weights``) so the result is bit-identical
to ``interpolate_era5_to_patch``. The weights are precomputed once and registered as buffers,
so the whole op is an indexed gather + weighted sum -- pure tensor algebra that runs on CUDA
with no CPU round-trip. Feed it the COARSE LR (kept small on disk / cheap to transfer) and
upsample as the first step of the forward pass.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

import data_acquisition.data_utils as du


class BilinearUpsampler(nn.Module):
    """Bilinearly resample a coarse regular ERA5 grid onto the curvilinear HR patch grid.

    Parameters
    ----------
    lr_lat, lr_lon : 1-D source coords as stored (``lat`` descending, ``lon`` in [0, 360)).
    hr_lat, hr_lon : 2-D target patch coords (``hr_lat``/``hr_lon`` from the store).
    """

    def __init__(self, lr_lat, lr_lon, hr_lat, hr_lon):
        super().__init__()
        src_lat = np.asarray(lr_lat)
        self.flip_lat = bool(src_lat[0] > src_lat[-1])  # bilinear_weights needs ascending
        if self.flip_lat:
            src_lat = src_lat[::-1]
        tgt_lat = np.asarray(hr_lat)
        tgt_lon = du._to_360(np.asarray(hr_lon))

        idx, w = du.bilinear_weights(src_lat, np.asarray(lr_lon), tgt_lat, tgt_lon)
        if np.isnan(w).any():
            raise ValueError("upsampler has out-of-bounds target points (NaN weights); "
                             "the stored LR box does not fully bracket the patch")

        self.out_hw = tuple(tgt_lat.shape)
        self.register_buffer("idx", torch.as_tensor(idx, dtype=torch.long))
        self.register_buffer("w", torch.as_tensor(w, dtype=torch.float32))

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        """``lr`` (..., C, H_src, W_src) -> (..., C, H_patch, W_patch)."""
        if self.flip_lat:
            lr = torch.flip(lr, dims=[-2])
        *lead, c, hs, ws = lr.shape
        flat = lr.reshape(*lead, c, hs * ws)
        gathered = flat[..., self.idx]              # (..., C, Npts, 4)
        out = (gathered * self.w).sum(-1)           # (..., C, Npts)
        return out.reshape(*lead, c, *self.out_hw)


def build_upsampler_from_store(store) -> BilinearUpsampler:
    """Construct a :class:`BilinearUpsampler` from a built sample store's coordinates."""
    import xarray as xr

    z = xr.open_zarr(store)
    return BilinearUpsampler(z["lat"].values, z["lon"].values,
                             z["hr_lat"].values, z["hr_lon"].values)
