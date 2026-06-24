"""Assemble and write LR/HR training samples to a zarr store.

One "sample" = one timestep over the single fixed 448x448 patch. We store the HR target
(CARRA2) at native resolution and the LR input (ERA5) on its COARSE grid -- the bilinear
upsampling to the patch grid happens at train time via
``data_utils.bilinear_weights`` / ``apply_bilinear`` (so nothing is duplicated on disk).

Store schema (see channel manifest)::

    hr             (time, hr_channel=3,  y=448, x=448)   CARRA2: t2m, u10, v10
    lr             (time, lr_channel=12, lat,   lon)      ERA5 coarse box (12 channels)
    land_sea_mask  (y, x)                                  CARRA2, static (written once)
    coords: time, hr_channel, lr_channel,
            hr_lat/hr_lon (y, x), lat/lon (1-D)

New days are appended along ``time`` so the store grows incrementally (the download ->
crop -> discard batch driver calls :func:`write_day` once per day).
"""

from __future__ import annotations

import os

import numpy as np
import xarray as xr

HR_VARS = ("t2m", "u10", "v10")


def build_day_dataset(patch: xr.Dataset, lr_stack: xr.DataArray,
                      *, hr_vars=HR_VARS) -> xr.Dataset:
    """Assemble one day's (8-timestep) sample Dataset from a CARRA2 patch + ERA5 LR stack.

    Parameters
    ----------
    patch : cropped CARRA2 patch with the HR variables (from ``crop_carra2_patch`` on a
        dataset opened via ``open_carra2``); must carry ``y_dim``/``x_dim`` attrs.
    lr_stack : the ERA5 12-channel coarse stack (from ``build_era5_lr_stack``), dims
        ``(time, channel, latitude, longitude)``.

    Returns
    -------
    xr.Dataset with ``hr`` and ``lr`` (both float32) and all coords -- but NOT the static
    land-sea mask (that is written once by :func:`write_day`).
    """
    y_dim, x_dim = patch.attrs["y_dim"], patch.attrs["x_dim"]

    hr = (xr.concat([patch[v] for v in hr_vars], dim="hr_channel")
          .transpose("time", "hr_channel", y_dim, x_dim)
          .astype("float32"))
    lr = lr_stack.transpose("time", "channel", "latitude", "longitude").astype("float32")

    return xr.Dataset(
        {
            "hr": (("time", "hr_channel", "y", "x"), hr.values),
            "lr": (("time", "lr_channel", "lat", "lon"), lr.values),
        },
        coords={
            "time": patch["time"].values,
            "hr_channel": list(hr_vars),
            "lr_channel": list(lr_stack["channel"].values),
            "hr_lat": (("y", "x"), np.asarray(patch["latitude"].values)),
            "hr_lon": (("y", "x"), np.asarray(patch["longitude"].values)),
            "lat": np.asarray(lr_stack["latitude"].values),
            "lon": np.asarray(lr_stack["longitude"].values),
        },
    )


def write_day(store: str | os.PathLike, day_ds: xr.Dataset,
              *, land_sea_mask: np.ndarray | None = None,
              attrs: dict | None = None) -> None:
    """Write/append one day's sample Dataset to a zarr store.

    On first call (store does not exist) the store is created, the static land-sea mask and
    store-level attrs are written, and per-sample chunking (time=1) is set. On later calls
    the day is appended along ``time``.

    Parameters
    ----------
    store : path to the zarr store.
    day_ds : output of :func:`build_day_dataset`.
    land_sea_mask : (y, x) static HR mask; written only at store creation.
    attrs : optional store-level metadata (center, levels, source, units, ...).
    """
    store = os.fspath(store)
    if not os.path.exists(store):
        init = day_ds
        if land_sea_mask is not None:
            init = init.assign(
                land_sea_mask=(("y", "x"), np.asarray(land_sea_mask, dtype="float32"))
            )
        if attrs:
            init = init.assign_attrs(attrs)
        encoding = {
            "hr": {"chunks": (1,) + day_ds["hr"].shape[1:]},
            "lr": {"chunks": (1,) + day_ds["lr"].shape[1:]},
        }
        init.to_zarr(store, mode="w", encoding=encoding)
    else:
        # Append only the time-varying fields; static vars/coords already in the store.
        # Re-attach the existing group attrs -- an append with an attr-less dataset would
        # otherwise overwrite (clear) the store-level metadata written at creation.
        import zarr

        existing = dict(zarr.open_group(store, mode="r").attrs)
        day_ds[["hr", "lr"]].assign_attrs(existing).to_zarr(store, append_dim="time")
