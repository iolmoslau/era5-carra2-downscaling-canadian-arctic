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
import pandas as pd
import xarray as xr

import data_acquisition.data_utils as du

HR_VARS = ("t2m", "u10", "v10")

# CARRA2 dynamic HR variables to download per day (CDS names) and the 3-hourly analysis times.
CARRA_DYNAMIC = ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind"]
ALL_TIMES = ["00:00", "03:00", "06:00", "09:00", "12:00", "15:00", "18:00", "21:00"]


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


# --------------------------------------------------------------------------------------
# Batch driver: download -> crop -> build -> write -> discard, one day at a time
# --------------------------------------------------------------------------------------


def _carra_request(variables, day: pd.Timestamp, times) -> dict:
    """Build a CARRA2 single-levels request dict for one day."""
    return {
        "level_type": "single_levels", "product_type": "analysis", "data_format": "netcdf",
        "variable": list(variables),
        "year": [f"{day.year:04d}"], "month": [f"{day.month:02d}"], "day": [f"{day.day:02d}"],
        "time": list(times),
        "area": [75, -150, 60, -113],  # ignored by CDS for CARRA2, but harmless
    }


def acquire_static_mask(center, patch_size, work_dir, *, dataset="reanalysis-pan-carra",
                        ref_day="2013-01-21") -> np.ndarray:
    """Download the CARRA2 land-sea mask once (single timestep), crop, return the (y,x) patch."""
    day = pd.Timestamp(ref_day)
    paths = du.download_carra2(dataset, _carra_request(["land_sea_mask"], day, ["00:00"]),
                               work_dir, basename="carra2_lsm_static")
    hr = du.open_carra2(paths)
    lsm = du.crop_carra2_patch(hr, center, patch_size)["lsm"]
    if "time" in lsm.dims:  # single-timestep requests collapse time to a scalar coord
        lsm = lsm.isel(time=0)
    mask = lsm.values.astype("float32")
    hr.close()
    for p in paths:
        os.remove(p)
    return mask


def build_dataset(store, center, patch_size, start, end, *, work_dir,
                  dataset="reanalysis-pan-carra", dynamic=CARRA_DYNAMIC, times=ALL_TIMES,
                  margin_cells=2, keep_downloads=False, attrs=None, mask_ref_day=None):
    """Build a sample zarr over a date range: per day download -> crop -> write -> discard.

    Each day's full-domain CARRA2 file is downloaded, the patch (all 8 timesteps) cropped, the
    ERA5 LR stack built (from ARCO), the day written/appended to the zarr, and the full-domain
    file deleted -- so large data never accumulates locally. Safe to resume: if ``store`` exists
    the new days are appended and the static mask is not re-fetched.

    Parameters
    ----------
    store : output zarr path.
    center : (lat, lon) patch centre.
    patch_size : HR cells per side (e.g. 448).
    start, end : inclusive date range (parseable by pandas); iterated daily.
    work_dir : scratch dir for the transient full-domain downloads.
    keep_downloads : keep the per-day CARRA2 files instead of deleting them.
    attrs : store-level metadata (written at store creation).
    """
    os.makedirs(work_dir, exist_ok=True)
    days = pd.date_range(start, end, freq="D")
    era5 = du.open_era5_lr()

    need_mask = not os.path.exists(store)
    mask = (acquire_static_mask(center, patch_size, work_dir, dataset=dataset,
                                ref_day=mask_ref_day or str(days[0].date()))
            if need_mask else None)

    for day in days:
        paths = du.download_carra2(dataset, _carra_request(dynamic, day, times), work_dir,
                                   basename=f"carra2_{day.strftime('%Y%m%d')}")
        hr = du.open_carra2(paths)
        patch = du.crop_carra2_patch(hr, center, patch_size)
        lr = du.build_era5_lr_stack(patch, era5=era5, margin_cells=margin_cells)
        day_ds = build_day_dataset(patch, lr)
        write_day(store, day_ds, land_sea_mask=(mask if need_mask else None),
                  attrs=(attrs if need_mask else None))
        need_mask = False
        hr.close()
        if not keep_downloads:
            for p in paths:
                os.remove(p)
        print(f"  wrote {day.date()}: +{day_ds.sizes['time']} steps -> {store}")

    return store
