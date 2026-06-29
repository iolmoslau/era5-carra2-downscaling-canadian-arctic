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

Chunks of samples are appended along ``time`` so the store grows incrementally. A "chunk"
is any span from one day up to one month (the download granularity); the batch driver calls
:func:`write_chunk` once per chunk.
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


def build_chunk_dataset(patch: xr.Dataset, lr_stack: xr.DataArray,
                        *, hr_vars=HR_VARS) -> xr.Dataset:
    """Assemble a chunk of samples (any number of timesteps) into the store schema.

    The timestamps come from whatever is in ``patch``/``lr_stack`` -- this is granularity
    agnostic, so the caller decides the chunk span (a day, a week, a whole month).

    Parameters
    ----------
    patch : cropped CARRA2 patch with the HR variables (from ``crop_carra2_patch`` on a
        dataset opened via ``open_carra2``); must carry ``y_dim``/``x_dim`` attrs.
    lr_stack : the ERA5 12-channel coarse stack (from ``build_era5_lr_stack*``), dims
        ``(time, channel, latitude, longitude)``, on the same timestamps as ``patch``.

    Returns
    -------
    xr.Dataset with ``hr`` and ``lr`` (both float32) and all coords -- but NOT the static
    land-sea mask (that is written once by :func:`write_chunk`).
    """
    y_dim, x_dim = patch.attrs["y_dim"], patch.attrs["x_dim"]

    hr = (xr.concat([patch[v] for v in hr_vars], dim="hr_channel")
          .transpose("time", "hr_channel", y_dim, x_dim)
          .astype("float32"))
    lr = lr_stack.transpose("time", "channel", "latitude", "longitude").astype("float32")

    ds = xr.Dataset(
        {
            "hr": (("time", "hr_channel", "y", "x"), hr.values),
            "lr": (("time", "lr_channel", "lat", "lon"), lr.values),
        },
        coords={
            "time": patch["time"].values,
            "hr_lat": (("y", "x"), np.asarray(patch["latitude"].values)),
            "hr_lon": (("y", "x"), np.asarray(patch["longitude"].values)),
            "lat": np.asarray(lr_stack["latitude"].values),
            "lon": np.asarray(lr_stack["longitude"].values),
        },
    )
    # Channel names live in attrs, NOT as string coord arrays: fixed-length unicode arrays
    # have no stable Zarr v3 spec (UnstableSpecificationWarning) and could become unreadable.
    ds.attrs["hr_channels"] = list(hr_vars)
    ds.attrs["lr_channels"] = [str(c) for c in lr_stack["channel"].values]
    return ds


def write_chunk(store: str | os.PathLike, chunk_ds: xr.Dataset,
                *, land_sea_mask: np.ndarray | None = None,
                attrs: dict | None = None) -> None:
    """Write/append one chunk's sample Dataset to a zarr store.

    On first call (store does not exist) the store is created, the static land-sea mask and
    store-level attrs are written, and per-sample chunking (time=1) is set. On later calls
    the chunk is appended along ``time``.

    Parameters
    ----------
    store : path to the zarr store.
    chunk_ds : output of :func:`build_chunk_dataset`.
    land_sea_mask : (y, x) static HR mask; written only at store creation.
    attrs : optional store-level metadata (center, levels, source, units, ...).
    """
    store = os.fspath(store)
    if not os.path.exists(store):
        init = chunk_ds
        if land_sea_mask is not None:
            init = init.assign(
                land_sea_mask=(("y", "x"), np.asarray(land_sea_mask, dtype="float32"))
            )
        if attrs:
            init = init.assign_attrs(attrs)
        encoding = {
            "hr": {"chunks": (1,) + chunk_ds["hr"].shape[1:]},
            "lr": {"chunks": (1,) + chunk_ds["lr"].shape[1:]},
        }
        init.to_zarr(store, mode="w", encoding=encoding)
    else:
        # Append only the time-varying fields; static vars/coords already in the store.
        # Re-attach the existing group attrs -- an append with an attr-less dataset would
        # otherwise overwrite (clear) the store-level metadata written at creation.
        import zarr

        existing = dict(zarr.open_group(store, mode="r").attrs)
        chunk_ds[["hr", "lr"]].assign_attrs(existing).to_zarr(store, append_dim="time")


# --------------------------------------------------------------------------------------
# Batch driver: per-month download -> crop each day -> write -> discard
# --------------------------------------------------------------------------------------


def _carra_request(variables, year, month, days, times) -> dict:
    """CARRA2 single-levels request for the given day-numbers within one month."""
    return {
        "level_type": "single_levels", "product_type": "analysis", "data_format": "netcdf",
        "variable": list(variables),
        "year": [f"{year:04d}"], "month": [f"{month:02d}"],
        "day": [f"{int(d):02d}" for d in days], "time": list(times),
        "area": [75, -150, 60, -113],  # ignored by CDS for CARRA2, but harmless
    }


def _era5_area_from_patch(patch, margin_cells):
    """Fixed ERA5 [N, W, S, E] area (deg, -180..180) bracketing the patch, rounded outward."""
    box = du.era5_box_from_patch(patch, margin_cells=margin_cells)
    w, e = float(du._to_pm180(box.lon_min)), float(du._to_pm180(box.lon_max))
    return [int(np.ceil(box.lat_max)), int(np.floor(w)),
            int(np.floor(box.lat_min)), int(np.ceil(e))]


def _acquire_mask_and_area(center, patch_size, work_dir, *, dataset, ref_day, margin_cells):
    """One-time CARRA2 land-sea-mask download; return ((y,x) mask, fixed ERA5 [N,W,S,E] area)."""
    paths = du.download_carra2(
        dataset, _carra_request(["land_sea_mask"], ref_day.year, ref_day.month,
                                [ref_day.day], ["00:00"]),
        work_dir, basename="carra2_lsm_static")
    hr = du.open_carra2(paths)
    patch = du.crop_carra2_patch(hr, center, patch_size)
    lsm = patch["lsm"]
    if "time" in lsm.dims:  # single-timestep requests collapse time to a scalar coord
        lsm = lsm.isel(time=0)
    mask = lsm.values.astype("float32")
    area = _era5_area_from_patch(patch, margin_cells)
    hr.close()
    for p in paths:
        os.remove(p)
    return mask, area


def _group_by_month(days):
    groups: dict = {}
    for d in days:
        groups.setdefault((d.year, d.month), []).append(d)
    return sorted(groups.items())


def build_dataset(store, center, patch_size, start, end, *, work_dir,
                  carra_dataset="reanalysis-pan-carra", dynamic=CARRA_DYNAMIC, times=ALL_TIMES,
                  levels=du.LR_LEVELS, margin_cells=2, chunk_days=1,
                  keep_downloads=False, attrs=None):
    """Build the sample zarr over [start, end], batching downloads BY MONTH.

    Per month: one CARRA2 request (full domain, dynamic HR vars) + one ERA5 CDS request
    (single+pressure levels, area-subset); then assemble/write the month's days in chunks of
    ``chunk_days``; then delete the month's files. Batching downloads by month amortises the
    large per-request CDS latency (previously paid per-day). ERA5 comes from CDS area-subset
    (tiny) rather than ARCO. Resumable: appends if the store already exists. The static
    land-sea mask and fixed ERA5 area are obtained once.

    Parameters
    ----------
    store : output zarr path.
    center : (lat, lon) patch centre.
    patch_size : HR cells per side (e.g. 448).
    start, end : inclusive date range (parseable by pandas); grouped by calendar month.
    work_dir : scratch dir for the transient downloads (deleted per month unless keep_downloads).
    chunk_days : days assembled+written per :func:`write_chunk` call (1 = day .. >=31 = whole
        month). Downloads stay monthly; this only sets the write granularity / memory per write.
    attrs : store-level metadata (written at store creation).
    """
    os.makedirs(work_dir, exist_ok=True)
    days = pd.date_range(start, end, freq="D")
    need_mask = not os.path.exists(store)

    # Resumability: skip timestamps already in the store, so a timed-out/requeued job appends
    # only the missing months and never duplicates samples.
    existing = set()
    if os.path.exists(store):
        existing = {pd.Timestamp(t) for t in xr.open_zarr(store)["time"].values}

    mask, area = _acquire_mask_and_area(center, patch_size, work_dir, dataset=carra_dataset,
                                        ref_day=days[0], margin_cells=margin_cells)
    print(f"static mask acquired; fixed ERA5 area [N,W,S,E] = {area}")

    for (yr, mo), grp in _group_by_month(days):
        expected = [pd.Timestamp(f"{d.strftime('%Y-%m-%d')} {hh}") for d in grp for hh in times]
        if existing.issuperset(expected):
            print(f"  skip {yr}-{mo:02d}: already in store")
            continue

        daynums = [d.day for d in grp]
        cpaths = du.download_carra2(
            carra_dataset, _carra_request(dynamic, yr, mo, daynums, times),
            work_dir, basename=f"carra2_{yr}{mo:02d}")
        sl, pl = du.download_era5_cds(work_dir, year=yr, month=mo, days=daynums, times=times,
                                      area=area, levels=levels)

        hr = du.open_carra2(cpaths)
        patch_month = du.crop_carra2_patch(hr, center, patch_size)
        era5 = du.open_era5_cds(sl, pl)

        # Within the month, assemble + write in chunks of `chunk_days` (skipping any already there).
        for i in range(0, len(grp), chunk_days):
            chunk_dates = {d.date() for d in grp[i:i + chunk_days]}
            chunk_times = [t for t in patch_month["time"].values
                           if pd.Timestamp(t).date() in chunk_dates
                           and pd.Timestamp(t) not in existing]
            if not chunk_times:
                continue
            patch_chunk = patch_month.sel(time=chunk_times)
            lr = du.build_era5_lr_stack_cds(era5, patch_chunk["time"].values, levels=levels)
            chunk_ds = build_chunk_dataset(patch_chunk, lr)
            write_chunk(store, chunk_ds, land_sea_mask=(mask if need_mask else None),
                        attrs=(attrs if need_mask else None))
            need_mask = False
            existing.update(pd.Timestamp(t) for t in chunk_times)
            print(f"  wrote {min(chunk_dates)}..{max(chunk_dates)}: "
                  f"+{chunk_ds.sizes['time']} steps -> {store}")

        hr.close()
        era5.close()
        if not keep_downloads:
            for p in (*cpaths, *sl, *pl):
                os.remove(p)

    return store
