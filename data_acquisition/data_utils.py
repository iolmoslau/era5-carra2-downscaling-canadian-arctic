"""Data acquisition & preprocessing utilities for ERA5 (LR) -> CARRA2 (HR) super-resolution.

Step 1 scope: *lazy* openers + cropping. No interpolation / sample writing yet.

Grid conventions you must keep in mind
--------------------------------------
ERA5 (ARCO-ERA5, the low-resolution input)
    * Regular lat/lon grid, 0.25 deg, hourly, global, 1940-present.
    * ``latitude``  is DESCENDING (90 -> -90).
    * ``longitude`` is in [0, 360).
    * Opened *lazily* straight from Google Cloud Storage Zarr -- nothing is downloaded
      until you call ``.compute()`` / ``.load()``.

CARRA2 (Copernicus pan-Arctic Regional Reanalysis, the high-resolution target)
    * 2.5 km, 3-hourly, polar-stereographic projection.
    * The grid is CURVILINEAR: ``latitude``/``longitude`` are 2-D fields indexed by
      projected dimensions (conventionally ``y``, ``x``). You crop in (y, x) index space.
    * CDS-only: there is no lazy cloud source. You must DOWNLOAD it (we request NetCDF to
      avoid GRIB tooling) and then open the local files lazily.

Typical Step-1 workflow
-----------------------
    1. download_carra2(...)            # fetch a small CARRA2 batch from CDS -> local .nc
    2. ds_hr = open_carra2(paths)      # lazy open, normalised coords
    3. patch = crop_carra2_patch(ds_hr, center_latlon=(lat, lon), size=128)
                                       # snap to nearest cell, crop NxN HR box == target grid
    4. box   = era5_box_from_patch(patch, margin_cells=2)
    5. ds_lr = open_era5_arco()
       lr    = select_era5_box(ds_lr, box)   # ERA5 subset that brackets the patch
"""

from __future__ import annotations

import os
import zipfile
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import xarray as xr

# --------------------------------------------------------------------------------------
# ERA5 (low-resolution input) -- lazy cloud Zarr
# --------------------------------------------------------------------------------------

ARCO_ERA5_URL = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"


def open_era5_arco(
    variables: str | Sequence[str] | None = None,
    level: int | Sequence[int] | None = None,
    *,
    url: str = ARCO_ERA5_URL,
    chunks: dict | str | None = None,
) -> xr.Dataset:
    """Lazily open the ARCO-ERA5 analysis-ready Zarr store from GCS.

    Nothing is read over the network beyond metadata; the returned Dataset is backed by
    dask and only fetches bytes when you compute a selection.

    Parameters
    ----------
    variables : optional variable name(s) to keep (e.g. ``"2m_temperature"`` or a list).
        ``None`` keeps every variable.
    level : optional pressure level(s) in hPa to select (only relevant for variables
        defined on the ``level`` dimension). ``None`` keeps all levels.
    url : Zarr store URL (override only for testing/alt stores).
    chunks : passed to ``xr.open_zarr``. ``None`` keeps the store's native chunking.

    Returns
    -------
    xr.Dataset with ``latitude`` (descending), ``longitude`` (0..360), ``time`` and,
    where applicable, ``level``.
    """
    ds = xr.open_zarr(url, chunks=chunks, storage_options={"token": "anon"})

    if variables is not None:
        if isinstance(variables, str):
            variables = [variables]
        ds = ds[list(variables)]

    if level is not None and "level" in ds.dims:
        ds = ds.sel(level=level)

    return ds


# --------------------------------------------------------------------------------------
# CARRA2 (high-resolution target) -- CDS download + lazy local open
# --------------------------------------------------------------------------------------


def download_carra2(
    dataset: str,
    request: dict,
    out_dir: str | os.PathLike,
    *,
    basename: str,
    unzip: bool = True,
    skip_existing: bool = True,
) -> list[str]:
    """Download one CARRA2 request from the CDS and return the local NetCDF path(s).

    Thin wrapper around ``cdsapi.Client().retrieve``. ``out_dir`` is parameterised so the
    same call works for small local test batches now and on the FIR cluster later.

    NetCDF requests from the CDS frequently arrive as a ``.zip`` containing one or more
    ``.nc`` files; when ``unzip`` is True we detect that and extract, returning the paths
    to the extracted NetCDF files.

    Parameters
    ----------
    dataset : CDS dataset id, e.g. ``"reanalysis-pan-carra-single-levels"``.
    request : the request dict (variable, levels, year/month/day, time, area, data_format,
        ...). Confirm exact keys from the dataset's "Show API request" form on CDS.
        NOTE: for ``reanalysis-pan-carra`` the lat/lon ``area`` key is SILENTLY IGNORED --
        you always receive the full pan-Arctic domain (~2869x2869, ~133 MB/variable/day).
        Crop to the region of interest locally after download (see crop_carra2_patch).
    out_dir : directory to write into (created if missing).
    basename : file stem for the downloaded target (no extension).
    unzip : extract NetCDF from a returned zip archive.
    skip_existing : if the expected output already exists, skip the network call.

    Returns
    -------
    list[str] of NetCDF file paths written.
    """
    import cdsapi  # imported lazily so the module imports without CDS configured

    out_dir = os.fspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # We don't know up-front whether CDS returns .nc or .zip, so download to a neutral name.
    target = os.path.join(out_dir, f"{basename}.download")
    nc_glob_existing = sorted(
        p for p in _listdir_full(out_dir)
        if os.path.basename(p).startswith(basename) and p.endswith(".nc")
    )
    if skip_existing and nc_glob_existing:
        return nc_glob_existing

    cdsapi.Client().retrieve(dataset, request, target)

    # Classify what we actually got.
    if zipfile.is_zipfile(target) and unzip:
        written: list[str] = []
        with zipfile.ZipFile(target) as zf:
            members = [m for m in zf.namelist() if m.endswith(".nc")]
            for m in members:
                # Prefix each extracted file with `basename` so skip_existing is reliable
                # and files from different periods never collide.
                dest = os.path.join(out_dir, f"{basename}_{os.path.basename(m)}")
                with zf.open(m) as src, open(dest, "wb") as dst:
                    dst.write(src.read())
                written.append(dest)
        os.remove(target)
        return sorted(written)

    # Otherwise assume a single NetCDF; give it a .nc extension.
    final = os.path.join(out_dir, f"{basename}.nc")
    os.replace(target, final)
    return [final]


def batch_download_carra2(
    dataset: str,
    base_request: dict,
    *,
    start: str,
    end: str,
    times: Sequence[str],
    out_dir: str | os.PathLike,
    freq: str = "MS",
    **download_kwargs,
) -> list[str]:
    """Download a date range as small per-period requests (default: one file per month).

    Keeps each transfer small enough for this machine; the identical loop scales on the
    cluster by pointing ``out_dir`` at project storage and widening the range.

    Parameters
    ----------
    dataset, base_request : as in :func:`download_carra2`. ``base_request`` supplies the
        non-temporal keys (variable, levels, area, data_format, product_type, ...);
        year/month/day/time are filled in per period.
    start, end : inclusive date range bounds parseable by ``pandas`` (e.g. "2018-01").
    times : analysis times to request, e.g. ``["00:00", "03:00", ..., "21:00"]``.
    out_dir : output directory.
    freq : pandas offset alias for batching ("MS" = month start -> one file per month).
    download_kwargs : forwarded to :func:`download_carra2`.

    Returns
    -------
    list[str] of all NetCDF paths written across periods.
    """
    periods = pd.date_range(start=start, end=end, freq=freq)
    all_paths: list[str] = []
    for period_start in periods:
        period_end = period_start + pd.tseries.frequencies.to_offset(freq) - pd.Timedelta(days=1)
        days = pd.date_range(period_start, min(period_end, pd.Timestamp(end)))
        req = dict(base_request)
        req["year"] = [f"{period_start.year:04d}"]
        req["month"] = [f"{period_start.month:02d}"]
        req["day"] = sorted({f"{d.day:02d}" for d in days})
        req["time"] = list(times)
        basename = f"carra2_{period_start.year:04d}{period_start.month:02d}"
        all_paths += download_carra2(dataset, req, out_dir, basename=basename, **download_kwargs)
    return all_paths


def open_carra2(
    paths: str | os.PathLike | Sequence[str | os.PathLike],
    *,
    chunks: dict | str | None = "auto",
    **open_kwargs,
) -> xr.Dataset:
    """Lazily open downloaded CARRA2 NetCDF file(s) with normalised coordinate names.

    Ensures the curvilinear ``latitude``/``longitude`` 2-D coordinates are present and the
    projected dims are exposed via the ``y_dim``/``x_dim`` attributes on the Dataset for
    downstream cropping.

    Parameters
    ----------
    paths : a single path or a list of paths (uses ``open_mfdataset`` for >1).
    chunks : dask chunking ("auto" keeps it lazy).
    open_kwargs : forwarded to the xarray opener.

    Returns
    -------
    xr.Dataset (lazy) with ``.attrs['y_dim']`` and ``.attrs['x_dim']`` set.
    """
    if isinstance(paths, (str, os.PathLike)):
        ds = _drop_level_scalars(xr.open_dataset(paths, chunks=chunks, **open_kwargs))
    else:
        # CDS splits a multi-variable NetCDF request by vertical level type, so each file
        # carries a different scalar level coord (heightAboveGround = 2 / 10 / ...). Drop
        # those per-file before combining, otherwise the merge conflicts.
        ds = xr.open_mfdataset(
            list(paths), chunks=chunks, combine="by_coords", compat="override",
            coords="minimal", preprocess=_drop_level_scalars, **open_kwargs,
        )

    ds = _normalise_carra2_coords(ds)
    return ds


def _drop_level_scalars(ds: xr.Dataset) -> xr.Dataset:
    """Drop scalar (0-d) level/step coords that differ between per-variable CARRA2 files."""
    drop = [c for c in ds.coords if ds[c].ndim == 0 and c not in ("time", "valid_time")]
    return ds.drop_vars(drop, errors="ignore")


def _normalise_carra2_coords(ds: xr.Dataset) -> xr.Dataset:
    """Standardise CARRA2 lat/lon coordinate names and record the projected dim names."""
    rename = {}
    if "latitude" not in ds and "lat" in ds:
        rename["lat"] = "latitude"
    if "longitude" not in ds and "lon" in ds:
        rename["lon"] = "longitude"
    if rename:
        ds = ds.rename(rename)

    if "latitude" not in ds.coords or "longitude" not in ds.coords:
        raise ValueError(
            "CARRA2 dataset is missing 'latitude'/'longitude' coordinates; "
            f"found coords {list(ds.coords)}."
        )

    lat = ds["latitude"]
    if lat.ndim != 2:
        raise ValueError(
            f"Expected 2-D curvilinear 'latitude' for CARRA2, got {lat.ndim}-D with "
            f"dims {lat.dims}. Check the downloaded grid."
        )

    # The projected dims are simply the dims of the 2-D latitude field (y, x order).
    y_dim, x_dim = lat.dims
    ds.attrs["y_dim"] = str(y_dim)
    ds.attrs["x_dim"] = str(x_dim)
    return ds


# --------------------------------------------------------------------------------------
# Cropping (CARRA2) and ERA5 box derivation
# --------------------------------------------------------------------------------------


def nearest_grid_index(ds: xr.Dataset, lat: float, lon: float) -> tuple[int, int]:
    """Return the (y, x) index of the CARRA2 cell nearest to a geographic point.

    Uses the haversine great-circle distance over the 2-D lat/lon fields, so it is correct
    near the pole where simple lat/lon differences distort.

    Parameters
    ----------
    ds : a CARRA2 Dataset opened via :func:`open_carra2`.
    lat, lon : target point in degrees. ``lon`` may be given in [-180, 180] or [0, 360).

    Returns
    -------
    (y_index, x_index) into the projected dims.
    """
    glat = np.deg2rad(np.asarray(ds["latitude"].values))
    glon = np.deg2rad(_to_pm180(np.asarray(ds["longitude"].values)))
    plat = np.deg2rad(lat)
    plon = np.deg2rad(_to_pm180(lon))

    dlat = glat - plat
    dlon = glon - plon
    a = np.sin(dlat / 2.0) ** 2 + np.cos(plat) * np.cos(glat) * np.sin(dlon / 2.0) ** 2
    dist = 2.0 * np.arcsin(np.sqrt(a))  # angular distance; argmin is all we need

    j, i = np.unravel_index(int(np.argmin(dist)), dist.shape)
    return int(j), int(i)


def crop_carra2_patch(
    ds: xr.Dataset,
    center_latlon: tuple[float, float],
    size: int | tuple[int, int],
) -> xr.Dataset:
    """Crop a fixed-size HR box around the cell nearest to ``center_latlon``.

    The cropped patch (and its 2-D lat/lon) defines the *target* grid that ERA5 will later
    be interpolated onto.

    Parameters
    ----------
    ds : CARRA2 Dataset from :func:`open_carra2`.
    center_latlon : (lat, lon) of the desired patch centre.
    size : patch size in HR cells; an int for a square ``size x size`` box, or ``(ny, nx)``.

    Returns
    -------
    xr.Dataset (lazy) sliced to the patch, preserving ``y_dim``/``x_dim`` attrs.

    Raises
    ------
    ValueError if the requested box would fall outside the CARRA2 domain.
    """
    y_dim, x_dim = ds.attrs["y_dim"], ds.attrs["x_dim"]
    ny, nx = (size, size) if isinstance(size, int) else size

    j, i = nearest_grid_index(ds, *center_latlon)
    j0, j1 = j - ny // 2, j - ny // 2 + ny
    i0, i1 = i - nx // 2, i - nx // 2 + nx

    ydim_len, xdim_len = ds.sizes[y_dim], ds.sizes[x_dim]
    if j0 < 0 or i0 < 0 or j1 > ydim_len or i1 > xdim_len:
        raise ValueError(
            f"Patch of size ({ny}, {nx}) centred at cell (y={j}, x={i}) falls outside the "
            f"CARRA2 domain (y:0..{ydim_len}, x:0..{xdim_len}). Move the centre inward or "
            f"shrink the patch."
        )

    return ds.isel({y_dim: slice(j0, j1), x_dim: slice(i0, i1)})


@dataclass
class Era5Box:
    """A lon/lat selection box for ERA5, in ERA5's own conventions (lon in [0, 360))."""

    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    crosses_antimeridian: bool  # True if the box wraps across the 0/360 seam


def era5_box_from_patch(patch: xr.Dataset, margin_cells: int = 2, era5_res: float = 0.25) -> Era5Box:
    """Compute the ERA5 lon/lat box that fully brackets a CARRA2 patch (plus a margin).

    The margin guarantees that every patch cell is surrounded by ERA5 cells so a later
    bilinear interpolation has a complete stencil even at the patch edges.

    Parameters
    ----------
    patch : a cropped CARRA2 patch from :func:`crop_carra2_patch`.
    margin_cells : number of extra ERA5 cells to pad on each side.
    era5_res : ERA5 grid spacing in degrees (0.25 for ARCO-ERA5).

    Returns
    -------
    Era5Box with bounds in ERA5 conventions; ``crosses_antimeridian`` flags a 0/360 wrap.
    """
    lat = np.asarray(patch["latitude"].values)
    lon180 = _to_pm180(np.asarray(patch["longitude"].values))

    margin = margin_cells * era5_res
    lat_min = float(np.nanmin(lat)) - margin
    lat_max = float(np.nanmax(lat)) + margin

    # Decide convention/wrap in [-180, 180] first, then convert to [0, 360).
    lon_span = float(np.nanmax(lon180) - np.nanmin(lon180))
    crosses = lon_span > 180.0  # patch straddles the antimeridian
    if crosses:
        # Keep the gap that does NOT contain the data: take min of positives, max of negatives.
        west = float(np.nanmin(lon180[lon180 > 0])) - margin
        east = float(np.nanmax(lon180[lon180 < 0])) + margin
        lon_min, lon_max = _to_360(west), _to_360(east)
    else:
        lon_min = _to_360(float(np.nanmin(lon180)) - margin)
        lon_max = _to_360(float(np.nanmax(lon180)) + margin)

    return Era5Box(lat_min, lat_max, lon_min, lon_max, crosses)


def select_era5_box(ds: xr.Dataset, box: Era5Box) -> xr.Dataset:
    """Lazily select the ERA5 subset described by ``box``.

    Honours ERA5's DESCENDING latitude (slice high->low) and [0, 360) longitude. If the box
    wraps the antimeridian, the two longitude segments are concatenated.

    Parameters
    ----------
    ds : an ERA5 Dataset from :func:`open_era5_arco`.
    box : an :class:`Era5Box` (typically from :func:`era5_box_from_patch`).

    Returns
    -------
    xr.Dataset (lazy) covering the box.
    """
    lat_sel = slice(box.lat_max, box.lat_min)  # descending latitude

    if not box.crosses_antimeridian:
        return ds.sel(latitude=lat_sel, longitude=slice(box.lon_min, box.lon_max))

    left = ds.sel(latitude=lat_sel, longitude=slice(box.lon_min, 360.0))
    right = ds.sel(latitude=lat_sel, longitude=slice(0.0, box.lon_max))
    return xr.concat([left, right], dim="longitude")


# --------------------------------------------------------------------------------------
# ERA5 LR channel stack (the low-resolution model input, stored on its native coarse grid)
# --------------------------------------------------------------------------------------

# The 12 LR channels (see channel manifest). Surface + pressure-levels @ {500, 850} + sea ice.
ERA5_LR_SURFACE = ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind"]
ERA5_LR_PLEVEL = ["temperature", "geopotential", "u_component_of_wind", "v_component_of_wind"]
ERA5_LR_OTHER = ["sea_ice_cover"]
LR_LEVELS = (500, 850)

_LR_SURFACE_SHORT = {"2m_temperature": "t2m", "10m_u_component_of_wind": "u10",
                     "10m_v_component_of_wind": "v10"}
_LR_PLEVEL_SHORT = {"temperature": "t", "geopotential": "z",
                    "u_component_of_wind": "u", "v_component_of_wind": "v"}
_LR_OTHER_SHORT = {"sea_ice_cover": "siconc"}


def lr_channel_names(levels=LR_LEVELS) -> list[str]:
    """Canonical, ordered LR channel names (e.g. t2m, u10, v10, t500, t850, ..., siconc)."""
    sfc = [_LR_SURFACE_SHORT[v] for v in ERA5_LR_SURFACE]
    pl = [f"{_LR_PLEVEL_SHORT[v]}{lev}" for v in ERA5_LR_PLEVEL for lev in levels]
    other = [_LR_OTHER_SHORT[v] for v in ERA5_LR_OTHER]
    return sfc + pl + other


def open_era5_lr(chunks=None) -> xr.Dataset:
    """Lazily open ARCO-ERA5 with only the variables needed for the LR stack."""
    return open_era5_arco(variables=ERA5_LR_SURFACE + ERA5_LR_PLEVEL + ERA5_LR_OTHER,
                          chunks=chunks)


def build_era5_lr_stack(patch: xr.Dataset, era5: xr.Dataset | None = None, *,
                        levels=LR_LEVELS, margin_cells: int = 2) -> xr.DataArray:
    """Build the coarse-grid ERA5 LR channel stack covering a CARRA2 patch.

    Returns the ERA5 box (NOT interpolated -- stored coarse, upsampled at train time) with
    all channels stacked on a ``channel`` axis: dims ``(time, channel, latitude, longitude)``.
    Times follow the patch's (3-hourly CARRA2) timestamps.

    Parameters
    ----------
    patch : a cropped CARRA2 patch (defines the box footprint and the timestamps).
    era5 : an already-open ERA5 dataset (from :func:`open_era5_lr`); opened if None.
    levels : pressure levels (hPa) for the pressure-level channels.
    margin_cells : ERA5-cell halo around the patch (matches :func:`era5_box_from_patch`,
        so the stored box has a full bilinear stencil for every patch cell at train time).
    """
    if era5 is None:
        era5 = open_era5_lr()

    box = era5_box_from_patch(patch, margin_cells=margin_cells)
    sub = select_era5_box(era5, box).sel(time=patch["time"].values, method="nearest")

    chans, names = [], []
    for v in ERA5_LR_SURFACE:
        chans.append(sub[v]); names.append(_LR_SURFACE_SHORT[v])
    for v in ERA5_LR_PLEVEL:
        for lev in levels:
            chans.append(sub[v].sel(level=lev)); names.append(f"{_LR_PLEVEL_SHORT[v]}{lev}")
    for v in ERA5_LR_OTHER:
        da_v = sub[v]
        if v == "sea_ice_cover":
            da_v = da_v.fillna(0.0)  # ERA5 sea ice is NaN over land -> no ice on land
        chans.append(da_v); names.append(_LR_OTHER_SHORT[v])

    # Drop the per-channel scalar 'level' coord so the channels concat cleanly.
    chans = [c.drop_vars("level", errors="ignore") for c in chans]
    da = xr.concat(chans, dim="channel").assign_coords(channel=names)
    da = da.transpose("time", "channel", "latitude", "longitude")
    da.name = "lr"
    return da


# --------------------------------------------------------------------------------------
# Interpolation (ERA5 regular grid -> CARRA2 curvilinear patch grid)
# --------------------------------------------------------------------------------------


def bilinear_weights(
    src_lat: np.ndarray,
    src_lon: np.ndarray,
    tgt_lat: np.ndarray,
    tgt_lon: np.ndarray,
):
    """Precompute bilinear gather indices + weights from a regular grid to target points.

    Geometry only -- depends on the grids, not on any field values -- so for a fixed patch
    this is computed once and reused for every timestamp. It is also trivially
    re-implementable in PyTorch for train-time upsampling, giving an identical result to the
    offline path (no train/offline skew).

    Parameters
    ----------
    src_lat, src_lon : 1-D, STRICTLY INCREASING source axes (degrees). ERA5 latitude is
        descending as delivered, so flip it (and the data) before calling.
    tgt_lat, tgt_lon : target coordinates (any shape, ravelled together) in the SAME
        longitude convention as ``src_lon``.

    Returns
    -------
    idx : (npts, 4) int flat indices into the ravelled ``(nlat, nlon)`` source grid -- the
        four cell corners (lower-lat/lower-lon, lower-lat/upper-lon, upper-lat/lower-lon,
        upper-lat/upper-lon) for each target point.
    w : (npts, 4) float bilinear weights (rows sum to 1). Targets outside the source grid
        get NaN weights, so :func:`apply_bilinear` returns NaN there -- matching the former
        ``RegularGridInterpolator(..., bounds_error=False, fill_value=nan)`` behaviour.
    """
    src_lat = np.asarray(src_lat)
    src_lon = np.asarray(src_lon)
    for nm, a in (("src_lat", src_lat), ("src_lon", src_lon)):
        if not np.all(np.diff(a) > 0):
            raise ValueError(f"{nm} must be strictly increasing")
    tlat = np.asarray(tgt_lat).ravel()
    tlon = np.asarray(tgt_lon).ravel()
    nlat, nlon = src_lat.size, src_lon.size

    # Locate the containing cell; clip so the 4-corner stencil is always in-range.
    i1 = np.clip(np.searchsorted(src_lat, tlat), 1, nlat - 1)
    j1 = np.clip(np.searchsorted(src_lon, tlon), 1, nlon - 1)
    i0, j0 = i1 - 1, j1 - 1
    ty = (tlat - src_lat[i0]) / (src_lat[i1] - src_lat[i0])
    tx = (tlon - src_lon[j0]) / (src_lon[j1] - src_lon[j0])

    idx = np.stack([i0 * nlon + j0, i0 * nlon + j1,
                    i1 * nlon + j0, i1 * nlon + j1], axis=-1)
    w = np.stack([(1 - ty) * (1 - tx), (1 - ty) * tx,
                  ty * (1 - tx), ty * tx], axis=-1)

    oob = ((tlat < src_lat[0]) | (tlat > src_lat[-1]) |
           (tlon < src_lon[0]) | (tlon > src_lon[-1]))
    w[oob] = np.nan
    return idx, w


def apply_bilinear(values: np.ndarray, idx: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Apply precomputed bilinear weights (from :func:`bilinear_weights`) to field values.

    Parameters
    ----------
    values : array with the source grid as its FIRST two axes -- ``(nlat, nlon, *extra)``
        (``extra`` may be empty, or e.g. a trailing time axis).
    idx, w : the gather indices and weights.

    Returns
    -------
    array of shape ``(npts, *extra)`` -- interpolated values at the target points.
    """
    values = np.asarray(values)
    nlat, nlon = values.shape[:2]
    extra = values.shape[2:]
    flat = values.reshape(nlat * nlon, -1)            # (ncells, E)
    out = np.einsum("pk,pke->pe", w, flat[idx])       # (npts, E); NaN weights -> NaN out
    return out.reshape((-1,) + extra)


def interpolate_era5_to_patch(era5_box: xr.Dataset, patch: xr.Dataset) -> xr.Dataset:
    """Bilinearly interpolate an ERA5 box onto a CARRA2 patch's curvilinear grid.

    Thin wrapper over :func:`bilinear_weights` (geometry, computed once) + :func:`apply_bilinear`
    (per variable). The ERA5 source is a *regular* lat/lon grid, so bilinear interpolation at
    the patch's 2-D (lat, lon) cell centres does the regridding -- no ``xesmf`` needed.

    Parameters
    ----------
    era5_box : a lazy ERA5 subset from :func:`select_era5_box` (must bracket the patch;
        use a margin via :func:`era5_box_from_patch` so edges have a full stencil).
    patch : a cropped CARRA2 patch from :func:`crop_carra2_patch`.

    Returns
    -------
    xr.Dataset on the patch grid (dims ``time, y, x``) carrying the patch's 2-D
    ``latitude``/``longitude`` coords. Variable NAMES are ERA5's (e.g. '2m_temperature').
    """
    if getattr(era5_box, "attrs", {}).get("crosses_antimeridian"):
        raise NotImplementedError("antimeridian-crossing boxes are not handled yet")

    y_dim, x_dim = patch.attrs["y_dim"], patch.attrs["x_dim"]

    # Target points: the patch's curvilinear cell centres, in ERA5's [0, 360) longitude.
    tgt_lat = np.asarray(patch["latitude"].values)
    tgt_lon = _to_360(np.asarray(patch["longitude"].values))
    ny, nx = tgt_lat.shape

    # Pull ERA5 at the patch timestamps (hourly ERA5 covers the 3-hourly CARRA2 times).
    times = patch["time"].values
    src = era5_box.sel(time=times, method="nearest").load()

    # bilinear_weights needs strictly increasing axes; ERA5 latitude is descending.
    src_lat = np.asarray(src["latitude"].values)
    src_lon = np.asarray(src["longitude"].values)
    flip_lat = src_lat[0] > src_lat[-1]
    if flip_lat:
        src_lat = src_lat[::-1]

    idx, w = bilinear_weights(src_lat, src_lon, tgt_lat, tgt_lon)

    out_vars = {}
    for name, da in src.data_vars.items():
        arr = da.transpose("latitude", "longitude", "time").values  # (nlat, nlon, nt)
        if flip_lat:
            arr = arr[::-1, :, :]
        res = apply_bilinear(arr, idx, w).reshape(ny, nx, len(times)).transpose(2, 0, 1)
        out_vars[name] = (("time", y_dim, x_dim), res)

    ds = xr.Dataset(
        out_vars,
        coords={
            "time": ("time", times),
            "latitude": ((y_dim, x_dim), np.asarray(patch["latitude"].values)),
            "longitude": ((y_dim, x_dim), np.asarray(patch["longitude"].values)),
        },
    )
    ds.attrs["y_dim"], ds.attrs["x_dim"] = y_dim, x_dim
    return ds


# --------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------


def _to_pm180(lon):
    """Convert longitude(s) to the [-180, 180) convention."""
    return (np.asarray(lon) + 180.0) % 360.0 - 180.0


def _to_360(lon):
    """Convert longitude(s) to the [0, 360) convention."""
    return np.asarray(lon) % 360.0


def _listdir_full(d: str) -> Iterable[str]:
    return (os.path.join(d, name) for name in os.listdir(d)) if os.path.isdir(d) else ()
