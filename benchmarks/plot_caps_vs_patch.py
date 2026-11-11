#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Draw the CAPS domain and our 448x448 patch together, for one valid time.

WHY: ``check_caps_coverage.py`` says 74.5% of the patch sits outside CAPS. That is a number, and
a number cannot distinguish "the supplied domain is centred somewhere else" from "I parsed the
coordinates wrongly". A picture can, and if the domain really is wrong it is worth knowing before
engineering around it -- re-requesting the extraction over the right boundaries is cheaper than
restricting every benchmark to a sliver.

Plotted in CAPS's OWN projection (north polar stereographic, vertical longitude 244 deg, which is
what ``orientationOfTheGridInDegrees`` reports), so the CAPS domain renders as a clean rectangle
and the patch as the rotated quadrilateral it is. Any mismatch is then obvious by eye.

READING GRIB WITHOUT cfgrib: corrdiff-env has no cffi, so cfgrib/eccodes/pygrib cannot import.
This shells out to eccodes' ``grib_get_data``, which emits "lat lon value" per point for any grid
and needs nothing from Python. Slower than a real reader and entirely adequate for one field.

    module load python/3.11 mpi4py/4.1.0 eccodes proj
    source ~/corrdiff-env/bin/activate

    python benchmarks/plot_caps_vs_patch.py \\
        --caps $SCRATCH/caps_inspect/caps/2021071512_006.grib2 \\
        --data-dir $PROJECT/data/derot --out figures/caps_vs_patch.png

The valid time is derived from the filename (``<init YYYYMMDDHH>_<lead HHH>.grib2`` -> init +
lead) and that same timestamp is pulled from our shard, so both panels show the same moment.
"""
from __future__ import annotations

import os

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
# Alliance's pyproj can leave PROJ_DATA unset -> cartopy CRS init raises DataDirError.
# Must run before cartopy loads pyproj. Mirrors visualization/plot_era5_carra2.py.
if not (os.environ.get("PROJ_DATA") or os.environ.get("PROJ_LIB")):
    try:
        import pyproj

        _cand = os.path.join(os.path.dirname(pyproj.__file__), "proj_dir", "share", "proj")
        if os.path.exists(os.path.join(_cand, "proj.db")):
            os.environ["PROJ_DATA"] = _cand
    except Exception:
        pass

import argparse  # noqa: E402
import io  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# CAPS grid, from grib_ls: polar_stereographic, 1465 x 825, Dx=Dy=2500 m, LaD=60,
# orientationOfTheGridInDegrees=244 (the meridian parallel to the grid's y axis).
CAPS_NI, CAPS_NJ = 1465, 825
CAPS_LON0, CAPS_LAT_TS = 244.0, 60.0
_NAME_RE = re.compile(r"(\d{10})_(\d{3})\.grib2$")


def valid_time_from_name(path: str) -> pd.Timestamp:
    """``2021071512_006.grib2`` -> 2021-07-15 18:00. Init + lead, both in the filename."""
    m = _NAME_RE.search(str(path))
    if not m:
        raise ValueError(f"cannot read an init/lead from {path!r} "
                         "(expected <YYYYMMDDHH>_<LLL>.grib2)")
    init, lead = m.groups()
    return pd.Timestamp(init[:4] + "-" + init[4:6] + "-" + init[6:8] + "T" + init[8:10]) \
        + pd.Timedelta(hours=int(lead))


def read_caps_field(path: str, short_name: str = "2t") -> tuple[np.ndarray, ...]:
    """lat/lon/value on the CAPS grid, via eccodes' CLI. Returns (lat2d, lon2d, val2d)."""
    out = subprocess.run(["grib_get_data", "-w", f"shortName={short_name}", str(path)],
                         capture_output=True, text=True)
    if out.returncode != 0 or not out.stdout.strip():
        raise RuntimeError(f"grib_get_data failed for {path}:\n{out.stderr[:800]}")
    df = pd.read_csv(io.StringIO(out.stdout), sep=r"\s+", header=0,
                     names=["lat", "lon", "value"])
    df = df[pd.to_numeric(df["lat"], errors="coerce").notna()]
    lat = df["lat"].to_numpy(float)
    lon = np.mod(df["lon"].to_numpy(float), 360.0)
    val = pd.to_numeric(df["value"], errors="coerce").to_numpy(float)
    n = lat.size
    if n != CAPS_NI * CAPS_NJ:
        # Do not silently reshape a grid that is not the one we documented.
        raise ValueError(f"{path} holds {n} points, expected {CAPS_NJ}x{CAPS_NI} = "
                         f"{CAPS_NJ * CAPS_NI}")
    shape = (CAPS_NJ, CAPS_NI)                     # i fastest (iScansNegatively=0)
    return lat.reshape(shape), lon.reshape(shape), val.reshape(shape)


def read_patch_field(data_dir: str, when: pd.Timestamp, var: str = "t2m"):
    """Our HR field at `when`, with its 2-D lat/lon. Falls back to the nearest stored time."""
    from dataloading.dataset import open_store, shard_path, store_exists

    store = shard_path(data_dir, when.year)
    if not store_exists(store):
        raise FileNotFoundError(f"no shard for {when.year} under {data_dir}")
    z = xr.open_zarr(open_store(store))
    names = list(z.attrs["hr_channels"])
    if var not in names:
        raise ValueError(f"{var} not in {names}")
    times = pd.DatetimeIndex(z["time"].values)
    if when in times:
        it = int(times.get_loc(when))
    else:
        it = int(np.abs(times - when).argmin())
        print(f"WARNING: {when} not in the shard; using nearest {times[it]}", file=sys.stderr)
    field = np.asarray(z["hr"].isel(time=it, hr_channel=names.index(var)).values, float)
    lat = np.asarray(z["hr_lat"].values, float)
    lon = np.mod(np.asarray(z["hr_lon"].values, float), 360.0)
    z.close()
    return lat, lon, field, times[it]


def outline(lat2d: np.ndarray, lon2d: np.ndarray):
    """The four edges of a 2-D grid, walked as one closed ring (for drawing a footprint)."""
    top, right = (lat2d[0, :], lon2d[0, :]), (lat2d[:, -1], lon2d[:, -1])
    bot, left = (lat2d[-1, ::-1], lon2d[-1, ::-1]), (lat2d[::-1, 0], lon2d[::-1, 0])
    la = np.concatenate([top[0], right[0], bot[0], left[0], top[0][:1]])
    lo = np.concatenate([top[1], right[1], bot[1], left[1], top[1][:1]])
    return la, lo


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--caps", required=True, help="one CAPS .grib2 member")
    ap.add_argument("--data-dir", required=True, help="directory holding our shards")
    ap.add_argument("--out", default="figures/caps_vs_patch.png")
    ap.add_argument("--var", default="2t", help="CAPS shortName to plot (default 2t)")
    ap.add_argument("--stride", type=int, default=3,
                    help="subsample CAPS for drawing; 1465x825 quads is slow (default 3)")
    ap.add_argument("--caps-dump", default="",
                    help="skip grib_get_data and read this saved 'lat lon value' text instead")
    args = ap.parse_args()

    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    when = valid_time_from_name(args.caps)
    print(f"CAPS member : {Path(args.caps).name}  -> valid {when}")

    if args.caps_dump:
        df = pd.read_csv(args.caps_dump, sep=r"\s+", header=0,
                         names=["lat", "lon", "value"])
        df = df[pd.to_numeric(df["lat"], errors="coerce").notna()]
        shape = (CAPS_NJ, CAPS_NI)
        c_lat = df["lat"].to_numpy(float).reshape(shape)
        c_lon = np.mod(df["lon"].to_numpy(float), 360.0).reshape(shape)
        c_val = pd.to_numeric(df["value"], errors="coerce").to_numpy(float).reshape(shape)
    else:
        c_lat, c_lon, c_val = read_caps_field(args.caps, args.var)

    p_lat, p_lon, p_val, p_time = read_patch_field(args.data_dir, when)
    print(f"CAPS   lat {c_lat.min():7.2f}..{c_lat.max():7.2f}   "
          f"lon {c_lon.min():7.2f}..{c_lon.max():7.2f}   "
          f"t2m {np.nanmin(c_val):.1f}..{np.nanmax(c_val):.1f} K")
    print(f"patch  lat {p_lat.min():7.2f}..{p_lat.max():7.2f}   "
          f"lon {p_lon.min():7.2f}..{p_lon.max():7.2f}   "
          f"t2m {np.nanmin(p_val):.1f}..{np.nanmax(p_val):.1f} K   at {p_time}")

    s = max(1, args.stride)
    cl, co, cv = c_lat[::s, ::s], c_lon[::s, ::s], c_val[::s, ::s]
    # One colour scale for both panels, or the eye reads a geometry problem as a physics one.
    allv = np.concatenate([cv[np.isfinite(cv)].ravel(), p_val[np.isfinite(p_val)].ravel()])
    vmin, vmax = np.percentile(allv, [1, 99])

    # CAPS's own projection: its domain is then a rectangle and the patch a rotated quad.
    proj = ccrs.NorthPolarStereo(central_longitude=CAPS_LON0 - 360.0, true_scale_latitude=CAPS_LAT_TS)
    pc = ccrs.PlateCarree()
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.4), subplot_kw={"projection": proj})

    p_ring_lat, p_ring_lon = outline(p_lat, p_lon)
    c_ring_lat, c_ring_lon = outline(c_lat, c_lon)

    for ax, zoom in zip(axes, (False, True)):
        m = ax.pcolormesh(co, cl, cv, transform=pc, vmin=vmin, vmax=vmax,
                          cmap="RdYlBu_r", shading="auto", zorder=1)
        ax.pcolormesh(p_lon, p_lat, p_val, transform=pc, vmin=vmin, vmax=vmax,
                      cmap="RdYlBu_r", shading="auto", zorder=3)
        ax.plot(c_ring_lon, c_ring_lat, transform=pc, color="k", lw=1.2, zorder=4,
                label="CAPS domain")
        ax.plot(p_ring_lon, p_ring_lat, transform=pc, color="crimson", lw=2.0, zorder=5,
                label="our 448x448 patch")
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), lw=0.4, zorder=6)
        ax.gridlines(draw_labels=False, lw=0.3, alpha=0.5, zorder=2)
        if zoom:
            pad = 6.0
            ax.set_extent([p_lon.min() - 360 - pad, p_lon.max() - 360 + pad,
                           p_lat.min() - 3, p_lat.max() + 3], crs=pc)
            ax.set_title("zoomed to the patch")
        else:
            ax.set_global()
            ax.set_extent([c_lon.min() - 360 - 4, c_lon.max() - 360 + 4,
                           max(c_lat.min() - 4, 20), min(c_lat.max() + 4, 89)], crs=pc)
            ax.set_title("CAPS domain (filled) + patch outline")
    axes[0].legend(loc="lower left", fontsize=8, framealpha=0.9)

    fig.suptitle(f"CAPS {args.var} vs CARRA2 patch t2m   valid {when}   "
                 f"(CAPS polar stereographic, lon0={CAPS_LON0:.0f}, 2.5 km)", fontsize=11)
    cb = fig.colorbar(m, ax=axes, orientation="vertical", shrink=0.82, pad=0.02)
    cb.set_label("2 m temperature [K]")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
