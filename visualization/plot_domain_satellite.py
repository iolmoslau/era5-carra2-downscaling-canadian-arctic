#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Satellite basemap of the study domain, in the CARRA2 native grid frame.

Same map conventions as every other figure in this repo (``visualization.plotting``): the
axes are the CARRA2 grid's own indices oriented north-up, the graticule is contoured from the
2-D lat/lon fields, coastlines and borders are mapped into index space, the scale bar comes
from the measured grid spacing, and communities are gold stars. The only difference is what is
underneath: instead of a model field, a satellite mosaic.

HOW THE IMAGERY GETS INTO INDEX SPACE
-------------------------------------
Tiles arrive in Web Mercator. Rather than reprojecting the mosaic (and then having to reproject
every overlay to match), each of the 448x448 grid cells is looked up in the mosaic by its own
(lat, lon): a forward sample, one output pixel at a time. That keeps the frame EXACTLY the one
the model sees -- the same array indices as the HR fields -- so this figure and a prediction
panel are pixel-for-pixel comparable.

The domain geometry is read from a built shard's ``hr_lat``/``hr_lon``, so the extent is
precisely the trained domain rather than a reconstruction of it.

NEEDS INTERNET -- run on a LOGIN node, like the other cartopy figures (WORKFLOW.md's collect
step). Tiles are cached under ~/.local/share/cartopy so re-runs are offline and instant.

Imagery is ESRI World Imagery, free for non-commercial use WITH ATTRIBUTION; the credit is
drawn on the figure and must stay there if it goes in the thesis.

    python -m visualization.plot_domain_satellite --store $PROJECT/data/derot/shard_2019.zarr \
        --out figures/domain_satellite.png
"""
from __future__ import annotations

import os

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

# Point pyproj/cartopy at their bundled PROJ data if the environment doesn't set it (the
# Alliance wheelhouse pyproj can leave PROJ_DATA unset -> cartopy CRS init raises
# DataDirError). Must run before cartopy loads pyproj. Mirrors tools/plot_sample_native.py.
if not (os.environ.get("PROJ_DATA") or os.environ.get("PROJ_LIB")):
    try:
        import pyproj

        _cand = os.path.join(os.path.dirname(pyproj.__file__), "proj_dir", "share", "proj")
        if os.path.exists(os.path.join(_cand, "proj.db")):
            os.environ["PROJ_DATA"] = _cand
    except Exception:
        pass

import argparse  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import cartopy.crs as ccrs  # noqa: E402
import shapely.geometry as sgeom  # noqa: E402
from cartopy.io.img_tiles import GoogleTiles  # noqa: E402

from dataloading.dataset import open_store, shard_path, store_exists  # noqa: E402
from visualization.plotting import (  # noqa: E402
    DEFAULT_STATIONS,
    add_geo_features,
    add_graticule,
    add_scalebar,
    add_stations,
    apply_orientation,
    grid_spacing_km,
    inverse_index_map,
    orient_north_up,
)

_OUTLINE = [pe.withStroke(linewidth=2.0, foreground="white")]


class _EsriImagery(GoogleTiles):
    """ESRI World Imagery tiles. Free for non-commercial use with attribution."""

    def _image_url(self, tile):
        x, y, z = tile
        return ("https://server.arcgisonline.com/ArcGIS/rest/services/"
                f"World_Imagery/MapServer/tile/{z}/{y}/{x}")


CREDIT = "Imagery: Esri, Maxar, Earthstar Geographics, and the GIS User Community"


def satellite_in_index_space(lat_o: np.ndarray, lon_o: np.ndarray, zoom: int,
                             cache: bool = True) -> np.ndarray:
    """Fetch a mosaic and sample it at every grid cell -> (ny, nx, 3) uint8 in index space."""
    tiler = _EsriImagery(desired_tile_form="RGB", cache=cache)
    merc = tiler.crs

    xy = merc.transform_points(ccrs.PlateCarree(), lon_o, lat_o)
    xm, ym = xy[..., 0], xy[..., 1]
    pad_x = 0.02 * (np.nanmax(xm) - np.nanmin(xm))
    pad_y = 0.02 * (np.nanmax(ym) - np.nanmin(ym))
    domain = sgeom.box(np.nanmin(xm) - pad_x, np.nanmin(ym) - pad_y,
                       np.nanmax(xm) + pad_x, np.nanmax(ym) + pad_y)

    img, extent, origin = tiler.image_for_domain(domain, zoom)
    img = np.asarray(img)
    x0, x1, y0, y1 = extent
    h, w = img.shape[:2]

    # forward sample: each output cell reads the mosaic pixel containing its own lat/lon
    col = np.clip(((xm - x0) / (x1 - x0) * (w - 1)).round().astype(int), 0, w - 1)
    frac_y = (ym - y0) / (y1 - y0)
    if origin == "upper":
        frac_y = 1.0 - frac_y
    row = np.clip((frac_y * (h - 1)).round().astype(int), 0, h - 1)

    out = img[row, col]
    if out.dtype != np.uint8:               # cartopy may hand back floats in [0, 1]
        out = (np.clip(out, 0, 1) * 255).astype(np.uint8)
    return out[..., :3]


def plot_domain(store: str, *, zoom: int = 7, stations=DEFAULT_STATIONS, scalebar_km: int = 100,
                title: str | None = None, out_path: str | None = None, cache: bool = True):
    """Render the satellite basemap with the standard overlays. Returns (fig, ax)."""
    with xr.open_zarr(open_store(store)) as z:
        lat2d = np.asarray(z["hr_lat"].values)
        lon2d = np.asarray(z["hr_lon"].values)
        attrs = dict(z.attrs)
    lon2d = np.where(lon2d > 180.0, lon2d - 360.0, lon2d)   # plotting wants [-180, 180)

    o = orient_north_up(lat2d, lon2d)
    lat_o, lon_o = apply_orientation(lat2d, o), apply_orientation(lon2d, o)
    ny, nx = lat_o.shape
    fx, fy = inverse_index_map(lat_o, lon_o)
    extent_ll = [lon_o.min(), lon_o.max(), lat_o.min(), lat_o.max()]
    spacing_km = grid_spacing_km(lat2d, lon2d)

    print(f"domain {nx}x{ny} @ ~{spacing_km:.2f} km  "
          f"lon [{extent_ll[0]:.2f}, {extent_ll[1]:.2f}]  lat [{extent_ll[2]:.2f}, {extent_ll[3]:.2f}]")
    print(f"fetching ESRI World Imagery at zoom {zoom} (needs internet; cached after first run)")
    rgb = satellite_in_index_space(lat_o, lon_o, zoom, cache=cache)

    fig, ax = plt.subplots(figsize=(9.5, 9.5))
    ax.imshow(rgb, origin="lower", extent=[0, nx - 1, 0, ny - 1], interpolation="bilinear",
              zorder=0)
    ix, iy = np.arange(nx), np.arange(ny)
    add_graticule(ax, ix, iy, lat_o, lon_o)
    add_geo_features(ax, fx, fy, extent_ll)
    if stations:
        add_stations(ax, fx, fy, stations, extent_ll)
    add_scalebar(ax, spacing_km, nx, ny, scalebar_km)

    ax.set_aspect("equal")
    ax.set_xlim(0, nx - 1)
    ax.set_ylim(0, ny - 1)
    ax.set_xlabel("grid index  (E →)")
    ax.set_ylabel("grid index  (N ↑)")
    if title is None:
        c = (attrs.get("center_lat"), attrs.get("center_lon"))
        centre = f"  |  centre {c[0]}, {c[1]}" if None not in c else ""
        title = f"Study domain — CARRA2 native grid, {nx}×{ny} @ ~{spacing_km:.2f} km{centre}"
    ax.set_title(title)
    ax.text(0.995, 0.005, CREDIT, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6, color="w", zorder=8,
            path_effects=[pe.withStroke(linewidth=1.6, foreground="0.2")])

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print("saved figure:", out_path)
    return fig, ax


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--store", help="a built shard (.zarr or .zarr.zip) to take the grid from")
    src.add_argument("--data-dir", help="shard directory; use with --year")
    ap.add_argument("--year", type=int, help="year to pick from --data-dir")
    ap.add_argument("--zoom", type=int, default=7,
                    help="tile zoom (default 7 ~ 478 m/px at 67degN, vs the 2.5 km grid)")
    ap.add_argument("--scalebar-km", type=int, default=100)
    ap.add_argument("--title")
    ap.add_argument("--no-stations", action="store_true")
    ap.add_argument("--no-cache", action="store_true", help="do not cache tiles on disk")
    ap.add_argument("--out", default="figures/domain_satellite.png")
    args = ap.parse_args()

    if args.store:
        store = args.store
    else:
        if args.year is None:
            ap.error("--data-dir requires --year")
        store = shard_path(args.data_dir, args.year)
    if not store_exists(store):
        ap.error(f"store not found: {store}")

    plot_domain(store, zoom=args.zoom,
                stations=None if args.no_stations else DEFAULT_STATIONS,
                scalebar_km=args.scalebar_km, title=args.title, out_path=args.out,
                cache=not args.no_cache)


if __name__ == "__main__":
    main()
