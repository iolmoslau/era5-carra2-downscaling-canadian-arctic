#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Does the CAPS domain actually contain our 448x448 patch, and how much de-rotation is needed?

TWO QUESTIONS THE CORNERS CANNOT ANSWER. ``grib_ls`` reports CAPS as polar stereographic,
1465 x 825 at 2500 m, first point (67.960 N, 219.239 E), last (64.496 N, 333.097 E). On a
projected grid those two corners do **not** bound a lat/lon box -- the footprint is a rotated
quadrilateral, so "is our patch inside?" is not answerable by comparing numbers. It has to be
measured per point.

The same dump answers the second question. CAPS winds are grid-relative
(``resolutionAndComponentFlags = 8``, i.e. Flag Table 3.3 bit 5 set), so the adapter must
de-rotate -- and the convergence angle of a polar stereographic grid varies with longitude, which
spans ~25 degrees across our patch. So this is a spatially varying rotation, not one constant, and
its range decides how much it matters.

Both come from one text dump of the CAPS geometry, which needs no Python GRIB bindings (there are
none in corrdiff-env -- cffi is missing):

    module load eccodes
    grib_get_data -w count=1 $SCRATCH/caps_inspect/caps/*.grib2 > $SCRATCH/caps_latlon.txt
    python benchmarks/check_caps_coverage.py --latlon $SCRATCH/caps_latlon.txt \
        --data-dir $PROJECT/data/derot --year 2020

The geometry is identical in every member, so this is done once for the whole archive.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

EARTH_R_KM = 6371.0
# "Outside the footprint" is measured in units of the SOURCE grid's own spacing, not in km.
# A point inside a grid is at most half a cell diagonal (0.71 spacings) from the nearest centre,
# so 1.0 is a generous cut that no interior point reaches while an exterior one blows past it.
# Hardcoding 2.5 km instead reported 70% of a fully-covering grid as "outside" purely because
# that grid was coarser -- coverage and resolution are different questions.
OUTSIDE_SPACINGS = 1.0


def load_latlon(path: str, ni: int, nj: int) -> tuple[np.ndarray, np.ndarray]:
    """Parse ``grib_get_data`` output into 2-D lat/lon on the CAPS grid.

    Scan order comes from the GRIB flags: ``iScansNegatively=0``, ``jScansPositively=1`` means i
    runs west->east fastest and j runs south->north, so the flat list reshapes as (nj, ni).
    """
    import pandas as pd  # noqa: PLC0415  (loadtxt is far slower on 1.2M rows)

    df = pd.read_csv(path, sep=r"\s+", comment="#", header=0,
                     names=["lat", "lon", "value"], engine="python", skip_blank_lines=True)
    df = df[pd.to_numeric(df["lat"], errors="coerce").notna()]      # drop any header/footer text
    lat = df["lat"].to_numpy(dtype=float)
    lon = df["lon"].to_numpy(dtype=float)
    if lat.size != ni * nj:
        raise ValueError(f"{path} holds {lat.size} points, expected {ni}*{nj} = {ni * nj}. "
                         f"Pass --ni/--nj if the grid differs.")
    return lat.reshape(nj, ni), lon.reshape(nj, ni)


def _unit_xyz(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    la, lo = np.deg2rad(lat.ravel()), np.deg2rad(lon.ravel())
    return np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def native_spacing_km(lat: np.ndarray, lon: np.ndarray, sample: int = 20000,
                      seed: int = 0) -> float:
    """The grid's own point spacing: median distance from a point to its nearest neighbour.

    Derived rather than assumed, so the coverage test works for any source grid -- CAPS at
    2.5 km today, a coarser dynamical model tomorrow.
    """
    from scipy.spatial import cKDTree  # noqa: PLC0415

    xyz = _unit_xyz(lat, lon)
    rng = np.random.default_rng(seed)
    idx = rng.choice(xyz.shape[0], size=min(sample, xyz.shape[0]), replace=False)
    tree = cKDTree(xyz)
    chord, _ = tree.query(xyz[idx], k=2)            # k=1 is the point itself
    arc = 2.0 * np.arcsin(np.clip(chord[:, 1] / 2.0, 0.0, 1.0))
    return float(np.median(arc) * EARTH_R_KM)


def nearest_km(src_lat, src_lon, dst_lat, dst_lon):
    """Great-circle distance from each dst point to the nearest src point, in km, plus its index.

    A KD-tree on unit-sphere Cartesian coordinates: chord distance is monotonic in great-circle
    distance, so nearest-by-chord is nearest-by-arc, and no projection assumption enters.
    """
    from scipy.spatial import cKDTree  # noqa: PLC0415

    tree = cKDTree(_unit_xyz(src_lat, src_lon))
    chord, idx = tree.query(_unit_xyz(dst_lat, dst_lon), k=1)
    arc = 2.0 * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0))
    return arc * EARTH_R_KM, idx


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--latlon", required=True, help="grib_get_data dump of one CAPS member")
    ap.add_argument("--ni", type=int, default=1465, help="CAPS points along i (default 1465)")
    ap.add_argument("--nj", type=int, default=825, help="CAPS points along j (default 825)")
    ap.add_argument("--data-dir", required=True, help="directory holding our shards")
    ap.add_argument("--year", type=int, default=2020, help="shard to read hr_lat/hr_lon from")
    args = ap.parse_args()

    import xarray as xr

    from dataloading.dataset import open_store, shard_path, store_exists
    from scripts.derotate_winds import grid_convergence

    caps_lat, caps_lon = load_latlon(args.latlon, args.ni, args.nj)
    caps_lon = np.mod(caps_lon, 360.0)
    print(f"CAPS grid   {caps_lat.shape[0]} x {caps_lat.shape[1]}  "
          f"lat {caps_lat.min():.2f}..{caps_lat.max():.2f}  "
          f"lon {caps_lon.min():.2f}..{caps_lon.max():.2f}")
    # Cross-check the parse against what grib_ls reported for the same file. If these two points
    # disagree with it, the dump was misread and every distance below is meaningless -- a check
    # the point COUNT cannot make, since a column mix-up preserves the count.
    print(f"  first point lat {caps_lat.ravel()[0]:.3f} lon {caps_lon.ravel()[0]:.3f}   "
          f"last point lat {caps_lat.ravel()[-1]:.3f} lon {caps_lon.ravel()[-1]:.3f}")
    print("  (grib_ls reported first 67.960/219.239, last 64.496/333.097 -- these must match)")

    store = shard_path(args.data_dir, args.year)
    if not store_exists(store):
        print(f"ERROR: no shard for {args.year} under {args.data_dir}", file=sys.stderr)
        return 1
    with xr.open_zarr(open_store(store)) as z:
        hr_lat = np.asarray(z["hr_lat"].values, dtype=float)
        hr_lon = np.mod(np.asarray(z["hr_lon"].values, dtype=float), 360.0)
    print(f"patch grid  {hr_lat.shape[0]} x {hr_lat.shape[1]}  "
          f"lat {hr_lat.min():.2f}..{hr_lat.max():.2f}  "
          f"lon {hr_lon.min():.2f}..{hr_lon.max():.2f}")

    # ---------------------------------------------------------------- coverage
    dist, idx = nearest_km(caps_lat, caps_lon, hr_lat, hr_lon)
    spacing = native_spacing_km(caps_lat, caps_lon)
    cut = OUTSIDE_SPACINGS * spacing
    outside = dist > cut
    print("\n---- coverage ------------------------------------------------")
    print(f"  CAPS native spacing (measured): {spacing:.2f} km")
    print(f"  nearest CAPS point: median {np.median(dist):.2f} km, "
          f"mean {dist.mean():.2f} km, max {dist.max():.2f} km")
    # 0.71 = half a cell diagonal, the interior bound for an ISOTROPIC grid. CAPS is isotropic
    # (Dx = Dy = 2500 m); on an anisotropic grid the median nearest spacing is the smaller axis
    # and interior points can sit a little beyond 0.71 of it, which is why the cut is at 1.0.
    print(f"  as a fraction of spacing: median {np.median(dist) / spacing:.2f}, "
          f"max {dist.max() / spacing:.2f}  (interior bound ~0.71 for an isotropic grid)")
    print(f"  outside cut: {cut:.2f} km = {OUTSIDE_SPACINGS:.1f} x spacing")
    print(f"  patch points outside: {int(outside.sum())} of {dist.size} "
          f"({100.0 * outside.mean():.2f}%)")
    if outside.any():
        oi = np.argwhere(outside.reshape(hr_lat.shape))
        print(f"  worst point: lat {hr_lat.ravel()[np.argmax(dist)]:.3f} "
              f"lon {hr_lon.ravel()[np.argmax(dist)]:.3f}  ({dist.max():.1f} km away)")
        print(f"  affected rows {oi[:, 0].min()}..{oi[:, 0].max()}, "
              f"cols {oi[:, 1].min()}..{oi[:, 1].max()}")

        # WHERE the covered part is, as a picture. Row/column ranges cannot distinguish a
        # contiguous diagonal strip (real geometry) from a scattered pattern (a parsing
        # artifact), and they read identically -- 0..447 in both cases. The map can.
        inside2d = (~outside).reshape(hr_lat.shape)
        k = 16                                        # a 16x16 map of 28x28-pixel blocks
        blocks = inside2d.reshape(k, 448 // k, k, 448 // k).mean(axis=(1, 3))
        ramp = " .:-=+*#@"
        print("\n  covered fraction per block (patch rows top=0, cols left=0):")
        for row in blocks:
            print("    " + "".join(ramp[min(int(v * len(ramp)), len(ramp) - 1)] for v in row))
        print(f"    ' ' = none, '@' = all. covered total {100 * inside2d.mean():.1f}%")
        if inside2d.any():
            print(f"  covered region: lat {hr_lat[inside2d].min():.2f}..{hr_lat[inside2d].max():.2f}"
                  f"  lon {hr_lon[inside2d].min():.2f}..{hr_lon[inside2d].max():.2f}")
        print("  => the patch is NOT fully inside CAPS. The comparison must either be "
              "restricted\n     to the covered sub-region, or CAPS filled/masked there -- "
              "and whichever it is\n     belongs in the write-up, not in silent NaNs.")
    else:
        print("  => every patch point sits inside the CAPS footprint. Straight regrid.")

    # ---------------------------------------------------------------- de-rotation magnitude
    # Convergence is a property of the CAPS grid, so compute it there and read it off at the
    # points the patch actually samples. grid_convergence is numerical (from the 2-D lat/lon),
    # hence projection-agnostic -- the same function that fixed CARRA2.
    alpha = np.rad2deg(grid_convergence(caps_lat, caps_lon)).ravel()[idx]
    print("\n---- de-rotation (winds are grid-relative: flags bit 5 set) --")
    print(f"  CAPS grid convergence over the patch: "
          f"{alpha.min():+.2f} .. {alpha.max():+.2f} deg  (mean {alpha.mean():+.2f})")
    print(f"  spread across the patch: {alpha.max() - alpha.min():.2f} deg "
          "-- a single constant rotation will not do")
    worst = np.max(np.abs(alpha))
    print(f"  worst |alpha| = {worst:.2f} deg -> a {worst:.1f} deg direction error if skipped, "
          f"and {100 * (1 - np.cos(np.deg2rad(worst))):.1f}% of the speed moved "
          "into the wrong component")
    return 0


if __name__ == "__main__":
    sys.exit(main())
