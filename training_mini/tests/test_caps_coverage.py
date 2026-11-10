# SPDX-License-Identifier: Apache-2.0
"""Does the CAPS domain contain our patch, and how much de-rotation does it need?

Two things here were wrong when first written, both found by actually running the script:

1. **Coverage was measured in km against an assumed 2.5 km spacing.** A grid that fully covered
   the patch but was *coarser* than 2.5 km came back as "70% of points outside", because every
   point sat further than one assumed half-cell from a grid centre. Coverage and resolution are
   different questions; the threshold now comes from the source grid's own measured spacing, so
   an interior point can never exceed 0.71 spacings however coarse the grid is.

2. **The convergence sign looked wrong.** ``grid_convergence`` returns the bearing of the grid
   **+x axis from true east, CCW positive**, which is ``-(lon - lambda0)`` for a north polar
   stereographic grid -- the negative of the meteorological convergence (grid north from true
   north, clockwise). Both are correct; they are different conventions. What matters is that it
   is self-consistent with ``derotate_uv``, which is pinned below. Guessing the sign instead
   would double the error rather than remove it.

    pytest training_mini/tests/test_caps_coverage.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "benchmarks"
for _p in (str(REPO), str(BENCH)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

pytest.importorskip("scipy")
cc = pytest.importorskip("check_caps_coverage")

# The real patch's bounds, from testing_data/shard_2011.zarr
PATCH_LAT = (61.380, 72.420)
PATCH_LON = (212.746, 242.979)


def _grid(lat_lo, lat_hi, lon_lo, lon_hi, n=90):
    lon, lat = np.meshgrid(np.linspace(lon_lo, lon_hi, n), np.linspace(lat_lo, lat_hi, n))
    return lat, lon


def _patch(n=60):
    return _grid(*PATCH_LAT, *PATCH_LON, n)


# ------------------------------------------------------------------ spacing is measured, not assumed
def test_native_spacing_tracks_the_grid_it_is_given():
    """Halving the point count must roughly double the measured spacing."""
    lat_a, lon_a = _grid(60, 70, 210, 240, 80)
    lat_b, lon_b = _grid(60, 70, 210, 240, 40)
    sa = cc.native_spacing_km(lat_a, lon_a)
    sb = cc.native_spacing_km(lat_b, lon_b)
    assert 1.7 < sb / sa < 2.3, (sa, sb)


def test_a_coarse_but_covering_grid_is_not_called_outside():
    """THE DEFECT: a covering grid coarser than 2.5 km read as 70% outside under a fixed km cut."""
    src_lat, src_lon = _grid(PATCH_LAT[0] - 0.5, PATCH_LAT[1] + 0.5,
                             PATCH_LON[0] - 1.0, PATCH_LON[1] + 1.0, 90)
    p_lat, p_lon = _patch()
    dist, _ = cc.nearest_km(src_lat, src_lon, p_lat, p_lon)
    spacing = cc.native_spacing_km(src_lat, src_lon)
    assert spacing > 2.5, "this fixture is deliberately coarser than CAPS"
    # The behaviour that matters: nothing is flagged as outside. (A ratio bound of 0.71 would
    # hold only for an ISOTROPIC grid; this lat/lon fixture is anisotropic, so its median nearest
    # spacing is the smaller axis and interior points reach ~0.78 of it. Hence the 1.0 cut.)
    assert not (dist > cc.OUTSIDE_SPACINGS * spacing).any()
    assert (dist / spacing).max() < 1.0


def test_a_clipped_grid_is_still_flagged():
    """Coverage must still fail when it genuinely should: source clipped 5 deg north of the patch."""
    src_lat, src_lon = _grid(PATCH_LAT[0] + 5.0, PATCH_LAT[1] + 0.5,
                             PATCH_LON[0] - 1.0, PATCH_LON[1] + 1.0, 90)
    p_lat, p_lon = _patch()
    dist, _ = cc.nearest_km(src_lat, src_lon, p_lat, p_lon)
    spacing = cc.native_spacing_km(src_lat, src_lon)
    outside = dist > cc.OUTSIDE_SPACINGS * spacing
    assert outside.any(), "the southern band of the patch is not covered and must be reported"
    assert dist.max() > 100.0, "an off-domain point should be far, not marginal"


def test_nearest_km_is_a_real_distance():
    """Sanity on the metric itself: one degree of latitude is ~111 km."""
    src_lat, src_lon = np.array([[60.0]]), np.array([[230.0]])
    dist, _ = cc.nearest_km(src_lat, src_lon, np.array([[61.0]]), np.array([[230.0]]))
    assert 110.0 < dist[0] < 112.0, dist


# ------------------------------------------------------------------ the parsing
def test_load_latlon_respects_the_scan_order(tmp_path):
    """iScansNegatively=0, jScansPositively=1 -> i fastest, so the flat list is (nj, ni)."""
    nj, ni = 7, 11
    lat, lon = _grid(60, 70, 210, 240, 1)[0], None
    lon, lat = np.meshgrid(np.linspace(210, 240, ni), np.linspace(60, 70, nj))
    p = tmp_path / "ll.txt"
    with p.open("w") as f:
        f.write("Latitude Longitude Value\n")
        for a, o in zip(lat.ravel(), lon.ravel()):
            f.write(f"{a:.5f} {o:.5f} 0.0\n")
    got_lat, got_lon = cc.load_latlon(str(p), ni, nj)
    assert got_lat.shape == (nj, ni)
    assert np.allclose(got_lat, lat) and np.allclose(got_lon, lon)


def test_a_point_count_that_contradicts_ni_nj_is_refused(tmp_path):
    p = tmp_path / "ll.txt"
    p.write_text("Latitude Longitude Value\n60.0 230.0 0.0\n61.0 231.0 0.0\n")
    with pytest.raises(ValueError, match="expected"):
        cc.load_latlon(str(p), 1465, 825)


# ------------------------------------------------------------------ the de-rotation convention
def test_grid_convergence_is_minus_the_meteorological_convergence():
    """For a north polar stereographic grid with vertical longitude lambda0, the function's
    angle (grid +x from true east, CCW) is -(lon - lambda0). CAPS has lambda0 = 244."""
    pyproj = pytest.importorskip("pyproj")
    from scripts.derotate_winds import grid_convergence

    LAM0, DX = 244.0, 2500.0
    crs = pyproj.CRS.from_proj4(
        f"+proj=stere +lat_0=90 +lat_ts=60 +lon_0={LAM0} +datum=WGS84 +units=m")
    fwd = pyproj.Transformer.from_crs(pyproj.CRS.from_epsg(4326), crs, always_xy=True)
    inv = pyproj.Transformer.from_crs(crs, pyproj.CRS.from_epsg(4326), always_xy=True)
    cx, cy = fwd.transform(-130.23, 67.2)
    n = 60
    X, Y = np.meshgrid(cx + (np.arange(n) - n // 2) * DX * 8,
                       cy + (np.arange(n) - n // 2) * DX * 8)
    lon, lat = inv.transform(X, Y)
    lon = np.mod(lon, 360.0)

    alpha = np.rad2deg(grid_convergence(lat, lon))
    conv = np.mod(lon - LAM0 + 180.0, 360.0) - 180.0
    interior = (slice(2, -2), slice(2, -2))
    assert np.nanmax(np.abs((alpha + conv)[interior])) < 0.5


def test_derotate_uv_agrees_with_that_convention():
    """A wind along grid +x must come out with earth bearing alpha. This is the pairing that
    matters: a sign error here doubles the rotation error instead of removing it."""
    from scripts.derotate_winds import derotate_uv

    alpha = np.deg2rad(np.array([-31.3, -14.0, 0.0, 1.0, 45.0]))
    u_e, v_e = derotate_uv(np.ones_like(alpha), np.zeros_like(alpha), alpha)
    assert np.allclose(np.arctan2(v_e, u_e), alpha, atol=1e-12)
    # and it must preserve speed, whatever the angle
    assert np.allclose(np.hypot(u_e, v_e), 1.0, atol=1e-12)


def test_caps_derotation_is_not_negligible_and_not_constant():
    """Over our patch, |alpha| = |lon - 244| runs ~1 to ~31 deg -- so the adapter needs a
    per-cell rotation, and skipping it would misdirect winds by up to 31 deg."""
    lo, hi = PATCH_LON
    mags = sorted(abs(x - 244.0) for x in (lo, hi))
    assert mags[0] < 2.0 and mags[1] > 30.0
