"""Plotting utilities for CARRA2 (HR) / ERA5 (LR) patches in the native grid frame.

The CARRA2 NetCDF from the CDS keeps no projected ``x``/``y`` coordinates and no CRS
parameters (only 2-D ``latitude``/``longitude`` on a polar-stereographic grid), so we plot
against the grid's own integer indices and reconstruct everything geographic from the 2-D
coordinate fields:

* **North up** -- the native (y, x) axes are rotated w.r.t. north; we transpose/flip the
  arrays so latitude increases upward and longitude increases rightward (``orient_north_up``).
* **Lat/lon graticule** -- contoured directly from the 2-D coordinate fields.
* **Coastlines & borders** -- Natural Earth geometries mapped into index space via an inverse
  (lon, lat) -> (col, row) interpolator (``inverse_index_map``). National/territorial borders
  are drawn dashed.
* **Scale bar** -- grid spacing (km) is measured with the haversine distance between adjacent
  cells, so the bar length is correct regardless of the unknown projection.
* **Station markers** -- small stars at given (lat, lon) points.

Run the self-test demo (rebuilds a patch from the cached scratch download):

    python -m visualization.plotting
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import cartopy.feature as cfeature
from scipy.interpolate import LinearNDInterpolator

import data_acquisition.data_utils as du

# Communities to mark (lat, lon in degrees).
DEFAULT_STATIONS = {
    "Inuvik": (68.3607, -133.7230),
    "Old Crow": (67.5706, -139.8391),
    "Paulatuk": (69.3515, -124.0758),
}

_OUTLINE = [pe.withStroke(linewidth=2.0, foreground="white")]


# --------------------------------------------------------------------------------------
# Orientation (native (y, x) -> "north up")
# --------------------------------------------------------------------------------------


def orient_north_up(lat2d: np.ndarray, lon2d: np.ndarray) -> dict:
    """Return the transpose/flip ops that put latitude up and longitude rightward.

    ``lon2d`` should already be in the [-180, 180) convention.
    """
    transpose = abs(np.nanmean(np.diff(lat2d, axis=1))) > abs(np.nanmean(np.diff(lat2d, axis=0)))
    lat_t = lat2d.T if transpose else lat2d
    lon_t = lon2d.T if transpose else lon2d
    flip0 = np.nanmean(np.diff(lat_t, axis=0)) < 0   # latitude must increase with row (upward)
    flip1 = np.nanmean(np.diff(lon_t, axis=1)) < 0   # longitude must increase with col (right)
    return {"transpose": transpose, "flip0": flip0, "flip1": flip1}


def apply_orientation(a: np.ndarray, o: dict) -> np.ndarray:
    """Apply an :func:`orient_north_up` spec to a 2-D array."""
    if o["transpose"]:
        a = a.T
    if o["flip0"]:
        a = a[::-1, :]
    if o["flip1"]:
        a = a[:, ::-1]
    return a


# --------------------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------------------


def grid_spacing_km(lat2d: np.ndarray, lon2d: np.ndarray) -> float:
    """Mean spacing (km) between adjacent grid cells near the patch centre (haversine)."""
    j, i = lat2d.shape[0] // 2, lat2d.shape[1] // 2

    def hav(la1, lo1, la2, lo2):
        r = 6371.0088
        p1, p2 = np.deg2rad(la1), np.deg2rad(la2)
        dphi, dl = np.deg2rad(la2 - la1), np.deg2rad(lo2 - lo1)
        h = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
        return 2 * r * np.arcsin(np.sqrt(h))

    dr = hav(lat2d[j, i], lon2d[j, i], lat2d[j + 1, i], lon2d[j + 1, i])
    dc = hav(lat2d[j, i], lon2d[j, i], lat2d[j, i + 1], lon2d[j, i + 1])
    return float(0.5 * (dr + dc))


def inverse_index_map(lat_o: np.ndarray, lon_o: np.ndarray, stride: int = 4):
    """Build (fx, fy): interpolators mapping (lon, lat) -> (col, row) in oriented index space.

    Downsampled by ``stride`` to keep the Delaunay triangulation cheap; points outside the
    patch return NaN (so off-patch geometry is naturally clipped).
    """
    ny, nx = lat_o.shape
    IX, IY = np.meshgrid(np.arange(nx), np.arange(ny))
    pts = np.column_stack([lon_o[::stride, ::stride].ravel(), lat_o[::stride, ::stride].ravel()])
    fx = LinearNDInterpolator(pts, IX[::stride, ::stride].ravel())
    fy = LinearNDInterpolator(pts, IY[::stride, ::stride].ravel())
    return fx, fy


# --------------------------------------------------------------------------------------
# Overlays
# --------------------------------------------------------------------------------------


def add_graticule(ax, ix, iy, lat_o, lon_o, dlat: int = 2, dlon: int = 5):
    """Overlay a lat/lon graticule contoured from the 2-D coordinate fields."""
    lat_lvls = np.arange(np.ceil(lat_o.min()), np.floor(lat_o.max()) + 1, dlat)
    lon_lvls = np.arange(np.ceil(lon_o.min() / dlon) * dlon,
                         np.floor(lon_o.max() / dlon) * dlon + 1, dlon)
    ca = ax.contour(ix, iy, lat_o, levels=lat_lvls, colors="k", linewidths=0.4, alpha=0.45)
    ax.clabel(ca, fmt="%g°N", fontsize=6)
    co = ax.contour(ix, iy, lon_o, levels=lon_lvls, colors="k", linewidths=0.4, alpha=0.45,
                    linestyles=":")
    ax.clabel(co, fmt="%g°", fontsize=6)


def _draw_feature(ax, feature, fx, fy, extent_ll, **kw):
    for geom in feature.intersecting_geometries(extent_ll):
        parts = geom.geoms if geom.geom_type.startswith("Multi") else [geom]
        for ln in parts:
            c = np.asarray(ln.coords)
            ax.plot(fx(c[:, 0], c[:, 1]), fy(c[:, 0], c[:, 1]), **kw)


def add_geo_features(ax, fx, fy, extent_ll, res: str = "50m"):
    """Draw coastlines (solid) plus national & territorial borders (dashed) in index space."""
    coast = cfeature.NaturalEarthFeature("physical", "coastline", res)
    national = cfeature.NaturalEarthFeature("cultural", "admin_0_boundary_lines_land", res)
    territorial = cfeature.NaturalEarthFeature("cultural", "admin_1_states_provinces_lines", res)
    _draw_feature(ax, coast, fx, fy, extent_ll, color="white", lw=1.1, zorder=3)
    _draw_feature(ax, national, fx, fy, extent_ll, color="k", lw=1.4,
                  linestyle=(0, (6, 3)), zorder=4)
    _draw_feature(ax, territorial, fx, fy, extent_ll, color="0.15", lw=0.9,
                  linestyle=(0, (3, 3)), zorder=4)


def add_stations(ax, fx, fy, stations: dict, extent_ll):
    """Mark communities with small stars + labels, skipping any outside the patch."""
    lon_min, lon_max, lat_min, lat_max = extent_ll
    for name, (lat, lon) in stations.items():
        if not (lat_min <= lat <= lat_max and lon_min <= lon <= lon_max):
            continue
        x, y = float(fx(lon, lat)), float(fy(lon, lat))
        if np.isnan(x) or np.isnan(y):
            continue
        ax.plot(x, y, marker="*", ms=12, mfc="gold", mec="k", mew=0.8, ls="none", zorder=6)
        ax.annotate(name, (x, y), xytext=(6, 4), textcoords="offset points", fontsize=8,
                    weight="bold", color="k", zorder=7, path_effects=_OUTLINE)


def add_scalebar(ax, spacing_km: float, nx: int, ny: int, length_km: int = 100):
    """Draw a horizontal scale bar (in km) using the measured grid spacing."""
    n = length_km / spacing_km  # length in index units
    x0, y0 = nx * 0.07, ny * 0.06
    ax.plot([x0, x0 + n], [y0, y0], color="k", lw=3, solid_capstyle="butt", zorder=7,
            path_effects=_OUTLINE)
    ax.text(x0 + n / 2, y0 + ny * 0.012, f"{length_km} km", ha="center", va="bottom",
            fontsize=8, weight="bold", zorder=7, path_effects=_OUTLINE)


# --------------------------------------------------------------------------------------
# Panels
# --------------------------------------------------------------------------------------


def plot_native_panel(ax, field_o, lat_o, lon_o, fx, fy, extent_ll, *, vmin, vmax, cmap,
                      spacing_km, title, stations=None, scalebar_km=100):
    """Render one field in oriented native-index space with all overlays."""
    ny, nx = field_o.shape
    ix, iy = np.arange(nx), np.arange(ny)
    m = ax.pcolormesh(ix, iy, field_o, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
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
    ax.set_title(title)
    return m


def plot_hr_lr_pair(hr_patch, interp, *, hr_var="t2m", lr_var=None, time_index=0,
                    stations=DEFAULT_STATIONS, cmap="turbo", scalebar_km=100,
                    units="K", suptitle=None, out_path=None):
    """Plot the HR CARRA2 patch and the ERA5-on-patch-grid side by side, North up.

    Parameters
    ----------
    hr_patch : cropped CARRA2 patch (from ``crop_carra2_patch``).
    interp : ERA5 interpolated onto the patch grid (from ``interpolate_era5_to_patch``).
    hr_var / lr_var : variable names (lr_var defaults to the first var in ``interp``).
    time_index : index into the time dimension.
    stations : {name: (lat, lon)} markers; pass ``None`` to disable.
    out_path : if given, save the figure there.

    Returns
    -------
    (fig, axes)
    """
    lr_var = lr_var or list(interp.data_vars)[0]

    lat2d = np.asarray(hr_patch["latitude"].values)
    lon2d = du._to_pm180(np.asarray(hr_patch["longitude"].values))
    o = orient_north_up(lat2d, lon2d)
    lat_o, lon_o = apply_orientation(lat2d, o), apply_orientation(lon2d, o)
    hr_field = apply_orientation(hr_patch[hr_var].isel(time=time_index).values, o)
    lr_field = apply_orientation(interp[lr_var].isel(time=time_index).values, o)

    fx, fy = inverse_index_map(lat_o, lon_o)
    extent_ll = [lon_o.min(), lon_o.max(), lat_o.min(), lat_o.max()]
    spacing_km = grid_spacing_km(lat2d, lon2d)
    vmin = float(np.nanmin([hr_field.min(), lr_field.min()]))
    vmax = float(np.nanmax([hr_field.max(), lr_field.max()]))

    fig, axes = plt.subplots(1, 2, figsize=(14, 7.8))
    common = dict(vmin=vmin, vmax=vmax, cmap=cmap, spacing_km=spacing_km,
                  stations=stations, scalebar_km=scalebar_km)
    m = plot_native_panel(axes[0], hr_field, lat_o, lon_o, fx, fy, extent_ll,
                          title=f"CARRA2 HR  {hr_field.shape[1]}×{hr_field.shape[0]}  "
                                f"(native ~{spacing_km:.2f} km grid)", **common)
    plot_native_panel(axes[1], lr_field, lat_o, lon_o, fx, fy, extent_ll,
                      title="ERA5 → patch grid  (bilinear, 0.25° source)", **common)

    if suptitle:
        fig.suptitle(suptitle, y=0.97)
    cbar = fig.colorbar(m, ax=axes, orientation="horizontal", fraction=0.05, pad=0.08)
    cbar.set_label(f"{lr_var}  [{units}]")

    if out_path:
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        print("saved figure:", out_path)
    return fig, axes


# --------------------------------------------------------------------------------------
# Self-test demo
# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    CENTER = (67.2, -130.23)
    PATCH = 448
    NC = "data_acquisition/_scratch_data/carra2_test_20130121.nc"

    hr = du.open_carra2(NC)
    patch = du.crop_carra2_patch(hr, CENTER, PATCH)
    box = du.era5_box_from_patch(patch, margin_cells=2)
    era5 = du.open_era5_arco(variables="2m_temperature")
    interp = du.interpolate_era5_to_patch(du.select_era5_box(era5, box), patch)

    t0 = str(patch["time"].values[0])[:16]
    plot_hr_lr_pair(
        patch, interp, hr_var="t2m",
        suptitle=f"2 m temperature  |  Little Chicago {CENTER}  |  {t0}",
        out_path="data_acquisition/_scratch_data/patch_native_oriented.png",
    )
