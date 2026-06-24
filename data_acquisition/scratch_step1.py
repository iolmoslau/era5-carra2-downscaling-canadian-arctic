"""Manual Step-1/2 verification harness (small batch, this machine).

NOT a library. Confirms: CARRA2 download -> lazy open -> square patch crop -> ERA5 box
-> lazy ERA5 select -> bilinear interpolation onto the patch grid -> native-coordinate
visualization (HR vs ERA5-on-HR-grid) with a lat/lon graticule and coastlines.

    python -m data_acquisition.scratch_step1
"""

import numpy as np
import data_acquisition.data_utils as du

CENTER = (67.2, -130.23)  # (lat, lon), Little Chicago, NWT
PATCH = 448               # HR cells per side
OUT_DIR = "data_acquisition/_scratch_data"  # local-only; delete after testing

# --- CARRA2 download (exact schema from the CDS "Show API request" panel) ---------------
DATASET = "reanalysis-pan-carra"
request = {
    "level_type": "single_levels",
    "variable": ["2m_temperature"],
    "product_type": "analysis",
    "time": ["00:00", "03:00", "06:00", "09:00", "12:00", "15:00", "18:00", "21:00"],
    "year": ["2013"],
    "month": ["01"],
    "day": ["21"],
    "data_format": "netcdf",
    "area": [75, -150, 60, -113],  # NOTE: ignored by CDS for CARRA2 -> full domain returned
}

paths = du.download_carra2(DATASET, request, OUT_DIR, basename="carra2_test_20130121")
print("downloaded:", paths)

# --- Lazy open + crop a true square patch ----------------------------------------------
hr = du.open_carra2(paths)
patch = du.crop_carra2_patch(hr, CENTER, PATCH)
print("patch dims:", dict(patch.sizes))

# --- ERA5 box -> lazy select -> bilinear interpolation onto the patch grid --------------
box = du.era5_box_from_patch(patch, margin_cells=2)
era5 = du.open_era5_arco(variables="2m_temperature")
lr = du.select_era5_box(era5, box)
interp = du.interpolate_era5_to_patch(lr, patch)   # ERA5 sampled on the HR (y, x) grid
iv = interp["2m_temperature"].isel(time=0).values
print("interp dims:", dict(interp.sizes),
      f"| t0 stats min {np.nanmin(iv):.1f} mean {np.nanmean(iv):.1f} max {np.nanmax(iv):.1f}",
      "| any NaN:", bool(np.isnan(iv).any()))

# --- Native-coordinate visualization ---------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
from scipy.interpolate import LinearNDInterpolator

hr_vals = patch["t2m"].isel(time=0).values
lat2d = patch["latitude"].values
lon2d = du._to_pm180(patch["longitude"].values)
ny, nx = lat2d.shape
ix, iy = np.arange(nx), np.arange(ny)
IX, IY = np.meshgrid(ix, iy)

vmin = float(np.nanmin([hr_vals.min(), iv.min()]))
vmax = float(np.nanmax([hr_vals.max(), iv.max()]))
extent_ll = [lon2d.min(), lon2d.max(), lat2d.min(), lat2d.max()]

# Inverse map (lon, lat) -> (col, row) for drawing coastlines in native index space.
# Downsample by 4 to keep the Delaunay triangulation cheap; placement stays accurate.
s = 4
pts_ll = np.column_stack([lon2d[::s, ::s].ravel(), lat2d[::s, ::s].ravel()])
fx = LinearNDInterpolator(pts_ll, IX[::s, ::s].ravel())
fy = LinearNDInterpolator(pts_ll, IY[::s, ::s].ravel())
_coast = cfeature.NaturalEarthFeature("physical", "coastline", "50m")


def native_panel(ax, vals, title):
    m = ax.pcolormesh(ix, iy, vals, cmap="turbo", vmin=vmin, vmax=vmax, shading="auto")
    # lat/lon graticule from the 2-D coordinate fields
    lat_lvls = np.arange(np.ceil(lat2d.min()), np.floor(lat2d.max()) + 1, 2)
    lon_lvls = np.arange(np.ceil(lon2d.min() / 5) * 5, np.floor(lon2d.max() / 5) * 5 + 1, 5)
    ca = ax.contour(ix, iy, lat2d, levels=lat_lvls, colors="k", linewidths=0.4, alpha=0.45)
    ax.clabel(ca, fmt="%g°N", fontsize=6)
    co = ax.contour(ix, iy, lon2d, levels=lon_lvls, colors="k", linewidths=0.4, alpha=0.45,
                    linestyles="--")
    ax.clabel(co, fmt="%g°", fontsize=6)
    # coastlines mapped into index space
    for geom in _coast.intersecting_geometries(extent_ll):
        parts = geom.geoms if geom.geom_type.startswith("Multi") else [geom]
        for ln in parts:
            c = np.asarray(ln.coords)
            ax.plot(fx(c[:, 0], c[:, 1]), fy(c[:, 0], c[:, 1]), color="white", lw=1.1)
    ax.set_aspect("equal")
    ax.set_xlim(0, nx - 1)
    ax.set_ylim(0, ny - 1)
    ax.set_xlabel("x grid index")
    ax.set_ylabel("y grid index")
    ax.set_title(title)
    return m


fig, axes = plt.subplots(1, 2, figsize=(14, 7))
m = native_panel(axes[0], hr_vals, f"CARRA2 HR  {nx}x{ny}  (native ~2.5 km grid)")
native_panel(axes[1], iv, "ERA5 -> patch grid  (bilinear, 0.25° source)")
fig.suptitle(f"2 m temperature  |  Little Chicago {CENTER}  |  {str(patch['time'].values[0])[:16]}",
             y=0.97)
cbar = fig.colorbar(m, ax=axes, orientation="horizontal", fraction=0.05, pad=0.09)
cbar.set_label("2 m temperature [K]")

out_png = OUT_DIR + "/patch_native_interp.png"
fig.savefig(out_png, dpi=130, bbox_inches="tight")
print("saved figure:", out_png)
