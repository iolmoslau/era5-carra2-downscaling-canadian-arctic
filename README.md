# thesis — ERA5 → CARRA2 super-resolution data pipeline

Tools to build paired low-/high-resolution training samples for downscaling, over a fixed
448×448 patch in northern Canada (Little Chicago, NWT) at CARRA2's 3-hourly cadence.

- **HR target** — CARRA2 (Copernicus pan-Arctic Regional Reanalysis), 2.5 km, polar-stereographic
  (`reanalysis-pan-carra`, CDS). Channels: `t2m`, `u10`, `v10` (+ static land–sea mask).
- **LR input** — ERA5, 0.25°, bilinearly upsampled at train time. Channels: surface `t2m/u10/v10`,
  pressure-level `t/z/u/v` @ 500 & 850 hPa, and sea-ice cover (12 total). Stored on the coarse grid.

## Layout
- `data_acquisition/data_utils.py` — lazy openers, CARRA2 download (CDS), cropping, ERA5 box
  derivation, the ERA5 LR channel stack (ARCO + CDS), and the bilinear regridding
  (`bilinear_weights` + `apply_bilinear`).
- `data_acquisition/dataset_builder.py` — `build_chunk_dataset` / `write_chunk` (zarr schema)
  and `build_dataset`: the month-batched download → crop → write → discard driver.
- `data_acquisition/scratch_step1.py` — end-to-end manual test harness.
- `dataloading/` — training layer: `PatchDataset` (one zarr store per split), `compute_norm_stats`
  (train-only), and `BilinearUpsampler` (coarse LR → patch grid on GPU at forward time).
- `visualization/plotting.py` — native-grid (North-up) plotting: lat/lon graticule, coastlines,
  national/territorial borders, scale bar, community markers.

## Splits
Train / val / test are **separate stores** built from distinct, contiguous time ranges
(range-based, not random — 3-hourly samples are strongly autocorrelated). Normalization stats
are computed on the **train** store only and reused for all splits; grid geometry is shared.

## Setup
Python env with `xarray, numpy, scipy, gcsfs, zarr, dask, pyproj, netCDF4, cartopy, cdsapi`.
CARRA2 access needs a [CDS](https://cds.climate.copernicus.eu) account, `~/.cdsapirc`, and the
CARRA2 licence accepted. ERA5 is read lazily from the public ARCO-ERA5 cloud Zarr (dev) and via
CDS area-subset (production build).

> Reanalysis data files are **not** tracked (see `.gitignore`) — they are downloaded/built locally.
