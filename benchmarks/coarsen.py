# SPDX-License-Identifier: Apache-2.0
"""HR <-> LR grid geometry shared by the benchmarks: footprint-mean coarsening and bilinear refining.

Promoted out of ``scripts/lr_hr_consistency.py`` so there is ONE implementation of each direction:

* :func:`build_binning` / :func:`coarsen` -- HR (448x448 curvilinear CARRA2 patch) onto the stored
  ERA5 LR grid by FOOTPRINT MEAN: every HR cell is binned into the LR cell that contains it and
  averaged. Not bilinear sampling -- an LR value is a grid-box quantity, so a box mean is the
  honest counterpart, and point sampling would keep 2.5 km variance.
* :class:`LRtoHR` -- LR onto the HR patch with exactly the bilinear weights the model uses at train
  time (``data_utils.bilinear_weights``, as wrapped by ``dataloading.upsample.BilinearUpsampler``),
  in numpy, so a benchmark and CorrDiff see the same interpolation of the same input.

Orientation: everything here takes and returns LR fields in the STORE's order (``lat`` descending,
``lon`` in [0, 360)); the flips needed by ``bilinear_weights`` happen inside.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data_acquisition.data_utils import apply_bilinear, bilinear_weights  # noqa: E402


# ----------------------------------------------------------------------------- HR -> LR
def build_binning(lat, lon, hr_lat, hr_lon, coverage: float = 0.5) -> dict:
    """Map every HR cell to the LR cell containing it (area-average weights).

    ``lat``/``lon`` are the stored 1-D LR axes, ``hr_lat``/``hr_lon`` the 2-D patch coordinates.
    Returns the flat LR index per HR cell, per-cell counts, and ``valid`` -- LR cells covered by at
    least ``coverage`` of a full cell's HR count (partial cells sit on the patch edge). ``counts``
    and ``valid`` are flat in ASCENDING-lat order; use :func:`store_order` to view them like a
    stored field.
    """
    lat = np.asarray(lat, dtype=float)                      # 1-D, stored descending
    lon = np.asarray(lon, dtype=float)                      # 1-D, [0, 360)
    flipped = bool(lat[0] > lat[-1])
    lat_asc = lat[::-1] if flipped else lat

    def edges(c: np.ndarray) -> np.ndarray:
        step = np.abs(np.diff(c)).mean()
        return np.concatenate([[c[0] - step / 2], (c[:-1] + c[1:]) / 2, [c[-1] + step / 2]])

    lat_edges, lon_edges = edges(lat_asc), edges(lon)
    hr_lat = np.asarray(hr_lat, dtype=float).ravel()
    hr_lon = np.mod(np.asarray(hr_lon, dtype=float), 360.0).ravel()  # match LR

    iy = np.digitize(hr_lat, lat_edges) - 1
    ix = np.digitize(hr_lon, lon_edges) - 1
    ny, nx = lat_asc.size, lon.size
    ok = (iy >= 0) & (iy < ny) & (ix >= 0) & (ix < nx)
    flat = np.where(ok, iy * nx + ix, 0)

    counts = np.bincount(flat[ok], minlength=ny * nx).astype(float)
    # Full cells all hold about the same number of HR cells; partial ones sit at the patch edge.
    full = np.median(counts[counts > 0]) if np.any(counts > 0) else 0.0
    valid = counts >= coverage * full
    return {"flat": flat, "ok": ok, "counts": counts, "valid": valid,
            "shape": (ny, nx), "flipped": flipped, "full": float(full)}


def build_binning_from_store(z, coverage: float = 0.5) -> dict:
    """:func:`build_binning` from an opened shard (``lat``, ``lon``, ``hr_lat``, ``hr_lon``)."""
    return build_binning(z["lat"].values, z["lon"].values,
                         z["hr_lat"].values, z["hr_lon"].values, coverage)


def store_order(flat_asc: np.ndarray, b: dict) -> np.ndarray:
    """A flat ascending-lat LR array (e.g. ``b["counts"]``) as a (ny, nx) field in store order."""
    out = np.asarray(flat_asc).reshape(b["shape"])
    return out[::-1] if b["flipped"] else out


def coarsen(field_hr: np.ndarray, b: dict) -> np.ndarray:
    """Area-average one HR field (y, x) onto the LR grid, oriented like the stored LR.

    LR cells containing no HR cell are NaN.
    """
    ny, nx = b["shape"]
    flat_vals = np.asarray(field_hr, dtype=float).ravel()[b["ok"]]
    sums = np.bincount(b["flat"][b["ok"]], weights=flat_vals, minlength=ny * nx)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(b["counts"] > 0, sums / b["counts"], np.nan)
    return store_order(mean, b)


def nearest_fill_index(has: np.ndarray):
    """Index arrays mapping every LR cell to the nearest cell where ``has`` is True.

    Used to extend coarsened HR (defined only under the patch footprint) to the ring of LR cells
    that the bilinear stencil reaches just outside it. Nearest in index space, which on a 0.25 deg
    grid over a few degrees is a fine proxy for distance.
    """
    from scipy.ndimage import distance_transform_edt  # noqa: PLC0415

    has = np.asarray(has, dtype=bool)
    if not has.any():
        raise ValueError("no LR cell is covered by the HR patch")
    _, (ii, jj) = distance_transform_edt(~has, return_distances=True, return_indices=True)
    return ii, jj


# ----------------------------------------------------------------------------- LR -> HR
class LRtoHR:
    """Bilinear LR (store order) -> HR patch, with the model's own weights.

    ``active`` is the flat store-order index set of LR cells any HR point's stencil touches. A
    benchmark only needs to compute anything on those cells; :meth:`scatter` puts an
    ``(..., n_active)`` vector back on the full grid with zeros elsewhere (never NaN: a NaN in a
    cell with zero weight would still poison the weighted sum).
    """

    def __init__(self, lat, lon, hr_lat, hr_lon):
        lat = np.asarray(lat, dtype=float)
        lon = np.asarray(lon, dtype=float)
        self.flip = bool(lat[0] > lat[-1])
        lat_asc = lat[::-1] if self.flip else lat
        self.lr_shape = (lat.size, lon.size)
        self.hr_shape = tuple(np.asarray(hr_lat).shape)
        idx, w = bilinear_weights(lat_asc, lon, np.asarray(hr_lat),
                                  np.mod(np.asarray(hr_lon, dtype=float), 360.0))
        if np.isnan(w).any():
            raise ValueError("HR points fall outside the LR grid (NaN bilinear weights); "
                             "the stored LR box does not bracket the patch")
        self.idx, self.w = idx, w

        ny, nx = self.lr_shape
        asc = np.unique(idx)                                  # ascending-lat flat indices
        iy, ix = np.divmod(asc, nx)
        if self.flip:
            iy = ny - 1 - iy
        self.active = np.sort(iy * nx + ix)                   # store-order flat indices

    def scatter(self, values: np.ndarray) -> np.ndarray:
        """``(..., n_active)`` -> ``(..., ny, nx)`` in store order, zeros off the active set."""
        values = np.asarray(values)
        lead = values.shape[:-1]
        out = np.zeros(lead + (self.lr_shape[0] * self.lr_shape[1],), dtype=np.float64)
        out[..., self.active] = values
        return out.reshape(lead + self.lr_shape)

    def __call__(self, field: np.ndarray) -> np.ndarray:
        """``(ny, nx, *extra)`` store-order field -> ``(H, W, *extra)`` on the HR patch."""
        f = np.asarray(field, dtype=np.float64)
        if f.shape[:2] != self.lr_shape:
            raise ValueError(f"expected LR field of shape {self.lr_shape}, got {f.shape[:2]}")
        if self.flip:
            f = f[::-1]
        out = apply_bilinear(f, self.idx, self.w)
        return out.reshape(self.hr_shape + f.shape[2:])
