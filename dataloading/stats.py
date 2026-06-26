"""Per-channel normalization statistics, computed on the TRAIN store only.

Stats must be derived from the training split alone (computing them on val/test leaks
information) and then applied unchanged to every split. Save once, reuse everywhere.
"""

from __future__ import annotations

import json
import os

import xarray as xr


def compute_norm_stats(store: str | os.PathLike, out_path: str | os.PathLike | None = None) -> dict:
    """Compute per-channel mean/std for ``hr`` and ``lr`` over a store (streamed via dask).

    Reductions run in float64 over (time, spatial) for each channel, so the result is one
    mean and one std per HR/LR channel. The land-sea mask is a 0/1 field and is not normalized.

    Parameters
    ----------
    store : zarr store to compute over (the TRAIN store).
    out_path : if given, write the stats dict as JSON here.

    Returns
    -------
    dict with ``hr_mean/hr_std`` (len 3), ``lr_mean/lr_std`` (len 12) and the channel-name
    lists ``hr_channels``/``lr_channels`` (for an explicit ordering check at load time).
    """
    z = xr.open_zarr(store)
    stats: dict = {}
    for var, spatial, chan in (("hr", ("time", "y", "x"), "hr_channel"),
                               ("lr", ("time", "lat", "lon"), "lr_channel")):
        da = z[var].astype("float64")
        stats[f"{var}_mean"] = da.mean(dim=spatial).compute().values.tolist()
        stats[f"{var}_std"] = da.std(dim=spatial).compute().values.tolist()
        stats[f"{var}_channels"] = [str(c) for c in z[chan].values]

    if out_path is not None:
        with open(out_path, "w") as f:
            json.dump(stats, f, indent=2)
    return stats


def load_norm_stats(path: str | os.PathLike) -> dict:
    """Load a stats dict previously written by :func:`compute_norm_stats`."""
    with open(path) as f:
        return json.load(f)
