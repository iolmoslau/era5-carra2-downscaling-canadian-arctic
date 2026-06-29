"""Per-channel normalization statistics, computed on the TRAIN store(s) only.

Stats must be derived from the training split alone (computing them on val/test leaks
information) and then applied unchanged to every split. Save once, reuse everywhere.
Accepts one store or several (e.g. per-year shards) and combines them exactly.
"""

from __future__ import annotations

import json
import os

import numpy as np
import xarray as xr


def compute_norm_stats(stores, out_path: str | os.PathLike | None = None) -> dict:
    """Per-channel mean/std for ``hr`` and ``lr`` over one or many stores (streamed).

    Combines stores exactly via summed first/second moments (no concatenation in memory),
    so per-year train shards give the same result as one big store.

    Parameters
    ----------
    stores : a single store path or a list of them (the TRAIN shards).
    out_path : if given, write the stats dict as JSON here.

    Returns
    -------
    dict with ``hr_mean/hr_std`` (len 3), ``lr_mean/lr_std`` (len 12) and the channel-name
    lists ``hr_channels``/``lr_channels`` (read from store attrs) for an ordering check.
    """
    if isinstance(stores, (str, os.PathLike)):
        stores = [stores]

    stats: dict = {}
    chan_names: dict = {}
    for var, chan_dim, spatial in (("hr", "hr_channel", ("time", "y", "x")),
                                   ("lr", "lr_channel", ("time", "lat", "lon"))):
        s = sq = n = None
        for store in stores:
            z = xr.open_zarr(store)
            chan_names[var] = list(z.attrs[f"{var}_channels"])
            da = z[var].astype("float64")
            part_s = da.sum(dim=spatial).compute().values            # per channel
            part_sq = (da ** 2).sum(dim=spatial).compute().values
            part_n = int(np.prod([z.sizes[d] for d in spatial]))
            s = part_s if s is None else s + part_s
            sq = part_sq if sq is None else sq + part_sq
            n = part_n if n is None else n + part_n
        mean = s / n
        var_ = np.maximum(sq / n - mean ** 2, 0.0)                   # guard fp negatives
        stats[f"{var}_mean"] = mean.tolist()
        stats[f"{var}_std"] = np.sqrt(var_).tolist()

    stats["hr_channels"] = chan_names["hr"]
    stats["lr_channels"] = chan_names["lr"]

    if out_path is not None:
        with open(out_path, "w") as f:
            json.dump(stats, f, indent=2)
    return stats


def load_norm_stats(path: str | os.PathLike) -> dict:
    """Load a stats dict previously written by :func:`compute_norm_stats`."""
    with open(path) as f:
        return json.load(f)
