# SPDX-License-Identifier: Apache-2.0
"""Write a benchmark's output in the schema ``generate.py``'s ``NetCDFWriter`` writes.

::

    root        lat(y,x)  lon(y,x)  time(time)   # i8 "hours since 1990-01-01", standard calendar
    truth/      <channel>(time, y, x)
    prediction/ <channel>(ensemble, time, y, x)
    input/      <channel>(time, y, x)            # optional

Hitting this exactly is what lets ``evaluate/eval_crps_mae.py`` and ``evaluate/plot_pdfs.py``
score a benchmark with no special-casing. Re-implemented rather than imported because
``training_mini/helpers/generate_helpers`` pulls in physicsnemo and torch, which a CPU-only
benchmark job has no reason to load. ``tests/test_bcsd.py`` pins the two against each other.

Writes one time at a time, so a 400-time file never has to sit in memory.
"""
from __future__ import annotations

import os

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import cftime
import netCDF4 as nc
import numpy as np
import pandas as pd

TIME_UNITS = "hours since 1990-01-01 00:00:00"
TIME_CALENDAR = "standard"


class SchemaWriter:
    """Context manager: ``with SchemaWriter(path, lat, lon, channels) as w: w.write(...)``."""

    def __init__(self, path, lat, lon, channels, input_channels=(), n_ensemble: int = 1,
                 attrs: dict | None = None):
        lat, lon = np.asarray(lat), np.asarray(lon)
        if lat.shape != lon.shape or lat.ndim != 2:
            raise ValueError("lat and lon must be 2-D and the same shape")
        self.channels = list(channels)
        self.input_channels = list(input_channels)
        self.n_ensemble = int(n_ensemble)
        self._n = 0

        f = nc.Dataset(path, "w")
        self._f = f
        f.createDimension("time")
        f.createDimension("ensemble")
        ny, nx = lat.shape
        f.createDimension("x", nx)
        f.createDimension("y", ny)

        v = f.createVariable("lat", "f", dimensions=("y", "x"))
        v[:] = lat
        v.standard_name, v.units = "latitude", "degrees_north"
        v = f.createVariable("lon", "f", dimensions=("y", "x"))
        v[:] = lon
        v.standard_name, v.units = "longitude", "degrees_east"
        v = f.createVariable("time", "i8", ("time",))
        v.calendar, v.units = TIME_CALENDAR, TIME_UNITS

        self.truth = f.createGroup("truth")
        self.prediction = f.createGroup("prediction")
        self.input = f.createGroup("input")
        for c in self.channels:
            self.truth.createVariable(c, "f", dimensions=("time", "y", "x"))
            self.prediction.createVariable(c, "f", dimensions=("ensemble", "time", "y", "x"))
        for c in self.input_channels:
            self.input.createVariable(c, "f", dimensions=("time", "y", "x"))
        for k, val in (attrs or {}).items():
            setattr(f, k, val)

    def write(self, time, truth: dict, prediction: dict, inputs: dict | None = None) -> int:
        """Append one time. ``prediction[c]`` is (y, x) or (ensemble, y, x)."""
        i = self._n
        t = pd.Timestamp(time)
        self._f["time"][i] = cftime.date2num(
            cftime.DatetimeGregorian(t.year, t.month, t.day, t.hour, t.minute, t.second),
            TIME_UNITS, TIME_CALENDAR)
        for c in self.channels:
            self.truth[c][i] = np.asarray(truth[c], dtype=np.float32)
            p = np.asarray(prediction[c], dtype=np.float32)
            p = p[None] if p.ndim == 2 else p
            if p.shape[0] != self.n_ensemble:
                raise ValueError(f"{c}: {p.shape[0]} members, writer expects {self.n_ensemble}")
            self.prediction[c][:, i] = p
        for c in self.input_channels:
            self.input[c][i] = np.asarray(inputs[c], dtype=np.float32)
        self._n += 1
        return i

    def close(self):
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def read_times(path) -> pd.DatetimeIndex:
    """The ``time`` variable of a schema NetCDF (e.g. ``full.nc``), as timestamps.

    Benchmarks take their times from here rather than re-sampling them, so "same times as the
    model" is a fact read off the file rather than a reproduced procedure.
    """
    with nc.Dataset(path) as f:
        v = f["time"]
        dates = cftime.num2date(v[:], v.units, getattr(v, "calendar", TIME_CALENDAR))
    return pd.DatetimeIndex([pd.Timestamp(d.isoformat()) for d in dates])
