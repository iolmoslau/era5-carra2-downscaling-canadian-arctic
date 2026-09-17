#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Archive a zarr directory store to a single-file zarr ``ZipStore`` (1 inode).

A ``*.zarr`` directory holds thousands of tiny chunk files (~5.9k for one of our shards), which
is what exhausts the project file-count (inode) quota. This packs the whole store into
``<store>.zip`` using **ZIP_STORED** -- no recompression, because zarr chunks are already
compressed, so reads stay as fast as the loose store (seek + read + the same decompress). The
dataset loader opens ``shard_YYYY.zarr.zip`` transparently (``dataloading.dataset.open_store``).

    python zip_shard.py --src shard_2020.zarr                 # -> shard_2020.zarr.zip
    python zip_shard.py --src shard_2020.zarr --remove-src    # and delete the loose store

Typical use: de-rotate the test years to $SCRATCH, then zip each into $PROJECT (1 inode each) so
they persist without re-derotating -- see the recipe in evaluate/README.md.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def verify_archive(src: str, dst: str, n_samples: int = 3, seed: int = 0) -> dict:
    """Confirm the archive reads back identically to the loose store it came from.

    `--remove-src` is irreversible and the shards are expensive to rebuild, so deletion is
    gated on this passing. Compares the time axis, the static mask and the store attrs in
    full, then `hr`/`lr` for a few random timesteps -- reading every timestep would cost as
    much as the zip itself for no extra confidence, since a corrupted ZIP_STORED entry fails
    loudly rather than returning subtly wrong bytes.
    """
    import numpy as np  # noqa: PLC0415
    import xarray as xr  # noqa: PLC0415

    from dataloading.dataset import open_store  # noqa: PLC0415

    with xr.open_zarr(open_store(src)) as a, xr.open_zarr(open_store(dst)) as b:
        if a.sizes["time"] != b.sizes["time"]:
            raise SystemExit(f"VERIFY FAILED: time length {a.sizes['time']} != {b.sizes['time']}")
        if not np.array_equal(np.asarray(a["time"].values), np.asarray(b["time"].values)):
            raise SystemExit("VERIFY FAILED: time coordinate differs")
        if not np.array_equal(np.asarray(a["land_sea_mask"].values),
                              np.asarray(b["land_sea_mask"].values)):
            raise SystemExit("VERIFY FAILED: land_sea_mask differs")
        for key in ("hr_channels", "lr_channels"):
            if list(a.attrs.get(key, [])) != list(b.attrs.get(key, [])):
                raise SystemExit(f"VERIFY FAILED: attr {key} differs")

        nt = a.sizes["time"]
        rng = np.random.default_rng(seed)
        idx = sorted(rng.choice(nt, size=min(n_samples, nt), replace=False).tolist())
        for i in idx:
            for var in ("hr", "lr"):
                if not np.array_equal(np.asarray(a[var].isel(time=i).values),
                                      np.asarray(b[var].isel(time=i).values)):
                    raise SystemExit(f"VERIFY FAILED: {var} differs at time index {i}")
    return {"n_time": nt, "checked": idx}


def zip_store(src: str, dst: str) -> None:
    if not os.path.isdir(src):
        sys.exit(f"not a directory store: {src}")
    tmp = dst + ".partial"
    # Faithful copy: every store file -> zip entry at its store-root-relative key. ZIP_STORED so
    # chunks are only packaged, not recompressed. Write to .partial then atomically rename, so a
    # crashed/killed run never leaves a half-written .zip that looks complete.
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        for root, _, files in os.walk(src):
            for fn in files:
                full = os.path.join(root, fn)
                zf.write(full, arcname=os.path.relpath(full, src))
    os.replace(tmp, dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="path to a *.zarr directory store")
    ap.add_argument("--dst", help="output .zip (default: <src>.zip)")
    ap.add_argument("--remove-src", action="store_true",
                    help="delete the loose store after a VERIFIED zip (implies --verify)")
    ap.add_argument("--verify", action="store_true",
                    help="read the archive back and compare against the loose store")
    ap.add_argument("--verify-samples", type=int, default=3,
                    help="random timesteps to compare when verifying (default 3)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="exit 0 if the archive is already there (for resumable batch runs)")
    args = ap.parse_args()

    src = args.src.rstrip("/")
    dst = args.dst or (src + ".zip")

    if args.skip_existing and os.path.isfile(dst):
        print(f"already archived, skipping: {dst}")
        return

    n_files = sum(len(f) for _, _, f in os.walk(src))
    zip_store(src, dst)
    print(f"wrote {dst}  ({os.path.getsize(dst) / 1e9:.2f} GB, 1 inode; loose store was {n_files} files)")

    # Deleting the loose store is irreversible and the shards are expensive to rebuild, so the
    # archive has to prove it reads back first.
    if args.verify or args.remove_src:
        info = verify_archive(src, dst, args.verify_samples)
        print(f"verified: {info['n_time']} timesteps, sampled {info['checked']} -- identical")

    if args.remove_src:
        shutil.rmtree(src)
        print(f"removed loose store {src}")


if __name__ == "__main__":
    main()
