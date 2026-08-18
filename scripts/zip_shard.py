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
                    help="delete the loose store after a successful zip")
    args = ap.parse_args()

    src = args.src.rstrip("/")
    dst = args.dst or (src + ".zip")
    n_files = sum(len(f) for _, _, f in os.walk(src))
    zip_store(src, dst)
    print(f"wrote {dst}  ({os.path.getsize(dst) / 1e9:.2f} GB, 1 inode; loose store was {n_files} files)")

    if args.remove_src:
        shutil.rmtree(src)
        print(f"removed loose store {src}")


if __name__ == "__main__":
    main()
