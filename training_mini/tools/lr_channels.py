#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Resolve the LR channel list for generation/evaluation, and check it against the checkpoint.

WHY THIS EXISTS: the network's input conv width is fixed at training time, while generation and
evaluation build the LR stack from ``dataset.lr_channels`` at run time. Three ways that goes
wrong, in increasing order of nastiness:

1. **Too many channels.** Feed the 12-channel stack to the 11-channel no-ice model and it dies at
   the first convolution -- after the GPU is allocated, the data opened and the model loaded.
   This happened: three generation jobs on 2026-09-24, the first attempts at generating from
   ``regression_3``.
2. **The override is unreachable.** ``conf/config_generate_era5_carra2_mini.yaml`` documents
   ``++dataset.lr_channels=[...]`` for the no-ice model, but neither ``slurm/generate.sh`` nor
   ``evaluate/run_eval.sh`` had any way to pass it. The documented fix could not be applied.
3. **The wrong 11 channels.** ``ERA5CARRA2Dataset`` checks the list is a *subset* of the store, so
   a typo fails loudly -- but dropping ``t500`` instead of ``siconc`` also yields 11 channels, and
   that MATCHES the conv width. It runs, silently, on a channel stack the model never saw. No
   error, just quietly wrong numbers.

So the list is resolved from the **data store** (never typed out), and its length is checked
against the **checkpoint** (which cannot be wrong about its own shape) before any GPU work.

    # the 11-channel no-ice stack, verified against the net that will consume it
    python tools/lr_channels.py --data-dir $DATA --drop siconc \
        --check-against $REG_CKPT --stage regression
    -> [t2m,u10,v10,t500,t850,z500,z850,u500,u850,v500,v850]

Prints a Hydra list literal on stdout for the shell to pass through; exits 1 with an explanation
on a mismatch, so the caller can simply `|| exit 1`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TOOLS = Path(__file__).resolve().parent
for _p in (str(_REPO_ROOT), str(_TOOLS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def store_channels(data_dir: str, year: int | None = None) -> list[str]:
    """The LR channel names the shards actually hold, in stored order.

    Read from the store rather than a hardcoded list so this cannot drift from the data, and so
    an ablation build with a different manifest is handled without a code change.
    """
    import xarray as xr  # noqa: PLC0415

    from dataloading.dataset import discover_shards, open_store, shard_path, store_exists  # noqa: PLC0415

    store = None
    if year is not None:
        cand = shard_path(data_dir, year)
        if store_exists(cand):
            store = cand
    if store is None:
        found = discover_shards(data_dir)
        if not found:
            raise FileNotFoundError(f"no shard_YYYY.zarr[.zip] stores under {data_dir}")
        store = found[0]
    with xr.open_zarr(open_store(store)) as z:
        return list(z.attrs["lr_channels"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, help="directory holding shard_YYYY.zarr[.zip]")
    ap.add_argument("--year", type=int, help="read the manifest from this year's shard")
    ap.add_argument("--drop", default="",
                    help="comma-separated channels to withhold, e.g. 'siconc' for the no-ice "
                         "model. Also the hook for the channel-ablation sweep.")
    ap.add_argument("--check-against", default="",
                    help="checkpoint whose input conv width must match the resolved list")
    ap.add_argument("--stage", default="regression", choices=("regression", "diffusion"),
                    help="which net --check-against is (they differ by img_out_channels)")
    args = ap.parse_args()

    try:
        channels = store_channels(args.data_dir, args.year)
    except Exception as e:
        print(f"ERROR: cannot read the LR channel manifest: {type(e).__name__}: {e}",
              file=sys.stderr)
        return 1

    drop = [c.strip() for c in args.drop.split(",") if c.strip()]
    unknown = [c for c in drop if c not in channels]
    if unknown:
        # Loud, because a silently ignored --drop is how you end up evaluating the wrong model.
        print(f"ERROR: cannot drop {unknown}: not in the store's channels {channels}",
              file=sys.stderr)
        return 1
    resolved = [c for c in channels if c not in drop]
    if not resolved:
        print("ERROR: --drop removed every channel", file=sys.stderr)
        return 1

    if args.check_against:
        from collect_run import channels_from_in, checkpoint_provenance  # noqa: PLC0415

        prov = checkpoint_provenance(args.check_against, args.stage)
        want = prov.get("lr_n")
        if want is None:
            # Could not read the checkpoint (physicsnemo absent, file unreadable). Warn and let
            # the job proceed: refusing here would block a run over a diagnostic we can't make.
            print(f"WARNING: could not read {args.check_against} to verify channels"
                  f" ({prov.get('error', 'unknown')}) -- proceeding with {len(resolved)}",
                  file=sys.stderr)
        elif want != len(resolved):
            print(f"ERROR: {args.stage} checkpoint expects {want} LR channels, but "
                  f"{len(resolved)} were resolved.", file=sys.stderr)
            print(f"       checkpoint: {args.check_against}", file=sys.stderr)
            print(f"       in_channels={prov.get('in_channels')} -> "
                  f"{want} LR + {prov['in_channels'] - want} fixed "
                  f"(lsm + grid + latents)", file=sys.stderr)
            print(f"       resolved:   {resolved}", file=sys.stderr)
            if want == len(channels) - 1 and "siconc" in channels and "siconc" not in drop:
                print("       This looks like the no-ice model. Re-run with NOICE=1 "
                      "(or --drop siconc).", file=sys.stderr)
            elif want == len(channels) and drop:
                print(f"       This looks like the full model. Drop the --drop {drop}.",
                      file=sys.stderr)
            return 1

    # Hydra list literal, no spaces: it is passed as one ++dataset.lr_channels=... argv element.
    print("[" + ",".join(resolved) + "]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
