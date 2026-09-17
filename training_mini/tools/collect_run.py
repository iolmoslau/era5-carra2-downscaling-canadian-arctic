#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Collect a training run's artifacts into training_mini/results/<name>/ and log it in runs.csv.

Per run it writes: loss_curve.png (from TensorBoard), sample_native.png + metrics.json (from a
generate.py NetCDF), and run_info.json (metadata); then upserts a summary row into results/runs.csv
(replacing any existing row with the same run name, so it's safe to re-run).

Run in an env with BOTH tensorboard and cartopy (corrdiff-env after setup_env.sh adds cartopy):

    python tools/collect_run.py --name regression_1 \
        --tensorboard $SCRATCH/corrdiff_mini_derot/tensorboard \
        --nc corrdiff_output.nc \
        --checkpoint $SCRATCH/corrdiff_mini_derot/checkpoints_regression/CorrDiffRegressionUNet.0.800000.mdlus \
        --hydra-config $REPO/logs/hydra/12345678 \
        --data $PROJECT/data/derot --stats $PROJECT/data/derot/stats_train_2011_2018.json \
        --error sigma --notes "de-rotated winds, 800k samples, 2xH100"

Provenance (audit P2-5)
-----------------------
Pass ``--checkpoint`` and the run records which channel set it was trained on, read from the
input conv width -- a fact about the weights, so it settles "was sea ice in?" when the config
name, the notes and memory disagree. ``--hydra-config`` points at the snapshot the SLURM scripts
write ($REPO/logs/hydra/<jobid>) and captures the RESOLVED config, which beats re-typing it:
``regression_1`` carries a hand-entered ``config`` field that nothing in this script wrote.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # training_mini/tools
TRAIN_DIR = HERE.parent                          # training_mini
RESULTS = TRAIN_DIR / "results"
CSV_PATH = RESULTS / "runs.csv"
FIELDS = ["run", "stage", "date", "config", "sea_ice", "lr_n", "in_channels",
          "train_samples", "checkpoint", "git", "notes",
          "rmse_t2m", "nrmse_t2m_pct", "bias_t2m",
          "rmse_u10", "nrmse_u10_pct", "bias_u10",
          "rmse_v10", "nrmse_v10_pct", "bias_v10",
          "ens_members", "ens_meanvar_t2m", "ens_meanvar_u10", "ens_meanvar_v10"]

# Channel arithmetic, mirroring how train.py sizes the input conv. Three terms beyond the LR
# channels themselves, and the third is easy to miss:
#     + include_lsm                       1   the static land-sea mask
#     + N_grid_channels                   4   sinusoidal positional embedding
#     + img_out_channels                  3   the LATENT the UNet denoises, concatenated with the
#                                             conditioning -- present for the regression net too,
#                                             which is why regression_step passes latents_shape
#     + img_out_channels                  3   again, only when hr_mean_conditioning (diffusion)
#
# regression: in = L + 1 + 4 + 3      = L + 8      (12 channels -> 20, 11 -> 19)
# diffusion:  in = L + 1 + 4 + 3 + 3  = L + 11     (12 channels -> 23, 11 -> 22)
#
# Confirmed against real checkpoints: regression_2 reads 20 and diffusion_2 reads 23, both
# giving L = 12, which the `input` group of a generate NetCDF corroborates BY NAME --
# 12 LR channels including siconc, plus lsm. Assumes the shipped defaults (include_lsm true,
# N_grid_channels 4, three HR outputs, no patching).
_STAGE_OFFSET = {"regression": 8, "diffusion": 11}
_FULL_LR, _NOICE_LR = 12, 11


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=str(TRAIN_DIR), text=True).strip()
    except Exception as e:
        print(f"  WARNING: git hash unavailable ({type(e).__name__}) -- run recorded without it",
              file=sys.stderr)
        return ""


def previous_collect(rdir: Path) -> dict:
    """The last run_info.json for this run, so a re-collect can top it up rather than blank it."""
    p = rdir / "run_info.json"
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception as e:
        print(f"  WARNING: ignoring unreadable {p} ({type(e).__name__})", file=sys.stderr)
        return {}


def prev_field(prev: dict, name: str) -> str:
    """Look a field up in either layout: the provenance block, or the older flat one.

    `regression_1` was collected before provenance was nested, and hand-edited besides, so both
    shapes are live in the repo.
    """
    if not prev:
        return ""
    got = (prev.get("provenance") or {}).get(name)
    if got in (None, ""):
        got = prev.get(name)
    return "" if got in (None, {}) else str(got)


def env_versions() -> dict:
    """Package versions, so a later comparison can tell whether the env moved underneath it."""
    out = {}
    for mod in ("physicsnemo", "torch", "numpy", "xarray", "zarr"):
        try:
            out[mod] = getattr(__import__(mod), "__version__", "?")
        except Exception:
            out[mod] = None
    return out


def checkpoint_provenance(path: str, stage: str) -> dict:
    """Back out which channels a checkpoint was trained on, from its input conv width.

    A checkpoint cannot be wrong about its own shape, so this settles "was sea ice in?" even
    when the config name, the notes and someone's memory disagree -- which is exactly the
    ambiguity that stalled the sea-ice run. It also keeps working for the channel-ablation
    variants, where `lr_n` will be neither 12 nor 11.

    The input conv is identified as the 4-D weight with the FEWEST input channels: every
    internal conv works on model_channels (64+), so the conditioning stack is unambiguous.
    """
    out = {"in_channels": None, "lr_n": None, "sea_ice": "unknown", "conv": None}
    if not path:
        return out
    try:
        from physicsnemo import Module  # noqa: PLC0415

        model = Module.from_checkpoint(str(path))
        convs = [(n, tuple(p.shape)) for n, p in model.named_parameters() if p.ndim == 4]
        if not convs:
            out["error"] = "no 4-D parameters in checkpoint"
            return out
        name, shape = min(convs, key=lambda ns: ns[1][1])
        out["conv"], out["in_channels"] = name, int(shape[1])
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        return out
    out.update(channels_from_in(out["in_channels"], stage))
    return out


def channels_from_in(in_channels: int, stage: str) -> dict:
    """Invert the channel arithmetic: input conv width -> LR channel count -> sea ice yes/no.

    Anything other than 12 (all channels) or 11 (the *_noice set) reports "unknown" rather than
    guessing -- which is the correct answer for the channel-ablation variants.
    """
    off = _STAGE_OFFSET.get(stage)
    if off is None or in_channels is None:
        return {"lr_n": None, "sea_ice": "unknown"}
    lr_n = in_channels - off
    return {"lr_n": lr_n,
            "sea_ice": "yes" if lr_n == _FULL_LR else "no" if lr_n == _NOICE_LR else "unknown"}


def nc_provenance(path: str) -> dict:
    """Channel names from a generate NetCDF's `input` group -- names, not arithmetic.

    `NetCDFWriter` creates one variable per `dataset.input_channels()`, so this states the
    channel set outright instead of inferring it. It corroborates rather than replaces the
    conv-width reading: the conv is a property of THIS checkpoint, whereas generate.py writes to
    one fixed corrdiff_output.nc that a later run overwrites (audit P2-4), so a recorded NetCDF
    path may no longer belong to this run. Disagreement between the two is itself a signal.
    """
    if not path or not Path(path).is_file():
        return {}
    try:
        import xarray as xr  # noqa: PLC0415

        with xr.open_dataset(path, group="input") as inp:
            names = [str(v) for v in inp.data_vars]
    except Exception as e:
        return {"nc_error": f"{type(e).__name__}: {e}"}
    lr = [n for n in names if n != "lsm"]
    return {"nc_input_channels": names, "nc_lr_n": len(lr),
            "nc_sea_ice": "yes" if "siconc" in lr else "no"}


def hydra_provenance(path: str) -> dict:
    """Pull the resolved config a run actually used, rather than what someone typed afterwards.

    `path` is the Hydra snapshot dir the SLURM scripts write ($REPO/logs/hydra/<jobid>) or the
    config.yaml inside it. Hand-entered provenance is how `regression_1` ended up with a config
    field nothing wrote and nobody can now vouch for.
    """
    if not path:
        return {}
    import yaml  # noqa: PLC0415

    p = Path(path)
    for cand in (p, p / "config.yaml", p / ".hydra" / "config.yaml"):
        if cand.is_file():
            cfg = yaml.safe_load(cand.read_text()) or {}
            ds, tr = cfg.get("dataset") or {}, cfg.get("training") or {}
            return {
                "hydra_config": str(cand),
                "model_name": (cfg.get("model") or {}).get("name"),
                "lr_channels": ds.get("lr_channels"),
                "include_lsm": ds.get("include_lsm"),
                "years": ds.get("years"),
                "stats_path": ds.get("stats_path"),
                "data_path": ds.get("data_path"),
                "training_duration": (tr.get("hp") or {}).get("training_duration"),
                "total_batch_size": (tr.get("hp") or {}).get("total_batch_size"),
            }
    print(f"  WARNING: no config.yaml under {path}", file=sys.stderr)
    return {}


def upsert_csv(row: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = []
    if CSV_PATH.exists():
        with open(CSV_PATH, newline="") as f:
            rows = [r for r in csv.DictReader(f) if r.get("run") != row["run"]]
    rows.append(row)
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="run name, e.g. regression_1 / diffusion_2")
    ap.add_argument("--tensorboard", help="TensorBoard log dir (usually OUTPUT_DIR/tensorboard)")
    ap.add_argument("--nc", help="generate.py output NetCDF (for the sample plot + metrics)")
    ap.add_argument("--checkpoint", help="checkpoint path, for the record")
    ap.add_argument("--train-samples", help="training length in processed samples")
    ap.add_argument("--notes", help="free-text notes")
    ap.add_argument("--error", default="sigma", help="error mode for the sample plot")
    ap.add_argument("--time", type=int, default=0, help="time index for the sample")
    # --- provenance: what was this run, exactly? ------------------------------------------
    ap.add_argument("--config",
                    help="Hydra config name used, e.g. config_training_era5_carra2_mini_regression")
    ap.add_argument("--hydra-config", default="",
                    help="$REPO/logs/hydra/<jobid> snapshot dir (or its config.yaml) -- the "
                         "RESOLVED config the run actually used; preferred over --config")
    ap.add_argument("--data", help="data dir the run read (e.g. $PROJECT/data/derot)")
    ap.add_argument("--stats", help="normalization stats JSON used")
    args = ap.parse_args()

    rdir = RESULTS / args.name
    rdir.mkdir(parents=True, exist_ok=True)
    stage = ("diffusion" if args.name.startswith("diffusion")
             else "regression" if args.name.startswith("regression") else "other")

    prev = previous_collect(rdir)
    carried = []

    def keep(name, supplied):
        """Supplied value wins; otherwise carry forward what the last collect recorded.

        collect_run has always been documented as idempotent, but that only held if you passed
        every original argument again -- omitting --nc replaced the row's metrics with blanks.
        Re-passing --nc is its own hazard, since generate.py writes one fixed corrdiff_output.nc
        that the next generation overwrites, so it may no longer hold this run's output. Topping
        up provenance must therefore not require re-supplying (or re-deriving) everything else.
        """
        if supplied is not None:
            return supplied
        was = prev_field(prev, name)
        if was:
            carried.append(name)
        return was

    checkpoint = keep("checkpoint", args.checkpoint)
    tensorboard = keep("tensorboard", args.tensorboard)
    nc = keep("nc", args.nc)
    train_samples = keep("train_samples", args.train_samples)
    notes = keep("notes", args.notes)
    cfg_name = keep("config", args.config)
    data_dir = keep("data", args.data)
    stats_path = keep("stats", args.stats)

    # Only EXPLICIT arguments redo work. A carried-over path is recorded but not re-plotted:
    # the figure already exists, and $SCRATCH tensorboard dirs get purged.
    if args.tensorboard:
        subprocess.run([sys.executable, str(HERE / "plot_losses.py"),
                        "--logdir", args.tensorboard,
                        "--out", str(rdir / "loss_curve.png"), "--logy"], check=True)

    metrics = prev.get("metrics") or {}
    if args.nc:
        subprocess.run([sys.executable, str(HERE / "plot_sample_native.py"),
                        "--nc", args.nc, "--out", str(rdir / "sample_native.png"),
                        "--metrics-out", str(rdir / "metrics.json"),
                        "--error", args.error, "--time", str(args.time)], check=True)
        metrics = json.load(open(rdir / "metrics.json")).get("channels", {})

    ck = checkpoint_provenance(checkpoint, stage)
    nc_p = nc_provenance(nc)
    hy = hydra_provenance(args.hydra_config)
    if nc_p.get("nc_sea_ice") and ck["sea_ice"] != "unknown" \
            and nc_p["nc_sea_ice"] != ck["sea_ice"]:
        print(f"  WARNING: checkpoint says sea_ice={ck['sea_ice']} but the NetCDF input group "
              f"says {nc_p['nc_sea_ice']}.\n           generate.py reuses one fixed output "
              f"path, so the NetCDF may belong to another run; trust the checkpoint.",
              file=sys.stderr)
    prov = {"config": cfg_name or hy.get("hydra_config", ""),
            "data": data_dir, "stats": stats_path or hy.get("stats_path", ""),
            "git": git_hash(), "versions": env_versions(), **ck, **nc_p, **hy}
    if carried:
        print(f"\ncarried forward from the previous collect: {', '.join(sorted(set(carried)))}")

    print("\nprovenance")
    print(f"  in_channels : {ck['in_channels']}  ({ck.get('conv') or 'n/a'})")
    print(f"  lr channels : {ck['lr_n']}      sea ice: {ck['sea_ice']}")
    if ck.get("error"):
        print(f"  NOTE: could not read the checkpoint ({ck['error']}).")
        print("        Channel provenance is the one field that cannot be faked -- "
              "re-run in corrdiff-env with --checkpoint to capture it.")
    if nc_p.get("nc_input_channels"):
        print(f"  nc inputs   : {nc_p['nc_lr_n']} LR + lsm, sea ice: {nc_p['nc_sea_ice']}")
    if hy.get("lr_channels") is not None:
        print(f"  lr_channels : {hy['lr_channels']}")

    info = {"run": args.name, "stage": stage,
            "date": datetime.date.today().isoformat(),
            "checkpoint": checkpoint, "train_samples": train_samples,
            "nc": nc, "tensorboard": tensorboard,
            "provenance": prov, "git": prov["git"],
            "notes": notes, "metrics": metrics}
    with open(rdir / "run_info.json", "w") as f:
        json.dump(info, f, indent=2)

    def g(ch, k):
        return metrics.get(ch, {}).get(k, "")
    row = {"run": args.name, "stage": stage, "date": info["date"],
           "config": Path(prov["config"]).name if prov["config"] else "",
           "sea_ice": ck["sea_ice"], "lr_n": ck["lr_n"] if ck["lr_n"] is not None else "",
           "in_channels": ck["in_channels"] if ck["in_channels"] is not None else "",
           "train_samples": train_samples, "checkpoint": checkpoint,
           "git": info["git"], "notes": notes}
    for ch in ["t2m", "u10", "v10"]:
        row[f"rmse_{ch}"] = g(ch, "rmse")
        row[f"nrmse_{ch}_pct"] = g(ch, "rmse_over_sigma_pct")
        row[f"bias_{ch}"] = g(ch, "bias")
        row[f"ens_meanvar_{ch}"] = g(ch, "ensemble_mean_var")   # blank for deterministic runs
    row["ens_members"] = next((metrics[ch]["ensemble_n"] for ch in ["t2m", "u10", "v10"]
                               if metrics.get(ch, {}).get("ensemble_n")), "")
    upsert_csv(row)

    print(f"\ncollected -> {rdir}")
    print(f"logged    -> {CSV_PATH}")


if __name__ == "__main__":
    main()
