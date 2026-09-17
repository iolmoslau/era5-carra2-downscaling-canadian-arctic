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

# Channel arithmetic, mirroring how train.py sizes the input conv:
#     in_channels = len(lr_channels) + include_lsm + N_grid_channels
#                   + img_out_channels   (only when hr_mean_conditioning)
# regression (hr_mean_conditioning false): in = L + 1 + 4     = L + 5
# diffusion  (hr_mean_conditioning true):  in = L + 1 + 4 + 3 = L + 8
# Assumes the shipped defaults include_lsm=true, N_grid_channels=4, 3 HR outputs.
_STAGE_OFFSET = {"regression": 5, "diffusion": 8}
_FULL_LR, _NOICE_LR = 12, 11


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=str(TRAIN_DIR), text=True).strip()
    except Exception as e:
        print(f"  WARNING: git hash unavailable ({type(e).__name__}) -- run recorded without it",
              file=sys.stderr)
        return ""


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
    ap.add_argument("--checkpoint", default="", help="checkpoint path, for the record")
    ap.add_argument("--train-samples", default="", help="training length in processed samples")
    ap.add_argument("--notes", default="", help="free-text notes")
    ap.add_argument("--error", default="sigma", help="error mode for the sample plot")
    ap.add_argument("--time", type=int, default=0, help="time index for the sample")
    # --- provenance: what was this run, exactly? ------------------------------------------
    ap.add_argument("--config", default="",
                    help="Hydra config name used, e.g. config_training_era5_carra2_mini_regression")
    ap.add_argument("--hydra-config", default="",
                    help="$REPO/logs/hydra/<jobid> snapshot dir (or its config.yaml) -- the "
                         "RESOLVED config the run actually used; preferred over --config")
    ap.add_argument("--data", default="", help="data dir the run read (e.g. $PROJECT/data/derot)")
    ap.add_argument("--stats", default="", help="normalization stats JSON used")
    args = ap.parse_args()

    rdir = RESULTS / args.name
    rdir.mkdir(parents=True, exist_ok=True)
    stage = ("diffusion" if args.name.startswith("diffusion")
             else "regression" if args.name.startswith("regression") else "other")

    if args.tensorboard:
        subprocess.run([sys.executable, str(HERE / "plot_losses.py"),
                        "--logdir", args.tensorboard,
                        "--out", str(rdir / "loss_curve.png"), "--logy"], check=True)

    metrics = {}
    if args.nc:
        subprocess.run([sys.executable, str(HERE / "plot_sample_native.py"),
                        "--nc", args.nc, "--out", str(rdir / "sample_native.png"),
                        "--metrics-out", str(rdir / "metrics.json"),
                        "--error", args.error, "--time", str(args.time)], check=True)
        metrics = json.load(open(rdir / "metrics.json")).get("channels", {})

    ck = checkpoint_provenance(args.checkpoint, stage)
    hy = hydra_provenance(args.hydra_config)
    prov = {"config": args.config or hy.get("hydra_config", ""),
            "data": args.data, "stats": args.stats or hy.get("stats_path", ""),
            "git": git_hash(), "versions": env_versions(), **ck, **hy}

    print("\nprovenance")
    print(f"  in_channels : {ck['in_channels']}  ({ck.get('conv') or 'n/a'})")
    print(f"  lr channels : {ck['lr_n']}      sea ice: {ck['sea_ice']}")
    if ck.get("error"):
        print(f"  NOTE: could not read the checkpoint ({ck['error']}).")
        print("        Channel provenance is the one field that cannot be faked -- "
              "re-run in corrdiff-env with --checkpoint to capture it.")
    if hy.get("lr_channels") is not None:
        print(f"  lr_channels : {hy['lr_channels']}")

    info = {"run": args.name, "stage": stage,
            "date": datetime.date.today().isoformat(),
            "checkpoint": args.checkpoint, "train_samples": args.train_samples,
            "nc": args.nc or "", "tensorboard": args.tensorboard or "",
            "provenance": prov, "git": prov["git"],
            "notes": args.notes, "metrics": metrics}
    with open(rdir / "run_info.json", "w") as f:
        json.dump(info, f, indent=2)

    def g(ch, k):
        return metrics.get(ch, {}).get(k, "")
    row = {"run": args.name, "stage": stage, "date": info["date"],
           "config": Path(prov["config"]).name if prov["config"] else "",
           "sea_ice": ck["sea_ice"], "lr_n": ck["lr_n"] if ck["lr_n"] is not None else "",
           "in_channels": ck["in_channels"] if ck["in_channels"] is not None else "",
           "train_samples": args.train_samples, "checkpoint": args.checkpoint,
           "git": info["git"], "notes": args.notes}
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
