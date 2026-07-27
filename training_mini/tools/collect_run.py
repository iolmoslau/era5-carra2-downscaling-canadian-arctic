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
        --error sigma --notes "de-rotated winds, 800k samples, 2xH100"
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
FIELDS = ["run", "stage", "date", "checkpoint", "git", "notes",
          "rmse_t2m", "nrmse_t2m_pct", "bias_t2m",
          "rmse_u10", "nrmse_u10_pct", "bias_u10",
          "rmse_v10", "nrmse_v10_pct", "bias_v10"]


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=str(TRAIN_DIR), text=True).strip()
    except Exception:
        return ""


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
    ap.add_argument("--notes", default="", help="free-text notes")
    ap.add_argument("--error", default="sigma", help="error mode for the sample plot")
    ap.add_argument("--time", type=int, default=0, help="time index for the sample")
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

    info = {"run": args.name, "stage": stage,
            "date": datetime.date.today().isoformat(),
            "checkpoint": args.checkpoint, "nc": args.nc or "",
            "tensorboard": args.tensorboard or "", "git": git_hash(),
            "notes": args.notes, "metrics": metrics}
    with open(rdir / "run_info.json", "w") as f:
        json.dump(info, f, indent=2)

    def g(ch, k):
        return metrics.get(ch, {}).get(k, "")
    row = {"run": args.name, "stage": stage, "date": info["date"],
           "checkpoint": args.checkpoint, "git": info["git"], "notes": args.notes}
    for ch in ["t2m", "u10", "v10"]:
        row[f"rmse_{ch}"] = g(ch, "rmse")
        row[f"nrmse_{ch}_pct"] = g(ch, "rmse_over_sigma_pct")
        row[f"bias_{ch}"] = g(ch, "bias")
    upsert_csv(row)

    print(f"\ncollected -> {rdir}")
    print(f"logged    -> {CSV_PATH}")


if __name__ == "__main__":
    main()
