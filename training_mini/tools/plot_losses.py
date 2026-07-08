#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Plot training + validation loss curves from the CorrDiff TensorBoard logs.

train.py writes `training_loss`, `training_loss_running_mean`, `validation_loss` and
`learning_rate` to TensorBoard (`training_mini/tensorboard/`). Resumed runs append new
event files; since the step axis is the global sample count, we merge all event files.

    # run from training_mini/ (CPU only; no GPU needed)
    python tools/plot_losses.py --logdir tensorboard --out loss_curves.png --logy
"""

from __future__ import annotations

import argparse
import glob
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402


def read_scalars(logdir: str) -> dict:
    files = sorted(glob.glob(os.path.join(logdir, "**", "events.out.tfevents.*"), recursive=True))
    if not files:
        raise SystemExit(f"No TensorBoard event files under {logdir!r}. "
                         "Run from training_mini/ or pass --logdir.")
    series: dict = defaultdict(dict)  # tag -> {step: value} (later files win on dup steps)
    for f in files:
        ea = EventAccumulator(f, size_guidance={"scalars": 0})
        ea.Reload()
        for tag in ea.Tags().get("scalars", []):
            for e in ea.Scalars(tag):
                series[tag][e.step] = e.value
    return series


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--logdir", default="tensorboard")
    ap.add_argument("--out", default="loss_curves.png")
    ap.add_argument("--logy", action="store_true", help="log-scale the loss axis")
    args = ap.parse_args()

    s = read_scalars(args.logdir)
    print("scalar tags found:", sorted(s))

    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False
    for tag, style, kw in [
        ("training_loss", "-", dict(alpha=0.4, linewidth=0.8)),
        ("training_loss_running_mean", "-", dict(linewidth=2.0)),
        ("validation_loss", "o-", dict(markersize=4, linewidth=1.5)),
    ]:
        if tag in s and s[tag]:
            steps = sorted(s[tag])
            ax.plot(steps, [s[tag][k] for k in steps], style, label=tag, **kw)
            plotted = True
    if not plotted:
        raise SystemExit(f"No loss tags to plot; found {sorted(s)}")

    ax.set_xlabel("samples processed")
    ax.set_ylabel("loss (summed over pixels/channels)")
    if args.logy:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title("CorrDiff-Mini regression: training vs validation loss")
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
