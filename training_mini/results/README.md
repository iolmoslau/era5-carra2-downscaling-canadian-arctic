# Training-run results

Systematic record of each CorrDiff-Mini training run. One subdirectory per run, named
`regression_<n>` or `diffusion_<n>` (n = run index), plus `runs.csv` as the index/spreadsheet.

```
results/
  runs.csv                 # one row per run (auto-filled metrics + your notes)
  regression_1/
    loss_curve.png         # training vs validation loss (from TensorBoard)
    sample_native.png      # input LR / truth / prediction / error, native grid
    metrics.json           # per-channel diagnostics (RMSE, RMSE/σ, bias, ...)
    run_info.json          # metadata: checkpoint, git commit, data, notes
  diffusion_1/
    ...
```

## Populate a run (one command)
After a run finishes training and you've generated a sample NetCDF (`generate.sh`), run — in
an env with tensorboard **and** cartopy (i.e. `corrdiff-env` once `setup_env.sh` has added
cartopy):

```bash
cd $REPO/training_mini
python tools/collect_run.py --name regression_1 \
  --tensorboard $SCRATCH/corrdiff_mini_derot/tensorboard \
  --nc corrdiff_output.nc \
  --checkpoint $SCRATCH/corrdiff_mini_derot/checkpoints_regression/CorrDiffRegressionUNet.0.800000.mdlus \
  --train-samples 800000 --error sigma \
  --notes "de-rotated winds; train 2011-18 / val 2019; 2x H100"
```

This creates `results/regression_1/` with the three artifacts and upserts the `runs.csv` row
(safe to re-run — it replaces the row for that run name). Each run's TensorBoard logs live in
its own `OUTPUT_DIR/tensorboard` (train.py writes them under `checkpoint_dir`), so loss curves
are per-run.

## runs.csv columns
- `run, stage, date, train_samples, checkpoint, git, notes` — identity, training length (in
  processed samples), and your free-text notes.
- `rmse_<ch>, nrmse_<ch>_pct, bias_<ch>` — per-channel absolute RMSE, RMSE/σ (%), and mean bias
  for t2m / u10 / v10, auto-filled from `metrics.json`.
- `ens_members, ens_meanvar_<ch>` — ensemble size and the per-channel spatial-mean variance of the
  generated distribution (diffusion runs with `NUM_ENS>1`; blank for deterministic runs). The
  `sample_native.png` also gets an "ensemble std" column showing where the model is uncertain.

## Adding diagnostics later
`metrics.json` is an extensible dict keyed by channel. To add a metric (e.g. spatial-spectrum
score, CRPS for diffusion ensembles), compute it in `tools/plot_sample_native.py` (add to the
`metrics[v]` dict) and add matching columns to `FIELDS` in `tools/collect_run.py`.
