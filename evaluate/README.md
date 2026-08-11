# evaluate/ — CRPS & MAE for CorrDiff vs. the regression-only net

Scores the trained model on a real sample of the 2019 validation year (not one or two
handpicked days), on the metrics the CorrDiff paper uses: **CRPS** and **MAE**, per channel.

## What it does

`run_eval.sh` runs `generate.py` twice over the same `N` 2019 times, then `eval_crps_mae.py`:

| run | `generate.py` mode | prediction | CRPS reduces to |
|-----|--------------------|------------|-----------------|
| **full** | `all` (regression mean + diffusion residual) | `NUM_ENS`-member ensemble | the ensemble CRPS |
| **reg**  | `regression` (deterministic mean) | 1 member | **MAE** (CRPS == MAE for a point forecast) |

So the regression-only net and the full diffusion model are scored on the **same axis** — a
deterministic forecast is just a 1-member ensemble.

### Metrics (per channel, averaged over all pixels and all times, in physical units)
- **`crps`** — ensemble CRPS, NRG estimator: `(1/M)Σ|xᵢ−y| − 1/(2M²)ΣΣ|xᵢ−xⱼ|`. First term =
  accuracy, second term = credit for spread. For `M=1` it is exactly the MAE.
- **`mae`** — MAE of the ensemble mean (the point forecast).
- **`rmse`** — RMSE of the ensemble mean (alongside; not the primary score).

The CRPS estimator is unit-tested against a brute-force pairwise computation, the `M=1 → MAE`
identity, and the analytic Gaussian CRPS.

## Run it (on fir)

```bash
# from $REPO, submit via slurm/submit.sh so logs go to $REPO/logs
N=100 NUM_ENS=15 \
  OUTPUT_DIR=$SCRATCH/corrdiff_runs/diffusion_2 \
  DATA_DIR=$PROJECT/data/derot \
  bash training_mini/slurm/submit.sh evaluate/run_eval.sh
```
- Checkpoints are auto-picked as the highest-step `.mdlus` in
  `$OUTPUT_DIR/checkpoints_{regression,diffusion}`. Override with `REG_CKPT=` / `RES_CKPT=` to
  pair a diffusion run with a regression checkpoint from a **different** run dir.
- `N` is the approximate number of 2019 times (the interval is snapped to a multiple of 3 h).
  Cost and disk scale with `N × NUM_ENS`.
- Outputs land in `$RESULT_DIR` (default `$OUTPUT_DIR/eval/`): `full.nc`, `reg.nc`,
  `metrics_crps_mae.json`.

## Score existing NetCDFs directly

`eval_crps_mae.py` is pure numpy/xarray (no GPU, no cartopy) — run it on a login node or
locally on any `generate.py` output, e.g. the `corrdiff_output.nc` you already have:

```bash
python evaluate/eval_crps_mae.py --nc full=full.nc reg=reg.nc --out metrics_crps_mae.json
```

## Baselines — TODO

The CorrDiff paper also compares against **bilinear/linear interpolation** and a **random
forest**. You wanted to reconsider these baselines before implementing them, so they're not
here yet. Adding one is just another labelled NetCDF with a `prediction` group in the same
format, then `--nc interp=interp.nc` alongside the others — the metric code is baseline-agnostic.
