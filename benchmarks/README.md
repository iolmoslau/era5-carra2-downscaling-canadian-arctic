# benchmarks — what CorrDiff gets compared against

`evaluate/` *scores* models and `scripts/` is operational tooling. A benchmark **is a model**, so
it lives here.

## The one rule

Every benchmark emits the **`NetCDFWriter` schema** that `generate.py` already writes:

```
root        lat(y,x)  lon(y,x)  time(time)
truth/      <channel>(time, y, x)
prediction/ <channel>(ensemble, time, y, x)
input/      <channel>(time, y, x)            # optional
```

Hit that and nothing in the evaluation code needs to change: `evaluate/eval_crps_mae.py` and
`evaluate/plot_pdfs.py` already take `--nc label=path` pairs, and the CRPS estimator collapses to
MAE at `ensemble=1`, so a deterministic benchmark lands on the same axis as a 32-member diffusion
ensemble with no special-casing.

Two further conditions make a comparison fair, and neither is enforced by the schema:

- **The same times.** Take them from the existing `full.nc`'s `time` variable rather than
  re-deriving them, so identity is a fact rather than a reproduced procedure.
- **Fit on train years only.** 2011–2018, matching `stats_train_2011_2018.json`. A benchmark
  tuned on 2020–22 would look excellent and mean nothing.

## Contents

| file | what |
|---|---|
| `inspect_caps.sh` | interrogate the CAPS archive: variables, units, grid, **wind frame**. Run this before writing any adapter. |
| `check_caps_coverage.py` | does the source domain contain our patch, and how much de-rotation does it need? Measures both from one `grib_get_data` dump. |
| `plot_caps_vs_patch.py` | draw the two footprints together for one valid time, in CAPS's own projection. |
| `coarsen.py` | footprint-mean HR→LR (`build_binning`/`coarsen`, now also used by `scripts/lr_hr_consistency.py`) and the model's own bilinear LR→HR (`LRtoHR`). One implementation of each direction. |
| `write_nc.py` | the schema emitter, shared by every benchmark; `read_times` takes the times from a model's NetCDF. |
| `bcsd.py` | Bias-Correction Spatial Disaggregation (Wood et al. 2004). `fit` on the train years, `predict` onto the test times. CPU-only. |
| `run_bcsd.sh` | fit → predict → score as one CPU SLURM job on fir. |

### Finding, 2026-11-10: the supplied CAPS extraction does not cover the patch

**Only 25.5% of the 448x448 patch falls inside the CAPS domain** — a band across the north,
`lat 67.94..72.42`, `lon 215.33..242.98`. The rest is up to **799 km** outside. Confirmed two
independent ways: measured from the delivered file's own coordinates, and reproduced to within
0.01% by reconstructing the grid analytically from the documented parameters (polar
stereographic, `lon_0 = 244`, `LaD = 60`, `Dx = Dy = 2500 m`, 1465 x 825, first point
67.960/219.239). So it is the delivered geometry, not a parsing error.

Everything *else* about the data is right: `2t`/`10u`/`10v`, K and m s-1, **2.5 km — the same
resolution as CARRA2**, 3-hourly valid times across 2020-2022. Only the domain placement is
wrong, which is why re-requesting the extraction is worth trying before redesigning around it.

Planned, per the roadmap:

| file | what |
|---|---|
| `adapt_caps.py` | CAPS GRIB2 → the schema: pick a lead, regrid to the patch, de-rotate winds if needed |

## BCSD

```bash
# on fir: fit (2011-18) -> predict on the model's exact test times -> score
NAME=bcsd TIMES_FROM=$SCRATCH/corrdiff_runs/diffusion_2/eval/full.nc \
  bash training_mini/slurm/submit.sh benchmarks/run_bcsd.sh
# if that full.nc has been purged, re-draw the same times as run_eval.sh did
NAME=bcsd YEARS="2020 2021 2022" N=400 SEED=0 \
  bash training_mini/slurm/submit.sh benchmarks/run_bcsd.sh
```

**What it does.** *BC*: CARRA2 is footprint-mean coarsened onto the ERA5 grid, and at each LR
cell ERA5 is quantile-mapped onto that coarse CARRA2 distribution, per calendar month (±1 month
pooled, ~700 samples per map). *SD*: the corrected coarse field becomes an anomaly from the coarse
CARRA2 climatology, the anomaly is interpolated with the model's own bilinear weights, and the
2.5 km CARRA2 climatology (month × hour of day) is added back (t2m) or multiplied in (wind speed).
All fine-scale structure therefore comes from climatology. That is the method's bet, and also its
limit: detail that isn't climatological, like a front or one night's cold pool, can't appear.

**Winds are speed + direction, not u + v**, as the roadmap required. Speed is quantile-mapped and
disaggregated multiplicatively. Direction gets a per-cell, per-month mean rotation (the same
magnitude-weighted circular estimator as the consistency check), then a per-pixel climatological
rotation of HR wind against the interpolated coarse wind, which is where terrain channelling
lives.

**Guards.** `predict` refuses a store whose `winds_frame` differs from the fit's. With
`--times-from`, it compares its truth with that file's `truth` group and fails beyond 1e-2: a
mismatch means BCSD was scored against a different store than the model was. It also warns when
asked to predict on a year it was fitted on.

**Sanity, local 2011 shard (in-sample year, 60 times, so not a result):** t2m MAE 1.77 vs 2.49 for
bilinear ERA5. That shard's winds are still grid-relative, and BCSD's direction correction
absorbed the ~100° rotation completely (u10 MAE 1.00 vs 3.56). `training_mini/tests/test_bcsd.py`
checks out-of-sample recovery of a known bias, rotation and fine-scale pattern on synthetic shards.
Disabling any one correction term fails the test.

## The trap to keep in mind

**Wind frame.** CARRA2's native winds are grid-relative and grid convergence is ~100° over this
domain; mixing frames cost a full retrain once already, and it does not show up in RMSE — only in
direction. Any new wind source gets its frame confirmed from the file, not from documentation.
`inspect_caps.sh` prints `winds(grid)`/`winds(earth)` (wgrib2) or `uvRelativeToGrid`
(eccodes: 1 = grid, 0 = earth) for exactly this reason.
