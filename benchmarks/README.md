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
| `coarsen.py` | binned footprint-mean HR→LR, promoted out of `scripts/lr_hr_consistency.py` so there is one implementation |
| `write_nc.py` | the schema emitter above, shared by every benchmark |
| `bcsd.py` | Bias-Correction Spatial Disaggregation (Wood et al. 2004), CPU-only |
| `adapt_caps.py` | CAPS GRIB2 → the schema: pick a lead, regrid to the patch, de-rotate winds if needed |

## The trap to keep in mind

**Wind frame.** CARRA2's native winds are grid-relative and grid convergence is ~100° over this
domain; mixing frames cost a full retrain once already, and it does not show up in RMSE — only in
direction. Any new wind source gets its frame confirmed from the file, not from documentation.
`inspect_caps.sh` prints `winds(grid)`/`winds(earth)` (wgrib2) or `uvRelativeToGrid`
(eccodes: 1 = grid, 0 = earth) for exactly this reason.
