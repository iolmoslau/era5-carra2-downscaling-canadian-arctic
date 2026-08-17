# evaluate/ — CRPS & MAE for CorrDiff vs. the regression-only net

Scores the trained model on a real sample of the 2019 validation year (not one or two
handpicked days), on the metrics the CorrDiff paper uses: **CRPS** and **MAE**, per channel.

## What it does

`run_eval.sh` draws **`N` random times** from the requested year(s) (`sample_times.py`, seeded),
runs `generate.py` twice over that **same** set of times, then `eval_crps_mae.py`:

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
NAME=diffusion_2 N=100 NUM_ENS=15 SEED=0 YEARS=2019 \
  OUTPUT_DIR=$SCRATCH/corrdiff_runs/diffusion_2 \
  DATA_DIR=$PROJECT/data/derot \
  bash training_mini/slurm/submit.sh evaluate/run_eval.sh
```

## Or specifying regression checkpoint:

```bash
REG=[path to regression checkpoint here]
NAME=diffusion_2 N=400 NUM_ENS=32 REG_CKPT=$REG YEARS="2020 2021" \
  OUTPUT_DIR=$SCRATCH/corrdiff_runs/diffusion_2 \
  DATA_DIR=$PROJECT/data/derot \
  bash training_mini/slurm/submit.sh evaluate/run_eval.sh
```

- **`NAME` is required** — it names the results folder (`results/<NAME>/eval/`). It is *not*
  guessed from `OUTPUT_DIR`, so an oddly-named scratch dir can't silently write to the wrong
  place; the job exits immediately if `NAME` is unset.
- **`N`** random times are drawn (without replacement) from **`YEARS`** (space-separated,
  default `2019`), with **`SEED`** for reproducibility — the full and regression passes use the
  identical set. Sampling reads the shard's real time index, so every pick is valid and times of
  day are unbiased (a fixed stride would hit only one hour). Cost/disk scale with `N × NUM_ENS`.
- Checkpoints are auto-picked as the highest-step `.mdlus` in
  `$OUTPUT_DIR/checkpoints_{regression,diffusion}`. Override with `REG_CKPT=` / `RES_CKPT=` to
  pair a diffusion run with a regression checkpoint from a **different** run dir.
- Sampling from multiple years (e.g. `YEARS="2018 2019"`) requires those shards in `$DATA_DIR`.
- **Outputs are split by size:**
  - `metrics_crps_mae.json` (small, the thing you keep) -> **`RESULT_DIR`**, default
    `training_mini/results/<NAME>/eval/` (e.g. `results/diffusion_2/eval/`). It's git-trackable —
    commit it with the run's results.
  - `full.nc` / `reg.nc` (several GB) -> **`NC_DIR`**, default `$OUTPUT_DIR/eval/` on `$SCRATCH`,
    since they're too big for the `$HOME` repo quota and `.nc` is gitignored anyway. They're
    intermediates — safe to delete after the metrics are computed. Set `NC_DIR=$RESULT_DIR` to
    force them alongside the metrics.

## Persisting de-rotated test shards (avoid the `$PROJECT` inode quota)

A loose `shard_YYYY.zarr` is ~5,900 tiny chunk files; a dozen shards exhausts the project
file-count (inode) quota. Archive each de-rotated test shard to a **single-file zarr ZipStore**
(1 inode, same ~4.7 GB) — the loader opens `shard_YYYY.zarr.zip` transparently, and reads are
byte-identical and just as fast (chunks are `ZIP_STORED`, so seek+read with no recompression).

```bash
# 1. de-rotate the test years to scratch (fast FS; SKIP_STATS keeps the train stats fixed)
DST_DIR=$SCRATCH/data/derot YEARS="2020 2021 2022" SKIP_STATS=1 \
  bash training_mini/slurm/submit.sh scripts/derotate_shards.sh

# 2. archive each into $PROJECT as a 1-inode zip, dropping the loose scratch copy (login node OK)
for y in 2020 2021 2022; do
  python scripts/zip_shard.py --src $SCRATCH/data/derot/shard_$y.zarr \
    --dst $PROJECT/data/derot/shard_$y.zarr.zip --remove-src
done

# 3. eval -- the loader picks up shard_YYYY.zarr.zip automatically; STATS resolves from DATA_DIR
REG=$(ls $SCRATCH/corrdiff_runs/regression_2/checkpoints_regression/*.mdlus | sort -t. -k3 -n | tail -1)
NAME=diffusion_2_test_2020_22 N=400 NUM_ENS=32 REG_CKPT=$REG \
  OUTPUT_DIR=$SCRATCH/corrdiff_runs/diffusion_2 \
  DATA_DIR=$PROJECT/data/derot YEARS="2020 2021 2022" \
  bash training_mini/slurm/submit.sh --time=4:00:00 evaluate/run_eval.sh
```

**The payoff:** the `.zip` shards persist in `$PROJECT` (backed up, not purged), so evaluating a
**new** model on the same test years is just step 3 — no re-derotating. `_resolve_stores` prefers
the `.zip` when present but still reads a loose `.zarr` if that's what's there, so a data dir can
mix archived test shards with loose train shards. `zip_shard.py` works on any shard, so you can
also archive raw/train shards to reclaim inodes.

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
