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

> **TODO (audit P2-2):** the CRPS estimator has **no unit tests**. The sorted-order identity in
> `crps_ensemble_map` deserves a brute-force cross-check, plus the `M=1 → MAE` identity and the
> analytic Gaussian CRPS. It is a headline number in the thesis; the `reg` pass reporting
> `crps == mae` exactly is consistent with the identity but is not a test.

## Run it (on fir)

```bash
# from $REPO, submit via slurm/submit.sh so logs go to $REPO/logs
NAME=diffusion_2 N=100 NUM_ENS=15 SEED=0 YEARS=2019 \
  REG_RUN=$SCRATCH/corrdiff_runs/regression_2 \
  RES_RUN=$SCRATCH/corrdiff_runs/diffusion_2 \
  DATA_DIR=$PROJECT/data/derot \
  bash training_mini/slurm/submit.sh evaluate/run_eval.sh
```

## Or pinning exact checkpoints:

```bash
NAME=diffusion_2 N=400 NUM_ENS=32 YEARS="2020 2021" \
  REG_CKPT=/path/to/CorrDiffRegressionUNet.0.NNN.mdlus \
  RES_CKPT=/path/to/EDMPrecondSuperResolution.0.NNN.mdlus \
  DATA_DIR=$PROJECT/data/derot \
  bash training_mini/slurm/submit.sh evaluate/run_eval.sh
```

- **`NAME` is required** — it names the results folder (`results/<NAME>/eval/`). It is *not*
  guessed from the run dirs, so an oddly-named scratch dir can't silently write to the wrong
  place; the job exits immediately if `NAME` is unset.
- **`SEED_BATCH`** is how many ensemble members are denoised in one sampler call. It defaults to
  the largest divisor of `NUM_ENS` that is ≤ 8, so `NUM_ENS=15` runs 3 sampler calls rather than
  15 — the diffusion pass used to be `NUM_ENS` sequential batch-of-one runs of an 18-step
  sampler, which leaves an H100 mostly idle. It **must divide `NUM_ENS`**, and the job refuses to
  start otherwise: physicsnemo sizes the diffusion latents from the conditioning batch rather
  than the seed count, so an uneven split emits more members than requested and then fails
  against the regression mean. Raise it for throughput, lower it if you hit OOM. Picking a
  `NUM_ENS` with plenty of divisors (16, 32) batches better than a prime-ish one (13, 15).
- **`N`** random times are drawn (without replacement) from **`YEARS`** (space-separated,
  default `2019`), with **`SEED`** for reproducibility — the full and regression passes use the
  identical set. Sampling reads the shard's real time index, so every pick is valid and times of
  day are unbiased (a fixed stride would hit only one hour). Cost/disk scale with `N × NUM_ENS`.
- **Two run dirs, not one.** Generation is `regression mean + diffusion residual`, and a
  diffusion run is trained against an *earlier* regression run (WORKFLOW.md §B), so the two nets
  normally live in different directories — a `diffusion_n/` dir holds only
  `checkpoints_diffusion`. Give both: `REG_RUN` (→ `checkpoints_regression/`) and `RES_RUN`
  (→ `checkpoints_diffusion/`). Each picks its highest-step `.mdlus` via `latest_ckpt`
  (`training_mini/slurm/common.sh`), which selects on the `<nimg>` in the filename — *not* on
  modification time, which a checkpoint archive/restore reorders. Pin an exact file with
  `REG_CKPT=` / `RES_CKPT=`. `OUTPUT_DIR=` still works and sets both, for a directory holding
  both nets.
  > Pairing a diffusion net with the **wrong** regression base produces quietly wrong metrics,
  > not an error — naming both runs in the command keeps the pairing visible at the call site.
- Sampling from multiple years (e.g. `YEARS="2018 2019"`) requires those shards in `$DATA_DIR`.
- **Outputs are split by size:**
  - `metrics_crps_mae.json` (small, the thing you keep) -> **`RESULT_DIR`**, default
    `training_mini/results/<NAME>/eval/` (e.g. `results/diffusion_2/eval/`). It's git-trackable —
    commit it with the run's results.
  - `full.nc` / `reg.nc` (several GB) -> **`NC_DIR`**, default `$RES_RUN/eval/` on `$SCRATCH`,
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
source $REPO/training_mini/slurm/common.sh
REG=$(latest_ckpt $SCRATCH/corrdiff_runs/regression_2/checkpoints_regression)
NAME=diffusion_2_test_2020_22 N=400 NUM_ENS=32 REG_CKPT=$REG \
  RES_RUN=$SCRATCH/corrdiff_runs/diffusion_2 \
  DATA_DIR=$PROJECT/data/derot YEARS="2020 2021 2022" \
  bash training_mini/slurm/submit.sh --time=5:30:00 evaluate/run_eval.sh
```

**The payoff:** the `.zip` shards persist in `$PROJECT` (backed up, not purged), so evaluating a
**new** model on the same test years is just step 3 — no re-derotating. `zip_shard.py` works on any
shard, so you can also archive raw/train shards to reclaim inodes.

Shard *location* is shared: `dataloading.dataset` owns `shard_path` / `discover_shards` /
`resolve_stores` / `open_store`, and the dataset loader, `make_stats.py`, `sample_times.py`,
`verify_shards.py`, `derotate_winds.py` and `trim_shard.py` all go through them. Per year the
`.zip` wins when present, otherwise the loose `.zarr` is used — so a data dir can freely mix
archived test shards with loose train shards, and no tool silently skips an archived year.

`STAGE=1` staging follows the same rules: `stage_shards` copies whichever form of each year is
present, and the years come from the config rather than a hardcoded list — so **train** shards can
be archived too. That is worth doing: staging nine loose shards measured ~6 min each (~54 min per
submission, at ~13 MB/s — metadata-bound, not bandwidth), which nine single-file archives should
cut to a few minutes.

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
