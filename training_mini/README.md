# training_mini — CorrDiff-Mini for ERA5 → CARRA2 downscaling

Trains a lightweight **CorrDiff-Mini** diffusion super-resolution model that downscales
projected ERA5 reanalysis (coarse, 0.25°) to high-resolution CARRA2 (2.5 km) over the
Canadian Arctic patch, using this repo's built zarr shards.

This directory **vendors** NVIDIA PhysicsNeMo's CorrDiff example and adds a thin adapter so
the existing data pipeline (`dataloading/`) plugs straight in.

## What's vendored vs. ours

Vendored from **NVIDIA/physicsnemo** `examples/weather/corrdiff` at commit
**`33ce3ec0ca65331f46ed0cbf131a881484a813ee`** (Apache-2.0): `train.py`, `generate.py`,
`helpers/`, `datasets/` (the reader package — `dataset.py` imports the sibling readers, so
they're kept as a unit), `conf/base/`, and `requirements.txt`. Unused example extras from
upstream (`inference/`, `score_samples.py`, upstream tests, and the other datasets'
top-level `config_*` entry files) were pruned to keep this training-only.

Ours:
- `datasets/era5_carra2.py` — `ERA5CARRA2Dataset` (CorrDiff `DownscalingDataset` adapter) +
  `LRConditioner` (GPU upsample + static land-sea-mask append).
- `conf/base/dataset/era5_carra2.yaml` + `conf/config_{training,generate}_era5_carra2_*` — Hydra configs.
- `tools/make_stats.py`, `tools/trim_shard.py`; `tests/test_era5_carra2_dataset.py`;
  `slurm/*.sh`; `kaggle/smoke_test.md`.

### Minimal edits to the vendored scripts
`train.py` and `generate.py` each get a small, clearly-commented hook (search
`[ERA5->CARRA2 adapter]`): if the dataset defines `make_conditioner()`, build an
`LRConditioner` on the device and apply it to `img_lr` right after the batch lands on GPU
(before patching / the UNet). This preserves this repo's design of keeping LR **coarse** on
disk and upsampling on GPU (see `dataloading/upsample.py`). No-op for upstream datasets.

## Design in one paragraph
`PatchDataset` serves coarse LR `(12,53,129)` + HR `(3,448,448)`. The adapter's
`__getitem__` returns **normalized coarse LR + HR** as `(img_clean, img_lr)`.
`input_channels()` reports the **post-conditioning** count (selected LR + `lsm`), so the
UNet is sized correctly. On GPU, `LRConditioner` bilinearly upsamples LR to 448×448 (reusing
`BilinearUpsampler`) and appends the normalized static land-sea mask. Because per-channel
normalization is affine and bilinear upsampling is linear, normalizing coarse then
upsampling is identical to the reverse — stats stay exact. The **two model variants** (with
/ without sea ice) differ only by the config's `lr_channels` list (drop `siconc`).

## Data layout & splits
- Shards: `shard_YYYY.zarr` **or** `shard_YYYY.zarr.zip` in one directory
  (`dataset.data_path`), mixed freely — per year, the 1-inode zip archive wins when present.
  `dataloading.dataset` resolves the form and `stage_shards` copies whichever exists, so no
  command changes either way. A loose store is ~5,900 tiny files; see `scripts/zip_shards.sh`.
- **Train 2011–2018, validate 2019, test 2020–2022.** The test years are held out for final
  evaluation only (`evaluate/run_eval.sh`); nothing is fitted on them, stats included.
- Stats: train-only (`tools/make_stats.py` over 2011–2018), one JSON reused by both variants.

## Install (GPU env — fir or Kaggle; NOT the M1 dev box)
```bash
pip install nvidia-physicsnemo
pip install -r ../requirements-train.txt
```

## Workflow (run from this directory)
```bash
# 0. one-time: train-only normalization stats
python tools/make_stats.py --data-dir /path/to/data \
  --years 2011 2012 2013 2014 2015 2016 2017 2018 \
  --out /path/to/data/stats_train_2011_2018.json

# 1. regression (mean predictor)
python train.py --config-name=config_training_era5_carra2_mini_regression
#    -> checkpoints_regression/*.mdlus

# 2. diffusion (residual), pointed at the regression checkpoint
python train.py --config-name=config_training_era5_carra2_mini_diffusion \
  ++training.io.regression_checkpoint_path=/path/to/regression.mdlus

# 3. generate on 2019
python generate.py --config-name=config_generate_era5_carra2_mini \
  ++generation.io.reg_ckpt_filename=/path/to/regression.mdlus \
  ++generation.io.res_ckpt_filename=/path/to/diffusion.mdlus
```
No-sea-ice variant: use the `*_noice` configs (regression + diffusion) — see
**With or without sea ice** below, since inference uses a different knob. Multi-GPU: launch
via `torchrun --standalone --nproc_per_node=<N> train.py ...` (see `slurm/`).

The configs reference `./data`; on the cluster the SLURM scripts symlink it to the real
shard directory.

## On fir

**Always submit through `slurm/submit.sh`, never bare `sbatch`.** SLURM cannot expand `$REPO` in
`#SBATCH` lines, so a bare `sbatch` drops the job's `.out`/`.err` in whatever directory you
launched from. `submit.sh` pins them to `$REPO/logs/` and forwards sbatch flags and env-var
prefixes verbatim. Paths below are from `$REPO`; `WORKFLOW.md` is the full per-run procedure
(train → generate → collect → archive) and this is the short form.

```bash
DATA=$PROJECT/data/derot                  # de-rotated shards (see scripts/derotate_shards.sh)
STATS=$DATA/stats_train_2011_2018.json

# ---- 1. REGRESSION (stage 1: the conditional mean) -------------------------------------------
# Re-submit the identical line to resume from the last checkpoint after a timeout.
OUT=$SCRATCH/corrdiff_runs/regression_3
DATA_DIR=$DATA STATS=$STATS OUTPUT_DIR=$OUT TRAIN_DURATION=1200000 \
  bash training_mini/slurm/submit.sh --gpus-per-node=h100:2 training_mini/slurm/train_regression.sh

# ...WITHOUT sea ice: pick the config. There is no NOICE flag for training (see below).
CONFIG=config_training_era5_carra2_mini_regression_noice \
DATA_DIR=$DATA STATS=$STATS OUTPUT_DIR=$OUT TRAIN_DURATION=1200000 \
  bash training_mini/slurm/submit.sh --gpus-per-node=h100:2 training_mini/slurm/train_regression.sh

# ---- 2. DIFFUSION (stage 2: the residual around that mean) -----------------------------------
# Needs the regression net it is built on, as arg 1 (or REG_CKPT=). That net normally lives in a
# DIFFERENT run directory, so it cannot be auto-discovered from this run's OUTPUT_DIR.
OUT=$SCRATCH/corrdiff_runs/diffusion_3
REG=$SCRATCH/corrdiff_runs/regression_3/checkpoints_regression/CorrDiffRegressionUNet.0.1200000.mdlus

CONFIG=config_training_era5_carra2_mini_diffusion_noice \
DATA_DIR=$DATA STATS=$STATS OUTPUT_DIR=$OUT TRAIN_DURATION=2000000 \
  bash training_mini/slurm/submit.sh --gpus-per-node=h100:2 \
    training_mini/slurm/train_diffusion.sh "$REG"
# The job refuses to start if $REG was trained on different channels than CONFIG declares --
# pairing a no-ice diffusion run with a with-ice regression base is one typo away, and it would
# otherwise fail only after the allocation, the staging pass and the model build.

# ---- 3. GENERATE a sample ---------------------------------------------------------------------
DATA_DIR=$DATA OUTPUT_DIR=$OUT \
  bash training_mini/slurm/submit.sh training_mini/slurm/generate.sh
#    -> training_mini/corrdiff_output.nc  (overwritten each time; collect before the next one)

# ...from a NO-ICE model: NOICE=1 here, not CONFIG (see below)
NOICE=1 DATA_DIR=$DATA OUTPUT_DIR=$OUT \
  bash training_mini/slurm/submit.sh training_mini/slurm/generate.sh

# ---- 4. FULL ensemble, once a diffusion net exists (mean + residual) --------------------------
REG=$SCRATCH/corrdiff_runs/regression_3/checkpoints_regression/CorrDiffRegressionUNet.0.1200000.mdlus
RES=$SCRATCH/corrdiff_runs/diffusion_3/checkpoints_diffusion/EDMPrecondSuperResolution.0.2000000.mdlus
NOICE=1 MODE=all NUM_ENS=15 REG_CKPT=$REG RES_CKPT=$RES DATA_DIR=$DATA \
  bash training_mini/slurm/submit.sh training_mini/slurm/generate.sh
```

### With or without sea ice — two different knobs

This catches people out, so it is worth stating plainly: **training picks a config, inference
passes a flag.** They are not interchangeable, and neither accepts the other's form.

| step | knob | why |
|---|---|---|
| `train_regression.sh` / `train_diffusion.sh` | `CONFIG=..._noice` | the config's `dataset.lr_channels` **is** the channel selection, and it also fixes the network's input width at build time |
| `generate.sh` / `run_eval.sh` | `NOICE=1` | no training config is involved; the channel list has to be overridden at inference to match the width already baked into the checkpoint |

The four training configs:

| config | stage | sea ice | LR channels |
|---|---|---|---|
| `config_training_era5_carra2_mini_regression` | regression | yes | 12 |
| `config_training_era5_carra2_mini_regression_noice` | regression | no | 11 |
| `config_training_era5_carra2_mini_diffusion` | diffusion | yes | 12 |
| `config_training_era5_carra2_mini_diffusion_noice` | diffusion | no | 11 |

With-ice is the default (`lr_channels: null` means all 12), so `CONFIG` only needs setting for
the no-ice variants. Give no-ice runs distinct names — mixing a no-ice checkpoint into a with-ice
run directory is unrecoverable confusion, since the directory is how `generate.sh` finds it.

### Finding a checkpoint path

The commands above spell checkpoint paths out in full so they can be copy-pasted. To let the
shell find the furthest-trained one instead:

```bash
source $REPO/training_mini/slurm/common.sh      # once per shell
REG=$(latest_ckpt $SCRATCH/corrdiff_runs/regression_3/checkpoints_regression)
```

`latest_ckpt` selects on the `nimg` in the filename, **not** mtime, which copying reorders — so
it stays correct after an archive/restore. **Sourcing is purely for your own shell**: every job
script sources `common.sh` itself, and resuming a training run needs none of it, since
PhysicsNeMo's `load_checkpoint` picks the highest `nimg` in `checkpoint_dir` on its own.

### `generate.sh` knobs

| var | default | what it does |
|---|---|---|
| `OUTPUT_DIR` | `$SCRATCH/corrdiff_mini` | the run to generate from; the **furthest-trained** `checkpoints_regression/*.mdlus` under it is used |
| `REG_CKPT` / `RES_CKPT` | from `OUTPUT_DIR` | pin exact checkpoints instead — needed when the diffusion net was trained against a *different* regression run |
| `DATA_DIR` | `$PROJECT/data` | shard directory; `shard_YYYY.zarr` and `.zarr.zip` both work |
| `MODE` | `regression` | `regression` (deterministic mean) · `diffusion` · `all` (mean + residual) |
| `NUM_ENS` | `1` | ensemble members per time; >1 only means anything for `diffusion`/`all` |
| `SEED_BATCH` | largest divisor of `NUM_ENS` ≤ 8 | members denoised per sampler call. **Must divide `NUM_ENS`** — the job refuses to start otherwise |
| `NOICE` | `0` | `1` withholds `siconc` → the 11-channel no-ice stack |
| `DROP` | *(empty)* | comma-separated channels to withhold; the general form of `NOICE`, and the hook for ablations |

**On `NOICE` / `DROP`:** the channel list is read from the shard's own attributes and its length
checked against the checkpoint's input conv width *before* the GPU does anything, so a mismatch
fails in seconds with the fix named rather than as a conv shape error minutes in. Getting this
wrong is not hypothetical — it killed three generation jobs on 2026-09-24. And passing the *wrong*
11 channels is worse than passing 12: the shapes still match, so the run completes and reports
plausible numbers for a stack the model never saw. That is why the list is never typed by hand.

Scoring a run over many times (CRPS/MAE, test years) is `evaluate/run_eval.sh` — see
`evaluate/README.md`. It takes the same `NOICE`/`DROP` flags.

Edit `--account` / `--mail-user` / `module load` lines to match the cluster.

## Smoke test first
See `kaggle/smoke_test.md` — a few-step end-to-end run on a Kaggle GPU with a trimmed shard.

## Local adapter test (M1, CPU, no physicsnemo)
```bash
pytest tests/test_era5_carra2_dataset.py -q     # runs against ../testing_data/shard_2011.zarr
```

## Re-vendoring later
Re-copy the upstream example over this directory, then re-apply our files and the two
`[ERA5->CARRA2 adapter]` hooks in `train.py`/`generate.py`. Bump the commit hash above.
