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
- Shards: `shard_YYYY.zarr` in one directory (`dataset.data_path`).
- Initial model: **train 2011–2018, validate 2019** (test set: future).
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
No-sea-ice variant: use the `*_noice` configs (regression + diffusion). Multi-GPU: launch
via `torchrun --standalone --nproc_per_node=<N> train.py ...` (see `slurm/`).

The configs reference `./data`; on the cluster the SLURM scripts symlink it to the real
shard directory.

## On fir
```bash
sbatch slurm/train_regression.sh
sbatch slurm/train_diffusion.sh /path/to/regression.mdlus
```
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
