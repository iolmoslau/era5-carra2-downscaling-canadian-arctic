# Kaggle GPU smoke test (CorrDiff-Mini, ERA5 -> CARRA2)

Goal: prove the **end-to-end pipeline runs on a CUDA GPU** (adapter -> conditioner ->
regression -> diffusion -> generate) with a tiny slice of data and a handful of steps,
*before* spending fir hours. This is a plumbing test, not a real training run.

Kaggle GPUs (T4 / P100) are **Turing/Pascal** and do **not** support bf16, so we override
`fp_optimizations=fp32` for the smoke test.

## 1. Locally: make a small shard to upload (~1 month)

```bash
# from the repo root, on the dev machine
python training_mini/tools/trim_shard.py \
  --src testing_data/shard_2011.zarr \
  --dst testing_data/shard_2011_smoke.zarr --steps 240   # ~1 month at 3-hourly
```

Zip `testing_data/shard_2011_smoke.zarr` and upload it as a **Kaggle Dataset** (a few
hundred MB instead of ~4 GB). Also push this repo to GitHub (or upload it as a dataset).

## 2. On Kaggle (Notebook, Accelerator = GPU T4 x2 or P100)

```bash
# --- setup ---
pip -q install nvidia-physicsnemo
pip -q install -r /kaggle/working/thesis/requirements-train.txt   # or the vendored requirements.txt

cd /kaggle/working/thesis/training_mini
SMOKE=/kaggle/input/<your-dataset>/shard_2011_smoke.zarr   # single .zarr -> `years` is ignored
STATS=/kaggle/working/stats_smoke.json

# --- train-only stats over the smoke shard ---
python tools/make_stats.py --stores "$SMOKE" --out "$STATS"

# --- STAGE 1: regression (~a few dozen steps) ---
python train.py --config-name=config_training_era5_carra2_mini_regression \
  ++dataset.data_path="$SMOKE" ++dataset.stats_path="$STATS" \
  ~validation \
  ++training.hp.training_duration=2000 \
  ++training.hp.total_batch_size=2 ++training.hp.batch_size_per_gpu=1 \
  ++training.perf.fp_optimizations=fp32 ++training.perf.dataloader_workers=2 \
  ++training.io.print_progress_freq=100 ++training.io.save_checkpoint_freq=1000

REG_CKPT=$(find . -name "*.mdlus" | grep -i regress | head -1)
echo "regression checkpoint: $REG_CKPT"

# --- STAGE 2: diffusion (~a few dozen steps) ---
python train.py --config-name=config_training_era5_carra2_mini_diffusion \
  ++dataset.data_path="$SMOKE" ++dataset.stats_path="$STATS" \
  ~validation \
  ++training.io.regression_checkpoint_path="$REG_CKPT" \
  ++training.hp.training_duration=2000 \
  ++training.hp.total_batch_size=2 ++training.hp.batch_size_per_gpu=1 \
  ++training.perf.fp_optimizations=fp32 ++training.perf.dataloader_workers=2 \
  ++training.io.print_progress_freq=100 ++training.io.save_checkpoint_freq=1000

RES_CKPT=$(find . -name "*.mdlus" | grep -iv regress | head -1)

# --- generate on a couple of in-range timestamps ---
python generate.py --config-name=config_generate_era5_carra2_mini \
  ++dataset.data_path="$SMOKE" ++dataset.stats_path="$STATS" \
  ++generation.io.reg_ckpt_filename="$REG_CKPT" \
  ++generation.io.res_ckpt_filename="$RES_CKPT" \
  '++generation.times=[2011-01-01T00:00:00,2011-01-05T12:00:00]'
```

## 3. What "passing" looks like
- Regression + diffusion loops run without shape/dtype errors and loss is finite/decreasing.
- Two `*.mdlus` checkpoints are produced.
- `generate.py` writes a NetCDF with `input` / `truth` / `prediction` groups at 448x448.
- Sanity-plot a predicted field (reuse `visualization/plotting.py`) and eyeball it.

## Troubleshooting
- **OOM on T4**: drop to `batch_size_per_gpu=1`, `total_batch_size=1`; if still tight,
  the full-image 448x448 diffusion is the heavy part — reduce steps or use a smaller crop.
- **bf16 error**: ensure `++training.perf.fp_optimizations=fp32` (T4/P100 have no bf16).
- **`times` not found**: the timestamps in `generation.times` must exist in the smoke shard
  (3-hourly from 2011-01-01). Use the first few days of January.
- **apex / group-norm import errors**: the mini configs do not use apex group norm; leave
  `training.perf.use_apex_gn` unset (default False).
