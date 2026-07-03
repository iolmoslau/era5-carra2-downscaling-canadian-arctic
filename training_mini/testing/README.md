# Kaggle smoke-test kernel

Headless GPU run of the CorrDiff-Mini pipeline (adapter → conditioner → regression →
diffusion → generate) on a trimmed shard. See the notebook's intro for what "passing" means.

## Files
- `corrdiff_mini_smoke.ipynb` — the kernel that runs the smoke test.
- `kernel-metadata.json` — push config (GPU + internet + dataset mount).

## One-time setup on this machine
```bash
pip install kaggle
# put your API token at ~/.kaggle/kaggle.json (Kaggle → Account → Create New Token), then:
chmod 600 ~/.kaggle/kaggle.json
kaggle --version
```

## 1. Make + upload the smoke dataset (the trimmed shard)
```bash
# from the repo root, on the dev machine: trim shard_2011 to ~1 month into its own folder
mkdir -p /tmp/shard-2011-smoke
python training_mini/tools/trim_shard.py \
  --src testing_data/shard_2011.zarr \
  --dst /tmp/shard-2011-smoke/shard_2011_smoke.zarr --steps 240

# dataset metadata (edit the username in `id`)
cat > /tmp/shard-2011-smoke/dataset-metadata.json <<'JSON'
{ "title": "shard-2011-smoke", "id": "KAGGLE_USERNAME/shard-2011-smoke", "licenses": [{"name": "CC0-1.0"}] }
JSON

kaggle datasets create -p /tmp/shard-2011-smoke --dir-mode zip
```
## 2. Edit `kernel-metadata.json`
Replace `KAGGLE_USERNAME` in **both** `id` and `dataset_sources` with your Kaggle username
(and match the dataset slug from step 1). Also update `REPO_URL` / `BRANCH` in the notebook's
Parameters cell, and make sure that branch is pushed to a reachable (public) GitHub repo.

## 3. Push + run + fetch results

**Use a T4, not a P100.** physicsnemo needs `torch>=2.10`, and torch 2.10 dropped Tesla P100
(sm_60) support — so a P100 fails fast in the notebook's GPU-check cell. **An API push defaults
to P100**, so force T4:
```bash
# if your kaggle CLI supports it (check `kaggle kernels push --help`):
kaggle kernels push -p training_mini/testing --accelerator NvidiaTeslaT4
# otherwise: push once, then in the kernel's Settings on kaggle.com set
#   Accelerator = "GPU T4 x2" (it persists across re-pushes), and re-run.

kaggle kernels status  KAGGLE_USERNAME/corrdiff-mini-smoke        # QUEUED/RUNNING/COMPLETE/ERROR
kaggle kernels output  KAGGLE_USERNAME/corrdiff-mini-smoke -p ./kaggle_out   # logs + artifacts
```
Re-push after edits (it versions automatically). Runs are non-interactive — you get the log and
outputs after completion. GPU sessions cap at ~12 h with a weekly quota.

## Notes
- The notebook auto-locates the `.zarr` under `/kaggle/input`, so the dataset slug only needs to
  match in `dataset_sources`.
- `enable_internet: true` is required so the kernel can `pip install nvidia-physicsnemo` and
  `git clone` the repo.
- physicsnemo needs `torch>=2.10` (already on the Kaggle image) — so we do **not** reinstall
  torch. Pin `PHYSICSNEMO_SPEC` in the notebook's Parameters cell only if a specific physicsnemo
  release is needed.
