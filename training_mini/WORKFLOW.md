# Per-run workflow: train → generate → collect results

The repeatable procedure for every training run. Conventions:

- **One `OUTPUT_DIR` per run**, named after the run: `$SCRATCH/corrdiff_runs/<name>`. This keeps
  each run's checkpoints (`checkpoints_regression/` or `checkpoints_diffusion/`) *and* its
  TensorBoard logs (`tensorboard/`) isolated, so loss curves are per-run.
- **Corrected data**: `DATA=$PROJECT/data/derot` with `STATS=$DATA/stats_train_2011_2018.json`
  (from `scripts/derotate_shards.sh`).
- Result name matches the run: `regression_<n>` / `diffusion_<n>`.
- Train/generate run as GPU jobs; **collect runs on a login node** (cartopy needs internet for
  the Natural Earth shapefiles) in `corrdiff-env` (which must have cartopy + tensorboard).

---

## A. Regression run  (example name: `regression_2`)

```bash
NAME=regression_2
OUT=$SCRATCH/corrdiff_runs/$NAME
DATA=$PROJECT/data/derot

# 1. TRAIN  (re-submit the same line until it reaches TRAIN_DURATION -- it resumes)
DATA_DIR=$DATA  STATS=$DATA/stats_train_2011_2018.json  OUTPUT_DIR=$OUT  TRAIN_DURATION=800000 \
  bash training_mini/slurm/submit.sh --gpus=h100:2 training_mini/slurm/train_regression.sh

# 2. GENERATE a sample on the 2019 validation year (deterministic mean)
DATA_DIR=$DATA  OUTPUT_DIR=$OUT \
  bash training_mini/slurm/submit.sh training_mini/slurm/generate.sh
#    -> writes training_mini/corrdiff_output.nc

# 3. COLLECT (login node, corrdiff-env with cartopy)
module load python/3.11 mpi4py/4.1.0 && source ~/corrdiff-env/bin/activate
cd $REPO/training_mini
REG=$(ls -t $OUT/checkpoints_regression/*.mdlus | head -1)
module load proj
python tools/collect_run.py --name $NAME \
  --tensorboard $OUT/tensorboard \
  --nc corrdiff_output.nc \
  --checkpoint "$REG" \
  --train-samples 800000 --error sigma \
  --notes "de-rotated winds; train 2011-18 / val 2019; 2x H100"

# 4. COMMIT the result
git add results/$NAME && git commit -m "$NAME results" && git push
```

## B. Diffusion run  (example name: `diffusion_1`, built on `regression_2`)

```bash
NAME=diffusion_1
OUT=$SCRATCH/corrdiff_runs/$NAME
DATA=$PROJECT/data/derot
REG=$(ls -t $SCRATCH/corrdiff_runs/regression_2/checkpoints_regression/*.mdlus | head -1)

# 1. TRAIN diffusion on the regression checkpoint (re-submit to resume)
DATA_DIR=$DATA  STATS=$DATA/stats_train_2011_2018.json  OUTPUT_DIR=$OUT  TRAIN_DURATION=2000000 \
  bash training_mini/slurm/submit.sh --gpus=h100:2 training_mini/slurm/train_diffusion.sh "$REG"

# 2. GENERATE an ensemble (regression mean + diffusion residual). NUM_ENS members per input
#    time gives the spread -> per-channel variance in metrics.json / runs.csv.
RES=$(ls -t $OUT/checkpoints_diffusion/*.mdlus | head -1)
MODE=all  NUM_ENS=15  REG_CKPT="$REG"  RES_CKPT="$RES"  DATA_DIR=$DATA \
  bash training_mini/slurm/submit.sh training_mini/slurm/generate.sh

# 3. COLLECT (login node)
module load python/3.11 mpi4py/4.1.0 && source ~/corrdiff-env/bin/activate
cd $REPO/training_mini
python tools/collect_run.py --name $NAME \
  --tensorboard $OUT/tensorboard \
  --nc corrdiff_output.nc \
  --checkpoint "$RES" \
  --train-samples 2000000 --error sigma \
  --notes "diffusion on regression_2; 4-member ensemble"

# 4. COMMIT
git add results/$NAME && git commit -m "$NAME results" && git push
```

---

## Where the logs go
Always submit through **`slurm/submit.sh`** (as above). SLURM can't expand `$REPO` in `#SBATCH`
lines, so a bare `sbatch` drops its `.out`/`.err` in whatever directory you launched from (that's
why logs scattered to `~/logs`, `$REPO/logs`, `$REPO/training_mini/logs`). `submit.sh` pins them.
Everything then lands under **`$REPO/logs/`**:

| path | what | who writes it |
|------|------|---------------|
| `$REPO/logs/<jobname>_<jobid>.out` / `.err` | SLURM stdout/stderr | `submit.sh` (`--output/--error`) |
| `$REPO/logs/hydra/<jobid>/` | resolved config snapshot + Hydra job log | `hydra.run.dir` override |
| `$REPO/logs/wandb/` | wandb offline runs | `wandb.results_dir` override |
| `$REPO/logs/generate.log` | generation run log | `generate.py` via `CORRDIFF_LOG_DIR` |
| `$OUT/tensorboard/` | **per-run** loss curves (stays on `$SCRATCH`) | `train.py` (intentional — not a scattered log) |

The per-run `tensorboard/` deliberately lives with the run's checkpoints under `$OUT`, so
`collect_run.py --tensorboard $OUT/tensorboard` finds the right curves.

## Notes
- **No-sea-ice variants**: add `CONFIG=config_training_era5_carra2_mini_regression_noice`
  (or `..._diffusion_noice`) to the train step, and for generation add
  `'++dataset.lr_channels=[t2m,u10,v10,t500,t850,z500,z850,u500,u850,v500,v850]'` — use a
  distinct run name (e.g. `regression_3_noice`).
- **Generate overwrites** `corrdiff_output.nc`, so collect right after generating (before the
  next run's generation). `collect_run` copies the plot into `results/<name>/`.
- **collect_run is idempotent** — re-running with the same `--name` replaces its `runs.csv` row
  and overwrites its folder, so you can re-collect after fixing a note or metric.
- Adding a new diagnostic later: compute it in `tools/plot_sample_native.py` (add to the
  `metrics[v]` dict) and add matching columns to `FIELDS` in `tools/collect_run.py`.
