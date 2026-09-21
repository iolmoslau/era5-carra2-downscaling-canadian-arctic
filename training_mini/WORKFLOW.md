# Per-run workflow: train → generate → collect results

The repeatable procedure for every training run. Conventions:

- **One `OUTPUT_DIR` per run**, named after the run: `$SCRATCH/corrdiff_runs/<name>`. This keeps
  each run's checkpoints (`checkpoints_regression/` or `checkpoints_diffusion/`) *and* its
  TensorBoard logs (`tensorboard/`) isolated, so loss curves are per-run.
- **Corrected data**: `DATA=$PROJECT/data/derot` with `STATS=$DATA/stats_train_2011_2018.json`
  (from `scripts/derotate_shards.sh`).
- **Shards are `shard_YYYY.zarr.zip` where archived, `shard_YYYY.zarr` otherwise** — per year,
  independently. Nothing in the commands below changes either way: `dataloading.dataset` resolves
  the form, and `stage_shards` copies whichever is present. Archive with `scripts/zip_shards.sh`
  before a run of resubmissions (see *Staging* below); details in `evaluate/README.md`.
- Result name matches the run: `regression_<n>` / `diffusion_<n>`.
- Train/generate run as GPU jobs; **collect runs on a login node** (cartopy needs internet for
  the Natural Earth shapefiles) in `corrdiff-env` (which must have cartopy + tensorboard).
- **Pick checkpoints with `latest_ckpt`**, never `ls -t`. Checkpoints are named
  `<Model>.0.<nimg>.mdlus`, and `latest_ckpt` selects on that `<nimg>` (training progress).
  Modification times agree with it only until you archive/restore a run (section C), after which
  `ls -t` silently returns an arbitrary checkpoint. Source it once per shell:
  ```bash
  source $REPO/training_mini/slurm/common.sh
  ```
  This applies wherever *you* name a checkpoint: the diffusion run's regression base (step B.1)
  and the generate/eval checkpoints. **Resuming a training run needs none of it** —
  `train_regression.sh` / `train_diffusion.sh` only pass `checkpoint_dir`, and PhysicsNeMo's
  `load_checkpoint` picks the highest `<nimg>` in that directory itself, so a re-submit always
  continues from the furthest-trained checkpoint regardless of mtimes.

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
source $REPO/training_mini/slurm/common.sh
cd $REPO/training_mini
REG=$(latest_ckpt $OUT/checkpoints_regression)
module load proj
python tools/collect_run.py --name $NAME \
  --tensorboard $OUT/tensorboard \
  --nc corrdiff_output.nc \
  --checkpoint "$REG" \
  --train-samples 1500000 --error sigma \
  --hydra-config $REPO/logs/hydra/<jobid> \
  --data $DATA --stats $DATA/stats_train_2011_2018.json \
  --notes "Regression two pushed to 1.5 M samples"

# 4. COMMIT the result
git add results/$NAME && git commit -m "$NAME results" && git push
```

## B. Diffusion run  (example name: `diffusion_1`, built on `regression_2`)

```bash
NAME=diffusion_1
OUT=$SCRATCH/corrdiff_runs/$NAME
DATA=$PROJECT/data/derot
REG=$(latest_ckpt $SCRATCH/corrdiff_runs/regression_2/checkpoints_regression)

# 1. TRAIN diffusion on the regression checkpoint (re-submit to resume)
DATA_DIR=$DATA  STATS=$DATA/stats_train_2011_2018.json  OUTPUT_DIR=$OUT  TRAIN_DURATION=2000000 \
  bash training_mini/slurm/submit.sh --gpus=h100:2 training_mini/slurm/train_diffusion.sh "$REG"

# 2. GENERATE an ensemble (regression mean + diffusion residual). NUM_ENS members per input
#    time gives the spread -> per-channel variance in metrics.json / runs.csv.
RES=$(latest_ckpt $OUT/checkpoints_diffusion)
MODE=all  NUM_ENS=15  REG_CKPT="$REG"  RES_CKPT="$RES"  DATA_DIR=$DATA \
  bash training_mini/slurm/submit.sh training_mini/slurm/generate.sh

# 3. COLLECT (login node)
module load python/3.11 mpi4py/4.1.0 && source ~/corrdiff-env/bin/activate
source $REPO/training_mini/slurm/common.sh
cd $REPO/training_mini
module load proj
python tools/collect_run.py --name $NAME \
  --tensorboard $OUT/tensorboard \
  --nc corrdiff_output.nc \
  --checkpoint "$RES" \
  --train-samples 2000000 --error sigma \
  --hydra-config $REPO/logs/hydra/<jobid> \
  --data $DATA --stats $DATA/stats_train_2011_2018.json \
  --notes "diffusion on regression_2; 4-member ensemble"

# 4. COMMIT
git add results/$NAME && git commit -m "$NAME results" && git push
```

## C. Archive a run's checkpoint  (only for runs worth keeping)

`$SCRATCH` is **purged by file age (~60 days) and is not backed up**, so checkpoints left there
disappear. The `results/<name>/` diagnostics are safe (they're in git), but the model weights are
not. Do this for any run you want to preserve — you don't have to keep them all, and you only need
the **final** checkpoint, not the whole history.

```bash
bash scripts/archive_run.sh regression_2 regression_2_extended diffusion_2
```

Each run lands at **`$PROJECT/trained_models/<name>/checkpoint/`**. Override `RUNS_DIR` /
`DEST_ROOT` for a different layout; `FORCE=1` re-does a run already archived.

The script picks the final checkpoint with `latest_ckpt` (highest `nimg`, never mtime — copying
is precisely what scrambles mtimes), byte-verifies every copy with `cmp` rather than trusting
`cp`'s exit status, and writes a `manifest.json` recording the step kept, the source path and the
run's channel provenance carried over from `results/<name>/run_info.json`, so the archive stands
alone.

- The `.pt` (optimizer) is only needed if you might **resume/extend** training later. For pure
  inference/generation you can keep just the `.mdlus` and drop the `.pt` to save space.
- **Diffusion runs need their base regression checkpoint too**: generation is `reg + residual`,
  so archiving a `diffusion_n` alone isn't reproducible — make sure the `regression_n` it was built
  on is also archived. The manifest says so, but it can't say *which* regression run; name both in
  the same `archive_run.sh` invocation and the pairing stays visible.
- **Restore** is the reverse copy back into a scratch `OUT/checkpoints_*` dir; then generate (or
  resume) exactly as above. Run training/generation from `$SCRATCH`, not `$PROJECT` — `$PROJECT`
  is cold storage, `$SCRATCH` is the fast filesystem.

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
- **If a job dies with `resume model weights: N '.mdlus' checkpoint(s) ... could not be loaded`**
  — that is the `[thesis]` guard in `train.py`, and it is doing its job. Upstream CorrDiff
  swallows checkpoint-load failures (`except Exception: pass`), which silently restarts training
  from random weights while the sample counter, LR schedule and checkpoint names all keep looking
  correct. The usual cause is a `.mdlus` truncated by preemption mid-write. Fix it by deleting the
  bad checkpoint (the run resumes from the next one down) or pointing `OUTPUT_DIR` at a fresh
  directory. `CORRDIFF_ALLOW_BAD_CHECKPOINT=1` restores the old behaviour if you really do want to
  press on — it passes through `submit.sh` like any other env var:
  ```bash
  CORRDIFF_ALLOW_BAD_CHECKPOINT=1 DATA_DIR=$DATA OUTPUT_DIR=$OUT \
    bash training_mini/slurm/submit.sh --gpus=h100:2 training_mini/slurm/train_regression.sh
  ```
  A **missing** optimizer `.pt` is only a warning, not an error — restoring a `.mdlus`-only
  archive (section C) resumes fine, with Adam moments reset and a transient loss bump.
- **Staging (`STAGE`, default `1`)**: both train scripts copy the shards the config needs
  (`dataset.years` + `validation.years`, read off the config by `config_years` — not a hardcoded
  list) to node-local `$SLURM_TMPDIR` before training. Outside a job `$SLURM_TMPDIR` is unset and
  staging turns itself off with a warning, reading from `DATA_DIR` in place; `STAGE=0` forces
  that. A **loose** `shard_YYYY.zarr` is ~5,900 tiny files and measured **~6 min to stage, ~54 min
  per submission** — metadata cost, not bandwidth. Running `scripts/zip_shards.sh` over the train
  years first turns that into nine large sequential reads, and since training is resubmitted
  repeatedly on the opportunistic queue it pays for itself within a couple of resubmissions. The
  archives are verified against the loose stores and the originals are kept; come back with
  `REMOVE_SRC=1` to reclaim the inodes only after a real training job has run off them.
- **Checkpointing dials** (both train scripts): `CKPT_FREQ` = samples between checkpoints
  (config default 5000, i.e. ~every 78 steps at `total_batch_size: 64`), `KEEP_CKPTS` = how many
  to retain (default `-1`, keep everything). Writes are synchronous behind a barrier and stall
  both GPUs, so raising `CKPT_FREQ` buys throughput at the cost of losing more progress to
  preemption. `KEEP_CKPTS` prunes on the *next* save, so set it when starting a run rather than
  partway through one you may want to bisect.
  ```bash
  CKPT_FREQ=50000 KEEP_CKPTS=3 DATA_DIR=$DATA OUTPUT_DIR=$OUT \
    bash training_mini/slurm/submit.sh --gpus=h100:2 training_mini/slurm/train_regression.sh
  ```
- **No-sea-ice variants**: add `CONFIG=config_training_era5_carra2_mini_regression_noice`
  (or `..._diffusion_noice`) to the train step, and for generation add
  `'++dataset.lr_channels=[t2m,u10,v10,t500,t850,z500,z850,u500,u850,v500,v850]'` — use a
  distinct run name (e.g. `regression_3_noice`). With-ice is the **default** (`lr_channels: null`
  = all 12), and every run collected so far is with-ice — `runs.csv` reads `sea_ice=yes, lr_n=12`
  for all five, confirmed from the checkpoints and corroborated by the NetCDF `input` group names.
  So the sea-ice comparison is the *no-ice* run, not the other way round.
- **Generate overwrites** `corrdiff_output.nc`, so collect right after generating (before the
  next run's generation). `collect_run` copies the plot into `results/<name>/`.
- **collect_run tops up rather than overwrites** — re-running with the same `--name` replaces its
  `runs.csv` row, but only fields you pass *explicitly* are redone; everything else is carried
  forward from the previous collect and reported as `carried`. That matters because
  `corrdiff_output.nc` is one fixed overwritten path: re-collecting to add provenance without
  `--nc` keeps the metrics from the original generation instead of blanking them, and re-passing
  `--nc` would score whatever generation ran most recently. Pass `--nc` only when you mean to
  rescore.
- **Provenance is read, not typed.** `--hydra-config $REPO/logs/hydra/<jobid>` is preferred over
  `--config`: it snapshots the config the run *resolved*, not the name someone remembered.
  Independently of both, `--checkpoint` gives `collect_run` the input-conv width, from which it
  recovers the LR channel count and whether sea ice was in — a fact about the weights, which is
  what settles a disagreement between the config name, the notes and memory.
- Adding a new diagnostic later: compute it in `tools/plot_sample_native.py` (add to the
  `metrics[v]` dict) and add matching columns to `FIELDS` in `tools/collect_run.py`.
