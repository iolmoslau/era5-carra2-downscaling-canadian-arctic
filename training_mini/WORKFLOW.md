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
  Modification times agree with it only until you archive/restore a run (section D), after which
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
  bash training_mini/slurm/submit.sh --gpus-per-node=h100:2 training_mini/slurm/train_regression.sh
#    NO-ICE variant: prepend CONFIG=config_training_era5_carra2_mini_regression_noice
#    (training selects channels by CONFIG; generation uses NOICE=1 -- see the note below)

# 2. GENERATE a sample on the 2019 validation year (deterministic mean)
DATA_DIR=$DATA  OUTPUT_DIR=$OUT \
  bash training_mini/slurm/submit.sh training_mini/slurm/generate.sh
#    -> writes training_mini/corrdiff_output.nc
#    NO-ICE model: prepend NOICE=1

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
  bash training_mini/slurm/submit.sh --gpus-per-node=h100:2 training_mini/slurm/train_diffusion.sh "$REG"
#    NO-ICE variant: prepend CONFIG=config_training_era5_carra2_mini_diffusion_noice, and point
#    $REG at the NO-ICE regression run. The job refuses to start if the base was trained on
#    different channels than CONFIG declares -- pairing a no-ice diffusion net with a with-ice
#    regression base is one typo away, and it would otherwise fail only after the allocation,
#    the staging pass and the model build. Example, built on regression_3:
#      NAME=diffusion_3
#      REG=$(latest_ckpt $SCRATCH/corrdiff_runs/regression_3/checkpoints_regression)
#      CONFIG=config_training_era5_carra2_mini_diffusion_noice \
#      DATA_DIR=$DATA STATS=$DATA/stats_train_2011_2018.json \
#      OUTPUT_DIR=$SCRATCH/corrdiff_runs/diffusion_3 TRAIN_DURATION=2000000 \
#        bash training_mini/slurm/submit.sh --gpus-per-node=h100:2 \
#          training_mini/slurm/train_diffusion.sh "$REG"

# 2. GENERATE an ensemble (regression mean + diffusion residual). NUM_ENS members per input
#    time gives the spread -> per-channel variance in metrics.json / runs.csv.
RES=$(latest_ckpt $OUT/checkpoints_diffusion)
MODE=all  NUM_ENS=15  REG_CKPT="$REG"  RES_CKPT="$RES"  DATA_DIR=$DATA \
  bash training_mini/slurm/submit.sh training_mini/slurm/generate.sh
#    NO-ICE pair: prepend NOICE=1 (both nets must be the no-ice ones)

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

## C. Evaluate over many times  (CRPS / MAE on the test years)

Sections A and B generate a *sample* — a couple of times, for a plot and a sanity check. The
numbers in `runs.csv` come from that, so **they are not a result**: two times from 2019 is a
diagnostic, not a score. A real comparison needs `evaluate/run_eval.sh`, which draws N random
times from the chosen years (seeded, so every model is scored on the *same* times) and reports
per-channel CRPS, MAE and RMSE.

```bash
DATA=$PROJECT/data/derot

# Regression net alone -- no diffusion stage needed. CRPS == MAE at one member.
EVAL_MODE=reg  NAME=regression_3_test  N=400  YEARS="2020 2021 2022"  NOICE=1 \
  REG_RUN=$SCRATCH/corrdiff_runs/regression_3  DATA_DIR=$DATA \
  bash training_mini/slurm/submit.sh --time=2:00:00 evaluate/run_eval.sh

# Full model vs regression-only, once the diffusion stage exists
NAME=diffusion_3_test  N=400  NUM_ENS=32  YEARS="2020 2021 2022"  NOICE=1 \
  REG_RUN=$SCRATCH/corrdiff_runs/regression_3 \
  RES_RUN=$SCRATCH/corrdiff_runs/diffusion_3  DATA_DIR=$DATA \
  bash training_mini/slurm/submit.sh --time=5:30:00 evaluate/run_eval.sh

# Commit the metrics (small, tracked); the NetCDFs stay on scratch and are gitignored
git add training_mini/results/$NAME/eval && git commit -m "$NAME eval" && git push
```

- **`NAME` is required** and is not guessed, so an oddly-named scratch dir cannot write into the
  wrong `results/<name>/eval/`.
- **Two run dirs**, because a diffusion run directory holds only `checkpoints_diffusion/` — its
  regression base lives in the run it was trained against (section B). Pairing the wrong base
  produces quietly wrong metrics, not an error, so name both at the call site.
- **`EVAL_MODE=reg`** scores the regression net on its own, which is the only way to evaluate a
  regression run before its diffusion stage is trained — and the right tool for the sea-ice
  comparison at the regression stage, where no residual is involved. It skips the full pass and
  deliberately does **not** score a `full.nc` left behind by an earlier evaluation.
- **`NOICE=1`** for a no-ice model, exactly as in generation. The channel list is checked against
  the checkpoint before the GPU does anything.
- **Same times for every model.** `N` and `SEED` fix the draw, so `N=400 SEED=0` over the same
  `YEARS` gives the identical times to any other run — that is what makes two CRPS numbers
  comparable. Changing either re-draws and breaks the comparison.
- **Where the outputs go:** `metrics_crps_mae.json` → `results/<NAME>/eval/` (small, commit it);
  `full.nc` / `reg.nc` → `$NC_DIR`, defaulting to the evaluated run's `eval/` on `$SCRATCH`
  (several GB, gitignored, safe to delete once the metrics exist).
- Cost scales with `N * NUM_ENS`. The 400-time, 32-member run took **4h09m** on one H100, so set
  `--time` accordingly; `EVAL_MODE=reg` is a single deterministic pass and far cheaper.

Details, including the `.zarr.zip` test-shard workflow: `evaluate/README.md`.

---

## D. Archive a run's checkpoint  (only for runs worth keeping)

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
    bash training_mini/slurm/submit.sh --gpus-per-node=h100:2 training_mini/slurm/train_regression.sh
  ```
  A **missing** optimizer `.pt` is only a warning, not an error — restoring a `.mdlus`-only
  archive (section D) resumes fine, with Adam moments reset and a transient loss bump.
- **netCDF4 drags in MPI, and MPI cannot start in a batch step.** Alliance's netCDF4 is the
  parallel build, so importing it imports `mpi4py`, which calls `MPI_Init_thread`. OpenMPI sees
  `SLURM_*`, assumes an `srun` launch, and tries to reach a PMIx server — an `sbatch` *batch*
  step has none, so it aborts with `Unreachable in ext3x_client.c at line 111`. `common.sh` now
  exports **`MPI4PY_RC_INITIALIZE=0`**, which skips `MPI_Init` and leaves netCDF4 fully usable;
  nothing here needs a live communicator (single-rank, serial NetCDF write, and distributed
  training uses NCCL). Export `MPI4PY_RC_INITIALIZE=1` to opt back in.
  This is why every `generate.py` job from 2026-09-24 died within seconds while `regression_3`
  trained for 3h21m the day before: `train.py` never imports netCDF4, `generate.py` does. It is
  also why `collect_run.py` kept working — it runs on a **login node**, where there are no
  `SLURM_*` variables to mislead OpenMPI — and why `salloc` cannot reproduce it: that shell is
  itself `srun`-launched, so it *has* a PMIx server. Reproduce with
  `sbatch scripts/probe_mpi.sh`.
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
- **Ask for GPUs with `--gpus-per-node`, never `--gpus`.** `--gpus=h100:3` is a *job-level* count
  and SLURM may satisfy it as **three nodes holding one GPU each**. Training launches with
  `torchrun --standalone --nnodes=1`, which only ever uses the first node, so the run trains on
  one GPU while the whole allocation bills. This happened: job `60887527` ran 12 h that way, and
  `sacct` showed it plainly — `.extern` (the whole allocation) `cpu=48,gpu=3`, `.batch` (first
  node only) `cpu=16,gpu=1`. Both train scripts now carry `#SBATCH --nodes=1` and
  `--gpus-per-node`, and `require_single_node` refuses a multi-node allocation up front.
  The line to skim in any job's `.out` is `Launching regression (...) on N H100`.
- **Valid GPU counts are constrained by the batch size.** CorrDiff requires
  `batch_size_per_gpu × accumulation_rounds × world_size == total_batch_size` with both factors
  computed by integer division, so a count that merely divides `total_batch_size` is not enough.
  At the shipped `64 / 4`: **1, 2, 4, 8, 16** — not 3, 5, 6 or 7. `require_world_size` checks this
  before staging; `batch_size_per_gpu: "auto"` does not get you out of it.
  Changing the count between resubmissions is otherwise safe — `cur_nimg` and the LR schedule are
  both absolute in samples, so a run started on 1 GPU resumes correctly on 4.
- **Checkpointing dials** (both train scripts): `CKPT_FREQ` = samples between checkpoints
  (config default 5000, i.e. ~every 78 steps at `total_batch_size: 64`), `KEEP_CKPTS` = how many
  to retain (default `-1`, keep everything). Writes are synchronous behind a barrier and stall
  both GPUs, so raising `CKPT_FREQ` buys throughput at the cost of losing more progress to
  preemption. `KEEP_CKPTS` prunes on the *next* save, so set it when starting a run rather than
  partway through one you may want to bisect.
  ```bash
  CKPT_FREQ=50000 KEEP_CKPTS=3 DATA_DIR=$DATA OUTPUT_DIR=$OUT \
    bash training_mini/slurm/submit.sh --gpus-per-node=h100:2 training_mini/slurm/train_regression.sh
  ```
- **No-sea-ice variants — two different knobs.** Training picks a **`CONFIG`**; generation and
  evaluation pass **`NOICE=1`**. They are not interchangeable, and the reason is structural: a
  training config's `lr_channels` fixes the network's input conv width when the model is built,
  whereas at inference that width is already baked into the checkpoint and the *dataset* has to be
  told to match it. Both forms are shown inline in sections A and B above.

  | step | knob |
  |---|---|
  | `train_regression.sh` / `train_diffusion.sh` | `CONFIG=config_training_era5_carra2_mini_{regression,diffusion}_noice` |
  | `generate.sh` / `run_eval.sh` | `NOICE=1` |

  Use a distinct run name for a no-ice run (`regression_3`, `diffusion_3`) — the run directory is
  how `generate.sh` finds a checkpoint, so mixing variants in one is unrecoverable confusion.
  `NOICE=1` is shorthand for `DROP=siconc`; `DROP` takes any comma-separated channel list, which
  is also the hook for the channel-ablation sweep. The list itself is **read from the data store**
  by `tools/lr_channels.py` and its length checked against the checkpoint's input conv width
  before the GPU does anything. Forgetting the flag used to mean a conv shape error minutes into
  the job (it killed three generation jobs on 2026-09-24); passing the *wrong* 11 channels was
  worse, since the shapes still matched and the run reported plausible numbers for a stack the
  model never saw. With-ice is the **default** (`lr_channels: null` = all 12), so `CONFIG` is only
  needed for the no-ice variants. The five runs in `runs.csv` are all with-ice
  (`sea_ice=yes, lr_n=12`, confirmed from the checkpoints and corroborated by the NetCDF `input`
  group names); **`regression_3` is the first no-ice run** — 11 channels, 1,200,000 samples,
  completed 2026-09-23 — so it is the no-ice side of the sea-ice comparison.
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
