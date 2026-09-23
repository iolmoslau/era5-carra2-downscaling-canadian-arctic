# SPDX-License-Identifier: Apache-2.0
"""A GPU count that merely divides ``total_batch_size`` is not enough.

``compute_num_accumulation_rounds`` (helpers/train_helpers.py, vendored) takes TWO integer
divisions and then demands the product come back exact::

    batch_gpu_total = total_batch_size // world_size
    bpg             = min(batch_size_per_gpu, batch_gpu_total)
    rounds          = batch_gpu_total // bpg
    require  bpg * rounds * world_size == total_batch_size

So at the shipped ``total_batch_size: 64, batch_size_per_gpu: 4``, three GPUs give
``64//3 = 21``, ``21//4 = 5``, ``4*5*3 = 60`` and the run raises. This actually happened: a
regression run trained on 2 GPUs was resubmitted with ``--gpus=h100:3`` after a timeout. The
ValueError lands *after* the allocation is granted and the shards are staged, so the guard runs
as a preflight instead, before staging.

    pytest training_mini/tests/test_world_size.py -q
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

TRAIN_DIR = Path(__file__).resolve().parents[1]
COMMON_SH = TRAIN_DIR / "slurm" / "common.sh"
CONF = TRAIN_DIR / "conf"

pytestmark = pytest.mark.skipif(not COMMON_SH.exists(), reason="slurm/common.sh not present")


def sh(script: str):
    return subprocess.run(
        ["bash", "-c", f'set -euo pipefail\nsource "{COMMON_SH}"\n{script}'],
        capture_output=True, text=True,
    )


def fits(world, total, bpg) -> bool:
    return sh(f"_ws_fits {world} {total} {bpg}").returncode == 0


def reference(world: int, total: int, bpg) -> bool:
    """compute_num_accumulation_rounds, transcribed. The bash must agree with this."""
    if world <= 0:
        return False
    batch_gpu_total = total // world
    if batch_gpu_total <= 0:
        return False
    if not isinstance(bpg, int) or bpg > batch_gpu_total:
        bpg = batch_gpu_total
    rounds = batch_gpu_total // bpg
    return rounds > 0 and bpg * rounds * world == total


# ------------------------------------------------------------------ the case that bit
@pytest.mark.parametrize("world,ok", [
    (1, True), (2, True), (3, False), (4, True),
    (5, False), (6, False), (7, False), (8, True),
])
def test_shipped_config_gpu_counts(world, ok):
    """total=64, bpg=4 -- the four mini configs. 3 GPUs is the one that was actually submitted."""
    assert fits(world, 64, 4) is ok


def test_dividing_the_batch_is_not_sufficient():
    """The whole point: 64 % 8 == 0 for world=8 AND for the failing counts' intuition.

    16 divides 64 evenly as a GPU count, and works; but 32 would leave batch_gpu_total=2 < bpg=4,
    which the min() rescues -- while 3, which does not divide 64, fails. Divisibility of
    total_batch_size by world_size is neither necessary nor sufficient on its own.
    """
    assert 64 % 3 != 0 and not fits(3, 64, 4)
    assert 64 % 5 != 0 and not fits(5, 64, 4)
    assert fits(32, 64, 4)          # bpg capped to 2 by batch_gpu_total, 2*1*32 == 64


@pytest.mark.parametrize("world", range(1, 17))
@pytest.mark.parametrize("total,bpg", [(64, 4), (64, 8), (256, 4), (48, 6), (12, 5)])
def test_bash_matches_the_vendored_python(world, total, bpg):
    assert fits(world, total, bpg) is reference(world, total, bpg)


def test_auto_batch_per_gpu_does_not_rescue_an_odd_gpu_count():
    """``batch_size_per_gpu: "auto"`` -- base_all.yaml's default -- is NOT a way out.

    It resolves to ``total // world_size``, which throws away the remainder: at world=3 that is
    21, one accumulation round, 21*1*3 = 63 != 64. So "auto" fits exactly when world_size divides
    total_batch_size, and an odd GPU count fails either way.
    """
    for world in (1, 2, 4, 8, 16, 32, 64):
        assert fits(world, 64, "auto"), world           # divides 64
    for world in (3, 5, 6, 7):
        assert not fits(world, 64, "auto"), world       # does not


# ------------------------------------------------------------------ the guard's behaviour
def test_guard_rejects_three_and_names_the_usable_counts():
    out = sh("require_world_size 3 64 4")
    assert out.returncode == 1
    assert "Usable --gpus counts here: 1 2 4 8 16" in out.stderr
    # the reassurance matters as much as the rejection: the user must not conclude that the
    # half-trained run is now stuck at whatever GPU count it started on
    assert "absolute in samples" in out.stderr


def test_guard_passes_the_count_the_run_started_on():
    assert sh("require_world_size 2 64 4").returncode == 0


def test_unpinned_batch_is_not_an_error():
    """A config that does not set training.hp.total_batch_size gives config_hp "" -- the guard
    has nothing to check and must not block the job."""
    assert sh('require_world_size 3 "" 4').returncode == 0
    assert sh('require_world_size "" 64 4').returncode == 0


# ------------------------------------------------------------------ reading it out of the config
@pytest.mark.parametrize("name", [
    "config_training_era5_carra2_mini_regression",
    "config_training_era5_carra2_mini_regression_noice",
    "config_training_era5_carra2_mini_diffusion",
    "config_training_era5_carra2_mini_diffusion_noice",
])
def test_config_hp_reads_what_the_configs_declare(name):
    path = CONF / f"{name}.yaml"
    if not path.exists():
        pytest.skip(f"{name} not present")
    declared = (yaml.safe_load(path.read_text())["training"]["hp"])
    for key in ("total_batch_size", "batch_size_per_gpu"):
        got = sh(f'config_hp "{path}" {key}').stdout.strip()
        assert got == str(declared[key])


def test_every_shipped_config_runs_on_two_gpus():
    """The documented submission is --gpus=h100:2; none of the configs may quietly forbid it."""
    for path in sorted(CONF.glob("config_training_era5_carra2_mini_*.yaml")):
        hp = yaml.safe_load(path.read_text())["training"]["hp"]
        assert reference(2, hp["total_batch_size"], hp["batch_size_per_gpu"]), path.name


# ------------------------------------------------------------------ ranks vs allocated GPUs
# The other half of the same incident: a job that asks for 3 GPUs but whose SLURM_GPUS_ON_NODE
# is unset launches ONE rank -- world=1 satisfies the batch arithmetic, so it trains happily on
# a third of the allocation for the full walltime and nothing ever fails.
def test_more_ranks_than_devices_is_an_error():
    out = sh("check_gpu_alloc 3 2")
    assert out.returncode == 1
    assert "only 2 CUDA device(s) are visible" in out.stderr


def test_fewer_ranks_than_devices_warns_but_runs():
    """Waste must not kill a job that is otherwise valid -- but it must be impossible to miss."""
    out = sh("check_gpu_alloc 1 3")
    assert out.returncode == 0
    assert "2 idle" in out.stderr
    assert "SLURM_GPUS_ON_NODE" in out.stderr


def test_matching_counts_are_silent():
    out = sh("check_gpu_alloc 2 2")
    assert out.returncode == 0 and out.stderr == ""


def test_undeterminable_device_count_does_not_block():
    """No nvidia-smi (a login node, this laptop) must not make the guard refuse to run."""
    assert sh('check_gpu_alloc 2 ""').returncode == 0
    assert sh('check_gpu_alloc "" 2').returncode == 0


def test_visible_gpus_counts_the_cuda_list():
    assert sh("CUDA_VISIBLE_DEVICES=0,1,2 visible_gpus").stdout.strip() == "3"
    assert sh("CUDA_VISIBLE_DEVICES=1 visible_gpus").stdout.strip() == "1"


# ------------------------------------------------------------------ NPROC derivation
# $SLURM_GPUS_ON_NODE is a claim; the visible devices are the fact. sbatch --export=ALL means an
# unset one keeps whatever the submitting shell had, so it can be stale or simply wrong.
NPROC_SNIPPET = '''
NPROC="${NPROC:-}"
if [[ ! "$NPROC" =~ ^[1-9][0-9]*$ ]]; then
  NPROC=$(visible_gpus)
  [[ "$NPROC" =~ ^[1-9][0-9]*$ ]] || NPROC="${SLURM_GPUS_ON_NODE:-1}"
fi
echo "$NPROC"
'''


def nproc(env: str) -> str:
    return sh(f"{env}\n{NPROC_SNIPPET}").stdout.strip()


def test_visible_devices_beat_a_stale_slurm_variable():
    """The incident: SLURM_GPUS_ON_NODE=1 inherited from the submitting shell while three GPUs
    were actually allocated. The device list must win."""
    assert nproc("export CUDA_VISIBLE_DEVICES=0,1,2 SLURM_GPUS_ON_NODE=1") == "3"


def test_explicit_nproc_beats_everything():
    """The escape hatch the warning points at, for when neither source can be trusted."""
    assert nproc("export NPROC=2 CUDA_VISIBLE_DEVICES=0,1,2,3 SLURM_GPUS_ON_NODE=4") == "2"


def test_a_junk_nproc_is_ignored_rather_than_passed_to_torchrun():
    assert nproc("export NPROC=abc CUDA_VISIBLE_DEVICES=0,1") == "2"
    assert nproc("export NPROC=0 CUDA_VISIBLE_DEVICES=0,1") == "2"


def test_falls_back_when_no_devices_are_visible():
    """A CPU-only shell must not end up launching zero ranks."""
    assert nproc("unset CUDA_VISIBLE_DEVICES; visible_gpus() { printf ''; }; "
                 "export SLURM_GPUS_ON_NODE=2") == "2"
    assert nproc("unset CUDA_VISIBLE_DEVICES SLURM_GPUS_ON_NODE; "
                 "visible_gpus() { printf '0\\n'; }") == "1"
