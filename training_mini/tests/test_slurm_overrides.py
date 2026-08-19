# SPDX-License-Identifier: Apache-2.0
"""The train scripts' env passthroughs emit the Hydra keys they claim to.

``CKPT_FREQ`` / ``KEEP_CKPTS`` reach ``train.py`` as ``++training.io.*`` overrides. Hydra's
``++`` *creates* a key when it is absent, so a typo (``save_ckpt_freq``) is accepted in silence
and simply does nothing -- you would only notice by counting checkpoint files afterwards. These
tests check the emitted key names against the config that actually defines them, and run the
real passthrough lines under ``set -euo pipefail`` to confirm they neither fire when unset nor
abort the job (``[[ -n ... ]] && CMD+=(...)`` returns non-zero when the guard fails).

    pytest training_mini/tests/test_slurm_overrides.py -q
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

SLURM = Path(__file__).resolve().parents[1] / "slurm"
BASE_TRAINING = Path(__file__).resolve().parents[1] / "conf/base/training/base_all.yaml"
SCRIPTS = ["train_regression.sh", "train_diffusion.sh"]

# env var -> the Hydra key it must set, and the section of base_all.yaml defining it
DIALS = {
    "CKPT_FREQ": ("++training.io.save_checkpoint_freq", "io", "save_checkpoint_freq"),
    "KEEP_CKPTS": ("++training.io.save_n_recent_checkpoints", "io", "save_n_recent_checkpoints"),
    "TRAIN_DURATION": ("++training.hp.training_duration", "hp", "training_duration"),
    "TOTAL_BATCH": ("++training.hp.total_batch_size", "hp", "total_batch_size"),
    "BATCH_PER_GPU": ("++training.hp.batch_size_per_gpu", "hp", "batch_size_per_gpu"),
}


@pytest.fixture(scope="module")
def base_training_cfg() -> dict:
    return yaml.safe_load(BASE_TRAINING.read_text())


def passthrough_lines(script: Path) -> list[str]:
    """The `[[ -n "${VAR:-}" ]] && CMD+=(...)` lines, lifted verbatim from the script."""
    return [ln for ln in script.read_text().splitlines()
            if ln.strip().startswith("[[ -n") and "CMD+=" in ln]


def run_lines(lines: list[str], env: dict) -> list[str]:
    """Execute those lines under the same shell options the job uses."""
    prog = "set -euo pipefail\nCMD=()\n" + "\n".join(lines) + \
           '\nprintf "%s\\n" ${CMD[@]+"${CMD[@]}"}\n'
    out = subprocess.run(["bash", "-c", prog], capture_output=True, text=True,
                         env={**os.environ, **env})
    assert out.returncode == 0, f"exit {out.returncode}: {out.stderr}"
    return [ln for ln in out.stdout.splitlines() if ln]


@pytest.mark.parametrize("name", SCRIPTS)
def test_keys_exist_in_the_config_that_defines_them(name, base_training_cfg):
    """A mistyped ++key would be silently accepted by Hydra and do nothing."""
    text = (SLURM / name).read_text()
    for var, (key, section, leaf) in DIALS.items():
        assert f"{key}=${var}" in text, f"{name}: {var} does not emit {key}"
        assert leaf in base_training_cfg[section], \
            f"{key} targets training.{section}.{leaf}, absent from {BASE_TRAINING.name}"


@pytest.mark.parametrize("name", SCRIPTS)
def test_no_dial_fires_when_unset(name):
    lines = passthrough_lines(SLURM / name)
    assert lines, f"{name}: found no passthrough lines to test"
    env = {v: "" for v in DIALS}          # explicitly empty, as an unset var would be
    assert run_lines(lines, env) == []


@pytest.mark.parametrize("name", SCRIPTS)
def test_dials_emit_exactly_their_override_when_set(name):
    lines = passthrough_lines(SLURM / name)
    emitted = run_lines(lines, {v: "" for v in DIALS} |
                        {"CKPT_FREQ": "50000", "KEEP_CKPTS": "3"})
    assert emitted == [
        "++training.io.save_checkpoint_freq=50000",
        "++training.io.save_n_recent_checkpoints=3",
    ]


@pytest.mark.parametrize("name", SCRIPTS)
def test_passthroughs_survive_set_e_when_all_unset(name):
    """`[[ -n ... ]] && CMD+=(...)` returns 1 when the guard fails; must not kill the job."""
    lines = passthrough_lines(SLURM / name)
    prog = "set -euo pipefail\nCMD=()\n" + "\n".join(lines) + "\necho SURVIVED\n"
    out = subprocess.run(["bash", "-c", prog], capture_output=True, text=True,
                         env={**os.environ, **{v: "" for v in DIALS}})
    assert out.returncode == 0 and "SURVIVED" in out.stdout, out.stderr


def test_both_scripts_expose_the_same_dials():
    """The two stages must stay tunable the same way."""
    got = {n: sorted(v for v in DIALS if f"${v}" in (SLURM / n).read_text()) for n in SCRIPTS}
    assert got[SCRIPTS[0]] == got[SCRIPTS[1]] == sorted(DIALS)
