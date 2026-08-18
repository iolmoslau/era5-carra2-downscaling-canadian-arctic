# SPDX-License-Identifier: Apache-2.0
"""``latest_ckpt`` selects on training progress, not modification time.

Checkpoints are named ``<Model>.0.<nimg>.mdlus`` where ``<nimg>`` is the processed-sample
count. Two idioms used to be in circulation and both are wrong:

  ``ls -t | head -1``           -- mtime order, which a checkpoint archive/restore
                                   (WORKFLOW.md section C) reorders arbitrarily.
  ``sort -t. -k3 -n | tail -1`` -- splits the FULL PATH on dots, so a dot anywhere in a parent
                                   directory shifts the field and picks the wrong file.

Both fail silently -- you generate and log metrics against the wrong model -- so this pins the
shared helper down against a run directory that triggers each flaw.

    pytest training_mini/tests/test_latest_ckpt.py -q
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

COMMON_SH = Path(__file__).resolve().parents[1] / "slurm" / "common.sh"

pytestmark = pytest.mark.skipif(not COMMON_SH.exists(), reason="slurm/common.sh not present")


def sh(script: str, cwd: Path) -> str:
    """Run `script` under bash with common.sh sourced, as the SLURM jobs do."""
    out = subprocess.run(
        ["bash", "-c", f'set -euo pipefail\nsource "{COMMON_SH}"\n{script}'],
        cwd=str(cwd), capture_output=True, text=True,
    )
    assert out.returncode == 0, f"exit {out.returncode}\nstdout={out.stdout}\nstderr={out.stderr}"
    return out.stdout.strip()


@pytest.fixture
def run_dir(tmp_path) -> Path:
    """A restored run dir: a dot in the parent path, and mtimes in reverse training order."""
    d = tmp_path / "corrdiff_runs" / "run_v1.2" / "checkpoints_regression"
    d.mkdir(parents=True)
    for n in (5000, 800000, 1500000):
        (d / f"CorrDiffRegressionUNet.0.{n}.mdlus").touch()
        (d / f"CorrDiffRegressionUNet.0.{n}.pt").touch()
    # the restore copy landed the least-trained checkpoint last -> newest mtime
    for i, n in enumerate((1500000, 800000, 5000)):
        stamp = 1_760_000_000 + i * 3600
        for suffix in (".mdlus", ".pt"):
            import os

            os.utime(d / f"CorrDiffRegressionUNet.0.{n}{suffix}", (stamp, stamp))
    return d


def test_picks_highest_nimg_despite_inverted_mtimes(run_dir):
    assert sh(f'latest_ckpt "{run_dir}"', run_dir).endswith(
        "CorrDiffRegressionUNet.0.1500000.mdlus")


def test_the_mtime_idiom_it_replaces_would_have_been_wrong(run_dir):
    """Guards the guard: confirm the fixture really does defeat `ls -t`."""
    wrong = sh(f'ls -t "{run_dir}"/*.mdlus | head -1', run_dir)
    assert wrong.endswith("CorrDiffRegressionUNet.0.5000.mdlus")


def test_dot_in_parent_path_does_not_shift_the_field(run_dir):
    """`sort -t. -k3 -n` reads a field of the whole path, so run_v1.2 breaks it."""
    wrong = sh(f'ls "{run_dir}"/*.mdlus | sort -t. -k3 -n | tail -1', run_dir)
    assert wrong.endswith("CorrDiffRegressionUNet.0.800000.mdlus")     # the flaw
    assert sh(f'latest_ckpt "{run_dir}"', run_dir).endswith("0.1500000.mdlus")  # fixed


def test_ckpt_nimg_reads_the_sample_count(run_dir):
    assert sh(f'ckpt_nimg "$(latest_ckpt "{run_dir}")"', run_dir) == "1500000"
    assert sh('ckpt_nimg /a.b/EDMPrecondSuperResolution.0.2000000.pt', run_dir) == "2000000"


def test_returns_nonzero_when_there_is_nothing_to_pick(run_dir, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    for target in (empty, tmp_path / "does_not_exist"):
        assert sh(f'rc=0; latest_ckpt "{target}" >/dev/null || rc=$?; echo $rc', tmp_path) == "1"


def test_ignores_files_that_do_not_follow_the_naming_scheme(tmp_path):
    d = tmp_path / "ckpts"
    d.mkdir()
    (d / "CorrDiffRegressionUNet.0.400000.mdlus").touch()
    (d / "latest.mdlus").touch()               # no nimg field
    (d / "checkpoint.final.mdlus").touch()     # non-numeric where nimg should be
    assert sh(f'latest_ckpt "{d}"', d).endswith("CorrDiffRegressionUNet.0.400000.mdlus")


def test_callers_use_the_helper_not_the_broken_idioms():
    """The whole point of P0-2: exactly one selection algorithm remains in the repo."""
    repo = COMMON_SH.parents[2]
    offenders = []
    for path in list(repo.glob("**/*.sh")) + list(repo.glob("**/*.md")):
        if ".git" in path.parts or path == COMMON_SH:
            continue
        text = path.read_text(errors="ignore")
        for bad in ("ls -t ", "sort -t. -k3"):
            for line in text.splitlines():
                if bad in line and ".mdlus" in line:
                    offenders.append(f"{path.relative_to(repo)}: {line.strip()}")
    assert not offenders, "checkpoint selection must go through latest_ckpt:\n" + "\n".join(offenders)
