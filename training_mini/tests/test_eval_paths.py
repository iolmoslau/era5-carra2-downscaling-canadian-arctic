# SPDX-License-Identifier: Apache-2.0
"""Evaluation needs TWO run dirs, because it needs two nets from different runs.

Generation is ``regression mean + diffusion residual``, and WORKFLOW.md section B trains a
diffusion run against an EARLIER regression run -- so a diffusion run dir holds only
``checkpoints_diffusion``. Auto-discovery from a single ``OUTPUT_DIR`` therefore could never
find the regression net for a real run: the script's default path was unreachable for the
layout its own workflow produces.

``resolve_eval_paths`` takes ``REG_RUN`` and ``RES_RUN`` (or explicit ``REG_CKPT`` /
``RES_CKPT``), keeps ``OUTPUT_DIR`` working as a fallback that sets both, and derives ``NC_DIR``
from the run being evaluated.

    pytest training_mini/tests/test_eval_paths.py -q
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

COMMON_SH = Path(__file__).resolve().parents[1] / "slurm" / "common.sh"
RUN_EVAL = Path(__file__).resolve().parents[2] / "evaluate" / "run_eval.sh"

pytestmark = pytest.mark.skipif(not COMMON_SH.exists(), reason="slurm/common.sh not present")


def resolve(env: dict):
    """Run resolve_eval_paths with `env` and report the resolved variables."""
    script = (
        f'set -uo pipefail\nsource "{COMMON_SH}"\n'
        "if resolve_eval_paths; then\n"
        '  echo "REG_CKPT=$REG_CKPT"; echo "RES_CKPT=$RES_CKPT"; echo "NC_DIR=$NC_DIR"\n'
        '  echo "REG_RUN=$REG_RUN"; echo "RES_RUN=$RES_RUN"\n'
        "else exit 1; fi\n"
    )
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                         env={"PATH": "/usr/bin:/bin", **env})
    parsed = dict(ln.split("=", 1) for ln in out.stdout.splitlines() if "=" in ln)
    return out.returncode, parsed, out.stderr


def make_run(root: Path, name: str, kind: str, nimgs=(5000, 800000)) -> Path:
    """A run dir holding checkpoints of one kind only, as real runs do."""
    model = {"regression": "CorrDiffRegressionUNet", "diffusion": "EDMPrecondSuperResolution"}[kind]
    d = root / name / f"checkpoints_{kind}"
    d.mkdir(parents=True)
    for n in nimgs:
        (d / f"{model}.0.{n}.mdlus").touch()
    return root / name


# ------------------------------------------------------------------ the layout that failed
def test_separate_run_dirs_resolve_both_nets(tmp_path):
    """The real layout: regression_2/ and diffusion_2/ are different runs."""
    reg = make_run(tmp_path, "regression_2", "regression")
    res = make_run(tmp_path, "diffusion_2", "diffusion", nimgs=(1000000, 2000000))
    rc, got, err = resolve({"REG_RUN": str(reg), "RES_RUN": str(res)})
    assert rc == 0, err
    assert got["REG_CKPT"].endswith("CorrDiffRegressionUNet.0.800000.mdlus")
    assert got["RES_CKPT"].endswith("EDMPrecondSuperResolution.0.2000000.mdlus")
    assert got["NC_DIR"] == f"{res}/eval"       # NetCDFs go with the run being evaluated


def test_a_diffusion_run_alone_is_rejected_with_guidance(tmp_path):
    """What actually happened: OUTPUT_DIR pointed at a diffusion run, which has no
    checkpoints_regression."""
    res = make_run(tmp_path, "diffusion_2", "diffusion")
    rc, _, err = resolve({"OUTPUT_DIR": str(res)})
    assert rc == 1
    assert "no regression checkpoint" in err
    assert "REG_RUN=" in err                    # says which knob to reach for
    assert "trained against" in err             # and why it is not in this dir


def test_missing_diffusion_net_is_rejected(tmp_path):
    reg = make_run(tmp_path, "regression_2", "regression")
    rc, _, err = resolve({"REG_RUN": str(reg)})
    assert rc == 1 and "no diffusion checkpoint" in err


# ------------------------------------------------------------------ explicit checkpoints
def test_explicit_checkpoints_need_no_run_dirs(tmp_path):
    reg = make_run(tmp_path, "regression_2", "regression")
    res = make_run(tmp_path, "diffusion_2", "diffusion", nimgs=(2000000,))
    rc, got, err = resolve({
        "REG_CKPT": f"{reg}/checkpoints_regression/CorrDiffRegressionUNet.0.5000.mdlus",
        "RES_CKPT": f"{res}/checkpoints_diffusion/EDMPrecondSuperResolution.0.2000000.mdlus",
    })
    assert rc == 0, err
    assert got["REG_CKPT"].endswith("0.5000.mdlus")      # honoured, not overridden by latest
    assert got["NC_DIR"] == f"{res}/eval"                # derived from the checkpoint's run dir


def test_explicit_checkpoint_wins_over_the_run_dir(tmp_path):
    reg = make_run(tmp_path, "regression_2", "regression")
    res = make_run(tmp_path, "diffusion_2", "diffusion", nimgs=(1000000, 2000000))
    rc, got, _ = resolve({
        "REG_RUN": str(reg), "RES_RUN": str(res),
        "RES_CKPT": f"{res}/checkpoints_diffusion/EDMPrecondSuperResolution.0.1000000.mdlus",
    })
    assert rc == 0
    assert got["RES_CKPT"].endswith("0.1000000.mdlus")
    assert got["REG_CKPT"].endswith("0.800000.mdlus")    # still auto for the other one


def test_nonexistent_explicit_checkpoint_is_rejected(tmp_path):
    reg = make_run(tmp_path, "regression_2", "regression")
    rc, _, err = resolve({"REG_RUN": str(reg), "RES_CKPT": str(tmp_path / "nope.mdlus")})
    assert rc == 1 and "no diffusion checkpoint" in err


# ------------------------------------------------------------------ backward compatibility
def test_output_dir_still_works_when_one_dir_holds_both(tmp_path):
    """Older commands and the pre-change README must keep working."""
    d = tmp_path / "corrdiff_mini"
    for kind, model, n in (("regression", "CorrDiffRegressionUNet", 800000),
                           ("diffusion", "EDMPrecondSuperResolution", 2000000)):
        (d / f"checkpoints_{kind}").mkdir(parents=True)
        (d / f"checkpoints_{kind}" / f"{model}.0.{n}.mdlus").touch()
    rc, got, err = resolve({"OUTPUT_DIR": str(d)})
    assert rc == 0, err
    assert got["REG_CKPT"].endswith("0.800000.mdlus")
    assert got["RES_CKPT"].endswith("0.2000000.mdlus")
    assert got["NC_DIR"] == f"{d}/eval"


def test_run_dirs_override_output_dir(tmp_path):
    both = tmp_path / "legacy"
    for kind, model in (("regression", "CorrDiffRegressionUNet"),
                        ("diffusion", "EDMPrecondSuperResolution")):
        (both / f"checkpoints_{kind}").mkdir(parents=True)
        (both / f"checkpoints_{kind}" / f"{model}.0.1.mdlus").touch()
    reg = make_run(tmp_path, "regression_9", "regression", nimgs=(900000,))
    rc, got, _ = resolve({"OUTPUT_DIR": str(both), "REG_RUN": str(reg)})
    assert rc == 0
    assert got["REG_CKPT"].endswith("0.900000.mdlus")    # REG_RUN wins
    assert got["RES_CKPT"].startswith(str(both))         # RES falls back to OUTPUT_DIR


def test_explicit_nc_dir_is_left_alone(tmp_path):
    reg = make_run(tmp_path, "regression_2", "regression")
    res = make_run(tmp_path, "diffusion_2", "diffusion")
    rc, got, _ = resolve({"REG_RUN": str(reg), "RES_RUN": str(res),
                          "NC_DIR": str(tmp_path / "elsewhere")})
    assert rc == 0 and got["NC_DIR"] == str(tmp_path / "elsewhere")


def test_nothing_set_at_all_is_rejected():
    rc, _, err = resolve({})
    assert rc == 1 and "no regression checkpoint" in err


# ------------------------------------------------------------------ the script uses it
def test_run_eval_delegates_and_documents_the_new_form():
    text = RUN_EVAL.read_text()
    assert "resolve_eval_paths || exit 1" in text
    assert "REG_RUN=" in text and "RES_RUN=" in text
    # the old single-dir auto-discovery must be gone from the script itself
    assert 'latest_ckpt "$OUTPUT_DIR/checkpoints_regression"' not in text
