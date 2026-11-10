# SPDX-License-Identifier: Apache-2.0
"""EVAL_MODE=reg must score a regression run that has no diffusion stage yet.

``resolve_eval_paths`` required BOTH nets, so there was no way to evaluate a regression run over
the test years until its diffusion stage existed -- which is exactly the sea-ice comparison
(regression_2 with ice vs regression_3 without, at a matched sample count).

    pytest training_mini/tests/test_eval_mode.py -q
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

TRAIN_DIR = Path(__file__).resolve().parents[1]
REPO = TRAIN_DIR.parent
COMMON_SH = TRAIN_DIR / "slurm" / "common.sh"

pytestmark = pytest.mark.skipif(not COMMON_SH.exists(), reason="slurm/common.sh not present")


def sh(script: str, **env):
    pre = "".join(f'export {k}="{v}"\n' for k, v in env.items())
    return subprocess.run(
        ["bash", "-c", f'set -uo pipefail\nsource "{COMMON_SH}"\n{pre}{script}'],
        capture_output=True, text=True,
    )


def _run_dirs(tmp_path, reg_nimg=1200000, res_nimg=None):
    reg = tmp_path / "regression_3" / "checkpoints_regression"
    reg.mkdir(parents=True)
    (reg / f"CorrDiffRegressionUNet.0.{reg_nimg}.mdlus").write_text("w")
    if res_nimg is not None:
        res = tmp_path / "diffusion_3" / "checkpoints_diffusion"
        res.mkdir(parents=True)
        (res / f"EDMPrecondSuperResolution.0.{res_nimg}.mdlus").write_text("w")
    return tmp_path


def test_reg_mode_resolves_without_a_diffusion_net(tmp_path):
    _run_dirs(tmp_path)
    out = sh('resolve_eval_paths && echo "REG=$REG_CKPT" && echo "NC=$NC_DIR"',
             EVAL_MODE="reg", REG_RUN=str(tmp_path / "regression_3"), RES_RUN="", NC_DIR="")
    assert out.returncode == 0, out.stderr
    assert "regression_3/checkpoints_regression/CorrDiffRegressionUNet.0.1200000.mdlus" in out.stdout
    # NetCDFs must land with the run being evaluated, which here is the regression run
    assert f"NC={tmp_path}/regression_3/eval" in out.stdout


def test_default_mode_still_demands_the_diffusion_net(tmp_path):
    _run_dirs(tmp_path)
    out = sh("resolve_eval_paths", REG_RUN=str(tmp_path / "regression_3"), RES_RUN="", NC_DIR="")
    assert out.returncode == 1
    assert "no diffusion checkpoint" in out.stderr
    assert "EVAL_MODE=reg" in out.stderr, "the error must name the way out"


def test_both_nets_still_resolve_in_the_default_mode(tmp_path):
    _run_dirs(tmp_path, res_nimg=2000000)
    out = sh('resolve_eval_paths && echo "NC=$NC_DIR"',
             REG_RUN=str(tmp_path / "regression_3"), RES_RUN=str(tmp_path / "diffusion_3"),
             NC_DIR="")
    assert out.returncode == 0, out.stderr
    assert f"NC={tmp_path}/diffusion_3/eval" in out.stdout, "default stays with the diffusion run"


def test_a_missing_regression_net_is_fatal_in_either_mode(tmp_path):
    (tmp_path / "empty").mkdir()
    for mode in ("all", "reg"):
        out = sh("resolve_eval_paths", EVAL_MODE=mode, REG_RUN=str(tmp_path / "empty"),
                 RES_RUN="", NC_DIR="")
        assert out.returncode == 1, mode
        assert "no regression checkpoint" in out.stderr


def test_run_eval_skips_the_full_pass_and_scores_reg_only():
    text = (REPO / "evaluate/run_eval.sh").read_text()
    assert 'EVAL_MODE="${EVAL_MODE:-all}"' in text, "must default to the full comparison"
    assert 'if [[ "$EVAL_MODE" == "reg" ]]; then' in text
    # The decisive one: reg mode must NOT score a full.nc it did not write this run. Slice only
    # as far as the `else`, or the window spills into the full-comparison branch.
    branch = text.split("CRPS / MAE per channel: regression only", 1)[1].split("else", 1)[0]
    assert "--nc reg=" in branch, branch
    assert "full=" not in branch, \
        "reg mode must not score a stale full.nc from an earlier evaluation"
