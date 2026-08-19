# SPDX-License-Identifier: Apache-2.0
"""A checkpoint that exists but will not load must stop the run, not restart it silently.

Upstream CorrDiff wraps all three ``load_checkpoint`` calls in ``except Exception: pass``, so a
truncated ``.mdlus`` (preemption mid-write), a transient filesystem error, or a shape mismatch
after a config change quietly restarts training from random weights -- with the sample counter,
LR schedule and checkpoint filenames all still looking correct. The ``[thesis]`` guard in
``train.py`` keeps the one legitimate fresh-start case (an empty checkpoint directory) and turns
everything else into a hard error.

``train.py`` imports physicsnemo / hydra / nvtx / wandb at module scope, none of which are
installed on a dev box, so the helpers are lifted out by AST instead of importing the module.
That also means these tests exercise the real vendored source, not a copy.

    pytest training_mini/tests/test_checkpoint_resume_guard.py -q
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

TRAIN_PY = Path(__file__).resolve().parents[1] / "train.py"
WANTED = {"checkpoint_list", "_checkpoints_present", "_handle_checkpoint_load_failure"}

pytestmark = pytest.mark.skipif(not TRAIN_PY.exists(), reason="train.py not present")


@pytest.fixture(scope="module")
def helpers() -> dict:
    """Exec just the checkpoint helpers out of the vendored train.py."""
    tree = ast.parse(TRAIN_PY.read_text())
    ns: dict = {"os": os}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in WANTED:
            module = ast.Module(body=[node], type_ignores=[])
            exec(compile(module, str(TRAIN_PY), "exec"), ns)  # noqa: S102
    missing = WANTED - set(ns)
    assert not missing, f"helpers missing from train.py: {sorted(missing)}"
    return ns


class Recorder:
    """Stand-in for logger0 (RankZeroLoggingWrapper)."""

    def __init__(self):
        self.warnings: list[str] = []

    def warning(self, msg):
        self.warnings.append(str(msg))


def ckpt_dir(tmp_path, mdlus=(), pt=()) -> Path:
    d = tmp_path / "checkpoints_regression"
    d.mkdir(parents=True, exist_ok=True)
    for n in mdlus:
        (d / f"CorrDiffRegressionUNet.0.{n}.mdlus").write_text("weights")
    for n in pt:
        (d / f"CorrDiffRegressionUNet.0.{n}.pt").write_text("optimizer")
    return d


# ------------------------------------------------------------------ legitimate fresh starts
def test_empty_dir_is_a_fresh_start(helpers, tmp_path):
    """The one case upstream is right about: nothing to resume from."""
    d = ckpt_dir(tmp_path)
    log = Recorder()
    assert helpers["_handle_checkpoint_load_failure"](str(d), OSError("x"), "w", log) is True
    assert log.warnings == []


def test_absent_dir_is_a_fresh_start(helpers, tmp_path):
    """First-ever run: get_checkpoint_dir names a directory that does not exist yet."""
    missing = tmp_path / "does_not_exist"
    assert helpers["_handle_checkpoint_load_failure"](str(missing), OSError("x"), "w") is True


# ------------------------------------------------------------------ the failure being closed
def test_unloadable_checkpoint_raises(helpers, tmp_path):
    d = ckpt_dir(tmp_path, mdlus=(5000, 800000, 1500000))
    with pytest.raises(RuntimeError) as ei:
        helpers["_handle_checkpoint_load_failure"](
            str(d), OSError("truncated"), "resume model weights", Recorder()
        )
    msg = str(ei.value)
    assert "resume model weights" in msg
    assert "1500000" in msg          # names the newest, so you know what to inspect
    assert str(d) in msg             # and where
    assert "CORRDIFF_ALLOW_BAD_CHECKPOINT" in msg   # and the way out


def test_original_error_is_chained_not_swallowed(helpers, tmp_path):
    d = ckpt_dir(tmp_path, mdlus=(400000,))
    cause = OSError("unexpected EOF")
    with pytest.raises(RuntimeError) as ei:
        helpers["_handle_checkpoint_load_failure"](str(d), cause, "w", Recorder())
    assert ei.value.__cause__ is cause


def test_escape_hatch_warns_instead_of_raising(helpers, tmp_path, monkeypatch):
    monkeypatch.setenv("CORRDIFF_ALLOW_BAD_CHECKPOINT", "1")
    d = ckpt_dir(tmp_path, mdlus=(800000,))
    log = Recorder()
    assert helpers["_handle_checkpoint_load_failure"](
        str(d), OSError("boom"), "resume model weights", log
    ) is True
    assert len(log.warnings) == 1
    warned = log.warnings[0]
    # the warning must still say what is being given up, and why it was allowed
    assert "CORRDIFF_ALLOW_BAD_CHECKPOINT=1" in warned
    assert "randomly initialised weights" in warned
    assert "boom" in warned                      # the underlying error is not hidden


def test_escape_hatch_only_on_exactly_1(helpers, tmp_path, monkeypatch):
    monkeypatch.setenv("CORRDIFF_ALLOW_BAD_CHECKPOINT", "true")
    d = ckpt_dir(tmp_path, mdlus=(800000,))
    with pytest.raises(RuntimeError):
        helpers["_handle_checkpoint_load_failure"](str(d), OSError("x"), "w", Recorder())


def test_works_without_a_logger(helpers, tmp_path, monkeypatch, capsys):
    """rank>0 paths may pass logger=None; must not blow up on log.warning."""
    monkeypatch.setenv("CORRDIFF_ALLOW_BAD_CHECKPOINT", "1")
    d = ckpt_dir(tmp_path, mdlus=(1,))
    assert helpers["_handle_checkpoint_load_failure"](str(d), OSError("x"), "w") is True
    assert "CORRDIFF_ALLOW_BAD_CHECKPOINT=1" in capsys.readouterr().out


# ------------------------------------------------------------------ optimizer .pt is optional
def test_weights_without_optimizer_is_allowed(helpers, tmp_path):
    """WORKFLOW.md section C archives .mdlus only -- resuming from that must not hard-fail."""
    d = ckpt_dir(tmp_path, mdlus=(1500000,))          # no .pt
    assert helpers["_handle_checkpoint_load_failure"](
        str(d), OSError("x"), "resume optimizer state", Recorder(), suffix=".pt"
    ) is True


def test_corrupt_optimizer_still_raises(helpers, tmp_path):
    d = ckpt_dir(tmp_path, mdlus=(1500000,), pt=(1500000,))
    with pytest.raises(RuntimeError, match=r"\.pt"):
        helpers["_handle_checkpoint_load_failure"](
            str(d), OSError("x"), "resume optimizer state", Recorder(), suffix=".pt"
        )


def test_checkpoints_present_filters_by_suffix_and_orders_by_nimg(helpers, tmp_path):
    d = ckpt_dir(tmp_path, mdlus=(5000, 1500000, 800000), pt=(800000,))
    mdlus = helpers["_checkpoints_present"](str(d))
    assert mdlus == [
        "CorrDiffRegressionUNet.0.5000.mdlus",
        "CorrDiffRegressionUNet.0.800000.mdlus",
        "CorrDiffRegressionUNet.0.1500000.mdlus",
    ]
    assert helpers["_checkpoints_present"](str(d), ".pt") == [
        "CorrDiffRegressionUNet.0.800000.pt"
    ]


# ------------------------------------------------------------------ the sites stay wired up
def test_every_load_checkpoint_call_is_guarded():
    """Regression guard: no `except Exception: pass` may return around load_checkpoint."""
    tree = ast.parse(TRAIN_PY.read_text())
    unguarded = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        calls = {n.func.id for n in ast.walk(node) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name)}
        if "load_checkpoint" not in calls:
            continue
        for handler in node.handlers:
            names = {n.func.id for n in ast.walk(handler) if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Name)}
            if "_handle_checkpoint_load_failure" not in names:
                unguarded.append(f"line {handler.lineno}")
    assert not unguarded, (
        "load_checkpoint failures must go through _handle_checkpoint_load_failure; "
        f"unguarded handlers at: {unguarded}"
    )
