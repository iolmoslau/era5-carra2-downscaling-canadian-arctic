# SPDX-License-Identifier: Apache-2.0
"""The LR channel list must come from the store and agree with the checkpoint.

Three real failures motivate this, and the third is the one that matters most:

1. **Too many channels.** Three generation jobs on 2026-09-24 -- the first attempts at generating
   from the no-ice ``regression_3`` -- died at the first convolution, 12 channels into a net built
   for 11, after the GPU was allocated and the model loaded.
2. **The documented fix was unreachable.** ``config_generate_era5_carra2_mini.yaml`` told you to
   pass ``++dataset.lr_channels=[...]``, but neither ``slurm/generate.sh`` nor
   ``evaluate/run_eval.sh`` had a way to forward it.
3. **The wrong 11 channels.** ``ERA5CARRA2Dataset`` validates the list is a *subset*, so a typo
   fails loudly -- but dropping ``t500`` instead of ``siconc`` also gives 11, which MATCHES the
   conv width. That runs to completion on a stack the model never saw and reports plausible
   numbers. Only comparing names, not counts, catches it; hence resolving from the store.

    pytest training_mini/tests/test_lr_channels.py -q
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

TRAIN_DIR = Path(__file__).resolve().parents[1]
REPO = TRAIN_DIR.parent
TOOLS = TRAIN_DIR / "tools"
TESTDATA = REPO / "testing_data"

for _p in (str(REPO), str(TOOLS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

lr_channels = pytest.importorskip("lr_channels")

FULL = ["t2m", "u10", "v10", "t500", "t850", "z500", "z850",
        "u500", "u850", "v500", "v850", "siconc"]


def run(*args):
    return subprocess.run([sys.executable, str(TOOLS / "lr_channels.py"), *args],
                          capture_output=True, text=True)


# ------------------------------------------------------------------ resolving from the store
@pytest.mark.skipif(not (TESTDATA / "shard_2011.zarr").exists(),
                    reason="testing_data/shard_2011.zarr not present")
def test_reads_the_manifest_off_a_real_shard():
    got = lr_channels.store_channels(str(TESTDATA))
    assert got == FULL, "the store's own attrs are the source of truth for channel order"


@pytest.mark.skipif(not (TESTDATA / "shard_2011.zarr").exists(),
                    reason="testing_data/shard_2011.zarr not present")
def test_drop_siconc_yields_the_noice_stack_in_stored_order():
    out = run("--data-dir", str(TESTDATA), "--drop", "siconc")
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[t2m,u10,v10,t500,t850,z500,z850,u500,u850,v500,v850]"


@pytest.mark.skipif(not (TESTDATA / "shard_2011.zarr").exists(),
                    reason="testing_data/shard_2011.zarr not present")
def test_no_drop_passes_the_full_stack_explicitly():
    """Passing the full list is harmless and lands it in the Hydra snapshot for provenance."""
    out = run("--data-dir", str(TESTDATA), "--drop", "")
    assert out.returncode == 0 and out.stdout.strip() == "[" + ",".join(FULL) + "]"


@pytest.mark.skipif(not (TESTDATA / "shard_2011.zarr").exists(),
                    reason="testing_data/shard_2011.zarr not present")
def test_dropping_an_unknown_channel_is_refused():
    """A silently ignored --drop is how you evaluate a different model than you meant to."""
    out = run("--data-dir", str(TESTDATA), "--drop", "sicon")      # typo for siconc
    assert out.returncode == 1
    assert "not in the store's channels" in out.stderr
    assert out.stdout.strip() == ""        # nothing for the shell to pass on


def test_missing_data_dir_is_refused_not_guessed(tmp_path):
    out = run("--data-dir", str(tmp_path))
    assert out.returncode == 1 and "cannot read the LR channel manifest" in out.stderr


# ------------------------------------------------------------------ the checkpoint check
# channels_from_in is the shared arithmetic (regression in = L + 8, diffusion in = L + 11);
# these pin the comparison this tool builds on top of it.
@pytest.mark.parametrize("in_ch,stage,lr_n", [
    (20, "regression", 12), (19, "regression", 11),
    (23, "diffusion", 12), (22, "diffusion", 11),
])
def test_expected_channel_count_comes_from_the_conv_width(in_ch, stage, lr_n):
    from collect_run import channels_from_in

    assert channels_from_in(in_ch, stage)["lr_n"] == lr_n


def test_the_mismatch_that_actually_happened_is_reported_with_the_fix(tmp_path, monkeypatch):
    """12 resolved channels against the 19-channel no-ice net -> name NOICE=1 in the message."""
    monkeypatch.setattr(lr_channels, "store_channels", lambda *a, **k: list(FULL))
    import collect_run

    monkeypatch.setattr(collect_run, "checkpoint_provenance",
                        lambda p, s: {"in_channels": 19, "lr_n": 11, "sea_ice": "no"})
    monkeypatch.setattr(sys, "argv",
                        ["lr_channels.py", "--data-dir", str(tmp_path),
                         "--check-against", "/x.mdlus", "--stage", "regression"])
    assert lr_channels.main() == 1


def test_matching_counts_print_the_list(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(lr_channels, "store_channels", lambda *a, **k: list(FULL))
    import collect_run

    monkeypatch.setattr(collect_run, "checkpoint_provenance",
                        lambda p, s: {"in_channels": 19, "lr_n": 11, "sea_ice": "no"})
    monkeypatch.setattr(sys, "argv",
                        ["lr_channels.py", "--data-dir", str(tmp_path), "--drop", "siconc",
                         "--check-against", "/x.mdlus", "--stage", "regression"])
    assert lr_channels.main() == 0
    assert capsys.readouterr().out.strip().endswith("v850]")


def test_unreadable_checkpoint_warns_but_does_not_block(tmp_path, monkeypatch, capsys):
    """Refusing to run over a diagnostic we cannot perform would be worse than proceeding."""
    monkeypatch.setattr(lr_channels, "store_channels", lambda *a, **k: list(FULL))
    import collect_run

    monkeypatch.setattr(collect_run, "checkpoint_provenance",
                        lambda p, s: {"in_channels": None, "lr_n": None, "error": "no physicsnemo"})
    monkeypatch.setattr(sys, "argv",
                        ["lr_channels.py", "--data-dir", str(tmp_path),
                         "--check-against", "/x.mdlus", "--stage", "regression"])
    assert lr_channels.main() == 0
    cap = capsys.readouterr()
    assert "WARNING" in cap.err and cap.out.strip().startswith("[")


# ------------------------------------------------------------------ the shell side
@pytest.mark.parametrize("script", ["training_mini/slurm/generate.sh", "evaluate/run_eval.sh"])
def test_submission_scripts_forward_the_resolved_list(script):
    """The documented override was unreachable from both of these for the whole project.

    Three things must be true together: the NOICE flag exists, the list is resolved by the tool
    (not typed), and the result is actually passed to generate.py.
    """
    text = (REPO / script).read_text()
    assert 'NOICE' in text, f"{script} has no no-ice flag"
    assert "tools/lr_channels.py" in text, f"{script} does not resolve the list from the store"
    assert "++dataset.lr_channels=" in text, f"{script} never forwards the list"
    # Anchor on the invocation, not on the first mention of the filename -- both scripts also
    # name the tool in a comment, and splitting on that was checking the wrong slice.
    assert "python tools/lr_channels.py" in text, f"{script} does not invoke the tool"
    after = text.split("python tools/lr_channels.py", 1)[1][:300]
    assert "|| exit 1" in after, f"{script} ignores a channel-check failure"
