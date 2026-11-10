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


# ------------------------------------------------------------------ training: the config decides
# Training has no NOICE flag -- the config's dataset.lr_channels IS the channel selection. The
# hazard is pairing a no-ice diffusion run with a WITH-ice regression base (regression_2 for
# regression_3), which costs an allocation, a staging pass and a model build before it fails.
def _cfg(tmp_path, channels):
    import textwrap
    p = tmp_path / "c.yaml"
    body = "null" if channels is None else "[" + ", ".join(channels) + "]"
    p.write_text(textwrap.dedent(f"""\
        dataset:
            lr_channels: {body}
        training:
            hp:
                total_batch_size: 64
        """))
    return p


def test_config_list_overrides_the_store(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(lr_channels, "store_channels", lambda *a, **k: list(FULL))
    cfg = _cfg(tmp_path, FULL[:-1])                       # the noice list
    monkeypatch.setattr(sys, "argv", ["lr_channels.py", "--data-dir", str(tmp_path),
                                      "--config", str(cfg)])
    assert lr_channels.main() == 0
    assert capsys.readouterr().out.strip() == "[" + ",".join(FULL[:-1]) + "]"


def test_null_lr_channels_means_every_channel(tmp_path, monkeypatch, capsys):
    """`lr_channels: null` is the with-ice default -- all 12, not an empty selection."""
    monkeypatch.setattr(lr_channels, "store_channels", lambda *a, **k: list(FULL))
    cfg = _cfg(tmp_path, None)
    monkeypatch.setattr(sys, "argv", ["lr_channels.py", "--data-dir", str(tmp_path),
                                      "--config", str(cfg)])
    assert lr_channels.main() == 0
    assert capsys.readouterr().out.strip() == "[" + ",".join(FULL) + "]"


def test_noice_config_against_withice_regression_base_is_refused(tmp_path, monkeypatch):
    """THE PAIRING HAZARD: an 11-channel config with a 12-channel (in=20) regression net."""
    monkeypatch.setattr(lr_channels, "store_channels", lambda *a, **k: list(FULL))
    import collect_run
    monkeypatch.setattr(collect_run, "checkpoint_provenance",
                        lambda p, s: {"in_channels": 20, "lr_n": 12, "sea_ice": "yes"})
    cfg = _cfg(tmp_path, FULL[:-1])
    monkeypatch.setattr(sys, "argv", ["lr_channels.py", "--data-dir", str(tmp_path),
                                      "--config", str(cfg), "--check-against", "/r2.mdlus",
                                      "--stage", "regression"])
    assert lr_channels.main() == 1


def test_matching_config_and_base_is_accepted(tmp_path, monkeypatch):
    monkeypatch.setattr(lr_channels, "store_channels", lambda *a, **k: list(FULL))
    import collect_run
    monkeypatch.setattr(collect_run, "checkpoint_provenance",
                        lambda p, s: {"in_channels": 19, "lr_n": 11, "sea_ice": "no"})
    cfg = _cfg(tmp_path, FULL[:-1])
    monkeypatch.setattr(sys, "argv", ["lr_channels.py", "--data-dir", str(tmp_path),
                                      "--config", str(cfg), "--check-against", "/r3.mdlus",
                                      "--stage", "regression"])
    assert lr_channels.main() == 0


def test_config_asking_for_an_absent_channel_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(lr_channels, "store_channels", lambda *a, **k: list(FULL))
    cfg = _cfg(tmp_path, ["t2m", "u10", "sst"])           # sst is not in this dataset
    monkeypatch.setattr(sys, "argv", ["lr_channels.py", "--data-dir", str(tmp_path),
                                      "--config", str(cfg)])
    assert lr_channels.main() == 1


def test_diffusion_training_checks_its_regression_base():
    text = (REPO / "training_mini/slurm/train_diffusion.sh").read_text()
    assert "--config \"conf/$CONFIG.yaml\"" in text, "train_diffusion.sh must check the pairing"
    assert "--check-against \"$REG_CKPT\"" in text


# ------------------------------------------------------------------ the advice must fit the caller
# A real failure (diffusion_3, 2026-11-10): `CONFIG=..._noice` was typed on its own line, so it
# was a plain shell variable, never exported, and sbatch --export=ALL passed the DEFAULT with-ice
# config to the job. The guard caught the mismatch but told the user to "re-run with NOICE=1" --
# which does nothing for a training job. Naming the wrong knob sends people round in circles.
def _mismatch(tmp_path, monkeypatch, capsys, *, use_config, ckpt_lr_n=11, ckpt_in=19):
    monkeypatch.setattr(lr_channels, "store_channels", lambda *a, **k: list(FULL))
    import collect_run
    monkeypatch.setattr(collect_run, "checkpoint_provenance",
                        lambda p, s: {"in_channels": ckpt_in, "lr_n": ckpt_lr_n})
    argv = ["lr_channels.py", "--data-dir", str(tmp_path),
            "--check-against", "/r3.mdlus", "--stage", "regression"]
    if use_config:
        cfg = tmp_path / "config_training_era5_carra2_mini_diffusion.yaml"
        cfg.write_text("dataset:\n    lr_channels: null\n")
        argv += ["--config", str(cfg)]
    monkeypatch.setattr(sys, "argv", argv)
    rc = lr_channels.main()
    return rc, capsys.readouterr().err


def test_training_caller_is_told_about_the_config_not_noice(tmp_path, monkeypatch, capsys):
    rc, err = _mismatch(tmp_path, monkeypatch, capsys, use_config=True)
    assert rc == 1
    assert "_noice config" in err
    assert "NOICE=1" not in err, "NOICE is not a training knob; advising it wastes a submission"


def test_training_caller_is_warned_about_the_export_gotcha(tmp_path, monkeypatch, capsys):
    """The config that REACHED the job is named, so an unexported CONFIG is self-evident."""
    rc, err = _mismatch(tmp_path, monkeypatch, capsys, use_config=True)
    assert "config_training_era5_carra2_mini_diffusion" in err
    assert "is NOT exported" in err


def test_inference_caller_is_told_about_noice(tmp_path, monkeypatch, capsys):
    rc, err = _mismatch(tmp_path, monkeypatch, capsys, use_config=False)
    assert rc == 1
    assert "NOICE=1" in err
    assert "_noice config" not in err, "CONFIG is not an inference knob"


def test_the_reverse_mismatch_names_the_other_direction(tmp_path, monkeypatch, capsys):
    """A _noice config against a with-ice base: 12-channel checkpoint, 11 resolved."""
    monkeypatch.setattr(lr_channels, "store_channels", lambda *a, **k: list(FULL))
    import collect_run
    monkeypatch.setattr(collect_run, "checkpoint_provenance",
                        lambda p, s: {"in_channels": 20, "lr_n": 12})
    cfg = tmp_path / "c.yaml"
    cfg.write_text("dataset:\n    lr_channels: [" + ", ".join(FULL[:-1]) + "]\n")
    monkeypatch.setattr(sys, "argv", ["lr_channels.py", "--data-dir", str(tmp_path),
                                      "--config", str(cfg), "--check-against", "/r2.mdlus",
                                      "--stage", "regression"])
    assert lr_channels.main() == 1
    err = capsys.readouterr().err
    assert "without _noice" in err or "with-ice base" in err


@pytest.mark.parametrize("script", ["train_regression.sh", "train_diffusion.sh"])
def test_training_scripts_echo_the_config_before_staging(script):
    """A wrong CONFIG must be visible in the first lines, not only at "Launching ..." after
    the staging pass has already been paid for."""
    text = (TRAIN_DIR / "slurm" / script).read_text()
    # Anchor on the invocation, not the first mention -- common.sh is sourced with a comment
    # listing stage_shards, which is 50 lines above the echo.
    head = text.split('stage_shards "$DATA_DIR"', 1)[0]
    assert "[config] CONFIG=$CONFIG" in head, f"{script} does not echo CONFIG before staging"
