# SPDX-License-Identifier: Apache-2.0
"""A collected run must record what it actually was, not what someone typed afterwards.

This is audit item P2-5, and it bit immediately: ``results/regression_1/run_info.json`` carries a
``config`` field that ``collect_run.py`` never wrote, so it was hand-entered and cannot be
vouched for -- which left "was sea ice in this run?" unanswerable from the repo.

The fix that matters is ``channels_from_in``: a checkpoint's input-conv width is a fact about the
weights, so it settles the channel set even when the config name, the notes and memory disagree.
It also keeps working for the channel-ablation runs, where the count is neither 12 nor 11.

    pytest training_mini/tests/test_run_provenance.py -q
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

collect_run = pytest.importorskip("collect_run")


# ------------------------------------------------------------------ the arithmetic that decides
@pytest.mark.parametrize("stage,in_ch,lr_n,ice", [
    # from train.py: regression in = L + lsm(1) + N_grid(4); diffusion adds img_out(3)
    ("regression", 17, 12, "yes"),
    ("regression", 16, 11, "no"),
    ("diffusion", 20, 12, "yes"),
    ("diffusion", 19, 11, "no"),
])
def test_channel_count_recovered_from_conv_width(stage, in_ch, lr_n, ice):
    got = collect_run.channels_from_in(in_ch, stage)
    assert got == {"lr_n": lr_n, "sea_ice": ice}


def test_ablation_widths_report_unknown_rather_than_guessing():
    """A run with 9 LR channels is neither the full nor the noice set; saying "no" would be a lie."""
    got = collect_run.channels_from_in(14, "regression")
    assert got["lr_n"] == 9 and got["sea_ice"] == "unknown"


def test_unknown_stage_does_not_invent_a_count():
    assert collect_run.channels_from_in(17, "other") == {"lr_n": None, "sea_ice": "unknown"}
    assert collect_run.channels_from_in(None, "regression")["lr_n"] is None


# ------------------------------------------------------------------ graceful degradation
def test_missing_checkpoint_is_recorded_as_unknown_not_crashed():
    out = collect_run.checkpoint_provenance("", "regression")
    assert out["in_channels"] is None and out["sea_ice"] == "unknown"


def test_unreadable_checkpoint_records_why(tmp_path):
    """collect_run also runs where physicsnemo is absent; it must degrade, not abort the collect."""
    bad = tmp_path / "CorrDiffRegressionUNet.0.1000.mdlus"
    bad.write_text("not a checkpoint")
    out = collect_run.checkpoint_provenance(str(bad), "regression")
    assert out["in_channels"] is None
    assert out.get("error"), "the reason must be recorded, not swallowed"


# ------------------------------------------------------------------ resolved-config capture
def test_hydra_snapshot_is_preferred_over_hand_entry(tmp_path):
    (tmp_path / "config.yaml").write_text(
        "model:\n  name: regression\n"
        "dataset:\n"
        "  lr_channels: [t2m, u10, v10, t500, t850, z500, z850, u500, u850, v500, v850]\n"
        "  include_lsm: true\n  years: [2011, 2012]\n"
        "  stats_path: /p/stats.json\n  data_path: /p/derot\n"
        "training:\n  hp:\n    training_duration: 2000000\n    total_batch_size: 64\n"
    )
    hy = collect_run.hydra_provenance(str(tmp_path))
    assert hy["model_name"] == "regression"
    assert len(hy["lr_channels"]) == 11          # the noice set, read from what actually ran
    assert hy["stats_path"] == "/p/stats.json"
    assert hy["training_duration"] == 2000000


def test_hydra_snapshot_accepts_the_dir_or_the_file(tmp_path):
    (tmp_path / "config.yaml").write_text("dataset:\n  years: [2019]\n")
    assert collect_run.hydra_provenance(str(tmp_path))["years"] == [2019]
    assert collect_run.hydra_provenance(str(tmp_path / "config.yaml"))["years"] == [2019]


def test_absent_hydra_snapshot_is_not_fatal(tmp_path):
    assert collect_run.hydra_provenance(str(tmp_path)) == {}
    assert collect_run.hydra_provenance("") == {}


# ------------------------------------------------------------------ the index stays readable
def test_runs_csv_carries_the_identifying_columns():
    """These four are what let you tell two runs apart at a glance."""
    for col in ("config", "sea_ice", "lr_n", "in_channels"):
        assert col in collect_run.FIELDS, f"runs.csv must record {col}"


def test_existing_rows_survive_a_schema_change(tmp_path, monkeypatch):
    """Adding columns must not destroy rows written by the older collect_run."""
    csv_path = tmp_path / "runs.csv"
    csv_path.write_text("run,stage,date,notes\nregression_1,regression,2026-07-08,old row\n")
    monkeypatch.setattr(collect_run, "RESULTS", tmp_path)
    monkeypatch.setattr(collect_run, "CSV_PATH", csv_path)

    collect_run.upsert_csv({"run": "regression_3", "stage": "regression", "date": "2026-09-17",
                            "config": "config_training_era5_carra2_mini_regression",
                            "sea_ice": "yes", "lr_n": 12, "in_channels": 17})
    rows = list(csv.DictReader(csv_path.open()))
    assert [r["run"] for r in rows] == ["regression_1", "regression_3"]
    assert rows[0]["notes"] == "old row"          # preserved
    assert rows[0]["sea_ice"] == ""               # unknowable for the old run, left blank
    assert rows[1]["sea_ice"] == "yes"


def test_upsert_replaces_rather_than_duplicates(tmp_path, monkeypatch):
    csv_path = tmp_path / "runs.csv"
    monkeypatch.setattr(collect_run, "RESULTS", tmp_path)
    monkeypatch.setattr(collect_run, "CSV_PATH", csv_path)
    for ice in ("no", "yes"):
        collect_run.upsert_csv({"run": "regression_3", "sea_ice": ice})
    rows = list(csv.DictReader(csv_path.open()))
    assert len(rows) == 1 and rows[0]["sea_ice"] == "yes"
