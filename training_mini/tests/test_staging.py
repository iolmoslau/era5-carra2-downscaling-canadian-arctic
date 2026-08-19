# SPDX-License-Identifier: Apache-2.0
"""Staging shards to node-local storage must not kill the job, and must track the config.

The old staging line was::

    cp -r "$DATA_DIR"/shard_20{11,12,13,14,15,16,17,18,19}.zarr "$SLURM_TMPDIR/data/"

Three problems: the year list was hardcoded and could drift from the config's ``dataset.years``;
it could not see a shard archived as ``shard_YYYY.zarr.zip``; and under ``set -euo pipefail`` one
missing year made ``cp`` return non-zero and killed the job at staging -- after the GPUs had been
allocated. Years now come from the config, both shard forms are found, and a missing year warns
rather than aborting (``ERA5CARRA2Dataset`` raises for any store the run actually needs, so that
check stays authoritative).

    pytest training_mini/tests/test_staging.py -q
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

TRAIN_DIR = Path(__file__).resolve().parents[1]
COMMON_SH = TRAIN_DIR / "slurm" / "common.sh"
CONF = TRAIN_DIR / "conf"
TRAIN_SCRIPTS = ["train_regression.sh", "train_diffusion.sh"]

pytestmark = pytest.mark.skipif(not COMMON_SH.exists(), reason="slurm/common.sh not present")


def sh(script: str, cwd: Path | None = None):
    """Run under the same shell options the SLURM jobs use."""
    return subprocess.run(
        ["bash", "-c", f'set -euo pipefail\nsource "{COMMON_SH}"\n{script}'],
        capture_output=True, text=True, cwd=str(cwd) if cwd else None,
    )


def make_data_dir(tmp_path, loose=(), zipped=()) -> Path:
    d = tmp_path / "data"
    d.mkdir(exist_ok=True)
    for y in loose:
        store = d / f"shard_{y}.zarr"
        store.mkdir()
        (store / "zarr.json").write_text("{}")
    for y in zipped:
        (d / f"shard_{y}.zarr.zip").write_text("archive")
    return d


# ------------------------------------------------------------------ locating a shard
def test_shard_src_prefers_the_archive(tmp_path):
    d = make_data_dir(tmp_path, loose=(2011,), zipped=(2011, 2020))
    assert sh(f"shard_src {d} 2011").stdout.strip() == f"{d}/shard_2011.zarr.zip"
    assert sh(f"shard_src {d} 2020").stdout.strip() == f"{d}/shard_2020.zarr.zip"


def test_shard_src_falls_back_to_the_loose_store(tmp_path):
    d = make_data_dir(tmp_path, loose=(2012,))
    assert sh(f"shard_src {d} 2012").stdout.strip() == f"{d}/shard_2012.zarr"


def test_shard_src_reports_absence(tmp_path):
    d = make_data_dir(tmp_path, loose=(2012,))
    out = sh(f"shard_src {d} 1999 || echo ABSENT")
    assert "ABSENT" in out.stdout and out.returncode == 0


# ------------------------------------------------------------------ the abort being fixed
def test_missing_year_does_not_abort_the_job(tmp_path):
    """THE P0-4 REGRESSION: one absent year used to kill the job after GPU allocation."""
    d = make_data_dir(tmp_path, loose=(2011, 2012))          # 2013 deliberately absent
    dest = tmp_path / "tmpdir"
    out = sh(f'stage_shards {d} {dest} 2011 2012 2013\necho REACHED_TRAINING')
    assert out.returncode == 0, out.stderr
    assert "REACHED_TRAINING" in out.stdout
    assert "2013" in out.stderr and "WARNING" in out.stderr
    assert sorted(p.name for p in dest.iterdir()) == ["shard_2011.zarr", "shard_2012.zarr"]


def test_the_old_hardcoded_cp_really_did_abort(tmp_path):
    """Guards the guard: confirm the fixture reproduces the original failure."""
    d = make_data_dir(tmp_path, loose=(2011, 2012))
    dest = tmp_path / "tmpdir"
    dest.mkdir()
    out = subprocess.run(
        ["bash", "-c", "set -euo pipefail\n"
         f'cp -r {d}/shard_20{{11,12,13}}.zarr {dest}/\necho REACHED_TRAINING'],
        capture_output=True, text=True,
    )
    assert out.returncode != 0
    assert "REACHED_TRAINING" not in out.stdout


def test_archived_shards_stage_too(tmp_path):
    """Unblocks P1-3: 9 loose shards are ~53k files; the archives are 9."""
    d = make_data_dir(tmp_path, loose=(2011,), zipped=(2012, 2013))
    dest = tmp_path / "tmpdir"
    assert sh(f"stage_shards {d} {dest} 2011 2012 2013").returncode == 0
    assert sorted(p.name for p in dest.iterdir()) == [
        "shard_2011.zarr", "shard_2012.zarr.zip", "shard_2013.zarr.zip"]


def test_staging_nothing_is_fatal(tmp_path):
    """A warning per year is right; staging zero shards is not recoverable."""
    d = make_data_dir(tmp_path)
    out = sh(f"stage_shards {d} {tmp_path / 'tmpdir'} 2011 2012")
    assert out.returncode == 1
    assert "staged 0 shards" in out.stderr


def test_all_years_present_is_quiet(tmp_path):
    d = make_data_dir(tmp_path, loose=(2011, 2012))
    out = sh(f"stage_shards {d} {tmp_path / 'tmpdir'} 2011 2012")
    assert out.returncode == 0
    assert "WARNING" not in out.stderr
    assert "staged 2 shard(s)" in out.stdout


# ------------------------------------------------------------------ years follow the config
@pytest.mark.parametrize("name", [p.name for p in CONF.glob("config_training_*.yaml")])
def test_config_years_unions_train_and_validation(name):
    cfg = yaml.safe_load((CONF / name).read_text())
    expected = sorted(set(cfg["dataset"]["years"]) | set(cfg["validation"]["years"]))
    got = sh(f'config_years "{CONF / name}"').stdout.split()
    assert [int(y) for y in got] == expected


def test_derived_years_match_the_list_that_was_hardcoded():
    """Behaviour-preserving for the shipped configs: still 2011..2019."""
    got = sh(f'config_years "{CONF / "config_training_era5_carra2_mini_regression.yaml"}"')
    assert got.stdout.split() == [str(y) for y in range(2011, 2020)]


@pytest.mark.parametrize("name", TRAIN_SCRIPTS)
def test_scripts_no_longer_hardcode_the_year_list(name):
    text = (TRAIN_DIR / "slurm" / name).read_text()
    offenders = [ln.strip() for ln in text.splitlines()
                 if "shard_20{" in ln and not ln.strip().startswith("#")]
    assert not offenders, f"{name} still brace-expands a fixed year list: {offenders}"
    assert "config_years" in text, f"{name} must derive YEARS from the config"


@pytest.mark.parametrize("name", TRAIN_SCRIPTS)
def test_scripts_allow_a_years_override(name):
    assert 'YEARS="${YEARS:-' in (TRAIN_DIR / "slurm" / name).read_text()


# ------------------------------------------------------------------ stats years, same bug class
def test_config_years_can_select_train_only(tmp_path):
    """Stats must cover dataset.years and NEVER validation.years (that would leak the held-out
    year into the normalization)."""
    cfg = CONF / "config_training_era5_carra2_mini_regression.yaml"
    parsed = yaml.safe_load(cfg.read_text())
    train = sh(f'config_years "{cfg}" dataset').stdout.split()
    val = sh(f'config_years "{cfg}" validation').stdout.split()
    assert [int(y) for y in train] == sorted(parsed["dataset"]["years"])
    assert [int(y) for y in val] == sorted(parsed["validation"]["years"])
    assert not set(train) & set(val), "train and validation years must not overlap"


def test_stats_block_is_not_hardcoded_and_reads_the_source_dir():
    """The stats build had the same hardcoded 2011..2018 list, against the STAGED dir -- so a
    YEARS subset would make make_stats exit on a store it could not find."""
    import re

    text = (TRAIN_DIR / "slurm" / "train_regression.sh").read_text()
    line = next(ln for ln in text.splitlines()
                if "make_stats.py" in ln and not ln.strip().startswith("#"))
    # reads the SOURCE dir, so a YEARS override cannot starve it of a train shard
    assert '--data-dir "$DATA_DIR"' in line, f"stats must read the source dir, got: {line.strip()}"
    # and takes its years from the config's dataset.years, not a literal list
    assert "--years $TRAIN_YEARS" in line, f"stats year list still hardcoded: {line.strip()}"
    assert re.search(r'TRAIN_YEARS="\$\{TRAIN_YEARS:-\$\(config_years .* dataset\)\}"', text)
