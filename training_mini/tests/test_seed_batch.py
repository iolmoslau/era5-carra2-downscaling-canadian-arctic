# SPDX-License-Identifier: Apache-2.0
"""``seed_batch_size`` must divide ``num_ensembles``, and the scripts must enforce it.

physicsnemo's ``diffusion_step`` sizes its latents from the conditioning batch rather than from
the seed count::

    latents_shape = [img_lr.shape[0], img_out_channels, img_shape[0], img_shape[1]]

and ``generate.py`` expands ``img_lr`` to ``seed_batch_size``. So when ``num_ensembles`` is not a
multiple of ``seed_batch_size``, ``tensor_split`` hands the last call a short batch, that call
still emits ``seed_batch_size`` members, and the total no longer matches the regression mean it
is added to -- surfacing as a tensor-shape error deep inside physicsnemo. ``seed_batch_size: 1``
is the only always-safe value, which is why it was the default, but it serialises the ensemble.

These tests pin the helper that picks a safe value and the guard that rejects an unsafe one, and
check the shipped configs are internally consistent.

    pytest training_mini/tests/test_seed_batch.py -q
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


def seed_batch_for(n, cap=None) -> int:
    out = sh(f"seed_batch_for {n}" + (f" {cap}" if cap is not None else ""))
    assert out.returncode == 0, out.stderr
    return int(out.stdout.strip())


# ------------------------------------------------------------------ picking a safe value
@pytest.mark.parametrize("n,expected", [
    (1, 1),      # regression pass
    (2, 2),
    (4, 4),
    (15, 5),     # the eval config default: 8 and 7 and 6 do not divide 15
    (16, 8),     # capped at 8 even though 16 divides itself
    (32, 8),     # the README's big-eval example
    (7, 7),      # prime below the cap
    (13, 1),     # prime above the cap -> nothing but 1 is safe
    (100, 5),    # 8,7,6 do not divide 100
])
def test_seed_batch_for_picks_largest_safe_divisor(n, expected):
    assert seed_batch_for(n) == expected


@pytest.mark.parametrize("n", [1, 2, 4, 7, 13, 15, 16, 32, 100])
def test_seed_batch_for_always_divides(n):
    assert n % seed_batch_for(n) == 0


def test_cap_is_honoured(n=32):
    assert seed_batch_for(n, 4) == 4
    assert seed_batch_for(n, 1) == 1
    assert seed_batch_for(n, 64) == 32       # cap above n clamps to n


# ------------------------------------------------------------------ rejecting an unsafe value
@pytest.mark.parametrize("n,b", [(15, 8), (15, 2), (32, 5), (4, 3), (7, 2)])
def test_require_divisor_rejects_uneven_splits(n, b):
    out = sh(f"require_divisor {n} {b}")
    assert out.returncode == 1
    assert f"must be a positive divisor of NUM_ENS={n}" in out.stderr


@pytest.mark.parametrize("n,b", [(15, 5), (15, 15), (15, 1), (32, 8), (4, 4), (1, 1)])
def test_require_divisor_accepts_even_splits(n, b):
    assert sh(f"require_divisor {n} {b}").returncode == 0


def test_require_divisor_lists_the_valid_choices():
    err = sh("require_divisor 15 8").stderr
    assert "Divisors of 15: 1 3 5 15" in err


@pytest.mark.parametrize("bad", ["0", "-4", "abc", ""])
def test_require_divisor_rejects_nonsense(bad):
    assert sh(f'require_divisor 16 "{bad}"').returncode == 1


def test_helper_output_pairs_are_self_consistent():
    """Whatever seed_batch_for picks must always pass require_divisor."""
    for n in range(1, 65):
        b = seed_batch_for(n)
        assert sh(f"require_divisor {n} {b}").returncode == 0, f"{n}/{b} disagree"


# ------------------------------------------------------------------ the shipped configs
@pytest.mark.parametrize("name", [
    "config_generate_era5_carra2_eval.yaml",
    "config_generate_era5_carra2_mini.yaml",
])
def test_shipped_generate_configs_are_internally_consistent(name):
    """A direct `python generate.py --config-name=...` must work without the shell wrapper."""
    gen = yaml.safe_load((CONF / name).read_text())["generation"]
    n, b = gen["num_ensembles"], gen["seed_batch_size"]
    assert b >= 1 and n % b == 0, f"{name}: seed_batch_size={b} does not divide num_ensembles={n}"


def test_run_eval_pins_the_regression_pass_to_one():
    """The regression pass runs num_ensembles=1, so its seed_batch_size must be 1 too."""
    text = (TRAIN_DIR.parent / "evaluate" / "run_eval.sh").read_text()
    block = text.split("inference_mode=regression", 1)[1]
    assert "++generation.num_ensembles=1" in block
    assert "++generation.seed_batch_size=1" in block
