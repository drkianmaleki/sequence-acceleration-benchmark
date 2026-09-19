"""
tests/test_generators.py
========================
Regression tests for the synthetic regime definitions (redesign v2).

The benchmark scores a method by comparing its estimate against
TRUTH[regime](n_f, L_true).  If TRUTH disagrees with the sequence that
GENERATORS actually produced, every method in that regime is scored against
a target the sequence never attains, and the resulting numbers are
meaningless.  That failure is silent; these tests make it loud.

Redesign v2 invariants added here: both registries require the asymptote
argument (no silent shared constant), every regime converges to its *own*
hidden L_true, and the held-out regimes obey the same contracts.

Run with:
    pytest tests/
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.config as CFG_MOD  # noqa: E402
from src.generators import (  # noqa: E402
    ALL_REGIME_NAMES, GAP, GENERATORS, HOLDOUT, HOLDOUT_REGIME_NAMES,
    INTRINSIC_NOISE_REGIMES, REGIME_NAMES, TRUTH, regime_functions,
    true_asymptote,
)

INTRINSIC_NOISE = set(INTRINSIC_NOISE_REGIMES)
DETERMINISTIC = [r for r in ALL_REGIME_NAMES if r not in INTRINSIC_NOISE]

N_GRID = np.arange(0, 5001)
L_STARS = [0.005, 0.01, 0.137, 0.5]


def test_registries_agree():
    """GENERATORS, TRUTH and GAP must describe exactly the same regimes."""
    assert set(GENERATORS) == set(TRUTH) == set(GAP)
    assert len(REGIME_NAMES) == 18
    assert len(HOLDOUT_REGIME_NAMES) == 6
    assert set(REGIME_NAMES).isdisjoint(HOLDOUT)
    assert ALL_REGIME_NAMES == REGIME_NAMES + HOLDOUT_REGIME_NAMES
    assert all(GAP[r].holdout == (r in HOLDOUT) for r in ALL_REGIME_NAMES)


def test_registries_require_the_asymptote():
    """The rejected design's three-argument call must fail, not fall back."""
    with pytest.raises(TypeError):
        GENERATORS["single_exp"](N_GRID, np.random.RandomState(0), 0.0)
    with pytest.raises(TypeError):
        TRUTH["single_exp"](N_GRID)


@pytest.mark.parametrize("L_star", L_STARS)
@pytest.mark.parametrize("regime", DETERMINISTIC)
def test_truth_matches_noiseless_generator(regime, L_star):
    """TRUTH(n, L) must equal the generator's output at sigma = 0, for every n."""
    produced = GENERATORS[regime](N_GRID, np.random.RandomState(0), 0.0, L_star, seed=0)
    target = np.asarray(TRUTH[regime](N_GRID, L_star, seed=0), dtype=float)
    assert np.allclose(produced, target, rtol=0.0, atol=1e-12), (
        f"{regime}: generator and TRUTH disagree by "
        f"{np.abs(produced - target).max():.6f} at n="
        f"{int(np.abs(produced - target).argmax())}"
    )


@pytest.mark.parametrize("regime", sorted(INTRINSIC_NOISE))
def test_intrinsic_noise_regimes_are_unbiased(regime):
    """For noisy regimes, TRUTH must be the mean the observations scatter about.

    Uses an asymptote well above the clip at zero, where the mean is exact;
    the clip's small upward bias at the smallest L_true is a documented
    property of the observation model, not a TRUTH mismatch.
    """
    L_star = 0.1
    draws = np.array([
        GENERATORS[regime](N_GRID, np.random.RandomState(seed), 0.0, L_star)
        for seed in range(200)
    ])
    empirical = draws.mean(axis=0)
    target = np.asarray(TRUTH[regime](N_GRID, L_star), dtype=float)
    assert np.abs(empirical[500:] - target[500:]).max() < 0.01


@pytest.mark.parametrize("regime", ALL_REGIME_NAMES)
def test_regime_converges_to_its_own_asymptote(regime):
    """Every regime must converge to its hidden L_true, not to a shared value."""
    for seed in (0, 1):
        _, truth, L_true = regime_functions(regime, seed)
        far = np.array([10**5, 10**6, 10**7], dtype=float)
        gaps = np.abs(np.asarray(truth(far), dtype=float) - L_true)
        assert gaps[-1] <= gaps[0] + 1e-12, f"{regime}: not approaching L_true"
        assert gaps[-1] < 0.10, f"{regime}: still {gaps[-1]:.4f} from L_true at n=1e7"
        assert CFG_MOD.L_TRUE_RANGE[0] <= L_true <= CFG_MOD.L_TRUE_RANGE[1]


@pytest.mark.parametrize("regime", ALL_REGIME_NAMES)
def test_gap_is_nonnegative_and_vanishes(regime):
    g = np.asarray(GAP[regime](N_GRID, seed=0), dtype=float)
    assert np.all(np.isfinite(g))
    assert np.all(g >= 0.0)
    assert float(GAP[regime](1e7, seed=0)) < 0.10


@pytest.mark.parametrize("regime", ALL_REGIME_NAMES)
def test_truth_accepts_scalars(regime):
    """evaluation.py calls float(truth(n_f)) with a scalar; that must work."""
    for n in (150, 1000, 5000, 50000):
        scalar = float(TRUTH[regime](n, 0.02, seed=0))
        vector = float(np.asarray(TRUTH[regime](np.array([n]), 0.02, seed=0))[0])
        assert np.isfinite(scalar)
        assert abs(scalar - vector) < 1e-12


@pytest.mark.parametrize("regime", ALL_REGIME_NAMES)
def test_generator_shape_and_finiteness(regime):
    """Generators must return finite values shaped like their input."""
    for sigma in (0.0, 0.005):
        out = GENERATORS[regime](N_GRID, np.random.RandomState(3), sigma, 0.03, seed=0)
        assert out.shape == N_GRID.shape
        assert np.all(np.isfinite(out))


def test_noise_is_reproducible_from_seed():
    """Identical seeds must give identical sequences, or nothing replicates."""
    for regime in ALL_REGIME_NAMES:
        L = true_asymptote(regime, 11)
        a = GENERATORS[regime](N_GRID, np.random.RandomState(11), 0.005, L, seed=11)
        b = GENERATORS[regime](N_GRID, np.random.RandomState(11), 0.005, L, seed=11)
        assert np.array_equal(a, b), f"{regime} is not reproducible from its seed"
