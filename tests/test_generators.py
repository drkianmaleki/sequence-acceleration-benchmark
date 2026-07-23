"""
tests/test_generators.py
========================
Regression tests for the synthetic regime definitions.

The benchmark scores a method by comparing its estimate against
TRUTH[regime](n_f).  If TRUTH disagrees with the sequence that GENERATORS
actually produced, every method in that regime is scored against a target
the sequence never attains, and the resulting numbers are meaningless.

That failure is silent: nothing raises, the run completes, and the output
looks plausible.  These tests exist to make it loud.

Run with:
    pytest tests/
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.config as CFG_MOD  # noqa: E402
from src.generators import GENERATORS, REGIME_NAMES, TRUTH  # noqa: E402

# Regimes whose observation model carries irreducible noise even at sigma = 0.
# For these, TRUTH is the noiseless mean rather than any single realisation.
INTRINSIC_NOISE = {"noisy_plateau", "heavy_noise_plat"}

DETERMINISTIC = [r for r in REGIME_NAMES if r not in INTRINSIC_NOISE]

N_GRID = np.arange(0, 5001)


def test_registries_agree():
    """GENERATORS and TRUTH must describe exactly the same set of regimes."""
    assert set(GENERATORS) == set(TRUTH)
    assert len(REGIME_NAMES) == 18


@pytest.mark.parametrize("regime", DETERMINISTIC)
def test_truth_matches_noiseless_generator(regime):
    """TRUTH(n) must equal the generator's output at sigma = 0, for every n.

    This is the invariant whose violation silently corrupted the staircase
    and delayed_plateau regimes: each trajectory used to be written twice,
    once in the generator and once in TRUTH, and the two copies drifted.
    """
    produced = GENERATORS[regime](N_GRID, np.random.RandomState(0), 0.0)
    target = np.asarray(TRUTH[regime](N_GRID), dtype=float)
    assert np.allclose(produced, target, rtol=0.0, atol=1e-12), (
        f"{regime}: generator and TRUTH disagree by "
        f"{np.abs(produced - target).max():.6f} at n="
        f"{int(np.abs(produced - target).argmax())}"
    )


@pytest.mark.parametrize("regime", sorted(INTRINSIC_NOISE))
def test_intrinsic_noise_regimes_are_unbiased(regime):
    """For noisy regimes, TRUTH must be the mean the observations scatter about."""
    draws = np.array([
        GENERATORS[regime](N_GRID, np.random.RandomState(seed), 0.0)
        for seed in range(200)
    ])
    empirical = draws.mean(axis=0)
    target = np.asarray(TRUTH[regime](N_GRID), dtype=float)
    # Compare on the tail, where the deterministic part has decayed and the
    # clipping at zero no longer biases the mean.
    assert np.abs(empirical[500:] - target[500:]).max() < 0.01


@pytest.mark.parametrize("regime", REGIME_NAMES)
def test_regime_converges_to_shared_asymptote(regime):
    """Every regime must converge to L_INF, as the benchmark design assumes.

    Some regimes converge very slowly (log_slow decays as 1/log n), so this
    checks the trend rather than a fixed tolerance at finite n.
    """
    far = np.array([10**5, 10**6, 10**7], dtype=float)
    values = np.asarray(TRUTH[regime](far), dtype=float)
    gaps = np.abs(values - CFG_MOD.L_INF)
    assert gaps[-1] <= gaps[0] + 1e-12, f"{regime}: not approaching L_INF"
    assert gaps[-1] < 0.10, f"{regime}: still {gaps[-1]:.4f} from L_INF at n=1e7"


@pytest.mark.parametrize("regime", REGIME_NAMES)
def test_truth_accepts_scalars(regime):
    """evaluation.py calls float(truth(n_f)) with a scalar; that must work."""
    for n in (150, 1000, 5000):
        scalar = float(TRUTH[regime](n))
        vector = float(np.asarray(TRUTH[regime](np.array([n])))[0])
        assert np.isfinite(scalar)
        assert abs(scalar - vector) < 1e-12


@pytest.mark.parametrize("regime", REGIME_NAMES)
def test_generator_shape_and_finiteness(regime):
    """Generators must return finite values shaped like their input."""
    for sigma in (0.0, 0.005):
        out = GENERATORS[regime](N_GRID, np.random.RandomState(3), sigma)
        assert out.shape == N_GRID.shape
        assert np.all(np.isfinite(out))


def test_noise_is_reproducible_from_seed():
    """Identical seeds must give identical sequences, or nothing replicates."""
    for regime in REGIME_NAMES:
        a = GENERATORS[regime](N_GRID, np.random.RandomState(11), 0.005)
        b = GENERATORS[regime](N_GRID, np.random.RandomState(11), 0.005)
        assert np.array_equal(a, b), f"{regime} is not reproducible from its seed"
