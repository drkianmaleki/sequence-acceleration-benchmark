"""
generators.py
=============
Synthetic convergence-sequence generators for the acceleration study.

Each regime is defined exactly once, by a mean function

    _mean_*(n) -> np.ndarray

returning the *noiseless* sequence value at arbitrary (scalar or array) n.
The two public registries are both derived from these mean functions:

    GENERATORS[name](n_array, rng, sigma) -> np.ndarray
        the observed sequence, i.e. the mean plus noise
    TRUTH[name](n) -> float
        the noiseless value at n, used as the ground-truth prediction target

Deriving both from a single definition guarantees that the generated
sequence and its prediction target cannot drift apart.

All regimes share the asymptote L = _L, so that s(n) -> _L as n -> infinity.

Regimes
-------
Original 14:
  single_exp, two_exp, three_exp, four_exp,
  power_law, rational_decay, mixed_pow_rat, multiphase,
  osc_exp, damped_osc_pow, log_slow, delayed_plateau,
  staircase, noisy_plateau

New 4 (Phase 0 addition):
  slow_osc_power    - oscillation whose period grows with n (harder for Wynn)
  log_oscillatory   - logarithmic base with superimposed oscillation
  heavy_noise_plat  - near-plateau buried in heavy-tailed (Laplace) noise
  broken_power_law  - exponent changes at a mid-sequence breakpoint

Two regimes carry intrinsic noise that is present even at sigma = 0, because
irreducible observation noise is the property under test:
  noisy_plateau     - fixed Gaussian noise, sd 0.003
  heavy_noise_plat  - fixed Laplace noise, scale 0.004
For these, TRUTH returns the noiseless mean, which is the quantity a method
is asked to predict.

Reference
---------
Maleki, K. (2026). Working paper.
"""

from typing import Callable, Dict

import numpy as np

# ── Shared asymptote ───────────────────────────────────────────────────────────
_L = 0.01


# ── Internal helpers ───────────────────────────────────────────────────────────

def _as_float_array(n) -> np.ndarray:
    """Coerce scalar or array-like n to a float array (0-d for scalars)."""
    return np.asarray(n, dtype=float)


def _gauss(s: np.ndarray, sigma: float, rng: np.random.RandomState) -> np.ndarray:
    """Add i.i.d. Gaussian noise to sequence s."""
    return s + sigma * rng.randn(*s.shape) if sigma > 0.0 else s.copy()


def _from_mean(name: str, mean_fn: Callable) -> Callable:
    """Build a generator returning mean_fn(n) plus i.i.d. Gaussian noise."""

    def generator(n: np.ndarray, rng: np.random.RandomState,
                  sigma: float) -> np.ndarray:
        return _gauss(mean_fn(n), sigma, rng)

    generator.__name__ = f"gen_{name}"
    generator.__qualname__ = f"gen_{name}"
    generator.__doc__ = mean_fn.__doc__
    return generator


# ── Mean functions: the single definition of each regime ──────────────────────

def _mean_single_exp(n) -> np.ndarray:
    """Single-component exponential decay: L + 0.75 exp(-0.040 n)."""
    n = _as_float_array(n)
    return _L + 0.75 * np.exp(-0.040 * n)


def _mean_two_exp(n) -> np.ndarray:
    """Two-component exponential: L + 0.50 exp(-0.018n) + 0.25 exp(-0.15n)."""
    n = _as_float_array(n)
    return _L + 0.50 * np.exp(-0.018 * n) + 0.25 * np.exp(-0.15 * n)


def _mean_three_exp(n) -> np.ndarray:
    """Three-component exponential mixture."""
    n = _as_float_array(n)
    return (_L + 0.30 * np.exp(-0.008 * n)
               + 0.25 * np.exp(-0.060 * n)
               + 0.20 * np.exp(-0.250 * n))


def _mean_four_exp(n) -> np.ndarray:
    """Four-component exponential mixture."""
    n = _as_float_array(n)
    return (_L + 0.20 * np.exp(-0.004 * n)
               + 0.20 * np.exp(-0.025 * n)
               + 0.15 * np.exp(-0.100 * n)
               + 0.10 * np.exp(-0.400 * n))


def _mean_power_law(n) -> np.ndarray:
    """Power-law decay: L + 0.80 / (n+1)^0.70."""
    n = _as_float_array(n)
    return _L + 0.80 / (n + 1.0) ** 0.70


def _mean_rational_decay(n) -> np.ndarray:
    """Rational (Pade-type) decay: L + 0.80 / (1 + 0.04n)."""
    n = _as_float_array(n)
    return _L + 0.80 / (1.0 + 0.04 * n)


def _mean_mixed_pow_rat(n) -> np.ndarray:
    """Mixed power-law + rational: L + 0.40/(n+1)^0.5 + 0.30/(1+0.02n)."""
    n = _as_float_array(n)
    return _L + 0.40 / (n + 1.0) ** 0.50 + 0.30 / (1.0 + 0.02 * n)


def _mean_multiphase(n) -> np.ndarray:
    """Multiphase: fast exponential + slow rational tail."""
    n = _as_float_array(n)
    return _L + 0.50 * np.exp(-0.06 * n) + 0.25 / (1.0 + 0.003 * n)


def _mean_osc_exp(n) -> np.ndarray:
    """Oscillatory exponential: exponential envelope x cosine modulation."""
    n = _as_float_array(n)
    return _L + 0.70 * np.exp(-0.030 * n) * (1.0 + 0.15 * np.cos(0.20 * n))


def _mean_damped_osc_pow(n) -> np.ndarray:
    """Damped oscillatory power-law: power-law x log-cosine modulation."""
    n = _as_float_array(n)
    return (_L + 0.80 / (n + 1.0) ** 0.60
               * (1.0 + 0.20 * np.cos(0.30 * np.log(n + 1.0))))


def _mean_log_slow(n) -> np.ndarray:
    """Logarithmically slow convergence: L + 0.80 / log(n + e)."""
    n = _as_float_array(n)
    return _L + 0.80 / np.log(n + np.e)


def _mean_delayed_plateau(n) -> np.ndarray:
    """Delayed plateau: smooth stall near n=40 before resuming decline.

    The Gaussian bump centred at n = 40 is the defining feature of this
    regime and is therefore part of the noiseless trajectory.
    """
    n = _as_float_array(n)
    bump = 0.06 * np.exp(-((n - 40.0) ** 2) / 200.0)
    return _L + 0.60 * np.exp(-0.010 * n) + 0.10 * np.exp(-0.200 * n) + bump


def _mean_staircase(n) -> np.ndarray:
    """Staircase decay: piecewise constant with sudden drops.

    Each entry of ``steps`` is (first index at which the level applies,
    height above L).  Later entries overwrite earlier ones, so the level in
    force at n is the one for the last threshold not exceeding n.  The final
    step has height 0, so the sequence reaches the shared asymptote L.
    """
    steps = [(0, 0.70), (20, 0.55), (40, 0.42), (65, 0.28), (90, 0.18),
             (130, 0.10), (170, 0.05), (240, 0.02), (320, 0.0)]
    n = _as_float_array(n)
    s = np.full(n.shape, _L, dtype=float)
    for threshold, height in steps:
        s = np.where(n >= threshold, _L + height, s)
    return np.maximum(s, _L)


def _mean_noisy_plateau(n) -> np.ndarray:
    """Near-plateau: L + 0.02 exp(-0.05n), observed under fixed noise."""
    n = _as_float_array(n)
    return _L + 0.02 * np.exp(-0.05 * n)


def _mean_slow_osc_power(n) -> np.ndarray:
    """Slowly oscillating power-law.

    The oscillation period grows as sqrt(n), making the frequency decrease
    over time.  Wynn epsilon struggles when the sign of the error term
    changes unpredictably; this tests that boundary.

        s(n) = L + 0.70 / (n+1)^0.65 * (1 + 0.25 * cos(sqrt(n+1)))
    """
    n = _as_float_array(n)
    return (_L + 0.70 / (n + 1.0) ** 0.65
               * (1.0 + 0.25 * np.cos(np.sqrt(n + 1.0))))


def _mean_log_oscillatory(n) -> np.ndarray:
    """Log-slow base with superimposed oscillation.

    Designed to challenge both Richardson (which assumes power-law) and
    Wynn epsilon (which struggles with oscillation on a log-slow base).

        s(n) = L + 0.80/log(n+e) * (1 + 0.20 * sin(0.15n))
    """
    n = _as_float_array(n)
    return _L + 0.80 / np.log(n + np.e) * (1.0 + 0.20 * np.sin(0.15 * n))


def _mean_heavy_noise_plat(n) -> np.ndarray:
    """Near-plateau: L + 0.05 exp(-0.10n), observed under heavy-tailed noise."""
    n = _as_float_array(n)
    return _L + 0.05 * np.exp(-0.10 * n)


def _mean_broken_power_law(n) -> np.ndarray:
    """Broken power-law: exponent changes at n=60.

    Before n=60 the sequence follows a steep power-law; after n=60 it
    switches to a shallower one.  Richardson fitted on the early window
    will systematically overestimate the final level.

        s(n) = L + 0.80/(n+1)^alpha(n)
        alpha(n) = 0.90  if n < 60
                 = 0.40  otherwise
    """
    n = _as_float_array(n)
    alpha = np.where(n < 60, 0.90, 0.40)
    return _L + 0.80 / (n + 1.0) ** alpha


# ── Generators with intrinsic noise ───────────────────────────────────────────
# These two regimes are defined by irreducible observation noise, so they
# carry their own noise term even when sigma = 0.

def gen_noisy_plateau(n: np.ndarray, rng: np.random.RandomState,
                      sigma: float) -> np.ndarray:
    """Near-plateau with intrinsic Gaussian noise (sd 0.003).

    The intrinsic noise is what defines this regime, so the sweep parameter
    sigma is deliberately not applied on top of it.
    """
    base_sigma = 0.003
    s = _mean_noisy_plateau(n) + base_sigma * rng.randn(*np.shape(n))
    return np.maximum(s, 0.0)


def gen_heavy_noise_plat(n: np.ndarray, rng: np.random.RandomState,
                         sigma: float) -> np.ndarray:
    """Near-plateau with Laplace (heavy-tailed) noise, scale 0.004.

    The deterministic component is tiny; the signal is almost entirely
    noise after a fast initial drop.  Tests the lower limit of all methods.
    Any sweep noise sigma is added on top of the intrinsic Laplace noise.
    """
    noise = rng.laplace(0.0, 0.004, size=np.shape(n))
    if sigma > 0.0:
        noise = noise + sigma * rng.randn(*np.shape(n))
    return np.maximum(_mean_heavy_noise_plat(n) + noise, 0.0)


# ── Registries ────────────────────────────────────────────────────────────────
# TRUTH and GENERATORS are both derived from the mean functions above, so no
# regime can be generated from one formula and scored against another.

_MEANS: Dict[str, Callable] = {
    "single_exp":       _mean_single_exp,
    "two_exp":          _mean_two_exp,
    "three_exp":        _mean_three_exp,
    "four_exp":         _mean_four_exp,
    "power_law":        _mean_power_law,
    "rational_decay":   _mean_rational_decay,
    "mixed_pow_rat":    _mean_mixed_pow_rat,
    "multiphase":       _mean_multiphase,
    "osc_exp":          _mean_osc_exp,
    "damped_osc_pow":   _mean_damped_osc_pow,
    "log_slow":         _mean_log_slow,
    "delayed_plateau":  _mean_delayed_plateau,
    "staircase":        _mean_staircase,
    "noisy_plateau":    _mean_noisy_plateau,
    # Phase 0 additions
    "slow_osc_power":   _mean_slow_osc_power,
    "log_oscillatory":  _mean_log_oscillatory,
    "heavy_noise_plat": _mean_heavy_noise_plat,
    "broken_power_law": _mean_broken_power_law,
}

# Regimes whose observation model is not simply "mean + sigma * Gaussian".
_INTRINSIC_NOISE_GENERATORS: Dict[str, Callable] = {
    "noisy_plateau":    gen_noisy_plateau,
    "heavy_noise_plat": gen_heavy_noise_plat,
}

TRUTH: Dict[str, Callable] = dict(_MEANS)

GENERATORS: Dict[str, Callable] = {
    name: _INTRINSIC_NOISE_GENERATORS.get(name, None) or _from_mean(name, mean_fn)
    for name, mean_fn in _MEANS.items()
}

# Module-level aliases so individual generators stay importable by name
# (e.g. ``from src.generators import gen_single_exp``).
gen_single_exp       = GENERATORS["single_exp"]
gen_two_exp          = GENERATORS["two_exp"]
gen_three_exp        = GENERATORS["three_exp"]
gen_four_exp         = GENERATORS["four_exp"]
gen_power_law        = GENERATORS["power_law"]
gen_rational_decay   = GENERATORS["rational_decay"]
gen_mixed_pow_rat    = GENERATORS["mixed_pow_rat"]
gen_multiphase       = GENERATORS["multiphase"]
gen_osc_exp          = GENERATORS["osc_exp"]
gen_damped_osc_pow   = GENERATORS["damped_osc_pow"]
gen_log_slow         = GENERATORS["log_slow"]
gen_delayed_plateau  = GENERATORS["delayed_plateau"]
gen_staircase        = GENERATORS["staircase"]
gen_slow_osc_power   = GENERATORS["slow_osc_power"]
gen_log_oscillatory  = GENERATORS["log_oscillatory"]
gen_broken_power_law = GENERATORS["broken_power_law"]

REGIME_NAMES = list(GENERATORS.keys())
