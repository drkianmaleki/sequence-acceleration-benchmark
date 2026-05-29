"""
generators.py
=============
Synthetic convergence-sequence generators for the acceleration study.

Each generator has the signature:
    gen_*(n_array, rng, sigma) -> np.ndarray

where
    n_array : integer indices, shape (N,)
    rng     : np.random.RandomState for reproducibility
    sigma   : noise standard deviation (0 = noiseless)

The companion TRUTH dict maps each regime name to a closed-form function
    truth(n) -> float
that returns the *noiseless* sequence value at an arbitrary integer n.
This is used to compute the ground-truth prediction target.

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

Reference
---------
Maleki, K. (2026). Working paper.
"""

import numpy as np
from typing import Callable, Dict, Tuple

# ── Shared asymptote ───────────────────────────────────────────────────────────
_L = 0.01


# ── Internal helpers ───────────────────────────────────────────────────────────

def _gauss(s: np.ndarray, sigma: float, rng: np.random.RandomState) -> np.ndarray:
    """Add i.i.d. Gaussian noise to sequence s."""
    return s + sigma * rng.randn(*s.shape) if sigma > 0.0 else s.copy()


# ── Original 14 generators ────────────────────────────────────────────────────

def gen_single_exp(n: np.ndarray, rng: np.random.RandomState,
                   sigma: float) -> np.ndarray:
    """Single-component exponential decay: L + 0.75 exp(-0.040 n)."""
    return _gauss(_L + 0.75 * np.exp(-0.040 * n), sigma, rng)


def gen_two_exp(n: np.ndarray, rng: np.random.RandomState,
                sigma: float) -> np.ndarray:
    """Two-component exponential: L + 0.50 exp(-0.018n) + 0.25 exp(-0.15n)."""
    return _gauss(_L + 0.50 * np.exp(-0.018 * n)
                     + 0.25 * np.exp(-0.15  * n), sigma, rng)


def gen_three_exp(n: np.ndarray, rng: np.random.RandomState,
                  sigma: float) -> np.ndarray:
    """Three-component exponential mixture."""
    return _gauss(_L + 0.30 * np.exp(-0.008 * n)
                     + 0.25 * np.exp(-0.060 * n)
                     + 0.20 * np.exp(-0.250 * n), sigma, rng)


def gen_four_exp(n: np.ndarray, rng: np.random.RandomState,
                 sigma: float) -> np.ndarray:
    """Four-component exponential mixture."""
    return _gauss(_L + 0.20 * np.exp(-0.004 * n)
                     + 0.20 * np.exp(-0.025 * n)
                     + 0.15 * np.exp(-0.100 * n)
                     + 0.10 * np.exp(-0.400 * n), sigma, rng)


def gen_power_law(n: np.ndarray, rng: np.random.RandomState,
                  sigma: float) -> np.ndarray:
    """Power-law decay: L + 0.80 / (n+1)^0.70."""
    return _gauss(_L + 0.80 / (n + 1.0) ** 0.70, sigma, rng)


def gen_rational_decay(n: np.ndarray, rng: np.random.RandomState,
                       sigma: float) -> np.ndarray:
    """Rational (Padé-type) decay: L + 0.80 / (1 + 0.04n)."""
    return _gauss(_L + 0.80 / (1.0 + 0.04 * n), sigma, rng)


def gen_mixed_pow_rat(n: np.ndarray, rng: np.random.RandomState,
                      sigma: float) -> np.ndarray:
    """Mixed power-law + rational: L + 0.40/(n+1)^0.5 + 0.30/(1+0.02n)."""
    return _gauss(_L + 0.40 / (n + 1.0) ** 0.50
                     + 0.30 / (1.0 + 0.02 * n), sigma, rng)


def gen_multiphase(n: np.ndarray, rng: np.random.RandomState,
                   sigma: float) -> np.ndarray:
    """Multiphase: fast exponential + slow rational tail."""
    return _gauss(_L + 0.50 * np.exp(-0.06 * n)
                     + 0.25 / (1.0 + 0.003 * n), sigma, rng)


def gen_osc_exp(n: np.ndarray, rng: np.random.RandomState,
                sigma: float) -> np.ndarray:
    """Oscillatory exponential: exponential envelope × cosine modulation."""
    return _gauss(_L + 0.70 * np.exp(-0.030 * n)
                     * (1.0 + 0.15 * np.cos(0.20 * n)), sigma, rng)


def gen_damped_osc_pow(n: np.ndarray, rng: np.random.RandomState,
                       sigma: float) -> np.ndarray:
    """Damped oscillatory power-law: power-law × log-cosine modulation."""
    return _gauss(_L + 0.80 / (n + 1.0) ** 0.60
                     * (1.0 + 0.20 * np.cos(0.30 * np.log(n + 1.0))),
                  sigma, rng)


def gen_log_slow(n: np.ndarray, rng: np.random.RandomState,
                 sigma: float) -> np.ndarray:
    """Logarithmically slow convergence: L + 0.80 / log(n + e)."""
    return _gauss(_L + 0.80 / np.log(n + np.e), sigma, rng)


def gen_delayed_plateau(n: np.ndarray, rng: np.random.RandomState,
                        sigma: float) -> np.ndarray:
    """Delayed plateau: smooth stall near n=40 before resuming decline."""
    bump = 0.06 * np.exp(-((n - 40.0) ** 2) / 200.0)
    s    = _L + 0.60 * np.exp(-0.010 * n) + 0.10 * np.exp(-0.200 * n) + bump
    return _gauss(s, sigma, rng)


def gen_staircase(n: np.ndarray, rng: np.random.RandomState,
                  sigma: float) -> np.ndarray:
    """Staircase decay: piecewise constant with sudden drops."""
    steps = [(0, 0.70), (20, 0.55), (40, 0.42),
             (65, 0.28), (90, 0.18), (130, 0.10), (170, 0.05)]
    s = np.full_like(n, _L, dtype=float)
    for threshold, value in reversed(steps):
        s = np.where(n >= threshold, _L + value, s)
    s = np.maximum(s, _L)
    return _gauss(s, sigma, rng)


def gen_noisy_plateau(n: np.ndarray, rng: np.random.RandomState,
                      sigma: float) -> np.ndarray:
    """Near-plateau with intrinsic Gaussian noise (sigma_base=0.003)."""
    base_sigma = 0.003
    s = _L + 0.02 * np.exp(-0.05 * n) + base_sigma * rng.randn(*n.shape)
    return np.maximum(s, 0.0)


# ── New 4 generators (Phase 0 addition) ───────────────────────────────────────

def gen_slow_osc_power(n: np.ndarray, rng: np.random.RandomState,
                       sigma: float) -> np.ndarray:
    """
    Slowly oscillating power-law.

    The oscillation period grows as sqrt(n), making the frequency
    decrease over time.  Wynn epsilon struggles when the sign of the
    error term changes unpredictably; this tests that boundary.

        s(n) = L + 0.70 / (n+1)^0.65 * (1 + 0.25 * cos(sqrt(n+1)))
    """
    s = _L + 0.70 / (n + 1.0) ** 0.65 * (1.0 + 0.25 * np.cos(np.sqrt(n + 1.0)))
    return _gauss(s, sigma, rng)


def gen_log_oscillatory(n: np.ndarray, rng: np.random.RandomState,
                        sigma: float) -> np.ndarray:
    """
    Log-slow base with superimposed oscillation.

    Designed to challenge both Richardson (which assumes power-law) and
    Wynn epsilon (which struggles with oscillation on log-slow base).

        s(n) = L + 0.80/log(n+e) * (1 + 0.20 * sin(0.15n))
    """
    s = _L + 0.80 / np.log(n + np.e) * (1.0 + 0.20 * np.sin(0.15 * n))
    return _gauss(s, sigma, rng)


def gen_heavy_noise_plat(n: np.ndarray, rng: np.random.RandomState,
                         sigma: float) -> np.ndarray:
    """
    Near-plateau with Laplace (heavy-tailed) noise.

    The deterministic component is tiny; the signal is almost entirely
    noise after a fast initial drop.  Tests the lower limit of all methods.

        s(n) = L + 0.05*exp(-0.10n) + Laplace(0, 0.004)
    """
    laplace_noise = rng.laplace(0.0, 0.004, size=n.shape)
    if sigma > 0:
        laplace_noise += sigma * rng.randn(*n.shape)
    s = _L + 0.05 * np.exp(-0.10 * n) + laplace_noise
    return np.maximum(s, 0.0)


def gen_broken_power_law(n: np.ndarray, rng: np.random.RandomState,
                         sigma: float) -> np.ndarray:
    """
    Broken power-law: exponent changes at n=60.

    Before n=60 the sequence follows a steep power-law; after n=60 it
    switches to a shallower one.  Richardson fitted on the early window
    will systematically overestimate the final level.

        s(n) = L + 0.80/(n+1)^alpha(n)
        alpha(n) = 0.90  if n < 60
                 = 0.40  otherwise
    """
    alpha = np.where(n < 60, 0.90, 0.40)
    s = _L + 0.80 / (n + 1.0) ** alpha
    return _gauss(s, sigma, rng)


# ── Truth functions (noiseless, evaluated at arbitrary n) ──────────────────────
# These return the *expected* sequence value at n, ignoring noise.

TRUTH: Dict[str, Callable] = {
    "single_exp":       lambda n: _L + 0.75 * np.exp(-0.040 * n),
    "two_exp":          lambda n: (_L + 0.50 * np.exp(-0.018 * n)
                                      + 0.25 * np.exp(-0.150 * n)),
    "three_exp":        lambda n: (_L + 0.30 * np.exp(-0.008 * n)
                                      + 0.25 * np.exp(-0.060 * n)
                                      + 0.20 * np.exp(-0.250 * n)),
    "four_exp":         lambda n: (_L + 0.20 * np.exp(-0.004 * n)
                                      + 0.20 * np.exp(-0.025 * n)
                                      + 0.15 * np.exp(-0.100 * n)
                                      + 0.10 * np.exp(-0.400 * n)),
    "power_law":        lambda n: _L + 0.80 / (n + 1.0) ** 0.70,
    "rational_decay":   lambda n: _L + 0.80 / (1.0 + 0.04 * n),
    "mixed_pow_rat":    lambda n: (_L + 0.40 / (n + 1.0) ** 0.50
                                      + 0.30 / (1.0 + 0.02 * n)),
    "multiphase":       lambda n: (_L + 0.50 * np.exp(-0.06 * n)
                                      + 0.25 / (1.0 + 0.003 * n)),
    "osc_exp":          lambda n: (_L + 0.70 * np.exp(-0.030 * n)
                                      * (1.0 + 0.15 * np.cos(0.20 * n))),
    "damped_osc_pow":   lambda n: (_L + 0.80 / (n + 1.0) ** 0.60
                                      * (1.0 + 0.20 * np.cos(0.30 * np.log(n + 1.0)))),
    "log_slow":         lambda n: _L + 0.80 / np.log(n + np.e),
    "delayed_plateau":  lambda n: (_L + 0.60 * np.exp(-0.010 * n)
                                      + 0.10 * np.exp(-0.200 * n)),
    "staircase":        lambda n: _L,            # converges to L
    "noisy_plateau":    lambda n: _L + 0.02 * np.exp(-0.05 * n),
    # New regimes
    "slow_osc_power":   lambda n: (_L + 0.70 / (n + 1.0) ** 0.65
                                      * (1.0 + 0.25 * np.cos(np.sqrt(n + 1.0)))),
    "log_oscillatory":  lambda n: (_L + 0.80 / np.log(n + np.e)
                                      * (1.0 + 0.20 * np.sin(0.15 * n))),
    "heavy_noise_plat": lambda n: _L + 0.05 * np.exp(-0.10 * n),
    "broken_power_law": lambda n: (_L + 0.80 / (n + 1.0) ** (0.90 if n < 60 else 0.40)
                                   if np.isscalar(n) else
                                   _L + 0.80 / (n + 1.0) ** np.where(n < 60, 0.90, 0.40)),
}


# ── Registry ───────────────────────────────────────────────────────────────────

GENERATORS: Dict[str, Callable] = {
    "single_exp":       gen_single_exp,
    "two_exp":          gen_two_exp,
    "three_exp":        gen_three_exp,
    "four_exp":         gen_four_exp,
    "power_law":        gen_power_law,
    "rational_decay":   gen_rational_decay,
    "mixed_pow_rat":    gen_mixed_pow_rat,
    "multiphase":       gen_multiphase,
    "osc_exp":          gen_osc_exp,
    "damped_osc_pow":   gen_damped_osc_pow,
    "log_slow":         gen_log_slow,
    "delayed_plateau":  gen_delayed_plateau,
    "staircase":        gen_staircase,
    "noisy_plateau":    gen_noisy_plateau,
    # Phase 0 additions
    "slow_osc_power":   gen_slow_osc_power,
    "log_oscillatory":  gen_log_oscillatory,
    "heavy_noise_plat": gen_heavy_noise_plat,
    "broken_power_law": gen_broken_power_law,
}

REGIME_NAMES = list(GENERATORS.keys())
