"""
generators.py
=============
Synthetic convergence-sequence generators for the acceleration study.

Redesign v2: every regime is defined once, by a *gap* function

    gap(n) -> np.ndarray          (noiseless distance above the asymptote)

and the noiseless mean of a sequence is

    mean(n, L_star) = L_star + gap(n)

The gap shapes are exactly those of the rejected design (the old mean minus
the old shared constant 0.01); only the asymptote changed.  The asymptote is
now hidden and heterogeneous:

    L_true(regime, seed) = 10 ** U(log10 0.005, log10 0.5),
    U drawn from RandomState(crc32(f"{regime}:{seed}:Lstar") % 2**31)

so it is deterministic, reproducible, and never shared across seeds.  A
"legacy" mode (config.ASYMPTOTE_MODE = "legacy") restores L_true = 0.01 for
every regime and seed; it exists only for the regression test that locks in
the referee's finding.

Public API
----------
    GAP[name](n, seed=None)                      gap above the asymptote
    mean(name, n, L_star, seed=None)             L_star + gap
    true_asymptote(name, seed, mode=None)        hidden L_true
    truth_fn(name, L_star, seed=None)            n -> mean(n)
    generator_fn(name, L_star, seed=None)        (n, rng, sigma) -> observed
    regime_functions(name, seed, mode=None)      (gen, truth, L_true) for one seed

    TRUTH[name](n, L_star, seed=None)            registry form of mean()
    GENERATORS[name](n, rng, sigma, L_star, seed=None)

Both registries *require* L_star; the old three-argument call fails loudly
so no caller can silently fall back to a shared asymptote.

Regimes
-------
Core 18 (used for method comparison and any selector / cascade training):
  single_exp, two_exp, three_exp, four_exp, power_law, rational_decay,
  mixed_pow_rat, multiphase, osc_exp, damped_osc_pow, log_slow,
  delayed_plateau, staircase, noisy_plateau, slow_osc_power,
  log_oscillatory, heavy_noise_plat, broken_power_law

Held-out 6 (evaluation only; no method fits these natively and they never
enter selector or cascade training; HOLDOUT flags them):
  stretched_exp   gap = 0.7 exp(-(n/40)^0.5)
  logistic_tail   gap = 0.6 / (1 + exp((n-60)/18))
  inv_sqrt_log    gap = 0.9 / (sqrt(n+1) log(n+e))
  random_knots    seeded 3-knot continuous piecewise power law
                  (knots ~ U{20..200}, exponents ~ U[0.3, 0.9]); the shape
                  depends on the sequence seed
  real_boot_a     smoothing-spline lower envelope of the recorded XGBoost
  real_boot_b     curves (adult, higgs) in results/real_data/, rescaled to
                  gap(0) = 0.7 and shifted to L_true

Two regimes carry intrinsic noise present even at sigma = 0, because
irreducible observation noise is the property under test:
  noisy_plateau     fixed Gaussian noise, sd 0.003
  heavy_noise_plat  fixed Laplace noise, scale 0.004
Their observations are clipped at zero (losses are non-negative); TRUTH is
the unclipped noiseless mean.

Reference
---------
Maleki, K. (2026). Working paper.
"""

import os
import zlib
from typing import Callable, Dict, Optional, Tuple

import numpy as np

import src.config as CFG_MOD


# ── Internal helpers ───────────────────────────────────────────────────────────

def _as_float_array(n) -> np.ndarray:
    """Coerce scalar or array-like n to a float array (0-d for scalars)."""
    return np.asarray(n, dtype=float)


def _gauss(s: np.ndarray, sigma: float, rng: np.random.RandomState) -> np.ndarray:
    """Add i.i.d. Gaussian noise to sequence s."""
    return s + sigma * rng.randn(*s.shape) if sigma > 0.0 else s.copy()


# ── Gap functions: the single definition of each core regime ──────────────────
# Each returns the noiseless distance above the asymptote.  The arithmetic is
# the rejected design's mean with the leading constant removed, so
# L_star + gap(n) at L_star = 0.01 reproduces the old means (see
# tests/test_redesign.py::test_legacy_mode_reproduces_old_means).

def _gap_single_exp(n) -> np.ndarray:
    """Single-component exponential decay: 0.75 exp(-0.040 n)."""
    n = _as_float_array(n)
    return 0.75 * np.exp(-0.040 * n)


def _gap_two_exp(n) -> np.ndarray:
    """Two-component exponential: 0.50 exp(-0.018n) + 0.25 exp(-0.15n)."""
    n = _as_float_array(n)
    return 0.50 * np.exp(-0.018 * n) + 0.25 * np.exp(-0.15 * n)


def _gap_three_exp(n) -> np.ndarray:
    """Three-component exponential mixture."""
    n = _as_float_array(n)
    return (0.30 * np.exp(-0.008 * n)
            + 0.25 * np.exp(-0.060 * n)
            + 0.20 * np.exp(-0.250 * n))


def _gap_four_exp(n) -> np.ndarray:
    """Four-component exponential mixture."""
    n = _as_float_array(n)
    return (0.20 * np.exp(-0.004 * n)
            + 0.20 * np.exp(-0.025 * n)
            + 0.15 * np.exp(-0.100 * n)
            + 0.10 * np.exp(-0.400 * n))


def _gap_power_law(n) -> np.ndarray:
    """Power-law decay: 0.80 / (n+1)^0.70."""
    n = _as_float_array(n)
    return 0.80 / (n + 1.0) ** 0.70


def _gap_rational_decay(n) -> np.ndarray:
    """Rational (Pade-type) decay: 0.80 / (1 + 0.04n)."""
    n = _as_float_array(n)
    return 0.80 / (1.0 + 0.04 * n)


def _gap_mixed_pow_rat(n) -> np.ndarray:
    """Mixed power-law + rational: 0.40/(n+1)^0.5 + 0.30/(1+0.02n)."""
    n = _as_float_array(n)
    return 0.40 / (n + 1.0) ** 0.50 + 0.30 / (1.0 + 0.02 * n)


def _gap_multiphase(n) -> np.ndarray:
    """Multiphase: fast exponential + slow rational tail."""
    n = _as_float_array(n)
    return 0.50 * np.exp(-0.06 * n) + 0.25 / (1.0 + 0.003 * n)


def _gap_osc_exp(n) -> np.ndarray:
    """Oscillatory exponential: exponential envelope x cosine modulation."""
    n = _as_float_array(n)
    return 0.70 * np.exp(-0.030 * n) * (1.0 + 0.15 * np.cos(0.20 * n))


def _gap_damped_osc_pow(n) -> np.ndarray:
    """Damped oscillatory power-law: power-law x log-cosine modulation."""
    n = _as_float_array(n)
    return (0.80 / (n + 1.0) ** 0.60
            * (1.0 + 0.20 * np.cos(0.30 * np.log(n + 1.0))))


def _gap_log_slow(n) -> np.ndarray:
    """Logarithmically slow convergence: 0.80 / log(n + e)."""
    n = _as_float_array(n)
    return 0.80 / np.log(n + np.e)


def _gap_delayed_plateau(n) -> np.ndarray:
    """Delayed plateau: smooth stall near n=40 before resuming decline.

    The Gaussian bump centred at n = 40 is the defining feature of this
    regime and is therefore part of the noiseless trajectory.
    """
    n = _as_float_array(n)
    bump = 0.06 * np.exp(-((n - 40.0) ** 2) / 200.0)
    return 0.60 * np.exp(-0.010 * n) + 0.10 * np.exp(-0.200 * n) + bump


def _gap_staircase(n) -> np.ndarray:
    """Staircase decay: piecewise constant with sudden drops.

    Each entry of ``steps`` is (first index at which the level applies,
    height above the asymptote).  Later entries overwrite earlier ones, so
    the level in force at n is the one for the last threshold not exceeding
    n.  The final step has height 0, so the sequence reaches its asymptote
    exactly at n = 320.
    """
    steps = [(0, 0.70), (20, 0.55), (40, 0.42), (65, 0.28), (90, 0.18),
             (130, 0.10), (170, 0.05), (240, 0.02), (320, 0.0)]
    n = _as_float_array(n)
    g = np.zeros(n.shape, dtype=float)
    for threshold, height in steps:
        g = np.where(n >= threshold, height, g)
    return np.maximum(g, 0.0)


def _gap_noisy_plateau(n) -> np.ndarray:
    """Near-plateau: 0.02 exp(-0.05n), observed under fixed noise."""
    n = _as_float_array(n)
    return 0.02 * np.exp(-0.05 * n)


def _gap_slow_osc_power(n) -> np.ndarray:
    """Slowly oscillating power-law.

    The oscillation period grows as sqrt(n), making the frequency decrease
    over time.  Wynn epsilon struggles when the sign of the error term
    changes unpredictably; this tests that boundary.

        gap(n) = 0.70 / (n+1)^0.65 * (1 + 0.25 * cos(sqrt(n+1)))
    """
    n = _as_float_array(n)
    return (0.70 / (n + 1.0) ** 0.65
            * (1.0 + 0.25 * np.cos(np.sqrt(n + 1.0))))


def _gap_log_oscillatory(n) -> np.ndarray:
    """Log-slow base with superimposed oscillation.

        gap(n) = 0.80/log(n+e) * (1 + 0.20 * sin(0.15n))
    """
    n = _as_float_array(n)
    return 0.80 / np.log(n + np.e) * (1.0 + 0.20 * np.sin(0.15 * n))


def _gap_heavy_noise_plat(n) -> np.ndarray:
    """Near-plateau: 0.05 exp(-0.10n), observed under heavy-tailed noise."""
    n = _as_float_array(n)
    return 0.05 * np.exp(-0.10 * n)


def _gap_broken_power_law(n) -> np.ndarray:
    """Broken power-law: exponent changes at n=60.

        gap(n) = 0.80/(n+1)^alpha(n),  alpha = 0.90 (n < 60) else 0.40
    """
    n = _as_float_array(n)
    alpha = np.where(n < 60, 0.90, 0.40)
    return 0.80 / (n + 1.0) ** alpha


# ── Held-out gap functions (evaluation only) ──────────────────────────────────

def _gap_stretched_exp(n) -> np.ndarray:
    """Stretched exponential: 0.7 exp(-(n/40)^0.5)."""
    n = _as_float_array(n)
    return 0.7 * np.exp(-np.sqrt(n / 40.0))


def _gap_logistic_tail(n) -> np.ndarray:
    """Logistic tail: 0.6 / (1 + exp((n-60)/18)).  The exponent is clipped
    to avoid overflow warnings at large n (the value is 0 there either way)."""
    n = _as_float_array(n)
    z = np.minimum((n - 60.0) / 18.0, 700.0)
    return 0.6 / (1.0 + np.exp(z))


def _gap_inv_sqrt_log(n) -> np.ndarray:
    """Inverse sqrt-log: 0.9 / (sqrt(n+1) log(n+e))."""
    n = _as_float_array(n)
    return 0.9 / (np.sqrt(n + 1.0) * np.log(n + np.e))


def random_knots_params(seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Seeded parameters of the random_knots regime: three sorted knots drawn
    without replacement from {20, ..., 200}, four exponents ~ U[0.3, 0.9],
    and the segment constants that make the piecewise power law continuous
    with gap(0) = 0.7.
    """
    rs = np.random.RandomState(
        zlib.crc32(f"random_knots:{seed}:knots".encode()) % 2**31)
    knots = np.sort(rs.choice(np.arange(20, 201), size=3, replace=False)).astype(float)
    exps = rs.uniform(0.3, 0.9, size=4)
    consts = [0.7]                                  # (0+1)^(-a) = 1  ->  gap(0) = 0.7
    for j in range(1, 4):
        k = knots[j - 1] + 1.0
        consts.append(consts[-1] * k ** (exps[j] - exps[j - 1]))
    return knots, exps, np.asarray(consts, dtype=float)


def _gap_random_knots(n, seed: Optional[int]) -> np.ndarray:
    """Seeded 3-knot continuous piecewise power law (shape depends on seed)."""
    if seed is None:
        raise ValueError("random_knots needs the sequence seed: gap(n, seed)")
    knots, exps, consts = random_knots_params(int(seed))
    n = _as_float_array(n)
    seg = np.searchsorted(knots, n, side="right")   # 0..3
    return consts[seg] * (n + 1.0) ** (-exps[seg])


_REAL_CURVES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "real_data", "real_data_curves.csv")
_REAL_BOOT_SOURCES = {"real_boot_a": "adult", "real_boot_b": "higgs"}
_REAL_BOOT_LAMBDA = 0.01          # smoothing on the log(n+1) axis
_REAL_BOOT_GAP0 = 0.7
_real_boot_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}


def real_boot_profile(dataset: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gap profile of a recorded real curve on its integer round grid.

    Construction: smoothing spline of the recorded validation loss against
    log(n+1) (lambda = 0.01, mild smoothing that keeps the steep start),
    lower envelope (running minimum, so the profile is non-increasing even
    where the real curve overfits and rises), rescaled to gap(0) = 0.7 with
    the envelope's final value as the floor.  Beyond the recorded range the
    gap is 0, so L_true is the exact limit of the regime.
    """
    if dataset in _real_boot_cache:
        return _real_boot_cache[dataset]
    from scipy.interpolate import make_smoothing_spline
    if not os.path.exists(_REAL_CURVES_PATH):
        raise FileNotFoundError(
            f"real_boot regimes need {_REAL_CURVES_PATH}")
    data = np.genfromtxt(_REAL_CURVES_PATH, delimiter=",", names=True)
    n = np.asarray(data["round"], dtype=float)
    y = np.asarray(data[dataset], dtype=float)
    x = np.log(n + 1.0)
    fitted = make_smoothing_spline(x, y, lam=_REAL_BOOT_LAMBDA)(x)
    env = np.minimum.accumulate(fitted)
    floor = env[-1]
    prof = _REAL_BOOT_GAP0 * (env - floor) / (env[0] - floor)
    _real_boot_cache[dataset] = (n, prof)
    return n, prof


def _make_gap_real_boot(dataset: str) -> Callable:
    def gap(n) -> np.ndarray:
        grid_n, prof = real_boot_profile(dataset)
        n = _as_float_array(n)
        return np.interp(n, grid_n, prof, left=prof[0], right=0.0)
    gap.__name__ = f"_gap_real_boot_{dataset}"
    gap.__doc__ = (f"Smoothing-spline lower envelope of the recorded "
                   f"{dataset} curve, rescaled to gap(0) = 0.7.")
    return gap


# ── Registries ────────────────────────────────────────────────────────────────

_CORE_GAPS: Dict[str, Callable] = {
    "single_exp":       _gap_single_exp,
    "two_exp":          _gap_two_exp,
    "three_exp":        _gap_three_exp,
    "four_exp":         _gap_four_exp,
    "power_law":        _gap_power_law,
    "rational_decay":   _gap_rational_decay,
    "mixed_pow_rat":    _gap_mixed_pow_rat,
    "multiphase":       _gap_multiphase,
    "osc_exp":          _gap_osc_exp,
    "damped_osc_pow":   _gap_damped_osc_pow,
    "log_slow":         _gap_log_slow,
    "delayed_plateau":  _gap_delayed_plateau,
    "staircase":        _gap_staircase,
    "noisy_plateau":    _gap_noisy_plateau,
    # Phase 0 additions
    "slow_osc_power":   _gap_slow_osc_power,
    "log_oscillatory":  _gap_log_oscillatory,
    "heavy_noise_plat": _gap_heavy_noise_plat,
    "broken_power_law": _gap_broken_power_law,
}

_HOLDOUT_GAPS: Dict[str, Callable] = {
    "stretched_exp": _gap_stretched_exp,
    "logistic_tail": _gap_logistic_tail,
    "inv_sqrt_log":  _gap_inv_sqrt_log,
    "random_knots":  _gap_random_knots,
    "real_boot_a":   _make_gap_real_boot("adult"),
    "real_boot_b":   _make_gap_real_boot("higgs"),
}

# Regimes whose *shape* depends on the sequence seed (gap needs the seed).
SEED_DEPENDENT_SHAPE = frozenset({"random_knots"})

# Regimes whose observation model is not simply "mean + sigma * Gaussian".
INTRINSIC_NOISE_REGIMES = frozenset({"noisy_plateau", "heavy_noise_plat"})


def _uniform_gap(name: str, fn: Callable, holdout: bool) -> Callable:
    """Wrap a gap function so every entry has the signature gap(n, seed=None)."""
    seed_dep = name in SEED_DEPENDENT_SHAPE

    if seed_dep:
        def gap(n, seed=None):
            return fn(n, seed)
    else:
        def gap(n, seed=None):
            return fn(n)

    gap.__name__ = f"gap_{name}"
    gap.__qualname__ = f"gap_{name}"
    gap.__doc__ = fn.__doc__
    gap.regime = name
    gap.holdout = holdout
    gap.seed_dependent = seed_dep
    return gap


GAP: Dict[str, Callable] = {}
GAP.update({k: _uniform_gap(k, f, holdout=False) for k, f in _CORE_GAPS.items()})
GAP.update({k: _uniform_gap(k, f, holdout=True) for k, f in _HOLDOUT_GAPS.items()})

REGIME_NAMES = list(_CORE_GAPS.keys())            # the 18 core regimes
HOLDOUT_REGIME_NAMES = list(_HOLDOUT_GAPS.keys())  # the 6 held-out regimes
ALL_REGIME_NAMES = REGIME_NAMES + HOLDOUT_REGIME_NAMES
HOLDOUT = frozenset(HOLDOUT_REGIME_NAMES)


def is_holdout(regime: str) -> bool:
    return regime in HOLDOUT


# ── Hidden true asymptote ─────────────────────────────────────────────────────

def asymptote_seed(regime: str, seed: int) -> int:
    """The RandomState seed from which L_true(regime, seed) is drawn."""
    return zlib.crc32(f"{regime}:{seed}:Lstar".encode()) % 2**31


def true_asymptote(regime: str, seed: int, mode: Optional[str] = None) -> float:
    """
    Hidden asymptote of the (regime, seed) sequence.

    hetero (default): 10 ** U(log10(L_TRUE_RANGE[0]), log10(L_TRUE_RANGE[1]))
                      drawn from RandomState(asymptote_seed(regime, seed)).
    legacy          : config.LEGACY_L_INF for every regime and seed (the
                      rejected design; regression test only).
    """
    if regime not in GAP:
        raise KeyError(f"unknown regime {regime!r}")
    mode = CFG_MOD.ASYMPTOTE_MODE if mode is None else mode
    if mode == "legacy":
        return float(CFG_MOD.LEGACY_L_INF)
    if mode != "hetero":
        raise ValueError(
            f"unknown ASYMPTOTE_MODE {mode!r}; expected one of "
            f"{CFG_MOD.ASYMPTOTE_MODES}")
    rs = np.random.RandomState(asymptote_seed(regime, int(seed)))
    lo, hi = np.log10(CFG_MOD.L_TRUE_RANGE[0]), np.log10(CFG_MOD.L_TRUE_RANGE[1])
    return float(10.0 ** rs.uniform(lo, hi))


# ── Means, truths and generators ──────────────────────────────────────────────

def mean(regime: str, n, L_star: float, seed: Optional[int] = None) -> np.ndarray:
    """Noiseless mean: L_star + gap(n)."""
    return L_star + GAP[regime](n, seed)


def truth_fn(regime: str, L_star: float, seed: Optional[int] = None) -> Callable:
    """Return n -> mean(regime, n, L_star) for a fixed asymptote."""
    gap = GAP[regime]

    def truth(n) -> np.ndarray:
        return L_star + gap(n, seed)

    truth.__name__ = f"truth_{regime}"
    truth.L_star = float(L_star)
    truth.regime = regime
    return truth


def gen_noisy_plateau(n, rng: np.random.RandomState, sigma: float,
                      L_star: float, seed: Optional[int] = None) -> np.ndarray:
    """Near-plateau with intrinsic Gaussian noise (sd 0.003), clipped at 0.

    The intrinsic noise is what defines this regime, so the sweep parameter
    sigma is deliberately not applied on top of it.
    """
    base_sigma = 0.003
    s = mean("noisy_plateau", n, L_star) + base_sigma * rng.randn(*np.shape(n))
    return np.maximum(s, 0.0)


def gen_heavy_noise_plat(n, rng: np.random.RandomState, sigma: float,
                         L_star: float, seed: Optional[int] = None) -> np.ndarray:
    """Near-plateau with Laplace (heavy-tailed) noise, scale 0.004, clipped at 0.

    Any sweep noise sigma is added on top of the intrinsic Laplace noise.
    """
    noise = rng.laplace(0.0, 0.004, size=np.shape(n))
    if sigma > 0.0:
        noise = noise + sigma * rng.randn(*np.shape(n))
    return np.maximum(mean("heavy_noise_plat", n, L_star) + noise, 0.0)


_INTRINSIC_NOISE_GENERATORS: Dict[str, Callable] = {
    "noisy_plateau":    gen_noisy_plateau,
    "heavy_noise_plat": gen_heavy_noise_plat,
}


def generator_fn(regime: str, L_star: float, seed: Optional[int] = None) -> Callable:
    """Return (n, rng, sigma) -> observed sequence for a fixed asymptote."""
    if regime in _INTRINSIC_NOISE_GENERATORS:
        base = _INTRINSIC_NOISE_GENERATORS[regime]

        def generator(n, rng, sigma):
            return base(n, rng, sigma, L_star, seed)
    else:
        truth = truth_fn(regime, L_star, seed)

        def generator(n, rng, sigma):
            return _gauss(truth(n), sigma, rng)

    generator.__name__ = f"gen_{regime}"
    generator.L_star = float(L_star)
    generator.regime = regime
    return generator


def regime_functions(regime: str, seed: int,
                     mode: Optional[str] = None) -> Tuple[Callable, Callable, float]:
    """
    Everything an evaluation loop needs for one (regime, seed):

        gen, truth, L_true = regime_functions(regime, seed)
        seq  = gen(n_array, rng, sigma)       # observed
        y_nf = truth(n_f)                     # noiseless target

    L_true is hidden from methods; hand them assumed_asymptote(...) instead.
    """
    L_true = true_asymptote(regime, seed, mode)
    return (generator_fn(regime, L_true, seed),
            truth_fn(regime, L_true, seed),
            L_true)


def _registry_truth(regime: str) -> Callable:
    def truth(n, L_star, seed=None):
        return mean(regime, n, L_star, seed)
    truth.__name__ = f"truth_{regime}"
    return truth


def _registry_generator(regime: str) -> Callable:
    def generator(n, rng, sigma, L_star, seed=None):
        return generator_fn(regime, L_star, seed)(n, rng, sigma)
    generator.__name__ = f"gen_{regime}"
    generator.__doc__ = GAP[regime].__doc__
    return generator


# TRUTH[name](n, L_star, seed=None) and GENERATORS[name](n, rng, sigma, L_star,
# seed=None).  L_star is required: the rejected design's three-argument call
# raises TypeError instead of silently using a shared constant.
TRUTH: Dict[str, Callable] = {name: _registry_truth(name) for name in GAP}
GENERATORS: Dict[str, Callable] = {name: _registry_generator(name) for name in GAP}

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
