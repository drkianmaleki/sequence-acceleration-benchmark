"""
diagnostics.py
==============
Perturbation diagnostics and trajectory feature extraction.

Perturbation diagnostics
------------------------
Two run-time diagnostics of one (method, window) pair, both IQRs of the
method's own estimates under small changes of its input:
    shift_iqr        -- IQR of estimates across small window-start shifts
                        (deterministic)
    perturb_iqr      -- IQR of estimates under tiny multiplicative value
                        perturbations
They are trust signals a deployment could compute; no score is formed from
them.

The perturbations are PAIRED: for one (regime, obs_idx, noise, seed) window
of length n the perturbation factors are one array F of shape
(perturb_trials, n), F = 1 + perturb_scale * Z with Z ~ N(0, 1), drawn once
from RandomState(perturbation_key(regime, seed, obs_idx, sigma)).  Every
method evaluated on that window, at every horizon g, is applied to the same
perturbed windows seq_win * F[t] (perturbation_factors,
perturb_iqr_with_factors).  A method's perturbation diagnostic therefore
depends neither on which other methods are evaluated nor on the order of
regimes, and two methods on the same window are compared on identical
perturbations.

Trajectory features
-------------------
Six scalar features are extracted from the observable window:
    log_log_slope    -- slope of log(s-L) vs log(n)  [Richardson exponent]
    curvature_idx    -- normalised mean |Δ²s|
    oscillation_idx  -- zero-crossing rate of Δs
    noise_var        -- variance of Δ²s  [proxy for noise level]
    richardson_r2    -- R² of 1-term Richardson fit on the window
    diff_ratio_cv    -- coefficient of variation of s[n]/s[n-1] ratios

Reference
---------
Maleki, K. (2026). Working paper.
"""

import zlib

import numpy as np
from typing import List, Optional, Tuple, Dict
from scipy.optimize import curve_fit


# ── Validity check ─────────────────────────────────────────────────────────────

def is_valid(v: float, cfg: dict) -> bool:
    """Return True if v is finite and within configured validity bounds."""
    return bool(np.isfinite(v)
                and cfg.get("min_valid", -0.5) <= v <= cfg.get("max_valid", 500.0))


# ── Perturbation diagnostics ───────────────────────────────────────────────────

def compute_shift_iqr(method_fn, seq: List[float], indices: List[int],
                      future_x: float, cfg: dict) -> float:
    """
    Window-shift IQR.

    Re-run method_fn on windows shifted by each value in cfg['win_shifts'].
    Return the IQR of valid estimates.  A small IQR means the estimate
    is insensitive to slight changes in the window start position.

    Returns np.inf if fewer than 2 valid shift estimates exist.
    """
    shifts = cfg.get("win_shifts", [-2, -1, 0, 1, 2])
    ests: List[float] = []
    for sh in shifts:
        start = max(0, sh)
        sh_seq = seq[start:]
        sh_idx = indices[start:]
        if len(sh_seq) < 3:
            continue
        v = method_fn(sh_seq, sh_idx, future_x, cfg)
        if is_valid(v, cfg):
            ests.append(v)
    if len(ests) < 2:
        return float("inf")
    return float(np.subtract(*np.percentile(ests, [75, 25])))


def perturbation_key(regime: str, seed: int, obs_idx: int, sigma: float) -> int:
    """The RandomState seed of one window's perturbation factors:
    (seed * 999 + obs_idx * 7 + int(sigma * 1e6) % 9973 + crc32(regime) % 100003) mod 2**32."""
    return (int(seed) * 999 + int(obs_idx) * 7 + int(sigma * 1e6) % 9973
            + zlib.crc32(str(regime).encode()) % 100003) % 2**32


def perturbation_factors(regime: str, seed: int, obs_idx: int, sigma: float,
                         n_points: int, n_trials: int, scale: float) -> np.ndarray:
    """
    The paired perturbation factors of one window: F of shape (n_trials,
    n_points), F = 1 + scale * Z, Z ~ N(0, 1), drawn once from
    RandomState(perturbation_key(regime, seed, obs_idx, sigma)).  Every
    method evaluated on the window, at every horizon, uses the same F.
    """
    rng = np.random.RandomState(perturbation_key(regime, seed, obs_idx, sigma))
    return 1.0 + float(scale) * rng.randn(int(n_trials), int(n_points))


def perturb_iqr_with_factors(method_fn, seq_win, idx_win, future_x: float, cfg: dict,
                             factors: np.ndarray) -> float:
    """
    Perturbation IQR of one method on one window under given paired factors:
    the IQR of its valid estimates on the perturbed windows seq_win * F[t]
    (a raised exception counts as an invalid estimate); NaN with fewer than
    two valid estimates.
    """
    arr = np.asarray(seq_win, dtype=float)
    ests: List[float] = []
    for f in np.asarray(factors, dtype=float):
        try:
            v = float(method_fn(list(arr * f), idx_win, future_x, cfg))
        except Exception:
            v = float("nan")
        if is_valid(v, cfg):
            ests.append(v)
    if len(ests) < 2:
        return float("nan")
    return float(np.subtract(*np.percentile(ests, [75, 25])))


# ── Trajectory feature extraction ─────────────────────────────────────────────

def extract_features(seq: List[float], indices: List[int],
                     cfg: dict) -> Dict[str, float]:
    """
    Extract six scalar features from an observable window.

    These features are intended as inputs to an adaptive regime
    classifier or as predictors of Richardson failure.

    Parameters
    ----------
    seq     : observed sequence values (the window)
    indices : corresponding integer indices
    cfg     : must contain 'L_inf' (the assumed asymptote L_hat) and 'ridge'

    Returns
    -------
    dict with keys:
        log_log_slope, curvature_idx, oscillation_idx,
        noise_var, richardson_r2, diff_ratio_cv
    """
    s   = np.asarray(seq, dtype=float)
    x   = np.asarray(indices, dtype=float)
    # cfg["L_inf"] is the ASSUMED asymptote L_hat (redesign v2: no default)
    L0 = min(float(cfg["L_inf"]), float(np.min(s)) * 0.9)
    out: Dict[str, float] = {}

    # 1. log-log slope (Richardson exponent estimate)
    shifted = s - L0
    pos     = shifted > 0
    if pos.sum() >= 3:
        log_x = np.log(np.maximum(x[pos], 1.0))
        log_s = np.log(shifted[pos])
        try:
            p = np.polyfit(log_x, log_s, 1)
            out["log_log_slope"] = float(p[0])   # negative = power-law decay
        except Exception:
            out["log_log_slope"] = float("nan")
    else:
        out["log_log_slope"] = float("nan")

    # 2. curvature index = mean|Δ²s| / range(s)
    if len(s) >= 3:
        d2s   = np.diff(s, 2)
        rng   = float(np.max(s) - np.min(s))
        out["curvature_idx"] = float(np.mean(np.abs(d2s))) / (rng + 1e-15)
    else:
        out["curvature_idx"] = float("nan")

    # 3. oscillation index = zero-crossing rate of Δs
    if len(s) >= 3:
        ds   = np.diff(s)
        zc   = float(np.sum(ds[:-1] * ds[1:] < 0))
        out["oscillation_idx"] = zc / max(len(ds) - 1, 1)
    else:
        out["oscillation_idx"] = float("nan")

    # 4. noise variance = var(Δ²s)
    if len(s) >= 4:
        out["noise_var"] = float(np.var(np.diff(s, 2)))
    else:
        out["noise_var"] = float("nan")

    # 5. Richardson R² (goodness of 1-term power-law fit on the window)
    x_safe = np.maximum(x, 1.0)
    try:
        def model(n, c, a):
            return L0 + c / n**a
        popt, _ = curve_fit(model, x_safe, s,
                            p0=[max(s[-1] - L0, 1e-4), 0.8],
                            bounds=([0, 0.05], [5, 4]),
                            maxfev=1000)
        s_pred = model(x_safe, *popt)
        ss_res = np.sum((s - s_pred) ** 2)
        ss_tot = np.sum((s - s.mean()) ** 2)
        r2 = 1.0 - ss_res / (ss_tot + 1e-20)
        out["richardson_r2"] = float(np.clip(r2, -10.0, 1.0))
    except Exception:
        out["richardson_r2"] = float("nan")

    # 6. Consecutive difference ratio CV = std / |mean| of s[n+1]/s[n]
    if len(s) >= 4:
        ratios = s[1:] / np.maximum(np.abs(s[:-1]), 1e-15)
        ratios = ratios[np.isfinite(ratios)]
        if len(ratios) >= 2:
            cv = float(np.std(ratios)) / (float(np.abs(np.mean(ratios))) + 1e-15)
            out["diff_ratio_cv"] = cv
        else:
            out["diff_ratio_cv"] = float("nan")
    else:
        out["diff_ratio_cv"] = float("nan")

    return out
