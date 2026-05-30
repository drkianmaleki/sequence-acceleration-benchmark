"""
trajectories.py
===============
Feature extraction, cascade selection, and accelerator application
for real XGBoost validation loss curves.

Takes raw loss curves produced by datasets.py, applies the Phase 2
six-feature extractor, maps each curve to its nearest synthetic regime,
selects a method via the two-rule cascade, and applies that method to
produce a predicted final loss.
"""

import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# ── Feature extraction (mirrors phase2._extract_features exactly) ──────────────

FEATURE_COLS = [
    'log_log_slope', 'curvature_idx', 'oscillation_idx',
    'noise_var', 'richardson_r2', 'diff_ratio_cv',
]


def extract_features(seq: np.ndarray, indices: np.ndarray, L_inf: float) -> dict:
    """
    Extract six scalar features from an observation window.
    Mirrors phase2._extract_features exactly so results are comparable.

    Parameters
    ----------
    seq     : observed values in the window
    indices : corresponding iteration indices (1-based)
    L_inf   : assumed asymptotic floor

    Returns
    -------
    dict with keys: log_log_slope, curvature_idx, oscillation_idx,
                    noise_var, richardson_r2, diff_ratio_cv
    """
    s   = np.asarray(seq,     dtype=float)
    x   = np.asarray(indices, dtype=float)
    out: dict = {}

    # Dynamic L0: guarantee shifted values are positive
    win_min = float(np.min(s))
    L0      = max(0.0, min(L_inf, win_min * 0.5))
    if win_min <= 0.0:
        L0 = 0.0

    # 1. log-log slope
    shifted = s - L0
    pos     = shifted > 0
    if pos.sum() >= 3:
        try:
            log_x = np.log(np.maximum(x[pos], 1.0))
            log_s = np.log(shifted[pos])
            p = np.polyfit(log_x, log_s, 1)
            out['log_log_slope'] = float(p[0])
        except Exception:
            out['log_log_slope'] = float('nan')
    else:
        out['log_log_slope'] = float('nan')

    # 2. curvature index
    if len(s) >= 3:
        d2s = np.diff(s, 2)
        rng = float(np.max(s) - np.min(s))
        out['curvature_idx'] = float(np.mean(np.abs(d2s))) / (rng + 1e-15)
    else:
        out['curvature_idx'] = float('nan')

    # 3. oscillation index
    if len(s) >= 3:
        ds = np.diff(s)
        zc = float(np.sum(ds[:-1] * ds[1:] < 0))
        out['oscillation_idx'] = zc / max(len(ds) - 1, 1)
    else:
        out['oscillation_idx'] = float('nan')

    # 4. noise variance
    if len(s) >= 4:
        out['noise_var'] = float(np.var(np.diff(s, 2)))
    else:
        out['noise_var'] = float('nan')

    # 5. Richardson R²
    x_safe = np.maximum(x, 1.0)
    try:
        def model(n, c, a):
            return L0 + c / n**a
        popt, _ = curve_fit(
            model, x_safe, s,
            p0     = [max(float(s[-1]) - L0, 1e-4), 0.8],
            bounds = ([0, 0.05], [5, 4]),
            maxfev = 1000,
        )
        s_pred = model(x_safe, *popt)
        ss_res = float(np.sum((s - s_pred) ** 2))
        ss_tot = float(np.sum((s - s.mean()) ** 2))
        r2     = 1.0 - ss_res / (ss_tot + 1e-20)
        out['richardson_r2'] = float(np.clip(r2, -10.0, 1.0))
    except Exception:
        out['richardson_r2'] = float('nan')

    # 6. Consecutive ratio CV
    if len(s) >= 4:
        ratios = s[1:] / np.maximum(np.abs(s[:-1]), 1e-15)
        ratios = ratios[np.isfinite(ratios)]
        if len(ratios) >= 2:
            cv = (float(np.std(ratios))
                  / (float(np.abs(np.mean(ratios))) + 1e-15))
            out['diff_ratio_cv'] = cv
        else:
            out['diff_ratio_cv'] = float('nan')
    else:
        out['diff_ratio_cv'] = float('nan')

    return out


# ── Cascade selector (Phase 2 two-rule cascade) ────────────────────────────────

def apply_cascade(features: dict) -> str:
    """
    Apply the Phase 2 two-rule cascade to select a method.

    Rules (from phase2_rules.csv, best precision):
      1. if log_log_slope > -0.10  → use rational_fit
      2. if richardson_r2  < 0.50  → use rational_fit
      3. otherwise                 → use richardson_1

    Returns the name of the selected method.
    """
    slope = features.get('log_log_slope', float('nan'))
    r2    = features.get('richardson_r2',  float('nan'))

    if np.isfinite(slope) and slope > -0.10:
        return 'rational_fit'
    if np.isfinite(r2) and r2 < 0.50:
        return 'rational_fit'
    return 'richardson_1'


# ── Accelerator application ────────────────────────────────────────────────────

# Default cfg matches what evaluation.py uses
_DEFAULT_CFG = dict(
    L_inf      = 0.01,
    ridge      = 1e-6,
    denom_tol  = 1e-10,
    min_valid  = -0.5,
    max_valid  = 500.0,
)


def apply_accelerator(
    method_name: str,
    seq:         np.ndarray,
    indices:     np.ndarray,
    future_x:    float,
    cfg:         dict = None,
) -> float:
    """
    Call the named accelerator on the observation window and return
    a predicted value at future_x.

    Parameters
    ----------
    method_name : key in src.accelerators.METHODS
    seq         : observation window values
    indices     : corresponding iteration indices
    future_x    : target iteration index to predict at
    cfg         : accelerator config dict (defaults to _DEFAULT_CFG)

    Returns
    -------
    Predicted value as float, or np.nan if the method fails.
    """
    from src.accelerators import METHODS

    if cfg is None:
        cfg = _DEFAULT_CFG.copy()

    fn = METHODS.get(method_name)
    if fn is None:
        return float('nan')

    try:
        result = fn(seq, indices, future_x, cfg)
        if not np.isfinite(result):
            return float('nan')
        # Sanity clamp: prediction must be positive and below observed max
        if result < 0 or result > float(np.max(seq)) * 2:
            return float('nan')
        return float(result)
    except Exception:
        return float('nan')


# ── Regime mapping ─────────────────────────────────────────────────────────────

def map_to_regime(
    features:         dict,
    regime_centroids: pd.DataFrame,
) -> str:
    """
    Map a feature vector to the nearest synthetic regime by Euclidean
    distance in feature space (using only finite features).
    """
    fvec = np.array([features.get(c, np.nan) for c in FEATURE_COLS])
    cent = regime_centroids[FEATURE_COLS].values

    valid = np.isfinite(fvec) & np.all(np.isfinite(cent), axis=0)
    if valid.sum() == 0:
        return 'unknown'

    dists = np.linalg.norm(cent[:, valid] - fvec[valid], axis=1)
    return regime_centroids.index[np.argmin(dists)]


# ── Curve processing ───────────────────────────────────────────────────────────

def process_curves(
    curves:               dict,
    obs_depths:           list  = [30, 60, 90],
    window_len:           int   = 60,
    L_inf:                float = 0.01,
    phase2_features_path: str   = None,
    future_x:             int   = None,
) -> pd.DataFrame:
    """
    For each curve and observation depth:
      1. Extract six Phase 2 features from the observation window
      2. Apply the two-rule cascade to select a method
      3. Apply that method to predict the final loss
      4. Map to nearest synthetic regime

    Parameters
    ----------
    curves               : {dataset_name: np.ndarray of shape (n_rounds,)}
    obs_depths           : observation cutoff points (round indices, 1-based)
    window_len           : number of points in the observation window
    L_inf                : assumed asymptotic floor
    phase2_features_path : path to phase2_features.csv for regime mapping
    future_x             : target round to predict (defaults to n_rounds)

    Returns
    -------
    DataFrame with one row per (dataset, obs_depth), columns:
        dataset, obs_depth, true_final, current_val,
        predicted_val, cascade_err, current_err,
        selected_method, nearest_regime,
        log_log_slope, curvature_idx, oscillation_idx,
        noise_var, richardson_r2, diff_ratio_cv
    """
    # Load regime centroids if available
    regime_centroids = None
    if phase2_features_path and os.path.exists(phase2_features_path):
        df_feat = pd.read_csv(phase2_features_path)
        regime_centroids = df_feat.groupby('regime')[FEATURE_COLS].mean()

    # Default cfg using the provided L_inf
    cfg = _DEFAULT_CFG.copy()
    cfg['L_inf'] = L_inf

    records = []
    for name, curve in curves.items():
        n_rounds   = len(curve)
        true_final = float(curve[-1])
        fx         = float(future_x if future_x is not None else n_rounds)

        for obs in obs_depths:
            if obs >= n_rounds:
                continue

            # Build observation window
            start  = max(0, obs - window_len)
            window = curve[start:obs]
            idxs   = np.arange(start + 1, obs + 1, dtype=float)

            # 1. Features
            feats = extract_features(window, idxs, L_inf)

            # 2. Cascade → method selection
            method = apply_cascade(feats)

            # 3. Apply accelerator → predicted final loss
            predicted_val = apply_accelerator(
                method_name = method,
                seq         = window,
                indices     = idxs,
                future_x    = fx,
                cfg         = cfg,
            )

            # 4. Regime mapping
            regime = (map_to_regime(feats, regime_centroids)
                      if regime_centroids is not None else 'unknown')

            # 5. Errors
            current_val  = float(curve[obs - 1])
            cascade_err  = (abs(predicted_val - true_final)
                            if np.isfinite(predicted_val) else float('nan'))
            current_err  = abs(current_val - true_final)

            row = dict(
                dataset          = name,
                obs_depth        = obs,
                true_final       = true_final,
                current_val      = current_val,
                predicted_val    = predicted_val,
                cascade_err      = cascade_err,
                current_err      = current_err,
                selected_method  = method,
                nearest_regime   = regime,
                **{f: feats.get(f, np.nan) for f in FEATURE_COLS},
            )
            records.append(row)

    return pd.DataFrame(records)
