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

import src.config as C
from src.asymptote import assumed_asymptote, resolve_mode

# ── Feature extraction (mirrors phase2._extract_features exactly) ──────────────

FEATURE_COLS = [
    'log_log_slope', 'curvature_idx', 'oscillation_idx',
    'noise_var', 'richardson_r2', 'diff_ratio_cv',
]


def extract_features(seq: np.ndarray, indices: np.ndarray, L_hat: float) -> dict:
    """
    Extract six scalar features from an observation window.
    Mirrors phase2._extract_features exactly so results are comparable.

    Parameters
    ----------
    seq     : observed values in the window
    indices : corresponding iteration indices (1-based)
    L_hat   : ASSUMED asymptote (src.asymptote.assumed_asymptote); never L_true

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
    L0      = max(0.0, min(float(L_hat), win_min * 0.5))
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

# Default cfg (numerical settings only).  'L_inf' -- the ASSUMED asymptote
# L_hat -- must be supplied per window; there is deliberately no default.
_DEFAULT_CFG = dict(
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
    if 'L_inf' not in cfg:
        raise ValueError("cfg['L_inf'] (the assumed asymptote L_hat) is required")

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
    assumed_mode:         str   = None,
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
    assumed_mode         : config.ASSUMED_L_MODES entry (default config value).
                           L_true is unknown for real curves, so only the
                           deployable modes 'zero' and 'winmin' are valid here.
    phase2_features_path : path to phase2_features.csv for regime mapping
    future_x             : target round to predict (defaults to n_rounds)

    Returns
    -------
    DataFrame with one row per (dataset, obs_depth), columns:
        dataset, obs_depth, true_final, current_val,
        predicted_val, cascade_err, current_err,
        selected_method, nearest_regime, assumed_mode, L_hat,
        log_log_slope, curvature_idx, oscillation_idx,
        noise_var, richardson_r2, diff_ratio_cv
    """
    # Load regime centroids if available
    regime_centroids = None
    if phase2_features_path and os.path.exists(phase2_features_path):
        df_feat = pd.read_csv(phase2_features_path)
        regime_centroids = df_feat.groupby('regime')[FEATURE_COLS].mean()

    assumed_mode = resolve_mode(assumed_mode)
    cfg = _DEFAULT_CFG.copy()

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

            # 0. Assumed asymptote for this window (no oracle on real curves)
            L_hat = assumed_asymptote(None, window, assumed_mode)
            cfg['L_inf'] = L_hat

            # 1. Features
            feats = extract_features(window, idxs, L_hat)

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
                assumed_mode     = assumed_mode,
                L_hat            = L_hat,
                **{f: feats.get(f, np.nan) for f in FEATURE_COLS},
            )
            records.append(row)

    return pd.DataFrame(records)


# ── Redesign v2: re-evaluation of RECORDED curves (no retraining) ─────────────

REAL_EVAL_METHODS = ['richardson_1', 'rational_fit']


def perturb_seed(dataset: str, depth: int) -> int:
    """Deterministic perturbation seed, crc32 of "dataset:depth" (the seeding
    of scripts/analyze_real_diagnostics_legacy.py).  The window is a property
    of (dataset, depth), so every target round of a window shares its draws."""
    import zlib
    return zlib.crc32(f"{dataset}:{depth}".encode()) % 2 ** 31


def perturb_iqr_real(window, idxs, method: str, future_x: float, cfg: dict,
                     seed: int, n_trials: int = None, scale: float = None) -> float:
    """
    Perturbation IQR of one method on one real window: n_trials evaluations
    on windows multiplied by 1 + scale * U(-1, 1) noise (config.PERTURB_TRIALS
    = 5, config.PERTURB_SCALE = 0.02), seeded deterministically.  The IQR of
    the finite estimates; NaN with fewer than three.  Identical draws and
    rule to scripts/analyze_real_diagnostics_legacy.py.
    """
    n_trials = C.PERTURB_TRIALS if n_trials is None else int(n_trials)
    scale = C.PERTURB_SCALE if scale is None else float(scale)
    window = np.asarray(window, dtype=float)
    rng = np.random.RandomState(seed)
    ests = []
    for _ in range(n_trials):
        pert = window * (1.0 + scale * rng.uniform(-1, 1, size=len(window)))
        e = apply_accelerator(method_name=method, seq=pert, indices=idxs,
                              future_x=future_x, cfg=cfg)
        if np.isfinite(e):
            ests.append(e)
    if len(ests) < 3:
        return float('nan')
    q75, q25 = np.percentile(ests, [75, 25])
    return float(q75 - q25)


def curve_minimum_table(curves: dict, last_round: int = 500) -> pd.DataFrame:
    """
    Per recorded curve: the argmin round (1-based), the minimum, the value at
    ``last_round`` (or the last recorded round if shorter) and the relative
    rise from the minimum to it.  A target round beyond the argmin is a
    post-minimum target: the curve has turned up and any extrapolation of the
    descent is chasing a minimum that has already passed.
    """
    rows = []
    for name, curve in curves.items():
        v = np.asarray(curve, dtype=float)
        n = len(v)
        i = int(np.nanargmin(v))
        end = min(last_round, n)
        v_end = float(v[end - 1])
        rows.append(dict(dataset=name, n_rounds=n, argmin_round=i + 1,
                         min_value=float(v[i]), value_at_round=v_end, at_round=end,
                         rise_from_min=((v_end - float(v[i])) / float(v[i])
                                        if abs(float(v[i])) > 1e-12 else float('nan'))))
    return pd.DataFrame(rows)


def _auc_fail_vs_succ(iqr: np.ndarray, fail: np.ndarray):
    """AUC = P(IQR_fail > IQR_succ) from the Mann-Whitney U statistic (ties one
    half) and the two-sided p-value; NaN when either group is empty."""
    from scipy.stats import mannwhitneyu
    iqr = np.asarray(iqr, dtype=float)
    fail = np.asarray(fail, dtype=bool)
    ok = np.isfinite(iqr)
    f, s = iqr[ok & fail], iqr[ok & ~fail]
    if len(f) == 0 or len(s) == 0:
        return float('nan'), float('nan'), len(f), len(s)
    u = mannwhitneyu(f, s, alternative='two-sided')
    return float(u.statistic) / (len(f) * len(s)), float(u.pvalue), len(f), len(s)


REAL_STRATA = (('all', None), ('pre_min', 0), ('post_min', 1))


def real_data_strata(df_sum: pd.DataFrame) -> pd.DataFrame:
    """
    Every real-data summary statistic three ways: all cells, pre-minimum
    targets (target_round <= argmin_round) and post-minimum targets.  One row
    per stratum: cell and failure counts (failure = cascade_skill >= 1, which
    on these curves coincides with improvement < 0), medians of cascade
    error / skill / improvement, the cascade's win rates against each
    deployable trivial, and the perturbation-diagnostic AUC (higher
    perturb_iqr read as 'failure') with its Mann-Whitney p-value.
    """
    from src.trivial import REFERENCE_TAGS
    rows = []
    for label, flag in REAL_STRATA:
        sub = df_sum if flag is None else df_sum[df_sum['post_min_target'] == flag]
        fail = (sub['cascade_skill'] >= 1.0).to_numpy()
        auc, p, n_f, n_s = _auc_fail_vs_succ(sub['perturb_iqr'].to_numpy(), fail)
        row = dict(stratum=label, n_cells=int(len(sub)), n_fail=int(fail.sum()),
                   fail_rate=(round(float(fail.mean()), 4) if len(sub) else float('nan')),
                   n_neg_improvement=int((sub['improvement'] < 0).sum()),
                   med_cascade_err=float(sub['cascade_err'].median()) if len(sub) else float('nan'),
                   med_cascade_skill=float(sub['cascade_skill'].median()) if len(sub) else float('nan'),
                   med_improvement=float(sub['improvement'].median()) if len(sub) else float('nan'),
                   med_perturb_iqr_fail=(float(sub.loc[fail, 'perturb_iqr'].median()) if fail.any() else float('nan')),
                   med_perturb_iqr_succ=(float(sub.loc[~fail, 'perturb_iqr'].median()) if (~fail).any() else float('nan')),
                   perturb_auc=auc, perturb_p=p, n_fail_auc=n_f, n_succ_auc=n_s,
                   perturb_ordering=('failing > succeeding' if auc > 0.5 else 'failing < succeeding' if auc < 0.5
                                     else 'no ordering') if np.isfinite(auc) else '')
        for _, tag in REFERENCE_TAGS:
            col = f'cascade_win_vs_{tag}'
            row[f'win_rate_vs_{tag}'] = (round(float(sub[col].mean()), 4) if (col in sub and len(sub)) else float('nan'))
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_recorded_curves(
    curves:               dict,
    depths:               list,
    targets:              list,
    window_len:           int   = 60,
    assumed_mode:         str   = None,
    phase2_features_path: str   = None,
):
    """
    Re-evaluate recorded real curves on a (depth, target-round) grid.

    For every dataset, observation depth and target round with
    depth < target <= n_rounds: build the window ending at the depth,
    compute L_hat from the assumed-asymptote mode (no oracle on real
    curves), extract features, apply the two-rule cascade, and predict the
    loss at the target round with richardson_1, rational_fit, the cascade's
    choice and the four trivial reference methods.  ``skill`` is
    err / err(best of the four trivial references) -- the hindsight
    best-of-four (strict) bar; ``skill_vs_*`` / ``win_vs_*`` are the
    fixed-reference ratios and win flags against each trivial separately
    (the summary carries them for the cascade as ``cascade_skill_vs_*`` /
    ``cascade_win_vs_*``).

    Report-2 review (decision 5): every cell also carries ``perturb_iqr``,
    the perturbation IQR of the routed method (``selected_method``) at that
    target round: config.PERTURB_TRIALS evaluations on 2%-perturbed windows,
    crc32-seeded per (dataset, depth) (perturb_iqr_real).  It is a cell-level
    quantity and is repeated on every row of the cell in df_long.  The
    summary records the provenance of the regime centroids:
    ``phase2_features_path``, ``phase2_features_rows`` (0 when the file was
    absent and the mapping is 'unknown') and ``git_head``.

    Prompt 5A: every cell also carries ``argmin_round`` (1-based round of the
    curve's minimum), ``rise_from_min`` (relative rise from the minimum to
    round 500) and ``post_min_target`` = (target_round > argmin_round), so
    every summary can be split into pre- and post-minimum targets
    (real_data_strata).

    Returns
    -------
    (df_long, df_summary)
        df_long    : one row per (dataset, obs_depth, target_round, method)
        df_summary : one row per (dataset, obs_depth, target_round)
    """
    from src.accelerators import METHODS
    from src.pipeline import git_head
    from src.trivial import (SKILL_REFERENCE_METHODS, best_reference_error,
                             skill_score, skill_vs_table)

    assumed_mode = resolve_mode(assumed_mode)
    regime_centroids = None
    feat_rows = 0
    feat_path = str(phase2_features_path) if phase2_features_path else ''
    if phase2_features_path and os.path.exists(phase2_features_path):
        df_feat = pd.read_csv(phase2_features_path)
        regime_centroids = df_feat.groupby('regime')[FEATURE_COLS].mean()
        feat_rows = int(len(df_feat))
    provenance = dict(phase2_features_path=feat_path,
                      phase2_features_rows=feat_rows,
                      git_head=git_head())

    cfg = _DEFAULT_CFG.copy()
    long_rows, summary_rows = [], []
    minima = curve_minimum_table(curves).set_index('dataset')

    for name, curve in curves.items():
        curve    = np.asarray(curve, dtype=float)
        n_rounds = len(curve)
        argmin_round  = int(minima.loc[name, 'argmin_round'])
        rise_from_min = float(minima.loc[name, 'rise_from_min'])

        for depth in depths:
            if depth >= n_rounds:
                continue
            start  = max(0, depth - window_len)
            window = curve[start:depth]
            idxs   = np.arange(start + 1, depth + 1, dtype=float)

            L_hat = assumed_asymptote(None, window, assumed_mode)
            cfg['L_inf'] = L_hat
            feats    = extract_features(window, idxs, L_hat)
            selected = apply_cascade(feats)
            regime   = (map_to_regime(feats, regime_centroids)
                        if regime_centroids is not None else 'unknown')
            current_val = float(curve[depth - 1])

            for target in targets:
                if depth >= target or target > n_rounds:
                    continue
                true_val    = float(curve[target - 1])
                fx          = float(target)
                current_err = abs(current_val - true_val)

                preds = {}
                for m in REAL_EVAL_METHODS:
                    preds[m] = apply_accelerator(m, window, idxs, fx, cfg)
                for m in SKILL_REFERENCE_METHODS:
                    try:
                        preds[m] = float(METHODS[m](list(window), list(idxs), fx, cfg))
                    except Exception:
                        preds[m] = float('nan')
                preds['cascade'] = preds[selected]

                errs = {m: (abs(p - true_val) if np.isfinite(p) else float('nan'))
                        for m, p in preds.items()}
                ref_err = best_reference_error(errs)
                ref_best = min((m for m in SKILL_REFERENCE_METHODS
                                if np.isfinite(errs[m])),
                               key=lambda m: errs[m], default='')

                # perturb_iqr of the routed method at this target (cell-level)
                p_iqr = perturb_iqr_real(window, idxs, selected, fx, cfg,
                                         perturb_seed(name, depth))

                post_min = int(target > argmin_round)

                for m, p in preds.items():
                    e = errs[m]
                    long_rows.append(dict(
                        dataset=name, obs_depth=depth, target_round=target,
                        argmin_round=argmin_round, post_min_target=post_min,
                        method=m, selected_method=selected,
                        is_cascade=int(m == 'cascade'),
                        is_trivial=int(m in SKILL_REFERENCE_METHODS),
                        prediction=p, true_val=true_val, error=e,
                        current_err=current_err, ref_error=ref_err,
                        skill=(skill_score(e, ref_err) if np.isfinite(e) else float('nan')),
                        **skill_vs_table(e, errs),
                        perturb_iqr=p_iqr,
                        L_hat=L_hat, assumed_mode=assumed_mode,
                        nearest_regime=regime,
                    ))

                casc_err = errs['cascade']
                summary_rows.append(dict(
                    dataset=name, obs_depth=depth, target_round=target,
                    argmin_round=argmin_round, rise_from_min=rise_from_min,
                    post_min_target=post_min,
                    selected_method=selected, cascade_pred=preds['cascade'],
                    true_val=true_val, cascade_err=casc_err,
                    richardson_err=errs['richardson_1'],
                    rational_err=errs['rational_fit'],
                    current_err=current_err, ref_error=ref_err,
                    ref_best_method=ref_best,
                    cascade_skill=(skill_score(casc_err, ref_err)
                                   if np.isfinite(casc_err) else float('nan')),
                    improvement=((current_err - casc_err) / current_err
                                 if (np.isfinite(casc_err) and current_err > 1e-10)
                                 else float('nan')),
                    **{f'cascade_{k}': v for k, v in skill_vs_table(casc_err, errs).items()},
                    perturb_iqr=p_iqr,
                    L_hat=L_hat, assumed_mode=assumed_mode, nearest_regime=regime,
                    **{f: feats.get(f, np.nan) for f in FEATURE_COLS},
                    **provenance,
                ))

    return pd.DataFrame(long_rows), pd.DataFrame(summary_rows)
