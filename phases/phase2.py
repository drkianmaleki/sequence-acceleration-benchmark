"""
phase2.py
=========
Phase 2 — Richardson Failure Condition Mapping (redesign v2).

Four components:
  1. Sweep evaluation   — run the reduced method pool across (obs_idx, noise,
                          gap stratum) grids to map where Richardson's rank falls.
  2. Feature extraction — compute six trajectory features for every sequence,
                          using a dynamic L0 baseline that prevents NaN on
                          near-converged windows.
  3. Correlation        — Spearman correlation between features and Richardson
                          relative rank; minimum 15 observations required per
                          per-regime correlation.
  4. Rule testing       — evaluate simple and compound threshold rules.

Redesign v2
-----------
  * Targets are gap-stratified: for every (regime, obs_idx) and every g in
    config.HORIZON_GAP_FRACTIONS the target index is
    n_f = first n > obs_idx with gap(n) <= g * gap(obs_idx), capped at
    50,000 with the achieved fraction recorded.  Records and aggregations
    are keyed by target_g; capped cells are flagged and excluded from every
    pooled cross-regime statistic (phase diagram mean, correlations, rules).
  * The pool is the existing 9 methods + constant_assumed + constant_oracle.
    The oracle is evaluation-only: it never enters a rank or a "best other".
  * Core 18 regimes only (this phase feeds selector / cascade training).
  * Methods receive L_hat (ASSUMED_L_MODE "zero"); L_true is hidden.
  * Every record carries L_true, L_hat, target_g, achieved_g, n_f, capped,
    skill (vs best-of-four trivial reference), is_trivial, is_oracle.

Reduced method set (9 methods covering all Phase 1 champions):
  current_value   baseline floor
  richardson_1    Phase 1 reference (flexible power-law)
  richardson_a10  Safer fixed-alpha variant (alpha=1)
  single_exp_fit  Phase 1 overall winner
  rational_fit    Short-horizon winner
  pade_22         Rational-decay specialist
  log_linear      Oscillatory-exponential specialist
  weniger_d2      Staircase specialist
  anderson_1      Stable limit-estimator fallback

Author : Kian Maleki
Date   : 2026-05-24 (v1), 2026-09-19 (redesign v2)
"""

import os, math, warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.optimize import curve_fit
from typing import List, Dict, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

import src.config as CFG_MOD
from src.accelerators import METHODS
from src.generators   import regime_functions
from src.asymptote    import assumed_asymptote
from src.pipeline     import (REFERENCE_METHODS, capped_block, exclude_capped,
                              horizon_meta, method_flags, resolve_regimes)
from src.trivial      import ORACLE_METHODS, best_reference_error, skill_score

# ── Method pool ────────────────────────────────────────────────────────────────
PHASE2_BASE_METHODS = [
    'current_value',
    'richardson_1',
    'richardson_a10',
    'single_exp_fit',
    'rational_fit',
    'pade_22',
    'log_linear',
    'weniger_d2',
    'anderson_1',
]
# Redesign v2: the trivial constant predictors are first-class comparators.
PHASE2_METHODS = PHASE2_BASE_METHODS + ['constant_assumed', 'constant_oracle']
# Ranks, "best other" and selector candidates never include the oracle.
RANK_POOL = [m for m in PHASE2_METHODS if m not in ORACLE_METHODS]

METHOD_COLOURS = {
    'current_value':    '#888888',
    'richardson_1':     '#f4a261',
    'richardson_a10':   '#e76f51',
    'single_exp_fit':   '#2196f3',
    'rational_fit':     '#1565c0',
    'pade_22':          '#e91e63',
    'log_linear':       '#00897b',
    'weniger_d2':       '#9c27b0',
    'anderson_1':       '#795548',
    'constant_assumed': '#212121',
    'constant_oracle':  '#000000',
}

FIG_DPI = 150
CELL_KEYS = ['regime', 'obs_idx', 'noise', 'target_g']


# ── Config builder ─────────────────────────────────────────────────────────────
def _cfg(future_idx: int, L_hat: float, L_true: Optional[float] = None) -> dict:
    """L_hat is the ASSUMED asymptote (src.asymptote); never L_true.  L_true is
    stored under cfg['L_true'] for the constant_oracle comparator only."""
    cfg = {
        'L_inf':          float(L_hat),
        'ridge':          CFG_MOD.RIDGE,
        'min_valid':      CFG_MOD.MIN_VALID,
        'max_valid':      CFG_MOD.MAX_VALID,
        'denom_tol':      CFG_MOD.DENOM_TOL,
        'win_shifts':     CFG_MOD.WIN_SHIFTS,
        'perturb_trials': CFG_MOD.PERTURB_TRIALS,
        'perturb_scale':  CFG_MOD.PERTURB_SCALE,
        'W_CAT':          CFG_MOD.W_CAT,
        'W_BEATS':        CFG_MOD.W_BEATS,
        'future_idx':     future_idx,
    }
    if L_true is not None:
        cfg['L_true'] = float(L_true)
    return cfg


def _valid(v: float, cfg: dict) -> bool:
    return bool(math.isfinite(v) and
                cfg['min_valid'] <= v <= cfg['max_valid'])


def _stab(vr: float, cr: float, br: float) -> float:
    return vr - CFG_MOD.W_CAT * cr + CFG_MOD.W_BEATS * br


def _med(series) -> float:
    vals = pd.Series(series).dropna()
    return float(vals.median()) if len(vals) else float('nan')


# ── Feature extraction (local, with dynamic L0 fix) ───────────────────────────

def _extract_features(seq: list, indices: list, L_hat: float) -> dict:
    """
    Extract six scalar features from an observable window.

    Dynamic L0 fix: uses L0 = min(L_hat, 0.5 * min(window)) so that
    shifted = s - L0 > 0 even when the sequence is near its limit.
    This prevents NaN on near-converged windows.  L_hat is the ASSUMED
    asymptote handed to the cascade; L_true is never used here.

    Features
    --------
    log_log_slope   : slope of log(s-L0) vs log(n)  [Richardson exponent est.]
    curvature_idx   : normalised mean |Delta^2 s|
    oscillation_idx : zero-crossing rate of Delta s
    noise_var       : variance of Delta^2 s  [proxy for noise level]
    richardson_r2   : R^2 of 1-term Richardson fit on the window
    diff_ratio_cv   : coefficient of variation of s[n+1]/s[n] ratios
    """
    s   = np.asarray(seq, dtype=float)
    x   = np.asarray(indices, dtype=float)
    out: dict = {}

    # Dynamic L0: guarantee all shifted values are positive
    win_min  = float(np.min(s))
    L0       = max(0.0, min(float(L_hat), win_min * 0.5))
    # If window min is non-positive (noisy plateau), subtract 0
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

    # 2. curvature index = mean|Delta^2 s| / range(s)
    if len(s) >= 3:
        d2s = np.diff(s, 2)
        rng = float(np.max(s) - np.min(s))
        out['curvature_idx'] = float(np.mean(np.abs(d2s))) / (rng + 1e-15)
    else:
        out['curvature_idx'] = float('nan')

    # 3. oscillation index = zero-crossing rate of Delta s
    if len(s) >= 3:
        ds = np.diff(s)
        zc = float(np.sum(ds[:-1] * ds[1:] < 0))
        out['oscillation_idx'] = zc / max(len(ds) - 1, 1)
    else:
        out['oscillation_idx'] = float('nan')

    # 4. noise variance = var(Delta^2 s)
    if len(s) >= 4:
        out['noise_var'] = float(np.var(np.diff(s, 2)))
    else:
        out['noise_var'] = float('nan')

    # 5. Richardson R^2 (goodness of 1-term power-law fit)
    x_safe = np.maximum(x, 1.0)
    try:
        def model(n, c, a):
            return L0 + c / n**a
        popt, _ = curve_fit(
            model, x_safe, s,
            p0=[max(float(s[-1]) - L0, 1e-4), 0.8],
            bounds=([0, 0.05], [5, 4]),
            maxfev=1000,
        )
        s_pred  = model(x_safe, *popt)
        ss_res  = float(np.sum((s - s_pred) ** 2))
        ss_tot  = float(np.sum((s - s.mean()) ** 2))
        r2      = 1.0 - ss_res / (ss_tot + 1e-20)
        out['richardson_r2'] = float(np.clip(r2, -10.0, 1.0))
    except Exception:
        out['richardson_r2'] = float('nan')

    # 6. Consecutive ratio CV = std / |mean| of s[n+1]/s[n]
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


FEATURE_COLS = [
    'log_log_slope', 'curvature_idx', 'oscillation_idx',
    'noise_var', 'richardson_r2', 'diff_ratio_cv',
]

# ── Candidate threshold rules ──────────────────────────────────────────────────
# Each entry: (feature, operator, threshold, alternative_method)
CANDIDATE_RULES = [
    # R2-based rules — primary detection signal
    ('richardson_r2', '<', 0.50, 'pade_22'),
    ('richardson_r2', '<', 0.50, 'rational_fit'),
    ('richardson_r2', '<', 0.50, 'weniger_d2'),
    ('richardson_r2', '<', 0.70, 'rational_fit'),
    ('richardson_r2', '<', 0.70, 'log_linear'),
    ('richardson_r2', '<', 0.85, 'rational_fit'),
    # Slope-based rules — targeting rational/plateau regimes
    ('log_log_slope', '>', -0.10, 'pade_22'),
    ('log_log_slope', '>', -0.10, 'rational_fit'),
    ('log_log_slope', '>', -0.30, 'rational_fit'),
    # Oscillation-based rules — targeting oscillatory regimes
    ('oscillation_idx', '>', 0.05, 'log_linear'),
    ('oscillation_idx', '>', 0.10, 'log_linear'),
    ('oscillation_idx', '>', 0.20, 'log_linear'),
    ('oscillation_idx', '>', 0.05, 'weniger_d2'),
    # Curvature-based rules
    ('curvature_idx', '>', 0.10, 'rational_fit'),
    ('curvature_idx', '>', 0.20, 'pade_22'),
    # Ratio-consistency rules — targeting staircase/plateau
    ('diff_ratio_cv', '<', 0.01, 'weniger_d2'),
    ('diff_ratio_cv', '<', 0.05, 'pade_22'),
    # Noise-based rules
    ('noise_var', '>', 1e-5, 'richardson_a10'),
    ('noise_var', '>', 1e-4, 'single_exp_fit'),
]


# =============================================================================
# 1.  SWEEP EVALUATION
# =============================================================================

def run_sweep(obs_idx_list: List[int],
              noise_list:   List[float],
              gap_fractions: List[float],
              n_seeds:      int,
              window_len:   int,
              out_dir:      str,
              core_regimes: Optional[List[str]] = None,
              verbose:      bool = True):
    """
    Evaluate PHASE2_METHODS across all (obs_idx, noise, gap stratum) grids on
    the core regimes.  Returns (df_agg, df_feat) DataFrames.
    """
    os.makedirs(out_dir, exist_ok=True)
    regimes = resolve_regimes(core_regimes, include_holdout=False)
    gap_fractions = [float(g) for g in gap_fractions]

    n_total = (len(obs_idx_list) * len(noise_list) * n_seeds * len(regimes))
    done    = 0

    sweep_records   = []
    feature_records = []

    # Only the observed prefix is needed: targets come from the noiseless truth.
    n_arr = np.arange(max(obs_idx_list) + 1, dtype=float)
    extra_refs = [m for m in REFERENCE_METHODS if m not in PHASE2_METHODS]

    for obs_idx in obs_idx_list:
        wl = min(window_len, obs_idx)

        for sigma in noise_list:
            for seed in range(n_seeds):
                rng = np.random.RandomState(
                    seed * 137 + int(sigma * 1e6) % 9973 + obs_idx * 7)

                for regime in regimes:
                    # Hidden per-(regime, seed) asymptote; methods never see L_true.
                    gen, truth, L_true = regime_functions(regime, seed)
                    seq_full = gen(n_arr, rng, sigma)

                    # Observation window
                    w_start  = max(0, obs_idx - wl + 1)
                    seq_win  = list(seq_full[w_start : obs_idx + 1])
                    idx_win  = list(range(w_start, obs_idx + 1))
                    curr_val = float(seq_full[obs_idx])

                    # Assumed asymptote handed to features and methods
                    L_hat    = assumed_asymptote(L_true, seq_win)
                    feats    = _extract_features(seq_win, idx_win, L_hat)
                    feat_row = {
                        'regime':  regime,
                        'obs_idx': obs_idx,
                        'noise':   sigma,
                        'seed':    seed,
                        'L_true':  L_true,
                        'L_hat':   L_hat,
                    }
                    feat_row.update(feats)
                    feature_records.append(feat_row)

                    # Method evaluations, one gap stratum at a time
                    for g in gap_fractions:
                        hm       = horizon_meta(regime, obs_idx, g, seed)
                        n_f      = hm['n_f']
                        true_val = float(truth(n_f))
                        curr_err = abs(curr_val - true_val)
                        cfg      = _cfg(n_f, L_hat, L_true)

                        ests: Dict[str, float] = {}
                        errs: Dict[str, float] = {}
                        for method in PHASE2_METHODS + extra_refs:
                            try:
                                est = float(METHODS[method](seq_win, idx_win, float(n_f), cfg))
                            except Exception:
                                est = float('nan')
                            ests[method] = est
                            errs[method] = abs(est - true_val) if _valid(est, cfg) else float('nan')
                        ref_err = best_reference_error(errs)

                        for method in PHASE2_METHODS:
                            est, err = ests[method], errs[method]
                            valid = math.isfinite(err)
                            cat   = (not valid) or (
                                curr_err > 1e-12 and err > CFG_MOD.CAT_MULT * curr_err)
                            beats = valid and curr_err > 1e-12 and err < curr_err
                            impv  = ((curr_err / err) if (valid and err > 1e-12)
                                     else (1.0 if valid else float('nan')))
                            rec = {
                                'method':       method,
                                'regime':       regime,
                                'obs_idx':      obs_idx,
                                'noise':        sigma,
                                'seed':         seed,
                                'L_true':       L_true,
                                'L_hat':        L_hat,
                                'valid':        int(valid),
                                'catastrophic': int(cat),
                                'beats':        int(beats),
                                'error':        err if valid else float('nan'),
                                'impv':         impv if math.isfinite(impv) else float('nan'),
                                'ref_error':    ref_err,
                                'skill':        skill_score(err, ref_err) if valid else float('nan'),
                            }
                            rec.update(hm)
                            rec.update(method_flags(method))
                            sweep_records.append(rec)

                    done += 1
                    if verbose and done % max(1, n_total // 20) == 0:
                        print(f'  [{done:>6}/{n_total}]  {100*done/n_total:5.1f}%'
                              f'  obs={obs_idx}  sigma={sigma:.3f}',
                              flush=True)

    if verbose:
        print(f'  [{n_total}/{n_total}] 100.0%  Done.\n')

    df_raw  = pd.DataFrame(sweep_records)
    df_feat = pd.DataFrame(feature_records)

    # ── Aggregate ──────────────────────────────────────────────────────────────
    agg = []
    for keys, grp in df_raw.groupby(['method'] + CELL_KEYS, sort=False):
        method, regime, obs_idx, sigma, g = keys
        vr = grp['valid'].mean()
        cr = grp['catastrophic'].mean()
        br = grp['beats'].mean()
        agg.append({
            'method':     method,   'regime':     regime,
            'obs_idx':    obs_idx,  'noise':      sigma,
            'target_g':   g,
            'n_f':        float(grp['n_f'].median()),
            'achieved_g': _med(grp['achieved_g']),
            'capped':     int(grp['capped'].max()),
            'L_true':     _med(grp['L_true']),
            'L_hat':      _med(grp['L_hat']),
            'is_trivial': int(grp['is_trivial'].iloc[0]),
            'is_oracle':  int(grp['is_oracle'].iloc[0]),
            'valid_rate': round(vr, 4),
            'cat_rate':   round(cr, 4),
            'beats_rate': round(br, 4),
            'med_error':  _med(grp['error']),
            'med_skill':  _med(grp['skill']),
            'stability':  round(_stab(vr, cr, br), 4),
            'n_seeds':    int(len(grp)),
        })

    df_agg = pd.DataFrame(agg)

    # ── Save ───────────────────────────────────────────────────────────────────
    p = os.path.join(out_dir, 'phase2_sweep_aggregated.csv')
    df_agg.to_csv(p, index=False)
    print(f'  Saved: {p}  ({len(df_agg)} rows)')

    p = os.path.join(out_dir, 'phase2_features.csv')
    df_feat.to_csv(p, index=False)
    print(f'  Saved: {p}  ({len(df_feat)} rows)')

    df_cap = capped_block(df_agg, keys=['regime', 'obs_idx', 'target_g', 'method'],
                          value_cols=['med_error', 'med_skill'])
    p = os.path.join(out_dir, 'phase2_capped.csv')
    df_cap.to_csv(p, index=False)
    print(f'  Saved: {p}  ({len(df_cap)} rows)')

    return df_agg, df_feat


# =============================================================================
# 2.  PHASE DIAGRAMS
# =============================================================================

def build_phase_diagrams(df_agg:  pd.DataFrame,
                         g:       float,
                         out_dir: str) -> pd.DataFrame:
    """
    For each (regime, obs_idx, noise) at stratum g, determine Richardson's
    rank among the RANK_POOL (oracle excluded) and the best alternative.
    Cells carry n_f / achieved_g / capped so pooled views can drop capped cells.
    """
    rows = []
    sub  = df_agg[(df_agg['target_g'] == g) & (df_agg['method'].isin(RANK_POOL))]

    for (regime, obs_idx, sigma), grp in sub.groupby(['regime', 'obs_idx', 'noise']):
        ranked = (grp.sort_values(['stability', 'med_error'], ascending=[False, True])
                     .reset_index(drop=True))
        ranked['rank'] = ranked.index + 1
        by_err = (grp[grp['med_error'].notna()].sort_values('med_error')
                     .reset_index(drop=True))
        by_err['err_rank'] = by_err.index + 1

        r1 = ranked[ranked['method'] == 'richardson_1']
        r1e = by_err[by_err['method'] == 'richardson_1']
        best = ranked.iloc[0]

        r1_stab  = float(r1['stability'].values[0]) if not r1.empty else float('nan')
        r1_rank  = int(r1['rank'].values[0])         if not r1.empty else 99
        r1_erank = int(r1e['err_rank'].values[0])    if not r1e.empty else 99
        best_sc  = float(best['stability'])
        best_m   = str(best['method'])
        margin   = round(best_sc - r1_stab, 4)

        rows.append({
            'regime':                 regime,
            'obs_idx':                obs_idx,
            'noise':                  sigma,
            'target_g':               g,
            'n_f':                    float(grp['n_f'].iloc[0]),
            'achieved_g':             float(grp['achieved_g'].iloc[0]),
            'capped':                 int(grp['capped'].max()),
            'richardson_stability':   round(r1_stab, 4),
            'richardson_rank':        r1_rank,
            'richardson_err_rank':    r1_erank,
            'best_method':            best_m,
            'best_stability':         round(best_sc, 4),
            'stability_margin':       margin,
            'richardson_wins':        int(r1_rank == 1),
        })

    df_pd = pd.DataFrame(rows)
    p = os.path.join(out_dir, f'phase2_phase_diagram_g{g:g}.csv')
    df_pd.to_csv(p, index=False)
    print(f'  Saved: {p}  ({len(df_pd)} rows)')
    return df_pd


# =============================================================================
# 3.  CORRELATION ANALYSIS
# =============================================================================

def run_correlation_analysis(df_feat: pd.DataFrame,
                             df_agg:  pd.DataFrame,
                             g:       float,
                             out_dir: str,
                             suffix:  str = ''):
    """
    Spearman correlations between trajectory features and Richardson's
    losing margin at stratum g.

    Global (ALL regimes) and per-regime.  Capped cells are excluded (the
    losing margin at a capped cell is measured against a non-target horizon).
    Per-regime requires >= 15 finite observations.
    """
    MIN_OBS = 15
    keys = ['regime', 'obs_idx', 'noise']

    sub = df_agg[(df_agg['target_g'] == g) & (df_agg['method'].isin(RANK_POOL))]
    sub = exclude_capped(sub)

    r1 = (sub[sub['method'] == 'richardson_1'][keys + ['stability']]
          .rename(columns={'stability': 'r1_stability'}))
    best_other = (sub[sub['method'] != 'richardson_1']
                  .groupby(keys)['stability'].max()
                  .reset_index()
                  .rename(columns={'stability': 'best_other_stability'}))

    merged = r1.merge(best_other, on=keys)
    merged['r1_losing_margin'] = merged['best_other_stability'] - merged['r1_stability']
    merged['r1_is_losing']     = (merged['r1_losing_margin'] > 0.05).astype(int)

    feat_avg = (df_feat.drop(columns=['seed'])
                       .groupby(keys)
                       .mean()
                       .reset_index())

    base = merged.merge(feat_avg, on=keys, how='left')
    base['target_g'] = g

    corr_rows = []

    # ── Global correlations ────────────────────────────────────────────────────
    for feat in FEATURE_COLS:
        mask = base[feat].notna() & base['r1_losing_margin'].notna()
        n    = int(mask.sum())
        if n < 10:
            continue
        r_m, p_m = spearmanr(base.loc[mask, feat], base.loc[mask, 'r1_losing_margin'])
        r_l, p_l = spearmanr(base.loc[mask, feat], base.loc[mask, 'r1_is_losing'])
        corr_rows.append({
            'regime':              'ALL',
            'target_g':            g,
            'feature':             feat,
            'spearman_vs_margin':  round(float(r_m), 4),
            'p_vs_margin':         round(float(p_m), 4),
            'spearman_vs_losing':  round(float(r_l), 4),
            'p_vs_losing':         round(float(p_l), 4),
            'n_obs':               n,
        })

    # ── Per-regime correlations ────────────────────────────────────────────────
    for regime in sorted(df_agg['regime'].unique()):
        rsub = base[base['regime'] == regime]
        for feat in FEATURE_COLS:
            mask = rsub[feat].notna() & rsub['r1_losing_margin'].notna()
            n    = int(mask.sum())
            if n < MIN_OBS:
                corr_rows.append({
                    'regime': regime, 'target_g': g, 'feature': feat,
                    'spearman_vs_margin': float('nan'), 'p_vs_margin': float('nan'),
                    'spearman_vs_losing': float('nan'), 'p_vs_losing': float('nan'),
                    'n_obs': n,
                })
                continue
            r_m, p_m = spearmanr(rsub.loc[mask, feat], rsub.loc[mask, 'r1_losing_margin'])
            corr_rows.append({
                'regime': regime, 'target_g': g, 'feature': feat,
                'spearman_vs_margin': round(float(r_m), 4),
                'p_vs_margin': round(float(p_m), 4),
                'spearman_vs_losing': float('nan'), 'p_vs_losing': float('nan'),
                'n_obs': n,
            })

    df_corr = pd.DataFrame(corr_rows)
    p = os.path.join(out_dir, f'phase2_correlations{suffix}.csv')
    df_corr.to_csv(p, index=False)
    print(f'  Saved: {p}  ({len(df_corr)} rows)')
    return df_corr, base


# =============================================================================
# 4.  SIMPLE RULE TESTING
# =============================================================================

def test_simple_rules(merged_full: pd.DataFrame,
                      df_feat:     pd.DataFrame,
                      df_agg:      pd.DataFrame,
                      g:           float,
                      out_dir:     str,
                      suffix:      str = '') -> pd.DataFrame:
    """
    For each candidate rule (feature, operator, threshold, alternative),
    compute precision, recall, and mean stability gain at stratum g.
    Capped cells are excluded.

    Precision = P(Richardson is losing | rule fires)
    Recall    = P(rule fires | Richardson is losing)
    Mean gain = mean(stability(alt) - stability(richardson_1)) when rule fires
    """
    keys = ['regime', 'obs_idx', 'noise']
    feat_avg = (df_feat.drop(columns=['seed'])
                        .groupby(keys)
                        .mean()
                        .reset_index())

    sub = exclude_capped(df_agg[(df_agg['target_g'] == g)
                                & (df_agg['method'].isin(RANK_POOL))])
    pivot = (sub.pivot_table(index=keys, columns='method', values='stability')
                .reset_index())
    pivot.columns.name = None

    ev = pivot.merge(feat_avg, on=keys, how='left')

    rule_rows = []
    for feat, op, thresh, alt in CANDIDATE_RULES:
        if feat not in ev.columns:
            continue
        if alt not in ev.columns or 'richardson_1' not in ev.columns:
            continue

        valid = (ev[feat].notna() & ev['richardson_1'].notna() & ev[alt].notna())
        sub_ev = ev[valid].copy()
        if len(sub_ev) < 5:
            continue

        fires        = sub_ev[feat] < thresh if op == '<' else sub_ev[feat] > thresh
        r1_is_losing = sub_ev[alt] > sub_ev['richardson_1']

        n_fire   = int(fires.sum())
        n_total  = len(sub_ev)
        n_losing = int(r1_is_losing.sum())

        precision = (float((fires & r1_is_losing).sum() / n_fire)
                     if n_fire > 0 else float('nan'))
        recall    = (float((fires & r1_is_losing).sum() / n_losing)
                     if n_losing > 0 else float('nan'))
        gain      = (float((sub_ev.loc[fires, alt]
                            - sub_ev.loc[fires, 'richardson_1']).mean())
                     if n_fire > 0 else float('nan'))

        rule_rows.append({
            'target_g':    g,
            'feature':     feat,
            'operator':    op,
            'threshold':   thresh,
            'alternative': alt,
            'n_fire':      n_fire,
            'n_total':     n_total,
            'fire_rate':   round(n_fire / n_total, 3),
            'precision':   round(precision, 3) if math.isfinite(precision) else float('nan'),
            'recall':      round(recall,    3) if math.isfinite(recall)    else float('nan'),
            'mean_gain':   round(gain,      4) if math.isfinite(gain)      else float('nan'),
        })

    cols = ['target_g', 'feature', 'operator', 'threshold', 'alternative', 'n_fire',
            'n_total', 'fire_rate', 'precision', 'recall', 'mean_gain']
    df_rules = (pd.DataFrame(rule_rows, columns=cols)
                  .sort_values('precision', ascending=False)
                  .reset_index(drop=True))
    p = os.path.join(out_dir, f'phase2_rules{suffix}.csv')
    df_rules.to_csv(p, index=False)
    print(f'  Saved: {p}  ({len(df_rules)} rows)')
    return df_rules


# =============================================================================
# 5.  FIGURES
# =============================================================================

def _save(fig, path: str):
    fig.savefig(path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')


def _nearest_idx(lst: list, val: float) -> int:
    """Return index of element in lst nearest to val (handles float precision)."""
    return min(range(len(lst)), key=lambda i: abs(lst[i] - val))


# ── Figure 1 — Phase diagram ───────────────────────────────────────────────────

def fig_p2_01_phase_diagram(df_pd:   pd.DataFrame,
                             out_dir: str) -> str:
    """2-D (obs_idx x noise) heat-map of Richardson's rank, per key regime."""
    key_regimes = [
        'rational_decay', 'osc_exp', 'staircase',
        'broken_power_law', 'single_exp', 'power_law',
    ]
    obs_vals   = sorted(df_pd['obs_idx'].unique())
    noise_vals = sorted(df_pd['noise'].unique())
    n_methods  = len(RANK_POOL)
    g          = float(df_pd['target_g'].iloc[0]) if len(df_pd) else float('nan')

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes_flat  = axes.flatten()

    def _draw(ax, df_sub, title):
        mat = np.full((len(noise_vals), len(obs_vals)), np.nan)
        for _, row in df_sub.iterrows():
            ni = _nearest_idx(noise_vals, row['noise'])
            oi = _nearest_idx(obs_vals,   row['obs_idx'])
            mat[ni, oi] = row['richardson_rank']
        im = ax.imshow(mat, cmap='RdYlGn_r', aspect='auto',
                       vmin=1, vmax=n_methods, interpolation='nearest')
        ax.set_xticks(range(len(obs_vals)))
        ax.set_xticklabels(obs_vals, fontsize=7, rotation=45)
        ax.set_yticks(range(len(noise_vals)))
        ax.set_yticklabels([f'{v:.3f}' for v in noise_vals], fontsize=7)
        ax.set_xlabel('obs_idx', fontsize=8)
        ax.set_ylabel('noise σ', fontsize=8)
        ax.set_title(title, fontsize=9, fontweight='bold')
        for ni in range(len(noise_vals)):
            for oi in range(len(obs_vals)):
                v = mat[ni, oi]
                if np.isfinite(v):
                    ax.text(oi, ni, f'{int(v)}', ha='center', va='center',
                            fontsize=7,
                            color='white' if v > 4 else 'black')
        return im

    # Aggregated across all regimes: capped cells excluded (pooled statistic)
    agg_df = (exclude_capped(df_pd).groupby(['obs_idx', 'noise'])
                   ['richardson_rank'].mean()
                   .reset_index())
    im = _draw(axes_flat[0], agg_df, 'ALL REGIMES (mean rank, capped excluded)')
    plt.colorbar(im, ax=axes_flat[0], label='Richardson rank', shrink=0.8)

    for k, regime in enumerate(key_regimes):
        ax  = axes_flat[k + 1]
        sub = df_pd[df_pd['regime'] == regime]
        cap = ' [CAP]' if (len(sub) and sub['capped'].max() == 1) else ''
        im  = _draw(ax, sub, regime.replace('_', '\n') + cap)
        plt.colorbar(im, ax=ax, label='Rank', shrink=0.8)

    axes_flat[7].axis('off')
    axes_flat[7].text(0.5, 0.5,
        f'Richardson Rank\n(1 = best of {n_methods}, {n_methods} = worst;\n'
        'oracle comparator excluded)\n\n'
        'Green = Richardson wins\n'
        'Red   = Richardson fails\n\n'
        'Rows    = noise level\n'
        'Columns = obs_idx\n'
        '[CAP] = some cells hit the horizon cap',
        ha='center', va='center', fontsize=10,
        transform=axes_flat[7].transAxes)

    fig.suptitle(
        f'Figure P2-1 — Richardson Rank Phase Diagram  '
        f'(obs_idx × noise)\n'
        f'Gap stratum g = {g:g}',
        fontsize=11, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(out_dir, 'figure_p2_01_phase_diagram.png')
    _save(fig, path)
    return path


# ── Figure 2 — Crossover by obs_idx ───────────────────────────────────────────

def fig_p2_02_crossover(df_pd:   pd.DataFrame,
                         out_dir: str) -> str:
    """Richardson rank vs obs_idx for each regime (sigma ≈ 0)."""
    sub = df_pd[df_pd['noise'] < 1e-9].copy()
    if sub.empty:
        min_noise = df_pd['noise'].min()
        sub = df_pd[df_pd['noise'] == min_noise].copy()

    obs_vals  = sorted(sub['obs_idx'].unique())
    ever_wins = {r: bool((grp['richardson_rank'] == 1).any())
                 for r, grp in sub.groupby('regime')}

    fig, ax = plt.subplots(figsize=(13, 7))
    for regime in sorted(sub['regime'].unique()):
        rsub = sub[sub['regime'] == regime].sort_values('obs_idx')
        if rsub.empty:
            continue
        wins   = ever_wins.get(regime, False)
        colour = '#2e7d32' if wins else '#c62828'
        lw     = 2.0       if wins else 1.0
        ax.plot(rsub['obs_idx'], rsub['richardson_rank'],
                lw=lw, color=colour, alpha=0.75)
        last = rsub.iloc[-1]
        ax.text(last['obs_idx'] + 1, last['richardson_rank'],
                regime[:8], fontsize=6, va='center', color=colour)

    ax.axhline(1, color='#555', lw=0.8, ls='--', alpha=0.5,
               label='Rank 1 (wins)')
    ax.axhline(3, color='#888', lw=0.8, ls=':', alpha=0.5,
               label='Rank 3 (top-tier)')
    ax.invert_yaxis()
    ax.set_xlabel('obs_idx  (observation depth)', fontsize=10)
    ax.set_ylabel(f'Richardson rank  (1 = best of {len(RANK_POOL)})', fontsize=10)
    ax.set_title(
        'Figure P2-2 — Richardson Rank vs Observation Depth\n'
        'Green = eventually wins; Red = never wins  (lowest noise level)',
        fontsize=10, fontweight='bold')
    ax.set_xticks(obs_vals)
    ax.set_xticklabels(obs_vals, fontsize=8)
    ax.set_ylim(len(RANK_POOL) + 0.5, 0.5)
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = os.path.join(out_dir, 'figure_p2_02_crossover.png')
    _save(fig, path)
    return path


# ── Figure 3 — Per-regime correlation heatmap ─────────────────────────────────

def fig_p2_03_correlations(df_corr: pd.DataFrame,
                            out_dir: str) -> str:
    """Spearman r per (feature x regime), excluding ALL and NaN-only regimes."""
    sub = df_corr[df_corr['regime'] != 'ALL'].copy()
    sub = sub[sub['spearman_vs_margin'].notna()]
    if sub.empty:
        print('  Fig P2-3 skipped: no per-regime correlations with enough data.')
        return ''

    pivot   = sub.pivot_table(index='feature', columns='regime',
                               values='spearman_vs_margin')
    pivot   = pivot.reindex(index=FEATURE_COLS)
    mat     = pivot.values.astype(float)
    regimes = list(pivot.columns)

    vmax = min(1.0, max(0.3, float(np.nanmax(np.abs(mat)))))
    fig, ax = plt.subplots(figsize=(max(10, len(regimes) * 0.85), 5))
    im = ax.imshow(mat, cmap='RdBu_r', aspect='auto',
                   vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(regimes)))
    ax.set_xticklabels([r.replace('_', '\n') for r in regimes],
                       fontsize=7.5, ha='center')
    ax.set_yticks(range(len(FEATURE_COLS)))
    ax.set_yticklabels(FEATURE_COLS, fontsize=9)
    for i in range(len(FEATURE_COLS)):
        for j in range(len(regimes)):
            v = mat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                        fontsize=7,
                        color='white' if abs(v) > 0.5 * vmax else '#333')
    plt.colorbar(im, ax=ax, shrink=0.7,
                 label='Spearman r  (vs Richardson losing-margin)')
    ax.set_title(
        'Figure P2-3 — Feature × Regime Correlation Heatmap\n'
        'Blue = feature↑ → Richardson loses more;  '
        'Red = feature↑ → Richardson wins more',
        fontsize=10, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(out_dir, 'figure_p2_03_correlations.png')
    _save(fig, path)
    return path


# ── Figure 4 — Global feature correlations bar chart ─────────────────────────

def fig_p2_04_global_correlations(df_corr: pd.DataFrame,
                                   out_dir: str) -> str:
    """Horizontal bar chart of global Spearman r values with significance flags."""
    sub = df_corr[df_corr['regime'] == 'ALL'].copy().dropna(
        subset=['spearman_vs_margin'])
    if sub.empty:
        print('  Fig P2-4 skipped.')
        return ''

    sub = sub.sort_values('spearman_vs_margin', key=abs, ascending=True)
    colours = ['#c62828' if r > 0 else '#1565c0'
               for r in sub['spearman_vs_margin']]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(range(len(sub)), sub['spearman_vs_margin'],
            color=colours, edgecolor='white', lw=0.5, height=0.6)
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels(sub['feature'], fontsize=10)
    ax.axvline(0, color='black', lw=0.7)
    ax.axvline( 0.30, color='#43a047', lw=1.2, ls='--', alpha=0.8,
                label='|r| = 0.30  (candidate threshold)')
    ax.axvline(-0.30, color='#43a047', lw=1.2, ls='--', alpha=0.8)

    for i, (_, row) in enumerate(sub.iterrows()):
        r = row['spearman_vs_margin']
        p = row['p_vs_margin']
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else
              ('*' if p < 0.05 else ''))
        offset = 0.01 if r >= 0 else -0.01
        ha     = 'left' if r >= 0 else 'right'
        ax.text(r + offset, i,
                f'{r:+.3f} {sig}', va='center', ha=ha,
                fontsize=8.5, fontweight='bold' if abs(r) >= 0.3 else 'normal')

    ax.set_xlabel('Spearman r  (vs Richardson losing-margin)', fontsize=10)
    ax.set_title(
        'Figure P2-4 — Global Feature Correlations with Richardson Failure\n'
        'Red = feature↑ → Richardson loses  |  '
        'Blue = feature↑ → Richardson wins',
        fontsize=10, fontweight='bold')
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = os.path.join(out_dir, 'figure_p2_04_global_correlations.png')
    _save(fig, path)
    return path


# ── Figure 5 — Rule precision vs recall ───────────────────────────────────────

def fig_p2_05_rules(df_rules: pd.DataFrame,
                    out_dir:  str) -> str:
    """Scatter of all threshold rules: precision (y) vs recall (x)."""
    valid = df_rules.dropna(subset=['precision', 'recall', 'mean_gain']).copy()
    if valid.empty:
        print('  Fig P2-5 skipped.')
        return ''

    colours = [METHOD_COLOURS.get(m, '#999') for m in valid['alternative']]
    sizes   = np.clip(valid['n_fire'] * 2, 20, 400)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(valid['recall'], valid['precision'],
               c=colours, s=sizes, alpha=0.8,
               edgecolors='black', linewidths=0.4)

    for _, row in valid.iterrows():
        gain_str = f'{row["mean_gain"]:+.2f}'
        ax.annotate(
            f"{row['feature'][:9]}{row['operator']}{row['threshold']}\n"
            f"→ {row['alternative'][:10]}  ({gain_str})",
            (row['recall'], row['precision']),
            textcoords='offset points', xytext=(5, 3),
            fontsize=5.5, alpha=0.85)

    ax.axvline(0.5, color='grey', lw=0.7, ls='--', alpha=0.4)
    ax.axhline(0.4, color='grey', lw=0.7, ls='--', alpha=0.4)
    ax.set_xlabel('Recall   (fraction of Richardson failures caught)', fontsize=10)
    ax.set_ylabel('Precision  (when rule fires, Richardson really is losing)',
                  fontsize=10)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(
        'Figure P2-5 — Threshold Rule Precision vs Recall\n'
        'Bubble size ∝ n_fire; gain annotation = mean stability gain when rule fires',
        fontsize=10, fontweight='bold')

    seen, handles = set(), []
    for m, c in METHOD_COLOURS.items():
        if m in valid['alternative'].values and m not in seen:
            handles.append(plt.scatter([], [], color=c, s=60, label=m))
            seen.add(m)
    ax.legend(handles=handles, fontsize=8, loc='lower right')

    fig.tight_layout()
    path = os.path.join(out_dir, 'figure_p2_05_rules.png')
    _save(fig, path)
    return path


# =============================================================================
# MASTER CALL
# =============================================================================

def make_all_figures(df_pd:    pd.DataFrame,
                     df_corr:  pd.DataFrame,
                     df_rules: pd.DataFrame,
                     g:        float,
                     out_dir:  str) -> list:
    print('\n  Generating figures ...')
    paths = [
        fig_p2_01_phase_diagram(df_pd, out_dir),
        fig_p2_02_crossover(df_pd, out_dir),
        fig_p2_03_correlations(df_corr, out_dir),
        fig_p2_04_global_correlations(df_corr, out_dir),
        fig_p2_05_rules(df_rules, out_dir),
    ]
    return [p for p in paths if p]
