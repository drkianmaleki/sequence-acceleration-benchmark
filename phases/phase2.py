"""
phase2.py
=========
Phase 2 — Richardson Failure Condition Mapping.

Four components:
  1. Sweep evaluation   — run a reduced 9-method set across (obs_idx, noise,
                          future_idx) grids to map where Richardson's rank falls.
  2. Feature extraction — compute six trajectory features for every sequence,
                          using a dynamic L0 baseline that prevents NaN on
                          near-converged windows.
  3. Correlation        — Spearman correlation between features and Richardson
                          relative rank; minimum 15 observations required per
                          per-regime correlation.
  4. Rule testing       — evaluate simple and compound threshold rules.

Changes from v1
---------------
  * Dynamic L0 in feature extraction: L0 = min(L_inf, 0.5*min(window))
    prevents NaN on near-converged sequences (fixes per-regime correlations).
  * Additional candidate rules: lower R2 thresholds and compound selectors.
  * Per-regime correlation guard raised from n>=5 to n>=15.
  * Noise-value matching in phase diagram uses nearest-value lookup.
  * Crossover figure uses near-zero comparison for sigma==0.

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
Date   : 2026-05-24
"""

import os, math, warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.optimize import curve_fit
from typing import List, Dict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

warnings.filterwarnings('ignore')

import config as CFG_MOD
from accelerators import METHODS
from generators   import GENERATORS, REGIME_NAMES, TRUTH

# ── Reduced method set ─────────────────────────────────────────────────────────
PHASE2_METHODS = [
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

METHOD_COLOURS = {
    'current_value':  '#888888',
    'richardson_1':   '#f4a261',
    'richardson_a10': '#e76f51',
    'single_exp_fit': '#2196f3',
    'rational_fit':   '#1565c0',
    'pade_22':        '#e91e63',
    'log_linear':     '#00897b',
    'weniger_d2':     '#9c27b0',
    'anderson_1':     '#795548',
}

FIG_DPI = 150

# ── Config builder ─────────────────────────────────────────────────────────────
def _cfg(future_idx: int) -> dict:
    return {
        'L_inf':          CFG_MOD.L_INF,
        'ridge':          CFG_MOD.RIDGE,
        'min_valid':      CFG_MOD.MIN_VALID,
        'max_valid':      CFG_MOD.MAX_VALID,
        'denom_tol':      CFG_MOD.DENOM_TOL,
        'win_shifts':     CFG_MOD.WIN_SHIFTS,
        'perturb_trials': CFG_MOD.PERTURB_TRIALS,
        'perturb_scale':  CFG_MOD.PERTURB_SCALE,
        'W_CAT':          CFG_MOD.W_CAT,
        'W_BEATS':        CFG_MOD.W_BEATS,
    }


def _valid(v: float, cfg: dict) -> bool:
    return bool(math.isfinite(v) and
                cfg['min_valid'] <= v <= cfg['max_valid'])


def _stab(vr: float, cr: float, br: float) -> float:
    return vr - CFG_MOD.W_CAT * cr + CFG_MOD.W_BEATS * br


# ── Feature extraction (local, with dynamic L0 fix) ───────────────────────────

def _extract_features(seq: list, indices: list, L_inf: float) -> dict:
    """
    Extract six scalar features from an observable window.

    Dynamic L0 fix: uses L0 = min(L_inf, 0.5 * min(window)) so that
    shifted = s - L0 > 0 even when the sequence is near its limit.
    This prevents NaN on near-converged windows.

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
    L0       = max(0.0, min(L_inf, win_min * 0.5))
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
              future_list:  List[int],
              n_seeds:      int,
              window_len:   int,
              out_dir:      str,
              verbose:      bool = True):
    """
    Evaluate PHASE2_METHODS across all (obs_idx, noise, future_idx) grids.
    Returns (df_agg, df_feat) DataFrames.
    """
    os.makedirs(out_dir, exist_ok=True)

    n_total = (len(obs_idx_list) * len(noise_list)
               * n_seeds * len(REGIME_NAMES))
    done    = 0

    sweep_records   = []
    feature_records = []

    n_arr_max = np.arange(max(future_list) + 200, dtype=float)

    for obs_idx in obs_idx_list:
        wl = min(window_len, obs_idx)

        for sigma in noise_list:
            for seed in range(n_seeds):
                rng = np.random.RandomState(
                    seed * 137 + int(sigma * 1e6) % 9973 + obs_idx * 7)

                for regime in REGIME_NAMES:
                    gen      = GENERATORS[regime]
                    truth    = TRUTH[regime]
                    seq_full = gen(n_arr_max, rng, sigma)

                    # Observation window
                    w_start  = max(0, obs_idx - wl + 1)
                    seq_win  = list(seq_full[w_start : obs_idx + 1])
                    idx_win  = list(range(w_start, obs_idx + 1))
                    curr_val = float(seq_full[obs_idx])

                    # Feature extraction with dynamic L0 fix
                    feats    = _extract_features(seq_win, idx_win,
                                                 CFG_MOD.L_INF)
                    feat_row = {
                        'regime':  regime,
                        'obs_idx': obs_idx,
                        'noise':   sigma,
                        'seed':    seed,
                    }
                    feat_row.update(feats)
                    feature_records.append(feat_row)

                    # Method evaluations
                    for fid in future_list:
                        true_val = float(truth(fid))
                        curr_err = abs(curr_val - true_val)
                        cfg      = _cfg(fid)

                        for method in PHASE2_METHODS:
                            fn = METHODS[method]
                            try:
                                est = fn(seq_win, idx_win, float(fid), cfg)
                            except Exception:
                                est = float('nan')

                            valid = _valid(est, cfg)
                            err   = abs(est - true_val) if valid else float('nan')
                            cat   = (not valid) or (
                                valid and curr_err > 1e-12
                                and err > CFG_MOD.CAT_MULT * curr_err)
                            beats = (valid and curr_err > 1e-12
                                     and err < curr_err)
                            impv  = ((curr_err / err)
                                     if (valid and err > 1e-12)
                                     else (1.0 if valid else float('nan')))

                            sweep_records.append({
                                'method':       method,
                                'regime':       regime,
                                'obs_idx':      obs_idx,
                                'noise':        sigma,
                                'future_idx':   fid,
                                'seed':         seed,
                                'valid':        int(valid),
                                'catastrophic': int(cat),
                                'beats':        int(beats),
                                'error':        err if valid else float('nan'),
                                'impv':         (impv if math.isfinite(impv)
                                                 else float('nan')),
                            })

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
    for keys, grp in df_raw.groupby(
            ['method', 'regime', 'obs_idx', 'noise', 'future_idx']):
        method, regime, obs_idx, sigma, fid = keys
        vr = grp['valid'].mean()
        cr = grp['catastrophic'].mean()
        br = grp['beats'].mean()
        errors = grp['error'].dropna().tolist()
        me = float(np.median(errors)) if errors else float('nan')
        sc = _stab(vr, cr, br)
        agg.append({
            'method':     method,   'regime':     regime,
            'obs_idx':    obs_idx,  'noise':      sigma,
            'future_idx': fid,      'valid_rate': round(vr, 4),
            'cat_rate':   round(cr, 4),
            'beats_rate': round(br, 4),
            'med_error':  me,       'stability':  round(sc, 4),
        })

    df_agg = pd.DataFrame(agg)

    # ── Save ───────────────────────────────────────────────────────────────────
    p = os.path.join(out_dir, 'phase2_sweep_aggregated.csv')
    df_agg.to_csv(p, index=False)
    print(f'  Saved: {p}  ({len(df_agg)} rows)')

    p = os.path.join(out_dir, 'phase2_features.csv')
    df_feat.to_csv(p, index=False)
    print(f'  Saved: {p}  ({len(df_feat)} rows)')

    return df_agg, df_feat


# =============================================================================
# 2.  PHASE DIAGRAMS
# =============================================================================

def build_phase_diagrams(df_agg:     pd.DataFrame,
                          default_fid: int,
                          out_dir:    str) -> pd.DataFrame:
    """
    For each (regime, obs_idx, noise) at the default horizon, determine
    Richardson's rank among the 9 methods and the best alternative.
    """
    rows = []
    sub  = df_agg[df_agg['future_idx'] == default_fid]

    for (regime, obs_idx, sigma), grp in sub.groupby(
            ['regime', 'obs_idx', 'noise']):
        ranked = (grp.sort_values('stability', ascending=False)
                     .reset_index(drop=True))
        ranked['rank'] = ranked.index + 1

        r1 = ranked[ranked['method'] == 'richardson_1']
        best = ranked.iloc[0]

        r1_stab  = float(r1['stability'].values[0]) if not r1.empty else float('nan')
        r1_rank  = int(r1['rank'].values[0])         if not r1.empty else 99
        best_sc  = float(best['stability'])
        best_m   = str(best['method'])
        margin   = round(best_sc - r1_stab, 4)

        rows.append({
            'regime':                 regime,
            'obs_idx':                obs_idx,
            'noise':                  sigma,
            'future_idx':             default_fid,
            'richardson_stability':   round(r1_stab, 4),
            'richardson_rank':        r1_rank,
            'best_method':            best_m,
            'best_stability':         round(best_sc, 4),
            'stability_margin':       margin,
            'richardson_wins':        int(r1_rank == 1),
        })

    df_pd = pd.DataFrame(rows)
    p = os.path.join(out_dir, f'phase2_phase_diagram_n{default_fid}.csv')
    df_pd.to_csv(p, index=False)
    print(f'  Saved: {p}  ({len(df_pd)} rows)')
    return df_pd


# =============================================================================
# 3.  CORRELATION ANALYSIS
# =============================================================================

def run_correlation_analysis(df_feat:     pd.DataFrame,
                              df_agg:      pd.DataFrame,
                              default_fid: int,
                              out_dir:     str):
    """
    Spearman correlations between trajectory features and Richardson's
    losing margin.

    Global (ALL regimes) and per-regime.
    Per-regime requires >= 15 finite observations (raised from 5 to avoid
    spurious correlations from tiny samples).
    """
    MIN_OBS = 15   # minimum valid observations for per-regime correlation

    # Richardson stability per grid point
    r1 = (df_agg[df_agg['method'] == 'richardson_1']
          [['regime', 'obs_idx', 'noise', 'future_idx', 'stability']]
          .rename(columns={'stability': 'r1_stability'}))

    # Best non-Richardson stability per grid point
    best_other = (df_agg[df_agg['method'] != 'richardson_1']
                  .groupby(['regime', 'obs_idx', 'noise', 'future_idx'])
                  ['stability'].max()
                  .reset_index()
                  .rename(columns={'stability': 'best_other_stability'}))

    merged = r1.merge(best_other,
                      on=['regime', 'obs_idx', 'noise', 'future_idx'])
    merged['r1_losing_margin'] = (merged['best_other_stability']
                                   - merged['r1_stability'])
    merged['r1_is_losing']     = (merged['r1_losing_margin'] > 0.05
                                   ).astype(int)

    # Average features across seeds (one value per grid point)
    feat_avg = (df_feat.drop(columns=['seed'])
                       .groupby(['regime', 'obs_idx', 'noise'])
                       .mean()
                       .reset_index())

    base = merged[merged['future_idx'] == default_fid].merge(
        feat_avg, on=['regime', 'obs_idx', 'noise'], how='left')

    corr_rows = []

    # ── Global correlations ────────────────────────────────────────────────────
    for feat in FEATURE_COLS:
        mask = base[feat].notna() & base['r1_losing_margin'].notna()
        n    = int(mask.sum())
        if n < 10:
            continue
        r_m, p_m = spearmanr(base.loc[mask, feat],
                               base.loc[mask, 'r1_losing_margin'])
        r_l, p_l = spearmanr(base.loc[mask, feat],
                               base.loc[mask, 'r1_is_losing'])
        corr_rows.append({
            'regime':              'ALL',
            'feature':             feat,
            'spearman_vs_margin':  round(float(r_m), 4),
            'p_vs_margin':         round(float(p_m), 4),
            'spearman_vs_losing':  round(float(r_l), 4),
            'p_vs_losing':         round(float(p_l), 4),
            'n_obs':               n,
        })

    # ── Per-regime correlations ────────────────────────────────────────────────
    for regime in REGIME_NAMES:
        sub = base[base['regime'] == regime]
        for feat in FEATURE_COLS:
            mask = sub[feat].notna() & sub['r1_losing_margin'].notna()
            n    = int(mask.sum())
            if n < MIN_OBS:
                # Record NaN row so all regimes appear in output
                corr_rows.append({
                    'regime':              regime,
                    'feature':             feat,
                    'spearman_vs_margin':  float('nan'),
                    'p_vs_margin':         float('nan'),
                    'spearman_vs_losing':  float('nan'),
                    'p_vs_losing':         float('nan'),
                    'n_obs':               n,
                })
                continue
            r_m, p_m = spearmanr(sub.loc[mask, feat],
                                   sub.loc[mask, 'r1_losing_margin'])
            corr_rows.append({
                'regime':              regime,
                'feature':             feat,
                'spearman_vs_margin':  round(float(r_m), 4),
                'p_vs_margin':         round(float(p_m), 4),
                'spearman_vs_losing':  float('nan'),
                'p_vs_losing':         float('nan'),
                'n_obs':               n,
            })

    df_corr = pd.DataFrame(corr_rows)
    p = os.path.join(out_dir, 'phase2_correlations.csv')
    df_corr.to_csv(p, index=False)
    print(f'  Saved: {p}  ({len(df_corr)} rows)')
    return df_corr, base


# =============================================================================
# 4.  SIMPLE RULE TESTING
# =============================================================================

def test_simple_rules(merged_full: pd.DataFrame,
                      df_feat:     pd.DataFrame,
                      df_agg:      pd.DataFrame,
                      default_fid: int,
                      out_dir:     str) -> pd.DataFrame:
    """
    For each candidate rule (feature, operator, threshold, alternative),
    compute precision, recall, and mean stability gain.

    Precision = P(Richardson is losing | rule fires)
    Recall    = P(rule fires | Richardson is losing)
    Mean gain = mean(stability(alt) - stability(richardson_1)) when rule fires
    """
    # Per grid-point: features + stability of each method
    feat_avg = (df_feat.drop(columns=['seed'])
                        .groupby(['regime', 'obs_idx', 'noise'])
                        .mean()
                        .reset_index())

    pivot = (df_agg[df_agg['future_idx'] == default_fid]
             .pivot_table(index=['regime', 'obs_idx', 'noise'],
                          columns='method',
                          values='stability')
             .reset_index())
    pivot.columns.name = None

    ev = pivot.merge(feat_avg, on=['regime', 'obs_idx', 'noise'], how='left')

    rule_rows = []
    for feat, op, thresh, alt in CANDIDATE_RULES:
        if feat not in ev.columns:
            continue
        if alt not in ev.columns or 'richardson_1' not in ev.columns:
            continue

        valid = (ev[feat].notna()
                 & ev['richardson_1'].notna()
                 & ev[alt].notna())
        sub = ev[valid].copy()
        if len(sub) < 5:
            continue

        fires        = sub[feat] < thresh if op == '<' else sub[feat] > thresh
        r1_is_losing = sub[alt] > sub['richardson_1']

        n_fire   = int(fires.sum())
        n_total  = len(sub)
        n_losing = int(r1_is_losing.sum())

        precision = (float((fires & r1_is_losing).sum() / n_fire)
                     if n_fire > 0 else float('nan'))
        recall    = (float((fires & r1_is_losing).sum() / n_losing)
                     if n_losing > 0 else float('nan'))
        gain      = (float((sub.loc[fires, alt]
                            - sub.loc[fires, 'richardson_1']).mean())
                     if n_fire > 0 else float('nan'))

        rule_rows.append({
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

    df_rules = (pd.DataFrame(rule_rows)
                  .sort_values('precision', ascending=False)
                  .reset_index(drop=True))
    p = os.path.join(out_dir, 'phase2_rules.csv')
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
    n_methods  = len(PHASE2_METHODS)

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

    # Aggregated across all regimes
    agg_df = (df_pd.groupby(['obs_idx', 'noise'])
                   ['richardson_rank'].mean()
                   .reset_index())
    im = _draw(axes_flat[0], agg_df, 'ALL REGIMES (mean rank)')
    plt.colorbar(im, ax=axes_flat[0], label='Richardson rank', shrink=0.8)

    for k, regime in enumerate(key_regimes):
        ax  = axes_flat[k + 1]
        sub = df_pd[df_pd['regime'] == regime]
        im  = _draw(ax, sub, regime.replace('_', '\n'))
        plt.colorbar(im, ax=ax, label='Rank', shrink=0.8)

    axes_flat[7].axis('off')
    axes_flat[7].text(0.5, 0.5,
        'Richardson Rank\n(1 = best of 9, 9 = worst)\n\n'
        'Green = Richardson wins\n'
        'Red   = Richardson fails\n\n'
        'Rows    = noise level\n'
        'Columns = obs_idx',
        ha='center', va='center', fontsize=10,
        transform=axes_flat[7].transAxes)

    fig.suptitle(
        f'Figure P2-1 — Richardson Rank Phase Diagram  '
        f'(obs_idx × noise)\n'
        f'Horizon = {df_pd["future_idx"].iloc[0]}',
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
        # Fallback: use lowest noise level
        min_noise = df_pd['noise'].min()
        sub = df_pd[df_pd['noise'] == min_noise].copy()

    obs_vals  = sorted(sub['obs_idx'].unique())
    ever_wins = (sub.groupby('regime')
                    .apply(lambda g: (g['richardson_rank'] == 1).any()))

    fig, ax = plt.subplots(figsize=(13, 7))
    for regime in REGIME_NAMES:
        rsub = sub[sub['regime'] == regime].sort_values('obs_idx')
        if rsub.empty:
            continue
        wins   = ever_wins.get(regime, False)
        colour = '#2e7d32' if wins else '#c62828'
        lw     = 2.0       if wins else 1.0
        ax.plot(rsub['obs_idx'], rsub['richardson_rank'],
                lw=lw, color=colour, alpha=0.75)
        if not rsub.empty:
            last = rsub.iloc[-1]
            ax.text(last['obs_idx'] + 1, last['richardson_rank'],
                    regime[:8], fontsize=6, va='center', color=colour)

    ax.axhline(1, color='#555', lw=0.8, ls='--', alpha=0.5,
               label='Rank 1 (wins)')
    ax.axhline(3, color='#888', lw=0.8, ls=':', alpha=0.5,
               label='Rank 3 (top-tier)')
    ax.invert_yaxis()
    ax.set_xlabel('obs_idx  (observation depth)', fontsize=10)
    ax.set_ylabel('Richardson rank  (1 = best of 9)', fontsize=10)
    ax.set_title(
        'Figure P2-2 — Richardson Rank vs Observation Depth\n'
        'Green = eventually wins; Red = never wins  (lowest noise level)',
        fontsize=10, fontweight='bold')
    ax.set_xticks(obs_vals)
    ax.set_xticklabels(obs_vals, fontsize=8)
    ax.set_ylim(len(PHASE2_METHODS) + 0.5, 0.5)
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

    # Legend for alternatives
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

def make_all_figures(df_pd:      pd.DataFrame,
                     df_corr:    pd.DataFrame,
                     df_rules:   pd.DataFrame,
                     default_fid: int,
                     out_dir:    str) -> list:
    print('\n  Generating figures ...')
    paths = [
        fig_p2_01_phase_diagram(df_pd, out_dir),
        fig_p2_02_crossover(df_pd, out_dir),
        fig_p2_03_correlations(df_corr, out_dir),
        fig_p2_04_global_correlations(df_corr, out_dir),
        fig_p2_05_rules(df_rules, out_dir),
    ]
    return [p for p in paths if p]
