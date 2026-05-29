"""
phase4.py
=========
Phase 4 — Stability Diagnostic Stress Testing.

Two diagnostics are tested as real-time trust signals:

  shift_IQR   : IQR of estimates across small window-start shifts.
                Low = consistent = likely reliable.

  perturb_IQR : IQR of estimates under tiny multiplicative perturbations
                of the window values (2% scale).
                Low = robust = likely reliable.

Four questions answered
-----------------------
Q1. Do diagnostics predict actual error?
    Spearman r between each diagnostic and |estimate - true_val|,
    globally and per method.

Q2. Can rejection rules built on diagnostics catch bad estimates?
    Both shift_IQR and perturb_IQR are thresholded.
    Precision, recall, and mean error saved are reported.

Q3. Does adding a perturb_IQR filter improve the Phase 2 cascade?
    Apply Phase 2 cascade; if chosen method has perturb_IQR > threshold,
    fall back to current_value.

Q4. Does the diagnostic-weighted ensemble outperform fixed methods?
    Weight each method by 1/(perturb_IQR + eps); compare to oracle
    and Phase 2 cascade.

Changes from v1
---------------
  * test_rejection_rules loops over both diagnostics.
  * true_val stored in raw records; ensemble comparison fixed.
  * obs_reliability reports both diagnostics.
  * Figure P4-02 shows two panels (one per diagnostic).
  * Figure P4-03 shows both diagnostics on same axes.
  * SyntaxWarnings fixed (raw strings for backslash sequences).

Author : Kian Maleki
Date   : 2026-05-24
"""

import os, math, warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from typing import List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')

import src.config as CFG_MOD
from src.accelerators import METHODS
from src.generators   import GENERATORS, REGIME_NAMES, TRUTH

# ── Method set ─────────────────────────────────────────────────────────────────
PHASE4_METHODS = [
    'current_value', 'richardson_1', 'richardson_a10',
    'single_exp_fit', 'rational_fit', 'pade_22',
    'log_linear', 'weniger_d2', 'anderson_1',
]

METHOD_COLOURS = {
    'current_value':  '#888888', 'richardson_1':   '#f4a261',
    'richardson_a10': '#e76f51', 'single_exp_fit': '#2196f3',
    'rational_fit':   '#1565c0', 'pade_22':        '#e91e63',
    'log_linear':     '#00897b', 'weniger_d2':     '#9c27b0',
    'anderson_1':     '#795548',
}

DIAGNOSTICS = ['shift_iqr', 'perturb_iqr']
FIG_DPI = 150


# ── Helpers ────────────────────────────────────────────────────────────────────
def _cfg(fid):
    return {
        'L_inf': CFG_MOD.L_INF, 'ridge': CFG_MOD.RIDGE,
        'min_valid': CFG_MOD.MIN_VALID, 'max_valid': CFG_MOD.MAX_VALID,
        'denom_tol': CFG_MOD.DENOM_TOL,
    }

def _valid(v, cfg):
    return bool(math.isfinite(v) and
                cfg['min_valid'] <= v <= cfg['max_valid'])

def _save(fig, path):
    fig.savefig(path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')


# =============================================================================
# 1.  DIAGNOSTIC COMPUTATION HELPERS
# =============================================================================

def _shift_iqr(seq_win, idx_win, future_x, method, cfg, shifts):
    fn, ests = METHODS[method], []
    for sh in shifts:
        s0 = max(0, sh)
        v  = fn(seq_win[s0:], idx_win[s0:], future_x, cfg)
        if _valid(v, cfg):
            ests.append(v)
    if len(ests) < 2:
        return float('nan')
    return float(np.subtract(*np.percentile(ests, [75, 25])))


def _perturb_iqr(seq_win, idx_win, future_x, method, cfg,
                  n_trials, scale, rng):
    fn  = METHODS[method]
    arr = np.asarray(seq_win, dtype=float)
    ests = []
    for _ in range(n_trials):
        v = fn(list(arr * (1.0 + scale * rng.randn(len(arr)))),
               idx_win, future_x, cfg)
        if _valid(v, cfg):
            ests.append(v)
    if len(ests) < 2:
        return float('nan')
    return float(np.subtract(*np.percentile(ests, [75, 25])))


# =============================================================================
# 2.  MAIN EVALUATION LOOP
# =============================================================================

def run_phase4(obs_idx_list, noise_list, future_list, n_seeds,
               window_len, shifts, perturb_trials, perturb_scale,
               out_dir, verbose=True):
    """
    For every (regime, obs_idx, noise, seed, method, horizon):
      central estimate, shift_IQR, perturb_IQR, error, catastrophic flag.
    true_val is stored so ensemble error can be computed later.
    """
    os.makedirs(out_dir, exist_ok=True)
    n_arr   = np.arange(max(future_list) + 200, dtype=float)
    n_total = len(obs_idx_list) * len(noise_list) * n_seeds * len(REGIME_NAMES)
    done    = 0
    records = []

    for obs_idx in obs_idx_list:
        wl = min(window_len, obs_idx)

        for sigma in noise_list:
            for seed in range(n_seeds):
                rng   = np.random.RandomState(
                    seed * 137 + int(sigma * 1e6) % 9973 + obs_idx * 7)
                rng_p = np.random.RandomState(seed * 999 + obs_idx)

                for regime in REGIME_NAMES:
                    seq_full = GENERATORS[regime](n_arr, rng, sigma)
                    truth_fn = TRUTH[regime]

                    w_start  = max(0, obs_idx - wl + 1)
                    seq_win  = list(seq_full[w_start : obs_idx + 1])
                    idx_win  = list(range(w_start, obs_idx + 1))
                    curr_val = float(seq_full[obs_idx])

                    for fid in future_list:
                        true_val = float(truth_fn(fid))
                        curr_err = abs(curr_val - true_val)
                        cfg      = _cfg(fid)

                        for method in PHASE4_METHODS:
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

                            s_iqr = _shift_iqr(seq_win, idx_win,
                                               float(fid), method, cfg, shifts)
                            p_iqr = _perturb_iqr(seq_win, idx_win,
                                                  float(fid), method, cfg,
                                                  perturb_trials,
                                                  perturb_scale, rng_p)

                            records.append({
                                'regime':      regime,
                                'obs_idx':     obs_idx,
                                'noise':       sigma,
                                'seed':        seed,
                                'method':      method,
                                'future_idx':  fid,
                                'true_val':    true_val,
                                'estimate':    est if valid else float('nan'),
                                'error':       err,
                                'valid':       int(valid),
                                'catastrophic':int(cat),
                                'curr_err':    curr_err,
                                'shift_iqr':   s_iqr,
                                'perturb_iqr': p_iqr,
                            })

                done += 1
                if verbose and done % max(1, n_total // 20) == 0:
                    print(f'  [{done:>6}/{n_total}]  {100*done/n_total:5.1f}%'
                          f'  obs={obs_idx}  sigma={sigma:.3f}', flush=True)

    if verbose:
        print(f'  [{n_total}/{n_total}] 100.0%  Done.\n')

    df = pd.DataFrame(records)
    p  = os.path.join(out_dir, 'phase4_raw.csv')
    df.to_csv(p, index=False)
    print(f'  Saved: {p}  ({len(df)} rows)')
    return df


# =============================================================================
# 3.  Q1 — DIAGNOSTIC CORRELATIONS WITH ERROR
# =============================================================================

def diagnostic_correlations(df: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    """
    Spearman r between each diagnostic and |error|.
    Computed globally (all methods) and per method.
    """
    rows = []
    for diag in DIAGNOSTICS:
        # Global
        mask = df[diag].notna() & df['error'].notna()
        n    = int(mask.sum())
        if n >= 10:
            r, p = spearmanr(df.loc[mask, diag], df.loc[mask, 'error'])
            rows.append({'method': 'ALL', 'diagnostic': diag,
                         'spearman_r': round(float(r), 4),
                         'p_value':    round(float(p), 6), 'n': n})

        # Per method
        for method in PHASE4_METHODS:
            sub  = df[df['method'] == method]
            mask = sub[diag].notna() & sub['error'].notna()
            n    = int(mask.sum())
            if n < 10:
                continue
            r, p = spearmanr(sub.loc[mask, diag], sub.loc[mask, 'error'])
            rows.append({'method': method, 'diagnostic': diag,
                         'spearman_r': round(float(r), 4),
                         'p_value':    round(float(p), 6), 'n': n})

    df_corr = pd.DataFrame(rows)
    p = os.path.join(out_dir, 'phase4_diagnostic_correlations.csv')
    df_corr.to_csv(p, index=False)
    print(f'  Saved: {p}  ({len(df_corr)} rows)')
    return df_corr


# =============================================================================
# 4.  Q2 — REJECTION RULE TESTING (both diagnostics)
# =============================================================================

def test_rejection_rules(df: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    """
    For each (diagnostic, method, threshold):
      precision = P(catastrophic | diagnostic > threshold)
      recall    = P(diagnostic > threshold | catastrophic)
      mean_err_saved = mean(curr_err - error) when rule fires;
                       positive = reject saved error vs keeping estimate.
    """
    thresholds = [0.001, 0.002, 0.005, 0.010, 0.020,
                  0.050, 0.100, 0.200, 0.500, 1.000]
    rows = []

    for diag in DIAGNOSTICS:
        for method in PHASE4_METHODS:
            sub   = df[(df['method'] == method) & df[diag].notna()].copy()
            if len(sub) < 10:
                continue
            n_bad = int(sub['catastrophic'].sum())

            for thresh in thresholds:
                fires    = sub[diag] > thresh
                n_fire   = int(fires.sum())
                if n_fire == 0:
                    continue
                true_pos = int((fires & (sub['catastrophic'] == 1)).sum())
                precision = true_pos / n_fire
                recall    = (true_pos / n_bad) if n_bad > 0 else float('nan')

                mask2 = fires & sub['error'].notna()
                saved = (float((sub.loc[mask2, 'curr_err']
                                - sub.loc[mask2, 'error']).mean())
                         if mask2.sum() > 0 else float('nan'))

                rows.append({
                    'diagnostic':     diag,
                    'method':         method,
                    'threshold':      thresh,
                    'n_fire':         n_fire,
                    'n_total':        len(sub),
                    'fire_rate':      round(n_fire / len(sub), 3),
                    'precision':      round(precision, 3),
                    'recall':         round(float(recall), 3)
                                      if math.isfinite(recall) else float('nan'),
                    'mean_err_saved': round(saved, 4)
                                      if math.isfinite(saved) else float('nan'),
                })

    df_rules = (pd.DataFrame(rows)
                  .sort_values(['diagnostic', 'method', 'threshold'])
                  .reset_index(drop=True))
    p = os.path.join(out_dir, 'phase4_rejection_rules.csv')
    df_rules.to_csv(p, index=False)
    print(f'  Saved: {p}  ({len(df_rules)} rows)')
    return df_rules


# =============================================================================
# 5.  Q3 — CASCADE + PERTURB_IQR FILTER
# =============================================================================

def _phase2_cascade_choice(slope, r2):
    if math.isfinite(slope) and slope > -0.10:
        return 'rational_fit'
    if math.isfinite(r2) and r2 < 0.50:
        return 'rational_fit'
    return 'richardson_1'


def cascade_with_filter(df: pd.DataFrame,
                         df_feat_path: str,
                         out_dir: str) -> pd.DataFrame:
    """
    Compare Phase 2 cascade with and without a perturb_IQR rejection filter.
    Filter: if chosen method perturb_IQR > threshold, fall back to
    current_value.
    """
    try:
        df_feat = pd.read_csv(df_feat_path)
    except Exception:
        print('  WARNING: Phase 2 features not found; cascade filter skipped.')
        return pd.DataFrame()

    feat_avg = (df_feat.drop(columns=['seed'])
                        .groupby(['regime', 'obs_idx', 'noise'])
                        .mean()
                        .reset_index())

    fid_default = df['future_idx'].max()
    sub_fid = df[df['future_idx'] == fid_default].copy()
    merged  = sub_fid.merge(feat_avg, on=['regime', 'obs_idx', 'noise'],
                            how='left')

    thresholds = [float('inf'), 1.0, 0.5, 0.2, 0.1, 0.05, 0.02]
    rows = []

    for threshold in thresholds:
        errs = []
        for (regime, obs_idx, sigma, seed), grp in merged.groupby(
                ['regime', 'obs_idx', 'noise', 'seed']):
            frow   = grp.iloc[0]
            slope  = float(frow.get('log_log_slope', float('nan')))
            r2     = float(frow.get('richardson_r2', float('nan')))
            chosen = _phase2_cascade_choice(slope, r2)

            ch_row = grp[grp['method'] == chosen]
            iqr    = (float(ch_row['perturb_iqr'].values[0])
                      if not ch_row.empty else float('nan'))

            if (threshold < float('inf') and math.isfinite(iqr)
                    and iqr > threshold):
                chosen = 'current_value'

            chosen_row = grp[grp['method'] == chosen]
            if chosen_row.empty or not math.isfinite(
                    chosen_row['error'].values[0]):
                chosen_row = grp[grp['method'] == 'current_value']

            err = (float(chosen_row['error'].values[0])
                   if not chosen_row.empty else float('nan'))
            errs.append(err)

        label = ('no_filter' if threshold == float('inf')
                 else f'perturb_iqr>{threshold}')
        rows.append({
            'filter':       label,
            'iqr_threshold': threshold,
            'mean_error':   round(float(np.nanmean(errs)),   6),
            'median_error': round(float(np.nanmedian(errs)), 6),
            'n':            len(errs),
        })

    df_filt = pd.DataFrame(rows)
    p = os.path.join(out_dir, 'phase4_cascade_filter.csv')
    df_filt.to_csv(p, index=False)
    print(f'  Saved: {p}  ({len(df_filt)} rows)')
    return df_filt


# =============================================================================
# 6.  Q4 — DIAGNOSTIC-WEIGHTED ENSEMBLE
# =============================================================================

def ensemble_comparison(df: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    """
    Compare selectors using true_val stored in records.
    Ensemble: weight each method by 1/(perturb_IQR + eps).
    """
    eps         = 0.01
    fid_default = df['future_idx'].max()
    sub         = df[df['future_idx'] == fid_default].copy()

    summary_rows = []

    for sel_name in ['oracle', 'fixed_rational', 'fixed_richardson',
                     'phase2_proxy', 'diag_ensemble']:
        errs = []

        for (regime, obs_idx, sigma, seed), grp in sub.groupby(
                ['regime', 'obs_idx', 'noise', 'seed']):

            grp_idx  = grp.set_index('method')
            true_val = float(grp['true_val'].iloc[0])

            if sel_name == 'oracle':
                best_err = float('inf')
                for m in PHASE4_METHODS:
                    if m not in grp_idx.index:
                        continue
                    e = grp_idx.loc[m, 'error']
                    if math.isfinite(e) and e < best_err:
                        best_err = e
                errs.append(best_err if best_err < float('inf') else float('nan'))

            elif sel_name == 'fixed_rational':
                errs.append(float(grp_idx.loc['rational_fit', 'error'])
                            if 'rational_fit' in grp_idx.index
                            else float('nan'))

            elif sel_name == 'fixed_richardson':
                errs.append(float(grp_idx.loc['richardson_1', 'error'])
                            if 'richardson_1' in grp_idx.index
                            else float('nan'))

            elif sel_name == 'phase2_proxy':
                # Proxy: min of rational_fit and richardson_1
                r1  = (float(grp_idx.loc['richardson_1', 'error'])
                       if 'richardson_1' in grp_idx.index else float('nan'))
                rat = (float(grp_idx.loc['rational_fit', 'error'])
                       if 'rational_fit' in grp_idx.index else float('nan'))
                best = min(x for x in [r1, rat] if math.isfinite(x)) \
                       if any(math.isfinite(x) for x in [r1, rat]) \
                       else float('nan')
                errs.append(best)

            elif sel_name == 'diag_ensemble':
                ests, ws = [], []
                for m in PHASE4_METHODS:
                    if m not in grp_idx.index:
                        continue
                    est_val = grp_idx.loc[m, 'estimate']
                    p_iqr   = grp_idx.loc[m, 'perturb_iqr']
                    if math.isfinite(est_val) and math.isfinite(p_iqr):
                        ests.append(est_val)
                        ws.append(1.0 / (p_iqr + eps))
                if ests:
                    w   = np.array(ws); w /= w.sum()
                    ens = float(np.dot(w, ests))
                    errs.append(abs(ens - true_val))
                else:
                    errs.append(float('nan'))

        vals = pd.Series(errs).dropna()
        summary_rows.append({
            'selector':     sel_name,
            'mean_error':   round(float(vals.mean()),   6),
            'median_error': round(float(vals.median()), 6),
            'n':            len(vals),
        })

    df_sum = pd.DataFrame(summary_rows)
    p = os.path.join(out_dir, 'phase4_ensemble.csv')
    df_sum.to_csv(p, index=False)
    print(f'  Saved: {p}  ({len(df_sum)} rows)')
    return df_sum


# =============================================================================
# 7.  RELIABILITY VS OBS_IDX
# =============================================================================

def obs_reliability(df: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    """
    For each (obs_idx, method, diagnostic): Spearman r(diagnostic, error)
    and precision of high-diagnostic = catastrophic.
    """
    rows = []
    fid  = df['future_idx'].max()
    sub  = df[df['future_idx'] == fid]

    for obs_idx in sorted(sub['obs_idx'].unique()):
        for method in PHASE4_METHODS:
            msub = sub[(sub['obs_idx'] == obs_idx)
                       & (sub['method'] == method)]
            for diag in DIAGNOSTICS:
                mask = msub[diag].notna() & msub['error'].notna()
                n    = int(mask.sum())
                if n < 5:
                    continue
                r, p = spearmanr(msub.loc[mask, diag],
                                  msub.loc[mask, 'error'])
                hi      = msub[diag] > 0.05
                cat     = msub['catastrophic'] == 1
                prec    = (float((hi & cat).sum() / hi.sum())
                           if hi.sum() > 0 else float('nan'))
                rows.append({
                    'obs_idx':     obs_idx,
                    'method':      method,
                    'diagnostic':  diag,
                    'spearman_r':  round(float(r), 4),
                    'p_value':     round(float(p), 6),
                    'n':           n,
                    'hi_iqr_prec': round(prec, 3)
                                   if math.isfinite(prec) else float('nan'),
                })

    df_rel = pd.DataFrame(rows)
    p = os.path.join(out_dir, 'phase4_obs_reliability.csv')
    df_rel.to_csv(p, index=False)
    print(f'  Saved: {p}  ({len(df_rel)} rows)')
    return df_rel


# =============================================================================
# 8.  FIGURES
# =============================================================================

def fig_p4_01_correlations(df_corr: pd.DataFrame, out_dir: str) -> str:
    """Grouped bar chart: Spearman r per (method, diagnostic)."""
    methods_all = ['ALL'] + PHASE4_METHODS
    ds_labels   = {'shift_iqr': 'shift IQR', 'perturb_iqr': 'perturb IQR'}
    ds_colours  = {'shift_iqr': '#1565c0', 'perturb_iqr': '#c62828'}

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(methods_all))
    w = 0.35

    for k, diag in enumerate(DIAGNOSTICS):
        sub = df_corr[df_corr['diagnostic'] == diag].set_index('method')
        rs  = [float(sub.loc[m, 'spearman_r'])
               if m in sub.index else float('nan')
               for m in methods_all]
        offset = (k - 0.5) * w
        ax.bar(x + offset, rs, w,
               label=ds_labels[diag],
               color=ds_colours[diag], alpha=0.82,
               edgecolor='white', lw=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', '\n') for m in methods_all],
                       fontsize=7.5)
    ax.axhline(0,    color='black', lw=0.7)
    ax.axhline(0.30, color='#43a047', lw=1.2, ls='--', alpha=0.8,
               label='|r| = 0.30 threshold')
    ax.set_ylabel('Spearman r  (diagnostic vs |error|)', fontsize=10)
    ax.set_title(
        'Figure P4-1 — Diagnostic Correlation with Actual Error\n'
        'Positive r = high diagnostic \u2192 high error  '
        '(diagnostic is informative)',
        fontsize=10, fontweight='bold')
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = os.path.join(out_dir, 'figure_p4_01_correlations.png')
    _save(fig, path)
    return path


def fig_p4_02_rejection_rules(df_rules: pd.DataFrame, out_dir: str) -> str:
    """Precision vs recall for both diagnostics, side-by-side panels."""
    if df_rules.empty:
        return ''

    key_methods = ['richardson_1', 'rational_fit', 'pade_22', 'log_linear']
    key_methods = [m for m in key_methods
                   if m in df_rules['method'].unique()]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, diag in zip(axes, DIAGNOSTICS):
        sub_d = df_rules[df_rules['diagnostic'] == diag]
        for method in key_methods:
            sub = sub_d[sub_d['method'] == method].dropna(
                subset=['precision', 'recall'])
            if sub.empty:
                continue
            ax.plot(sub['recall'], sub['precision'],
                    'o-', label=method,
                    color=METHOD_COLOURS.get(method, '#999'),
                    lw=1.8, markersize=5, alpha=0.85)
            for _, row in sub.iterrows():
                ax.annotate(f'{row["threshold"]:.3f}',
                            (row['recall'], row['precision']),
                            textcoords='offset points', xytext=(4, 2),
                            fontsize=6, alpha=0.7)

        ax.axvline(0.5, color='grey', lw=0.7, ls='--', alpha=0.4)
        ax.axhline(0.5, color='grey', lw=0.7, ls='--', alpha=0.4)
        ax.set_xlabel('Recall  (fraction of bad estimates caught)', fontsize=9)
        ax.set_title(diag.replace('_', ' '), fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)

    axes[0].set_ylabel('Precision  (fraction of rejections that were bad)',
                        fontsize=9)
    fig.suptitle(
        'Figure P4-2 — Rejection Rule Precision vs Recall\n'
        'Numbers = IQR threshold; top-right = ideal',
        fontsize=10, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(out_dir, 'figure_p4_02_rejection_rules.png')
    _save(fig, path)
    return path


def fig_p4_03_obs_reliability(df_rel: pd.DataFrame, out_dir: str) -> str:
    """Spearman r vs obs_idx for both diagnostics, richardson_1."""
    if df_rel.empty:
        return ''

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    obs_vals = sorted(df_rel['obs_idx'].unique())

    for ax, diag in zip(axes, DIAGNOSTICS):
        sub_d = df_rel[df_rel['diagnostic'] == diag]
        for method in ['richardson_1', 'rational_fit',
                        'pade_22', 'anderson_1']:
            msub = sub_d[sub_d['method'] == method].sort_values('obs_idx')
            if msub.empty:
                continue
            ax.plot(msub['obs_idx'], msub['spearman_r'],
                    'o-', label=method,
                    color=METHOD_COLOURS.get(method, '#999'),
                    lw=2, markersize=6, alpha=0.85)

        ax.axhline(0.30, color='#43a047', lw=1.2, ls='--', alpha=0.8,
                   label='|r| = 0.30')
        ax.axhline(0, color='black', lw=0.7)
        ax.set_xlabel('obs_idx  (observation depth)', fontsize=9)
        ax.set_title(diag.replace('_', ' '), fontsize=10, fontweight='bold')
        ax.set_xticks(obs_vals)
        ax.set_xticklabels(obs_vals, fontsize=8)
        ax.legend(fontsize=8)

    axes[0].set_ylabel('Spearman r  (diagnostic vs |error|)', fontsize=9)
    fig.suptitle(
        'Figure P4-3 — Diagnostic Reliability vs Observation Depth\n'
        'Higher r = diagnostic more informative at this depth',
        fontsize=10, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(out_dir, 'figure_p4_03_obs_reliability.png')
    _save(fig, path)
    return path


def fig_p4_04_iqr_vs_error(df: pd.DataFrame, out_dir: str) -> str:
    """Scatter: both diagnostics vs |error| for richardson_1 and pade_22."""
    key_methods = [m for m in ['richardson_1', 'pade_22']
                   if m in df['method'].unique()]
    if not key_methods:
        return ''

    fig, axes = plt.subplots(len(DIAGNOSTICS), len(key_methods),
                              figsize=(6 * len(key_methods),
                                       5 * len(DIAGNOSTICS)))
    if len(key_methods) == 1:
        axes = axes.reshape(-1, 1)

    for row_i, diag in enumerate(DIAGNOSTICS):
        for col_j, method in enumerate(key_methods):
            ax  = axes[row_i, col_j]
            sub = df[(df['method'] == method)
                     & df[diag].notna()
                     & df['error'].notna()].copy()
            colours = ['#c62828' if c else '#1565c0'
                       for c in sub['catastrophic']]
            ax.scatter(sub[diag], sub['error'],
                       c=colours, alpha=0.25, s=8, linewidths=0)
            ax.set_xscale('symlog', linthresh=1e-6)
            ax.set_yscale('symlog', linthresh=1e-6)
            ax.set_xlabel(diag.replace('_', ' ') + '  (symlog)', fontsize=8)
            ax.set_ylabel('|error|  (symlog)', fontsize=8)
            ax.set_title(method, fontsize=9, fontweight='bold')
            ax.legend(handles=[
                mpatches.Patch(color='#c62828', label='Catastrophic'),
                mpatches.Patch(color='#1565c0', label='Non-catastrophic'),
            ], fontsize=7)

    fig.suptitle(
        'Figure P4-4 — Diagnostic vs Actual Error\n'
        'Positive slope = high diagnostic \u2192 high error  '
        '(diagnostic is useful)',
        fontsize=10, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(out_dir, 'figure_p4_04_iqr_vs_error.png')
    _save(fig, path)
    return path


def fig_p4_05_regime_diagnostic(df: pd.DataFrame, out_dir: str) -> str:
    """
    For richardson_1: mean perturb_IQR and mean error by regime.
    If perturb_IQR is useful, high-error regimes should also have high IQR.
    """
    sub  = df[(df['method'] == 'richardson_1')
              & df['perturb_iqr'].notna()
              & df['error'].notna()]
    if sub.empty:
        return ''

    fid = sub['future_idx'].max()
    agg = (sub[sub['future_idx'] == fid]
           .groupby('regime')
           .agg(mean_piqr=('perturb_iqr', 'mean'),
                mean_err=('error',       'mean'),
                cat_rate=('catastrophic','mean'))
           .reset_index()
           .sort_values('mean_err', ascending=False))

    fig, ax = plt.subplots(figsize=(12, 6))
    x   = np.arange(len(agg))
    ax2 = ax.twinx()

    ax.bar(x, agg['mean_err'], color='#f4a261', alpha=0.7,
           label='Mean |error|')
    ax2.plot(x, agg['mean_piqr'], 'o-', color='#c62828',
             lw=2, markersize=7, label='Mean perturb IQR')

    ax.set_xticks(x)
    ax.set_xticklabels([r.replace('_', '\n') for r in agg['regime']],
                       fontsize=7.5)
    ax.set_ylabel('Mean |error|', fontsize=9, color='#f4a261')
    ax2.set_ylabel('Mean perturb IQR', fontsize=9, color='#c62828')
    ax.set_title(
        'Figure P4-5 — richardson_1: Mean Error and perturb IQR by Regime\n'
        'If diagnostic is useful: IQR and error should co-vary by regime',
        fontsize=10, fontweight='bold')

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
    fig.tight_layout()
    path = os.path.join(out_dir, 'figure_p4_05_regime_diagnostic.png')
    _save(fig, path)
    return path


# =============================================================================
# MASTER RUN FUNCTION
# =============================================================================

def run_all(obs_idx_list, noise_list, future_list, n_seeds,
            window_len, shifts, perturb_trials, perturb_scale,
            out_dir, phase2_feat_path=None, verbose=True):
    os.makedirs(out_dir, exist_ok=True)

    df = run_phase4(obs_idx_list, noise_list, future_list,
                    n_seeds, window_len, shifts,
                    perturb_trials, perturb_scale, out_dir, verbose)

    print('\n  Computing diagnostic correlations ...')
    df_corr = diagnostic_correlations(df, out_dir)

    print('\n  Testing rejection rules (both diagnostics) ...')
    df_rules = test_rejection_rules(df, out_dir)

    print('\n  Cascade + diagnostic filter ...')
    df_filt = pd.DataFrame()
    if phase2_feat_path and os.path.exists(phase2_feat_path):
        df_filt = cascade_with_filter(df, phase2_feat_path, out_dir)
    else:
        print('  (Phase 2 features not found; skipping cascade filter)')

    print('\n  Ensemble comparison ...')
    df_ens = ensemble_comparison(df, out_dir)

    print('\n  Obs-depth reliability ...')
    df_rel = obs_reliability(df, out_dir)

    print('\n  Generating figures ...')
    paths = [
        fig_p4_01_correlations(df_corr, out_dir),
        fig_p4_02_rejection_rules(df_rules, out_dir),
        fig_p4_03_obs_reliability(df_rel, out_dir),
        fig_p4_04_iqr_vs_error(df, out_dir),
        fig_p4_05_regime_diagnostic(df, out_dir),
    ]

    return {
        'raw':         df,
        'correlations': df_corr,
        'rules':       df_rules,
        'filter':      df_filt,
        'ensemble':    df_ens,
        'reliability': df_rel,
        'figures':     [p for p in paths if p],
    }
