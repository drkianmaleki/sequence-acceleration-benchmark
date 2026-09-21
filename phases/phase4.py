"""
phase4.py
=========
Phase 4 — Stability Diagnostic Stress Testing (redesign v2).

Two diagnostics are tested as real-time trust signals:

  shift_IQR   : IQR of estimates across small window-start shifts.
  perturb_IQR : IQR of estimates under tiny multiplicative perturbations
                of the window values (2% scale).

Four questions answered
-----------------------
Q1. Do diagnostics predict actual error?  (Spearman r, globally and per method)
Q2. Can rejection rules built on diagnostics catch bad estimates?
Q3. Does adding a perturb_IQR filter improve the Phase 2 cascade?
Q4. Does the diagnostic-weighted ensemble outperform fixed methods?

Redesign v2
-----------
  * Evaluation points are the three gap strata per (regime, obs_idx); every
    record carries target_g, achieved_g, n_f, capped, L_true, L_hat and a
    skill score against the hindsight best-of-four trivial reference (strict)
    plus the fixed-reference skill_vs_* / win_vs_* columns.
  * Core and held-out regimes are both evaluated (is_holdout flag).  Pooled
    analyses (Q1-Q4, reliability) use the core regimes with capped cells
    excluded; held-out correlations are reported separately; capped cells
    go to phase4_capped.csv.
  * The four trivial reference methods are evaluated for skill; diagnostics
    are computed for the nine accelerators of the diagnostic pool only.

Author : Kian Maleki
Date   : 2026-05-24 (v1), 2026-09-19 (redesign v2)
"""

import os, math, warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from typing import List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')

import src.config as CFG_MOD
from src.accelerators import METHODS
from src.generators   import regime_functions
from src.asymptote    import assumed_asymptote
from src.pipeline     import (REFERENCE_METHODS, capped_block, exclude_capped,
                              horizon_meta, is_holdout, method_flags,
                              resolve_regimes)
from src.trivial      import (REFERENCE_TAGS, best_reference_error, skill_score,
                              skill_vs_from_arrays, skill_vs_table)

# ── Method set ─────────────────────────────────────────────────────────────────
PHASE4_METHODS = [
    'current_value', 'richardson_1', 'richardson_a10',
    'single_exp_fit', 'rational_fit', 'pade_22',
    'log_linear', 'weniger_d2', 'anderson_1',
]
# Evaluated for skill but without diagnostics (they are constants of the window)
EXTRA_REFERENCES = [m for m in REFERENCE_METHODS if m not in PHASE4_METHODS]
EVAL_METHODS = PHASE4_METHODS + EXTRA_REFERENCES

METHOD_COLOURS = {
    'current_value':  '#888888', 'richardson_1':   '#f4a261',
    'richardson_a10': '#e76f51', 'single_exp_fit': '#2196f3',
    'rational_fit':   '#1565c0', 'pade_22':        '#e91e63',
    'log_linear':     '#00897b', 'weniger_d2':     '#9c27b0',
    'anderson_1':     '#795548',
}

DIAGNOSTICS = ['shift_iqr', 'perturb_iqr']
FIG_DPI = 150
CELL = ['regime', 'obs_idx', 'noise', 'seed', 'target_g']


# ── Helpers ────────────────────────────────────────────────────────────────────
def _cfg(fid, L_hat):
    # L_hat is the ASSUMED asymptote (src.asymptote); never L_true.
    return {
        'L_inf': float(L_hat), 'ridge': CFG_MOD.RIDGE,
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


def _headline(df: pd.DataFrame, default_g: Optional[float]) -> float:
    gs = set(df['target_g'].unique())
    g = CFG_MOD.HEADLINE_G if default_g is None else float(default_g)
    return g if g in gs else float(max(gs))


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

def run_phase4(obs_idx_list, noise_list, gap_fractions, n_seeds,
               window_len, shifts, perturb_trials, perturb_scale,
               out_dir, core_regimes=None, holdout_regimes=None, verbose=True):
    """
    For every (regime, obs_idx, noise, seed, method, gap stratum):
      central estimate, shift_IQR, perturb_IQR, error, catastrophic flag,
      skill, plus the horizon metadata.  true_val is stored so ensemble
      error can be computed later.
    """
    os.makedirs(out_dir, exist_ok=True)
    regimes = resolve_regimes(core_regimes, holdout_regimes, include_holdout=True)
    gap_fractions = [float(g) for g in gap_fractions]
    n_arr   = np.arange(max(obs_idx_list) + 1, dtype=float)
    n_total = len(obs_idx_list) * len(noise_list) * n_seeds * len(regimes)
    done    = 0
    records = []

    for obs_idx in obs_idx_list:
        wl = min(window_len, obs_idx)

        for sigma in noise_list:
            for seed in range(n_seeds):
                rng   = np.random.RandomState(
                    seed * 137 + int(sigma * 1e6) % 9973 + obs_idx * 7)
                rng_p = np.random.RandomState(seed * 999 + obs_idx)

                for regime in regimes:
                    # Hidden per-(regime, seed) asymptote; methods never see L_true.
                    gen, truth_fn, L_true = regime_functions(regime, seed)
                    seq_full = gen(n_arr, rng, sigma)

                    w_start  = max(0, obs_idx - wl + 1)
                    seq_win  = list(seq_full[w_start : obs_idx + 1])
                    idx_win  = list(range(w_start, obs_idx + 1))
                    curr_val = float(seq_full[obs_idx])
                    L_hat    = assumed_asymptote(L_true, seq_win)
                    hold     = is_holdout(regime)

                    for g in gap_fractions:
                        hm       = horizon_meta(regime, obs_idx, g, seed)
                        n_f      = hm['n_f']
                        true_val = float(truth_fn(n_f))
                        curr_err = abs(curr_val - true_val)
                        cfg      = _cfg(n_f, L_hat)

                        ests, errs = {}, {}
                        for method in EVAL_METHODS:
                            try:
                                est = float(METHODS[method](seq_win, idx_win, float(n_f), cfg))
                            except Exception:
                                est = float('nan')
                            ests[method] = est
                            errs[method] = abs(est - true_val) if _valid(est, cfg) else float('nan')
                        ref_err = best_reference_error(errs)

                        for method in EVAL_METHODS:
                            est, err = ests[method], errs[method]
                            valid = math.isfinite(err)
                            cat   = (not valid) or (
                                curr_err > 1e-12 and err > CFG_MOD.CAT_MULT * curr_err)

                            if method in PHASE4_METHODS:
                                s_iqr = _shift_iqr(seq_win, idx_win, float(n_f),
                                                   method, cfg, shifts)
                                p_iqr = _perturb_iqr(seq_win, idx_win, float(n_f),
                                                     method, cfg, perturb_trials,
                                                     perturb_scale, rng_p)
                            else:
                                s_iqr = p_iqr = float('nan')

                            rec = {
                                'regime':      regime,
                                'is_holdout':  hold,
                                'obs_idx':     obs_idx,
                                'noise':       sigma,
                                'seed':        seed,
                                'L_true':      L_true,
                                'L_hat':       L_hat,
                                'method':      method,
                                'true_val':    true_val,
                                'estimate':    est if valid else float('nan'),
                                'error':       err,
                                'valid':       int(valid),
                                'catastrophic':int(cat),
                                'curr_err':    curr_err,
                                'ref_error':   ref_err,
                                'skill':       skill_score(err, ref_err) if valid else float('nan'),
                                'shift_iqr':   s_iqr,
                                'perturb_iqr': p_iqr,
                            }
                            rec.update(skill_vs_table(err if valid else float('nan'), errs))
                            rec.update(hm)
                            rec.update(method_flags(method))
                            records.append(rec)

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

def diagnostic_correlations(df: pd.DataFrame, out_dir: str, suffix: str = '') -> pd.DataFrame:
    """Spearman r between each diagnostic and |error|, globally and per method."""
    rows = []
    for diag in DIAGNOSTICS:
        mask = df[diag].notna() & df['error'].notna()
        n    = int(mask.sum())
        if n >= 10:
            r, p = spearmanr(df.loc[mask, diag], df.loc[mask, 'error'])
            rows.append({'method': 'ALL', 'diagnostic': diag,
                         'spearman_r': round(float(r), 4),
                         'p_value':    round(float(p), 6), 'n': n})

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

    df_corr = pd.DataFrame(rows, columns=['method', 'diagnostic', 'spearman_r', 'p_value', 'n'])
    p = os.path.join(out_dir, f'phase4_diagnostic_correlations{suffix}.csv')
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
      mean_err_saved = mean(curr_err - error) when rule fires
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

    cols = ['diagnostic', 'method', 'threshold', 'n_fire', 'n_total', 'fire_rate',
            'precision', 'recall', 'mean_err_saved']
    df_rules = (pd.DataFrame(rows, columns=cols)
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
                        out_dir: str,
                        default_g: Optional[float] = None) -> pd.DataFrame:
    """
    Compare Phase 2 cascade with and without a perturb_IQR rejection filter
    at the headline stratum.  Filter: if the chosen method's perturb_IQR
    exceeds the threshold, fall back to current_value.
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

    g = _headline(df, default_g)
    sub_g  = df[df['target_g'] == g].copy()
    merged = sub_g.merge(feat_avg, on=['regime', 'obs_idx', 'noise'], how='inner',
                         suffixes=('', '_feat'))

    thresholds = [float('inf'), 1.0, 0.5, 0.2, 0.1, 0.05, 0.02]
    rows = []

    for threshold in thresholds:
        errs = []
        for _, grp in merged.groupby(['regime', 'obs_idx', 'noise', 'seed']):
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
            if chosen_row.empty or not math.isfinite(chosen_row['error'].values[0]):
                chosen_row = grp[grp['method'] == 'current_value']

            err = (float(chosen_row['error'].values[0])
                   if not chosen_row.empty else float('nan'))
            errs.append(err)

        label = ('no_filter' if threshold == float('inf')
                 else f'perturb_iqr>{threshold}')
        rows.append({
            'filter':        label,
            'iqr_threshold': threshold,
            'target_g':      g,
            'mean_error':    round(float(np.nanmean(errs)),   6) if errs else float('nan'),
            'median_error':  round(float(np.nanmedian(errs)), 6) if errs else float('nan'),
            'n':             len(errs),
        })

    df_filt = pd.DataFrame(rows)
    p = os.path.join(out_dir, 'phase4_cascade_filter.csv')
    df_filt.to_csv(p, index=False)
    print(f'  Saved: {p}  ({len(df_filt)} rows)')
    return df_filt


# =============================================================================
# 6.  Q4 — DIAGNOSTIC-WEIGHTED ENSEMBLE
# =============================================================================

def ensemble_comparison(df: pd.DataFrame, out_dir: str,
                        default_g: Optional[float] = None) -> pd.DataFrame:
    """
    Compare selectors using true_val stored in records, at the headline
    stratum.  Ensemble: weight each method by 1/(perturb_IQR + eps).  The
    four trivial references are reported as fixed selectors, and every
    selector gets a median skill against the hindsight best-of-four reference
    (strict) plus med_skill_vs_* / win_rate_vs_* against each trivial.
    """
    eps = 0.01
    g   = _headline(df, default_g)
    sub = df[df['target_g'] == g].copy()

    selectors = (['oracle', 'fixed_rational', 'fixed_richardson',
                  'phase2_proxy', 'diag_ensemble'] + REFERENCE_METHODS)
    errs = {s: [] for s in selectors}
    refs = []

    for _, grp in sub.groupby(['regime', 'obs_idx', 'noise', 'seed']):
        grp_idx  = grp.set_index('method')
        true_val = float(grp['true_val'].iloc[0])
        refs.append(float(grp['ref_error'].iloc[0]))

        def _e(m):
            return float(grp_idx.loc[m, 'error']) if m in grp_idx.index else float('nan')

        best_err = float('inf')
        for m in PHASE4_METHODS:
            e = _e(m)
            if math.isfinite(e) and e < best_err:
                best_err = e
        errs['oracle'].append(best_err if best_err < float('inf') else float('nan'))
        errs['fixed_rational'].append(_e('rational_fit'))
        errs['fixed_richardson'].append(_e('richardson_1'))
        cands = [x for x in (_e('richardson_1'), _e('rational_fit')) if math.isfinite(x)]
        errs['phase2_proxy'].append(min(cands) if cands else float('nan'))

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
            w = np.array(ws); w /= w.sum()
            errs['diag_ensemble'].append(abs(float(np.dot(w, ests)) - true_val))
        else:
            errs['diag_ensemble'].append(float('nan'))

        for m in REFERENCE_METHODS:
            errs[m].append(_e(m))

    refs = np.asarray(refs, dtype=float)
    ref_arrays = {tag: np.asarray(errs[name], dtype=float) for name, tag in REFERENCE_TAGS}
    summary_rows = []
    for sel in selectors:
        vals = np.asarray(errs[sel], dtype=float)
        ok = np.isfinite(vals)
        sk = np.array([skill_score(v, r) for v, r in zip(vals, refs)], dtype=float)
        sk = sk[~np.isnan(sk)]
        summary_rows.append({
            'selector':     sel,
            'target_g':     g,
            'is_trivial':   int(sel in REFERENCE_METHODS),
            'mean_error':   round(float(vals[ok].mean()),   6) if ok.any() else float('nan'),
            'median_error': round(float(np.median(vals[ok])), 6) if ok.any() else float('nan'),
            'med_skill':    round(float(np.median(sk)), 4) if sk.size else float('nan'),
            'n':            int(ok.sum()),
            **skill_vs_from_arrays(vals, ref_arrays),   # med_skill_vs_* / win_rate_vs_*
        })

    df_sum = pd.DataFrame(summary_rows)
    p = os.path.join(out_dir, 'phase4_ensemble.csv')
    df_sum.to_csv(p, index=False)
    print(f'  Saved: {p}  ({len(df_sum)} rows)')
    return df_sum


# =============================================================================
# 7.  RELIABILITY VS OBS_IDX
# =============================================================================

def obs_reliability(df: pd.DataFrame, out_dir: str,
                    default_g: Optional[float] = None) -> pd.DataFrame:
    """For each (obs_idx, method, diagnostic) at the headline stratum:
    Spearman r(diagnostic, error) and precision of high-diagnostic = catastrophic."""
    rows = []
    g    = _headline(df, default_g)
    sub  = df[df['target_g'] == g]

    for obs_idx in sorted(sub['obs_idx'].unique()):
        for method in PHASE4_METHODS:
            msub = sub[(sub['obs_idx'] == obs_idx) & (sub['method'] == method)]
            for diag in DIAGNOSTICS:
                mask = msub[diag].notna() & msub['error'].notna()
                n    = int(mask.sum())
                if n < 5:
                    continue
                r, p = spearmanr(msub.loc[mask, diag], msub.loc[mask, 'error'])
                hi      = msub[diag] > 0.05
                cat     = msub['catastrophic'] == 1
                prec    = (float((hi & cat).sum() / hi.sum())
                           if hi.sum() > 0 else float('nan'))
                rows.append({
                    'obs_idx':     obs_idx,
                    'target_g':    g,
                    'method':      method,
                    'diagnostic':  diag,
                    'spearman_r':  round(float(r), 4),
                    'p_value':     round(float(p), 6),
                    'n':           n,
                    'hi_iqr_prec': round(prec, 3)
                                   if math.isfinite(prec) else float('nan'),
                })

    cols = ['obs_idx', 'target_g', 'method', 'diagnostic', 'spearman_r', 'p_value',
            'n', 'hi_iqr_prec']
    df_rel = pd.DataFrame(rows, columns=cols)
    p = os.path.join(out_dir, 'phase4_obs_reliability.csv')
    df_rel.to_csv(p, index=False)
    print(f'  Saved: {p}  ({len(df_rel)} rows)')
    return df_rel


# =============================================================================
# 8.  FIGURES
# =============================================================================

def fig_p4_01_correlations(df_corr: pd.DataFrame, out_dir: str) -> str:
    """Grouped bar chart: Spearman r per (method, diagnostic)."""
    if df_corr.empty:
        return ''
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
        '(core regimes, capped cells excluded)  Positive r = high diagnostic '
        '→ high error',
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
    key_methods = [m for m in key_methods if m in df_rules['method'].unique()]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, diag in zip(axes, DIAGNOSTICS):
        sub_d = df_rules[df_rules['diagnostic'] == diag]
        for method in key_methods:
            sub = sub_d[sub_d['method'] == method].dropna(subset=['precision', 'recall'])
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

    axes[0].set_ylabel('Precision  (fraction of rejections that were bad)', fontsize=9)
    fig.suptitle(
        'Figure P4-2 — Rejection Rule Precision vs Recall\n'
        'Numbers = IQR threshold; top-right = ideal',
        fontsize=10, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(out_dir, 'figure_p4_02_rejection_rules.png')
    _save(fig, path)
    return path


def fig_p4_03_obs_reliability(df_rel: pd.DataFrame, out_dir: str) -> str:
    """Spearman r vs obs_idx for both diagnostics."""
    if df_rel.empty:
        return ''

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    obs_vals = sorted(df_rel['obs_idx'].unique())

    for ax, diag in zip(axes, DIAGNOSTICS):
        sub_d = df_rel[df_rel['diagnostic'] == diag]
        for method in ['richardson_1', 'rational_fit', 'pade_22', 'anderson_1']:
            msub = sub_d[sub_d['method'] == method].sort_values('obs_idx')
            if msub.empty:
                continue
            ax.plot(msub['obs_idx'], msub['spearman_r'],
                    'o-', label=method,
                    color=METHOD_COLOURS.get(method, '#999'),
                    lw=2, markersize=6, alpha=0.85)

        ax.axhline(0.30, color='#43a047', lw=1.2, ls='--', alpha=0.8, label='|r| = 0.30')
        ax.axhline(0, color='black', lw=0.7)
        ax.set_xlabel('obs_idx  (observation depth)', fontsize=9)
        ax.set_title(diag.replace('_', ' '), fontsize=10, fontweight='bold')
        ax.set_xticks(obs_vals)
        ax.set_xticklabels(obs_vals, fontsize=8)
        ax.legend(fontsize=8)

    axes[0].set_ylabel('Spearman r  (diagnostic vs |error|)', fontsize=9)
    g = float(df_rel['target_g'].iloc[0])
    fig.suptitle(
        f'Figure P4-3 — Diagnostic Reliability vs Observation Depth  (g = {g:g})\n'
        'Higher r = diagnostic more informative at this depth',
        fontsize=10, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(out_dir, 'figure_p4_03_obs_reliability.png')
    _save(fig, path)
    return path


def fig_p4_04_iqr_vs_error(df: pd.DataFrame, out_dir: str) -> str:
    """Scatter: both diagnostics vs |error| for richardson_1 and pade_22."""
    key_methods = [m for m in ['richardson_1', 'pade_22'] if m in df['method'].unique()]
    if not key_methods:
        return ''

    fig, axes = plt.subplots(len(DIAGNOSTICS), len(key_methods),
                              figsize=(6 * len(key_methods), 5 * len(DIAGNOSTICS)))
    axes = np.asarray(axes).reshape(len(DIAGNOSTICS), len(key_methods))

    for row_i, diag in enumerate(DIAGNOSTICS):
        for col_j, method in enumerate(key_methods):
            ax  = axes[row_i, col_j]
            sub = df[(df['method'] == method) & df[diag].notna() & df['error'].notna()].copy()
            colours = ['#c62828' if c else '#1565c0' for c in sub['catastrophic']]
            ax.scatter(sub[diag], sub['error'], c=colours, alpha=0.25, s=8, linewidths=0)
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
        'Figure P4-4 — Diagnostic vs Actual Error (core regimes, capped excluded)\n'
        'Positive slope = high diagnostic → high error',
        fontsize=10, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(out_dir, 'figure_p4_04_iqr_vs_error.png')
    _save(fig, path)
    return path


def fig_p4_05_regime_diagnostic(df: pd.DataFrame, out_dir: str,
                                default_g: Optional[float] = None) -> str:
    """richardson_1: mean perturb_IQR and mean error by regime at the headline stratum."""
    sub  = df[(df['method'] == 'richardson_1')
              & df['perturb_iqr'].notna()
              & df['error'].notna()]
    if sub.empty:
        return ''

    g   = _headline(sub, default_g)
    agg = (sub[sub['target_g'] == g]
           .groupby('regime')
           .agg(mean_piqr=('perturb_iqr', 'mean'),
                mean_err=('error',       'mean'),
                cat_rate=('catastrophic','mean'))
           .reset_index()
           .sort_values('mean_err', ascending=False))
    if agg.empty:
        return ''

    fig, ax = plt.subplots(figsize=(12, 6))
    x   = np.arange(len(agg))
    ax2 = ax.twinx()

    ax.bar(x, agg['mean_err'], color='#f4a261', alpha=0.7, label='Mean |error|')
    ax2.plot(x, agg['mean_piqr'], 'o-', color='#c62828', lw=2, markersize=7,
             label='Mean perturb IQR')

    ax.set_xticks(x)
    ax.set_xticklabels([r.replace('_', '\n') for r in agg['regime']], fontsize=7.5)
    ax.set_ylabel('Mean |error|', fontsize=9, color='#f4a261')
    ax2.set_ylabel('Mean perturb IQR', fontsize=9, color='#c62828')
    ax.set_title(
        f'Figure P4-5 — richardson_1: Mean Error and perturb IQR by Regime (g = {g:g})\n'
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

def run_all(obs_idx_list, noise_list, gap_fractions, n_seeds,
            window_len, shifts, perturb_trials, perturb_scale,
            out_dir, phase2_feat_path=None, core_regimes=None,
            holdout_regimes=None, default_g=None, verbose=True):
    os.makedirs(out_dir, exist_ok=True)

    df = run_phase4(obs_idx_list, noise_list, gap_fractions, n_seeds,
                    window_len, shifts, perturb_trials, perturb_scale,
                    out_dir, core_regimes, holdout_regimes, verbose)

    # Pooled analyses: core regimes, capped cells excluded.
    df_core = exclude_capped(df[df['is_holdout'] == 0])
    df_hold = exclude_capped(df[df['is_holdout'] == 1])
    g_head  = _headline(df, default_g)

    print('\n  Capped block ...')
    df_cap = capped_block(df, keys=['regime', 'is_holdout', 'obs_idx', 'target_g', 'method'],
                          value_cols=['error', 'skill'])
    p = os.path.join(out_dir, 'phase4_capped.csv')
    df_cap.to_csv(p, index=False)
    print(f'  Saved: {p}  ({len(df_cap)} rows)')

    print('\n  Computing diagnostic correlations (core) ...')
    df_corr = diagnostic_correlations(df_core, out_dir)
    if len(df_hold):
        print('  Computing diagnostic correlations (held-out) ...')
        diagnostic_correlations(df_hold, out_dir, suffix='_holdout')

    print('\n  Testing rejection rules (both diagnostics) ...')
    df_rules = test_rejection_rules(df_core, out_dir)

    print('\n  Cascade + diagnostic filter ...')
    df_filt = pd.DataFrame()
    if phase2_feat_path and os.path.exists(phase2_feat_path):
        df_filt = cascade_with_filter(df_core, phase2_feat_path, out_dir, g_head)
    else:
        print('  (Phase 2 features not found; skipping cascade filter)')

    print('\n  Ensemble comparison ...')
    df_ens = ensemble_comparison(df_core, out_dir, g_head)

    print('\n  Obs-depth reliability ...')
    df_rel = obs_reliability(df_core, out_dir, g_head)

    print('\n  Generating figures ...')
    paths = [
        fig_p4_01_correlations(df_corr, out_dir),
        fig_p4_02_rejection_rules(df_rules, out_dir),
        fig_p4_03_obs_reliability(df_rel, out_dir),
        fig_p4_04_iqr_vs_error(df_core, out_dir),
        fig_p4_05_regime_diagnostic(df_core, out_dir, g_head),
    ]

    return {
        'raw':          df,
        'capped':       df_cap,
        'correlations': df_corr,
        'rules':        df_rules,
        'filter':       df_filt,
        'ensemble':     df_ens,
        'reliability':  df_rel,
        'default_g':    g_head,
        'figures':      [p for p in paths if p],
    }
