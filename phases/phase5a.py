"""
phase5a.py  (v3)
================
Phase 5A — Full 51-Method Ensemble with Ablation.

v3 changes vs v2
----------------
  * Added threshold_ensemble variants: filter by perturb_IQR threshold,
    then equal-weight median.  Avoids EPS calibration problems.
  * Added capped_diag_ensemble: weight = 1/(IQR+EPS) but capped at
    5× median weight to prevent any single method from dominating.
  * SELECTORS list updated to include new variants.
  * EPS kept at 0.01 for uncapped variants.

Key ablation questions
-----------------------
Q1. Does 51 methods beat 9 methods?    diag_9 vs diag_51
Q2. Does weighting beat equal?         equal_51 vs diag_51
Q3. Does threshold filter beat equal?  equal_51 vs threshold_ens_010
Q4. How much does oracle improve?      oracle_9 vs oracle_51
Q5. Where is the residual gap?         per-regime decomposition
Q6. Are dangerous methods auto-IDed?   method weight ranking

Author : Kian Maleki
Date   : 2026-05-24
"""

import os, math, warnings
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from typing import List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')

import src.config as CFG_MOD
from src.accelerators import METHODS, METHOD_NAMES
from src.generators   import GENERATORS, REGIME_NAMES, TRUTH

PHASE2_METHODS = [
    'current_value', 'richardson_1', 'richardson_a10',
    'single_exp_fit', 'rational_fit', 'pade_22',
    'log_linear', 'weniger_d2', 'anderson_1',
]

DANGEROUS = {'neville_2', 'neville_3', 'neville_4',
             'pade_21', 'pade_31', 'pade_32', 'linear'}

EPS     = 0.01   # for continuous weighting
FIG_DPI = 150


# ── helpers ───────────────────────────────────────────────────────────────────
def _cfg(fid):
    return {'L_inf': CFG_MOD.L_INF, 'ridge': CFG_MOD.RIDGE,
            'min_valid': CFG_MOD.MIN_VALID, 'max_valid': CFG_MOD.MAX_VALID,
            'denom_tol': CFG_MOD.DENOM_TOL}

def _valid(v, cfg):
    return bool(math.isfinite(v) and cfg['min_valid'] <= v <= cfg['max_valid'])

def _save(fig, path):
    fig.savefig(path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')

def _save_csv(df, out_dir, fname):
    p = os.path.join(out_dir, fname)
    df.to_csv(p, index=False)
    print(f'  Saved: {p}  ({len(df)} rows)')


# ── perturb IQR ───────────────────────────────────────────────────────────────
def _perturb_iqr(seq_win, idx_win, future_x, method, cfg,
                  n_trials, scale, rng):
    fn   = METHODS[method]
    arr  = np.asarray(seq_win, dtype=float)
    ests = []
    for _ in range(n_trials):
        v = fn(list(arr * (1.0 + scale * rng.randn(len(arr)))),
               idx_win, future_x, cfg)
        if _valid(v, cfg):
            ests.append(v)
    if len(ests) < 2:
        return float('nan')
    return float(np.subtract(*np.percentile(ests, [75, 25])))


# ── Phase 2 cascade features ──────────────────────────────────────────────────
def _cascade_features(seq_win, idx_win):
    s  = np.asarray(seq_win, dtype=float)
    x  = np.asarray(idx_win, dtype=float)
    L0 = max(0.0, min(CFG_MOD.L_INF, float(np.min(s)) * 0.5))

    slope = float('nan')
    pos   = (s - L0) > 0
    if pos.sum() >= 3:
        try:
            slope = float(np.polyfit(
                np.log(np.maximum(x[pos], 1.0)),
                np.log((s - L0)[pos]), 1)[0])
        except Exception:
            pass

    r2 = float('nan')
    try:
        def model(n, c, a):
            return L0 + c / n**a
        popt, _ = curve_fit(model, np.maximum(x, 1.0), s,
                             p0=[max(float(s[-1]) - L0, 1e-4), 0.8],
                             bounds=([0, 0.05], [5, 4]), maxfev=800)
        s_pred = model(np.maximum(x, 1.0), *popt)
        ss_res = float(np.sum((s - s_pred) ** 2))
        ss_tot = float(np.sum((s - s.mean()) ** 2))
        r2     = float(np.clip(1.0 - ss_res / (ss_tot + 1e-20), -10.0, 1.0))
    except Exception:
        pass

    return slope, r2


def _phase2_cascade(slope, r2):
    if math.isfinite(slope) and slope > -0.10:
        return 'rational_fit'
    if math.isfinite(r2) and r2 < 0.50:
        return 'rational_fit'
    return 'richardson_1'


# =============================================================================
# 1.  MAIN EVALUATION LOOP
# =============================================================================

def run_phase5a(obs_idx_list, noise_list, future_list, n_seeds,
                window_len, perturb_trials, perturb_scale,
                out_dir, verbose=True):
    os.makedirs(out_dir, exist_ok=True)

    all_methods = list(METHOD_NAMES)
    n_arr       = np.arange(max(future_list) + 200, dtype=float)
    n_total     = (len(obs_idx_list) * len(noise_list)
                   * n_seeds * len(REGIME_NAMES))
    done        = 0
    records     = []

    print(f'  Methods in pool : {len(all_methods)}  '
          f'(including {len(DANGEROUS)} dangerous)')
    print(f'  Progress updates: every {max(1, n_total // 20)} '
          f'regime-groups  (~5% increments)\n')

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
                    slope, r2 = _cascade_features(seq_win, idx_win)

                    for fid in future_list:
                        true_val = float(truth_fn(fid))
                        curr_err = abs(curr_val - true_val)
                        cfg      = _cfg(fid)

                        for method in all_methods:
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

                            p_iqr = _perturb_iqr(
                                seq_win, idx_win, float(fid),
                                method, cfg,
                                perturb_trials, perturb_scale, rng_p)

                            records.append({
                                'regime':       regime,
                                'obs_idx':      obs_idx,
                                'noise':        sigma,
                                'seed':         seed,
                                'method':       method,
                                'future_idx':   fid,
                                'true_val':     true_val,
                                'estimate':     est if valid else float('nan'),
                                'error':        err,
                                'valid':        int(valid),
                                'catastrophic': int(cat),
                                'curr_err':     curr_err,
                                'perturb_iqr':  p_iqr,
                                'is_dangerous': int(method in DANGEROUS),
                                'casc_slope':   slope,
                                'casc_r2':      r2,
                            })

                    # progress counter INSIDE regime loop
                    done += 1
                    if verbose and done % max(1, n_total // 20) == 0:
                        print(f'  [{done:>6}/{n_total}]  '
                              f'{100*done/n_total:5.1f}%  '
                              f'regime={regime:<20}  '
                              f'obs={obs_idx}  sigma={sigma:.3f}',
                              flush=True)

    if verbose:
        print(f'\n  [{n_total}/{n_total}] 100.0%  Done.\n')

    df = pd.DataFrame(records)
    p  = os.path.join(out_dir, 'phase5a_raw.csv')
    df.to_csv(p, index=False)
    sz = os.path.getsize(p) // 1024 // 1024
    print(f'  Saved: {p}  ({len(df):,} rows,  {sz} MB)')
    return df


# =============================================================================
# 2.  ENSEMBLE COMPUTATION
# =============================================================================

def _compute_ensembles(grp, true_val, all_methods, phase2_methods,
                        slope, r2):
    mi = grp.set_index('method')

    def _err(m):
        return float(mi.loc[m,'error'])     if m in mi.index else float('nan')
    def _est(m):
        return float(mi.loc[m,'estimate'])  if m in mi.index else float('nan')
    def _iqr(m):
        return float(mi.loc[m,'perturb_iqr']) if m in mi.index else float('nan')

    out = {}

    # ── Fixed single methods ───────────────────────────────────────────────────
    out['fixed_rational']   = _err('rational_fit')
    out['fixed_richardson'] = _err('richardson_1')
    out['current_value']    = _err('current_value')
    out['phase2_cascade']   = _err(_phase2_cascade(slope, r2))

    # ── Oracles ───────────────────────────────────────────────────────────────
    e9  = [_err(m) for m in phase2_methods if math.isfinite(_err(m))]
    e51 = [_err(m) for m in all_methods    if math.isfinite(_err(m))]
    out['oracle_9']  = min(e9)  if e9  else float('nan')
    out['oracle_51'] = min(e51) if e51 else float('nan')

    # ── Equal-weight ensembles (median) ────────────────────────────────────────
    def _equal_ens(pool):
        ests = [_est(m) for m in pool if math.isfinite(_est(m))]
        return abs(float(np.median(ests)) - true_val) if ests else float('nan')

    out['equal_ensemble_51']   = _equal_ens(all_methods)
    out['equal_ensemble_9']    = _equal_ens(phase2_methods)
    out['equal_ensemble_safe'] = _equal_ens(
        [m for m in all_methods if m not in DANGEROUS])

    # ── Diagnostic-weighted ensemble (continuous 1/IQR, EPS=0.01) ─────────────
    def _diag_ens(pool):
        ests, ws = [], []
        for m in pool:
            e = _est(m); q = _iqr(m)
            if math.isfinite(e) and math.isfinite(q):
                ests.append(e); ws.append(1.0 / (q + EPS))
        if not ests:
            return float('nan')
        w = np.array(ws); w /= w.sum()
        return abs(float(np.dot(w, ests)) - true_val)

    out['diag_ensemble_51']   = _diag_ens(all_methods)
    out['diag_ensemble_9']    = _diag_ens(phase2_methods)
    out['diag_ensemble_safe'] = _diag_ens(
        [m for m in all_methods if m not in DANGEROUS])

    # ── Capped diagnostic ensemble (weight <= 5x median weight) ───────────────
    def _capped_diag_ens(pool):
        ests, ws = [], []
        for m in pool:
            e = _est(m); q = _iqr(m)
            if math.isfinite(e) and math.isfinite(q):
                ests.append(e); ws.append(1.0 / (q + EPS))
        if not ests:
            return float('nan')
        ws = np.array(ws, dtype=float)
        cap = 5.0 * float(np.median(ws))
        ws  = np.minimum(ws, cap)
        ws /= ws.sum()
        return abs(float(np.dot(ws, ests)) - true_val)

    out['capped_diag_51']  = _capped_diag_ens(all_methods)
    out['capped_diag_safe'] = _capped_diag_ens(
        [m for m in all_methods if m not in DANGEROUS])

    # ── Threshold ensemble: filter high-IQR, then equal weight ────────────────
    # threshold_010 : exclude methods with perturb_IQR > 0.10
    # threshold_050 : exclude methods with perturb_IQR > 0.50
    # threshold_safe: exclude only DANGEROUS by name, keep all valid IQR
    def _threshold_ens(pool, iqr_threshold, exclude_names=None):
        if exclude_names is None:
            exclude_names = set()
        ests = []
        for m in pool:
            if m in exclude_names:
                continue
            e = _est(m); q = _iqr(m)
            if not math.isfinite(e):
                continue
            # include if IQR is below threshold OR IQR is NaN
            # (NaN-IQR methods like weniger are always stable-ish; include them
            # only if not excluded by name and not dangerous)
            iqr_ok = (not math.isfinite(q)) or (q <= iqr_threshold)
            if iqr_ok:
                ests.append(e)
        return abs(float(np.median(ests)) - true_val) if ests else float('nan')

    out['threshold_ens_010'] = _threshold_ens(
        all_methods, 0.10, exclude_names=DANGEROUS)
    out['threshold_ens_050'] = _threshold_ens(
        all_methods, 0.50, exclude_names=DANGEROUS)
    out['threshold_ens_safe'] = _threshold_ens(
        all_methods, 1e9, exclude_names=DANGEROUS)   # exclude dangerous only

    return out


SELECTORS = [
    'oracle_51',
    'oracle_9',
    'threshold_ens_010',
    'threshold_ens_050',
    'threshold_ens_safe',
    'capped_diag_51',
    'capped_diag_safe',
    'diag_ensemble_51',
    'diag_ensemble_safe',
    'equal_ensemble_51',
    'equal_ensemble_safe',
    'diag_ensemble_9',
    'equal_ensemble_9',
    'phase2_cascade',
    'fixed_rational',
    'fixed_richardson',
    'current_value',
]

SELECTOR_COLOURS = {
    'oracle_51':          '#000000',
    'oracle_9':           '#444444',
    'threshold_ens_010':  '#2e7d32',
    'threshold_ens_050':  '#66bb6a',
    'threshold_ens_safe': '#a5d6a7',
    'capped_diag_51':     '#1a237e',
    'capped_diag_safe':   '#3949ab',
    'diag_ensemble_51':   '#7986cb',
    'diag_ensemble_safe': '#9fa8da',
    'equal_ensemble_51':  '#1565c0',
    'equal_ensemble_safe':'#42a5f5',
    'diag_ensemble_9':    '#ff9800',
    'equal_ensemble_9':   '#ffc107',
    'phase2_cascade':     '#e65100',
    'fixed_rational':     '#c62828',
    'fixed_richardson':   '#f4a261',
    'current_value':      '#bbbbbb',
}


# =============================================================================
# 3.  AGGREGATION
# =============================================================================

def aggregate_results(df, out_dir):
    all_methods    = list(METHOD_NAMES)
    phase2_methods = PHASE2_METHODS
    fid_default    = df['future_idx'].max()

    ens_rows, wgt_rows = [], []

    for (regime, obs_idx, sigma, seed, fid), grp in df.groupby(
            ['regime', 'obs_idx', 'noise', 'seed', 'future_idx']):

        true_val = float(grp['true_val'].iloc[0])
        slope    = float(grp['casc_slope'].iloc[0])
        r2       = float(grp['casc_r2'].iloc[0])

        out = _compute_ensembles(grp, true_val, all_methods,
                                  phase2_methods, slope, r2)
        row = {'regime': regime, 'obs_idx': obs_idx,
               'noise': sigma, 'seed': seed, 'future_idx': fid}
        row.update(out)
        ens_rows.append(row)

        mi = grp.set_index('method')
        for m in all_methods:
            if m not in mi.index:
                continue
            wgt_rows.append({
                'method':       m,
                'regime':       regime,
                'obs_idx':      obs_idx,
                'noise':        sigma,
                'seed':         seed,
                'future_idx':   fid,
                'perturb_iqr':  float(mi.loc[m, 'perturb_iqr']),
                'error':        float(mi.loc[m, 'error']),
                'is_dangerous': int(m in DANGEROUS),
            })

    df_ens = pd.DataFrame(ens_rows)
    df_wgt = pd.DataFrame(wgt_rows)

    # ── Global comparison ──────────────────────────────────────────────────────
    comp_rows = []
    for fid in sorted(df_ens['future_idx'].unique()):
        sub = df_ens[df_ens['future_idx'] == fid]
        for sel in SELECTORS:
            if sel not in sub.columns:
                continue
            vals = sub[sel].dropna()
            if len(vals) == 0:
                continue
            comp_rows.append({
                'selector':     sel,
                'future_idx':   fid,
                'mean_error':   round(float(vals.mean()),   6),
                'median_error': round(float(vals.median()), 6),
                'n':            len(vals),
            })
    df_comp = pd.DataFrame(comp_rows)
    _save_csv(df_comp, out_dir, 'phase5a_ensemble.csv')

    # ── Per-sigma comparison (key for understanding diagnostic behaviour) ───────
    sigma_rows = []
    for (sigma, fid), sub in df_ens.groupby(['noise', 'future_idx']):
        for sel in SELECTORS:
            if sel not in sub.columns:
                continue
            vals = sub[sel].dropna()
            if len(vals) == 0:
                continue
            sigma_rows.append({
                'noise': sigma, 'selector': sel, 'future_idx': fid,
                'mean_error':   round(float(vals.mean()),   6),
                'median_error': round(float(vals.median()), 6),
                'n': len(vals),
            })
    df_sigma = pd.DataFrame(sigma_rows)
    _save_csv(df_sigma, out_dir, 'phase5a_by_sigma.csv')

    # ── Per-regime ─────────────────────────────────────────────────────────────
    regime_rows = []
    for (regime, fid), sub in df_ens.groupby(['regime', 'future_idx']):
        for sel in SELECTORS:
            if sel not in sub.columns:
                continue
            vals = sub[sel].dropna()
            if len(vals) == 0:
                continue
            regime_rows.append({
                'regime': regime, 'selector': sel, 'future_idx': fid,
                'mean_error':   round(float(vals.mean()),   6),
                'median_error': round(float(vals.median()), 6),
                'n': len(vals),
            })
    df_regime = pd.DataFrame(regime_rows)
    _save_csv(df_regime, out_dir, 'phase5a_per_regime.csv')

    # ── Ablation ───────────────────────────────────────────────────────────────
    abl_rows = []
    for fid in sorted(df_ens['future_idx'].unique()):
        sub = df_ens[df_ens['future_idx'] == fid]
        comparisons = [
            # (better_selector, baseline, label)
            ('threshold_ens_010', 'equal_ensemble_51',  'threshold_vs_equal_51'),
            ('threshold_ens_010', 'fixed_rational',     'threshold_vs_rational'),
            ('capped_diag_51',    'equal_ensemble_51',  'capped_diag_vs_equal'),
            ('diag_ensemble_51',  'equal_ensemble_51',  'diag_vs_equal_51'),
            ('diag_ensemble_9',   'equal_ensemble_9',   'weighting_gain_9'),
            ('threshold_ens_010', 'threshold_ens_safe', 'filter_benefit'),
            ('equal_ensemble_51', 'equal_ensemble_9',   'pool_expansion_gain'),
        ]
        for sel_a, sel_b, label in comparisons:
            if sel_a not in sub.columns or sel_b not in sub.columns:
                continue
            idx = sub[sel_a].notna() & sub[sel_b].notna()
            if idx.sum() == 0:
                continue
            # positive = sel_b - sel_a > 0 → sel_a has LOWER error → sel_a is BETTER
            gain = float((sub.loc[idx, sel_b] - sub.loc[idx, sel_a]).mean())
            abl_rows.append({
                'comparison': label, 'future_idx': fid,
                'mean_improvement': round(gain, 6),
                'n': int(idx.sum()),
            })
    df_abl = pd.DataFrame(abl_rows)
    _save_csv(df_abl, out_dir, 'phase5a_ablation.csv')

    # ── Method reliability ─────────────────────────────────────────────────────
    wgt_agg = (df_wgt[df_wgt['future_idx'] == fid_default]
               .groupby('method')
               .agg(mean_piqr=('perturb_iqr', 'mean'),
                    med_piqr=('perturb_iqr',  'median'),
                    mean_err=('error',          'mean'),
                    is_dangerous=('is_dangerous','first'))
               .reset_index()
               .sort_values('mean_piqr'))
    _save_csv(wgt_agg, out_dir, 'phase5a_method_weights.csv')

    # ── Oracle gap by obs_idx ──────────────────────────────────────────────────
    gap_rows = []
    for (obs_idx, fid), sub in df_ens.groupby(['obs_idx', 'future_idx']):
        for sel in ['threshold_ens_010', 'threshold_ens_safe',
                    'diag_ensemble_51', 'equal_ensemble_51', 'fixed_rational']:
            if sel not in sub.columns:
                continue
            idx = sub[sel].notna() & sub['oracle_51'].notna()
            if idx.sum() == 0:
                continue
            gap = float((sub.loc[idx, sel] - sub.loc[idx, 'oracle_51']).mean())
            gap_rows.append({
                'obs_idx': obs_idx, 'future_idx': fid,
                'selector': sel, 'mean_gap_vs_oracle': round(gap, 6),
                'n': int(idx.sum()),
            })
    df_gap = pd.DataFrame(gap_rows)
    _save_csv(df_gap, out_dir, 'phase5a_oracle_gap.csv')

    return df_ens, df_comp, df_sigma, df_regime, df_abl, wgt_agg, df_gap


# =============================================================================
# 4.  FIGURES
# =============================================================================

def fig_p5a_01_comparison(df_comp, out_dir):
    """Horizontal bar chart of mean error per selector per horizon."""
    horizons = sorted(df_comp['future_idx'].unique())
    fig, axes = plt.subplots(1, len(horizons),
                              figsize=(6*len(horizons), 10), sharey=False)
    if len(horizons) == 1:
        axes = [axes]

    for ax, fid in zip(axes, horizons):
        sub  = df_comp[df_comp['future_idx'] == fid].set_index('selector')
        sels = [s for s in SELECTORS if s in sub.index]
        vals = [float(sub.loc[s, 'mean_error']) for s in sels]
        cols = [SELECTOR_COLOURS.get(s, '#999') for s in sels]

        ax.barh(range(len(sels)), vals, color=cols,
                edgecolor='white', lw=0.4, height=0.65)
        ax.set_yticks(range(len(sels)))
        ax.set_yticklabels([s.replace('_','\n') for s in sels], fontsize=6.5)
        ax.set_xlabel('Mean absolute error', fontsize=9)
        ax.set_title(f'Horizon = {fid}', fontsize=10, fontweight='bold')

        finite_vals = [v for v in vals if math.isfinite(v)]
        mx = max(finite_vals) if finite_vals else 1.0
        for i, v in enumerate(vals):
            if math.isfinite(v):
                ax.text(v + mx*0.01, i, f'{v:.5f}',
                        va='center', fontsize=6.5)

    fig.suptitle(
        'Figure P5A-1 — Full 51-Method Ensemble: Selector Comparison\n'
        'Dark green = threshold ensembles (Phase 5A);  '
        'Black = oracle upper bounds',
        fontsize=10, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(out_dir, 'figure_p5a_01_comparison.png')
    _save(fig, path)
    return path


def fig_p5a_02_by_sigma(df_sigma, out_dir):
    """
    Key figure: mean error by sigma level for threshold_ens_010,
    diag_ensemble_51, equal_ensemble_51, fixed_rational.
    Shows whether diagnostic adds value at higher noise.
    """
    key_sels = ['oracle_51', 'threshold_ens_010', 'capped_diag_51',
                 'equal_ensemble_51', 'fixed_rational', 'fixed_richardson']
    key_sels = [s for s in key_sels if s in df_sigma['selector'].unique()]
    sigmas   = sorted(df_sigma['noise'].unique())
    fid      = df_sigma['future_idx'].max()
    sub      = df_sigma[df_sigma['future_idx'] == fid]

    fig, ax = plt.subplots(figsize=(10, 6))
    for sel in key_sels:
        sv = sub[sub['selector'] == sel].sort_values('noise')
        if sv.empty:
            continue
        ax.plot(sv['noise'], sv['mean_error'],
                'o-', label=sel.replace('_',' '),
                color=SELECTOR_COLOURS.get(sel, '#999'),
                lw=2, markersize=7, alpha=0.9)

    ax.set_xlabel('Noise level (sigma)', fontsize=10)
    ax.set_ylabel('Mean absolute error', fontsize=10)
    ax.set_title(
        f'Figure P5A-2 — Selector Performance by Noise Level  '
        f'(horizon = {fid})\n'
        'Key question: does the diagnostic add value at higher sigma?',
        fontsize=10, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')
    fig.tight_layout()
    path = os.path.join(out_dir, 'figure_p5a_02_by_sigma.png')
    _save(fig, path)
    return path


def fig_p5a_03_method_weights(wgt_agg, out_dir):
    """Method reliability ranking by mean perturb_IQR."""
    df  = wgt_agg.copy().reset_index(drop=True)
    max_iqr = df['mean_piqr'].replace([float('nan')], float('nan')).max(skipna=True)
    plot_val = df['mean_piqr'].fillna(max_iqr * 1.1)
    colours  = ['#c62828' if row['is_dangerous'] else '#1565c0'
                for _, row in df.iterrows()]

    fig, ax = plt.subplots(figsize=(14, max(8, len(df) * 0.22)))
    ax.barh(range(len(df)), plot_val,
            color=colours, edgecolor='white', lw=0.3, height=0.75)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['method'], fontsize=6.5)
    ax.set_xlabel(
        'Mean perturb_IQR  (lower = more stable = higher diagnostic weight)\n'
        'NaN plotted at max+10%',
        fontsize=9)
    ax.set_title(
        'Figure P5A-3 — Method Reliability Ranking\n'
        'Red = Phase 1 dangerous methods.  '
        'Validation: ALL dangerous methods should be at right (high IQR).',
        fontsize=10, fontweight='bold')
    ax.legend(handles=[
        mpatches.Patch(color='#c62828', label='Dangerous (Phase 1)'),
        mpatches.Patch(color='#1565c0', label='Safe'),
    ], fontsize=9, loc='lower right')
    fig.tight_layout()
    path = os.path.join(out_dir, 'figure_p5a_03_method_weights.png')
    _save(fig, path)
    return path


def fig_p5a_04_obs_gap(df_gap, out_dir):
    """Oracle gap vs obs_idx for key selectors."""
    obs_vals = sorted(df_gap['obs_idx'].unique())
    key_sels = ['threshold_ens_010', 'threshold_ens_safe',
                 'equal_ensemble_51', 'fixed_rational']
    key_sels = [s for s in key_sels if s in df_gap['selector'].unique()]
    fid      = df_gap['future_idx'].max()
    sub      = df_gap[df_gap['future_idx'] == fid]

    fig, ax = plt.subplots(figsize=(10, 6))
    for sel in key_sels:
        sv = sub[sub['selector'] == sel].sort_values('obs_idx')
        if sv.empty:
            continue
        ax.plot(sv['obs_idx'], sv['mean_gap_vs_oracle'],
                'o-', label=sel.replace('_',' '),
                color=SELECTOR_COLOURS.get(sel, '#999'),
                lw=2, markersize=6, alpha=0.9)

    ax.axhline(0, color='black', lw=0.8, ls='--', alpha=0.4,
               label='Oracle (gap = 0)')
    ax.set_xlabel('obs_idx  (observation depth)', fontsize=10)
    ax.set_ylabel('Mean error gap vs oracle_51', fontsize=10)
    ax.set_title(
        f'Figure P5A-4 — Oracle Gap vs Observation Depth  (horizon = {fid})',
        fontsize=10, fontweight='bold')
    ax.set_xticks(obs_vals)
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = os.path.join(out_dir, 'figure_p5a_04_obs_gap.png')
    _save(fig, path)
    return path


# =============================================================================
# MASTER RUN FUNCTION
# =============================================================================

def run_all(obs_idx_list, noise_list, future_list, n_seeds,
            window_len, perturb_trials, perturb_scale,
            out_dir, verbose=True):
    os.makedirs(out_dir, exist_ok=True)
    default_fid = max(future_list)

    df = run_phase5a(obs_idx_list, noise_list, future_list,
                     n_seeds, window_len, perturb_trials,
                     perturb_scale, out_dir, verbose)

    print('\n  Aggregating ensemble results ...')
    df_ens, df_comp, df_sigma, df_regime, df_abl, wgt_agg, df_gap = \
        aggregate_results(df, out_dir)

    print('\n  Generating figures ...')
    paths = [
        fig_p5a_01_comparison(df_comp, out_dir),
        fig_p5a_02_by_sigma(df_sigma, out_dir),
        fig_p5a_03_method_weights(wgt_agg, out_dir),
        fig_p5a_04_obs_gap(df_gap, out_dir),
    ]

    return {
        'raw': df, 'ensemble': df_ens, 'comparison': df_comp,
        'by_sigma': df_sigma, 'regime': df_regime,
        'ablation': df_abl, 'weights': wgt_agg, 'gap': df_gap,
        'figures': [p for p in paths if p],
    }
