"""
phase5a.py  (v4, redesign v2)
=============================
Phase 5A — Full-Pool Ensemble with Ablation.

Key ablation questions
-----------------------
Q1. Does the 51-accelerator pool beat the 9-method pool?  diag_9 vs diag_51
Q2. Does weighting beat equal?         equal_51 vs diag_51
Q3. Does threshold filter beat equal?  equal_51 vs threshold_ens_010
Q4. How much does oracle improve?      oracle_9 vs oracle_51
Q5. Where is the residual gap?         per-regime decomposition
Q6. Are dangerous methods auto-IDed?   method weight ranking

Redesign v2
-----------
  * Evaluation points are the three gap strata per (regime, obs_idx); every
    raw record carries target_g, achieved_g, n_f, capped, L_true, L_hat,
    skill, is_trivial, is_oracle, is_holdout, is_dangerous.
  * The ensemble / oracle pool is the 51 accelerators.  The five trivial
    comparators are evaluated and reported as fixed reference selectors
    (constant_oracle labelled) but never mixed into an ensemble.
  * The dangerous set is read from the Phase-1 artifact
    (src.dangerous.load_dangerous); the phase refuses to run without it.
  * Core and held-out regimes are evaluated; pooled comparisons use the core
    regimes with capped cells excluded; held-out and capped blocks are
    written separately.  Every selector row carries a median skill.

Author : Kian Maleki
Date   : 2026-05-24 (v1), 2026-09-19 (redesign v2)
"""

import os, math, warnings
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
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
from src.dangerous    import load_dangerous
from src.pipeline     import (ACCEL_METHODS, capped_block, exclude_capped,
                              horizon_meta, is_holdout, method_flags,
                              resolve_regimes)
from src.trivial      import (TRIVIAL_METHOD_NAMES, best_reference_error,
                              skill_score)

PHASE2_METHODS = [
    'current_value', 'richardson_1', 'richardson_a10',
    'single_exp_fit', 'rational_fit', 'pade_22',
    'log_linear', 'weniger_d2', 'anderson_1',
]

POOL         = list(ACCEL_METHODS)                 # 51 accelerators (ensembles, oracles)
EVAL_METHODS = POOL + list(TRIVIAL_METHOD_NAMES)   # + 5 trivial comparators (reference rows)
TRIVIAL_SELECTORS = list(TRIVIAL_METHOD_NAMES)

EPS     = 0.01   # for continuous weighting
FIG_DPI = 150
CELL    = ['regime', 'obs_idx', 'noise', 'seed', 'target_g']


# ── helpers ───────────────────────────────────────────────────────────────────
def _cfg(fid, L_hat, L_true=None):
    # L_hat is the ASSUMED asymptote (src.asymptote); L_true only feeds the
    # constant_oracle comparator through cfg['L_true'].
    cfg = {'L_inf': float(L_hat), 'ridge': CFG_MOD.RIDGE,
           'min_valid': CFG_MOD.MIN_VALID, 'max_valid': CFG_MOD.MAX_VALID,
           'denom_tol': CFG_MOD.DENOM_TOL}
    if L_true is not None:
        cfg['L_true'] = float(L_true)
    return cfg

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

def _headline(df, default_g):
    gs = set(df['target_g'].unique())
    g = CFG_MOD.HEADLINE_G if default_g is None else float(default_g)
    return g if g in gs else float(max(gs))


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
def _cascade_features(seq_win, idx_win, L_hat):
    s  = np.asarray(seq_win, dtype=float)
    x  = np.asarray(idx_win, dtype=float)
    L0 = max(0.0, min(float(L_hat), float(np.min(s)) * 0.5))

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

def run_phase5a(obs_idx_list, noise_list, gap_fractions, n_seeds,
                window_len, perturb_trials, perturb_scale,
                out_dir, dangerous, core_regimes=None, holdout_regimes=None,
                verbose=True):
    os.makedirs(out_dir, exist_ok=True)
    regimes = resolve_regimes(core_regimes, holdout_regimes, include_holdout=True)
    gap_fractions = [float(g) for g in gap_fractions]
    dangerous = set(dangerous)

    n_arr   = np.arange(max(obs_idx_list) + 1, dtype=float)
    n_total = (len(obs_idx_list) * len(noise_list) * n_seeds * len(regimes))
    done    = 0
    records = []

    print(f'  Pool (ensembles): {len(POOL)} accelerators  '
          f'({len(dangerous & set(POOL))} dangerous per artifact)')
    print(f'  Reference rows  : {len(TRIVIAL_METHOD_NAMES)} trivial comparators '
          f'(oracle labelled, never pooled)')
    print(f'  Progress updates: every {max(1, n_total // 20)} '
          f'regime-groups  (~5% increments)\n')

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
                    slope, r2 = _cascade_features(seq_win, idx_win, L_hat)
                    hold     = is_holdout(regime)

                    for g in gap_fractions:
                        hm       = horizon_meta(regime, obs_idx, g, seed)
                        n_f      = hm['n_f']
                        true_val = float(truth_fn(n_f))
                        curr_err = abs(curr_val - true_val)
                        cfg      = _cfg(n_f, L_hat, L_true)

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
                            p_iqr = (_perturb_iqr(seq_win, idx_win, float(n_f), method, cfg,
                                                  perturb_trials, perturb_scale, rng_p)
                                     if method in POOL else float('nan'))
                            rec = {
                                'regime':       regime,
                                'is_holdout':   hold,
                                'obs_idx':      obs_idx,
                                'noise':        sigma,
                                'seed':         seed,
                                'L_true':       L_true,
                                'L_hat':        L_hat,
                                'method':       method,
                                'true_val':     true_val,
                                'estimate':     est if valid else float('nan'),
                                'error':        err,
                                'valid':        int(valid),
                                'catastrophic': int(cat),
                                'curr_err':     curr_err,
                                'ref_error':    ref_err,
                                'skill':        skill_score(err, ref_err) if valid else float('nan'),
                                'perturb_iqr':  p_iqr,
                                'is_dangerous': int(method in dangerous),
                                'casc_slope':   slope,
                                'casc_r2':      r2,
                            }
                            rec.update(hm)
                            rec.update(method_flags(method))
                            records.append(rec)

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

def _compute_ensembles(grp, true_val, pool, phase2_methods, slope, r2, dangerous):
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

    # ── Trivial comparators as fixed reference selectors (oracle labelled) ─────
    for m in TRIVIAL_SELECTORS:
        out[m] = _err(m)

    # ── Oracles ───────────────────────────────────────────────────────────────
    e9  = [_err(m) for m in phase2_methods if math.isfinite(_err(m))]
    e51 = [_err(m) for m in pool           if math.isfinite(_err(m))]
    out['oracle_9']  = min(e9)  if e9  else float('nan')
    out['oracle_51'] = min(e51) if e51 else float('nan')

    safe = [m for m in pool if m not in dangerous]

    # ── Equal-weight ensembles (median) ────────────────────────────────────────
    def _equal_ens(p_):
        ests = [_est(m) for m in p_ if math.isfinite(_est(m))]
        return abs(float(np.median(ests)) - true_val) if ests else float('nan')

    out['equal_ensemble_51']   = _equal_ens(pool)
    out['equal_ensemble_9']    = _equal_ens(phase2_methods)
    out['equal_ensemble_safe'] = _equal_ens(safe)

    # ── Diagnostic-weighted ensemble (continuous 1/IQR, EPS=0.01) ─────────────
    def _diag_ens(p_):
        ests, ws = [], []
        for m in p_:
            e = _est(m); q = _iqr(m)
            if math.isfinite(e) and math.isfinite(q):
                ests.append(e); ws.append(1.0 / (q + EPS))
        if not ests:
            return float('nan')
        w = np.array(ws); w /= w.sum()
        return abs(float(np.dot(w, ests)) - true_val)

    out['diag_ensemble_51']   = _diag_ens(pool)
    out['diag_ensemble_9']    = _diag_ens(phase2_methods)
    out['diag_ensemble_safe'] = _diag_ens(safe)

    # ── Capped diagnostic ensemble (weight <= 5x median weight) ───────────────
    def _capped_diag_ens(p_):
        ests, ws = [], []
        for m in p_:
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

    out['capped_diag_51']   = _capped_diag_ens(pool)
    out['capped_diag_safe'] = _capped_diag_ens(safe)

    # ── Threshold ensemble: filter high-IQR, then equal weight ────────────────
    def _threshold_ens(p_, iqr_threshold, exclude_names=None):
        if exclude_names is None:
            exclude_names = set()
        ests = []
        for m in p_:
            if m in exclude_names:
                continue
            e = _est(m); q = _iqr(m)
            if not math.isfinite(e):
                continue
            iqr_ok = (not math.isfinite(q)) or (q <= iqr_threshold)
            if iqr_ok:
                ests.append(e)
        return abs(float(np.median(ests)) - true_val) if ests else float('nan')

    out['threshold_ens_010']  = _threshold_ens(pool, 0.10, exclude_names=dangerous)
    out['threshold_ens_050']  = _threshold_ens(pool, 0.50, exclude_names=dangerous)
    out['threshold_ens_safe'] = _threshold_ens(pool, 1e9,  exclude_names=dangerous)

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
] + TRIVIAL_SELECTORS

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
    'constant_assumed':   '#212121',
    'constant_oracle':    '#9e9e9e',
    'window_mean':        '#616161',
    'window_min':         '#757575',
    'last_value':         '#bdbdbd',
}


# =============================================================================
# 3.  AGGREGATION
# =============================================================================

def _selector_summary(sub: pd.DataFrame, extra: dict) -> List[dict]:
    """mean / median error and median skill per selector on one slice."""
    rows = []
    refs = sub['ref_error'].to_numpy(dtype=float)
    for sel in SELECTORS:
        if sel not in sub.columns:
            continue
        vals = sub[sel].to_numpy(dtype=float)
        ok = np.isfinite(vals)
        if not ok.any():
            continue
        sk = np.array([skill_score(v, r) for v, r in zip(vals, refs)], dtype=float)
        sk = sk[~np.isnan(sk)]
        row = dict(extra)
        row.update({
            'selector':     sel,
            'is_trivial':   int(sel in TRIVIAL_SELECTORS),
            'is_oracle':    int(sel == 'constant_oracle'),
            'mean_error':   round(float(vals[ok].mean()),   6),
            'median_error': round(float(np.median(vals[ok])), 6),
            'med_skill':    round(float(np.median(sk)), 4) if sk.size else float('nan'),
            'n':            int(ok.sum()),
        })
        rows.append(row)
    return rows


def aggregate_results(df, out_dir, dangerous, default_g=None):
    pool           = POOL
    phase2_methods = PHASE2_METHODS
    dangerous      = set(dangerous)
    g_head         = _headline(df, default_g)

    ens_rows, wgt_rows = [], []

    for (regime, obs_idx, sigma, seed, g), grp in df.groupby(CELL):
        true_val = float(grp['true_val'].iloc[0])
        slope    = float(grp['casc_slope'].iloc[0])
        r2       = float(grp['casc_r2'].iloc[0])
        first    = grp.iloc[0]

        out = _compute_ensembles(grp, true_val, pool, phase2_methods, slope, r2, dangerous)
        row = {'regime': regime, 'is_holdout': int(first['is_holdout']),
               'obs_idx': obs_idx, 'noise': sigma, 'seed': seed, 'target_g': g,
               'n_f': float(first['n_f']), 'achieved_g': float(first['achieved_g']),
               'capped': int(first['capped']),
               'L_true': float(first['L_true']), 'L_hat': float(first['L_hat']),
               'ref_error': float(first['ref_error'])}
        row.update(out)
        ens_rows.append(row)

        mi = grp.set_index('method')
        for m in pool:
            if m not in mi.index:
                continue
            wgt_rows.append({
                'method':       m,
                'regime':       regime,
                'is_holdout':   int(first['is_holdout']),
                'obs_idx':      obs_idx,
                'noise':        sigma,
                'seed':         seed,
                'target_g':     g,
                'capped':       int(first['capped']),
                'perturb_iqr':  float(mi.loc[m, 'perturb_iqr']),
                'error':        float(mi.loc[m, 'error']),
                'skill':        float(mi.loc[m, 'skill']),
                'is_dangerous': int(m in dangerous),
            })

    df_ens = pd.DataFrame(ens_rows)
    df_wgt = pd.DataFrame(wgt_rows)

    core = exclude_capped(df_ens[df_ens['is_holdout'] == 0])
    hold = exclude_capped(df_ens[df_ens['is_holdout'] == 1])

    # ── Global comparison (core, capped excluded) ──────────────────────────────
    comp_rows = []
    for g in sorted(core['target_g'].unique()):
        comp_rows += _selector_summary(core[core['target_g'] == g], {'target_g': g, 'regime_set': 'core'})
    df_comp = pd.DataFrame(comp_rows)
    _save_csv(df_comp, out_dir, 'phase5a_ensemble.csv')

    hold_rows = []
    for g in sorted(hold['target_g'].unique()):
        hold_rows += _selector_summary(hold[hold['target_g'] == g], {'target_g': g, 'regime_set': 'holdout'})
    df_hold = pd.DataFrame(hold_rows, columns=df_comp.columns if len(df_comp) else None)
    _save_csv(df_hold, out_dir, 'phase5a_ensemble_holdout.csv')

    # ── Per-sigma comparison (core) ────────────────────────────────────────────
    sigma_rows = []
    for (sigma, g), sub in core.groupby(['noise', 'target_g']):
        sigma_rows += _selector_summary(sub, {'noise': sigma, 'target_g': g})
    df_sigma = pd.DataFrame(sigma_rows)
    _save_csv(df_sigma, out_dir, 'phase5a_by_sigma.csv')

    # ── Per-regime (all regimes, capped cells excluded per regime) ─────────────
    regime_rows = []
    pooled_all = exclude_capped(df_ens)
    for (regime, g), sub in pooled_all.groupby(['regime', 'target_g']):
        n_all = int(((df_ens['regime'] == regime) & (df_ens['target_g'] == g)).sum())
        regime_rows += _selector_summary(sub, {
            'regime': regime, 'is_holdout': int(sub['is_holdout'].iloc[0]),
            'target_g': g, 'n_capped_excluded': n_all - int(len(sub))})
    df_regime = pd.DataFrame(regime_rows)
    _save_csv(df_regime, out_dir, 'phase5a_per_regime.csv')

    # ── Ablation (core) ────────────────────────────────────────────────────────
    abl_rows = []
    for g in sorted(core['target_g'].unique()):
        sub = core[core['target_g'] == g]
        comparisons = [
            ('threshold_ens_010', 'equal_ensemble_51',  'threshold_vs_equal_51'),
            ('threshold_ens_010', 'fixed_rational',     'threshold_vs_rational'),
            ('capped_diag_51',    'equal_ensemble_51',  'capped_diag_vs_equal'),
            ('diag_ensemble_51',  'equal_ensemble_51',  'diag_vs_equal_51'),
            ('diag_ensemble_9',   'equal_ensemble_9',   'weighting_gain_9'),
            ('threshold_ens_010', 'threshold_ens_safe', 'filter_benefit'),
            ('equal_ensemble_51', 'equal_ensemble_9',   'pool_expansion_gain'),
            ('threshold_ens_010', 'constant_assumed',   'threshold_vs_constant_assumed'),
            ('fixed_rational',    'window_min',         'rational_vs_window_min'),
        ]
        for sel_a, sel_b, label in comparisons:
            if sel_a not in sub.columns or sel_b not in sub.columns:
                continue
            idx = sub[sel_a].notna() & sub[sel_b].notna()
            if idx.sum() == 0:
                continue
            gain = float((sub.loc[idx, sel_b] - sub.loc[idx, sel_a]).mean())
            abl_rows.append({
                'comparison': label, 'target_g': g,
                'mean_improvement': round(gain, 6),
                'n': int(idx.sum()),
            })
    df_abl = pd.DataFrame(abl_rows)
    _save_csv(df_abl, out_dir, 'phase5a_ablation.csv')

    # ── Method reliability (core, headline stratum) ────────────────────────────
    wsub = exclude_capped(df_wgt[(df_wgt['is_holdout'] == 0) & (df_wgt['target_g'] == g_head)])
    wgt_agg = (wsub.groupby('method')
                   .agg(mean_piqr=('perturb_iqr', 'mean'),
                        med_piqr=('perturb_iqr',  'median'),
                        mean_err=('error',          'mean'),
                        med_skill=('skill',         'median'),
                        is_dangerous=('is_dangerous','first'))
                   .reset_index()
                   .sort_values('mean_piqr'))
    wgt_agg['target_g'] = g_head
    _save_csv(wgt_agg, out_dir, 'phase5a_method_weights.csv')

    # ── Oracle gap by obs_idx (core) ───────────────────────────────────────────
    gap_rows = []
    for (obs_idx, g), sub in core.groupby(['obs_idx', 'target_g']):
        for sel in ['threshold_ens_010', 'threshold_ens_safe',
                    'diag_ensemble_51', 'equal_ensemble_51', 'fixed_rational',
                    'constant_assumed']:
            if sel not in sub.columns:
                continue
            idx = sub[sel].notna() & sub['oracle_51'].notna()
            if idx.sum() == 0:
                continue
            gap = float((sub.loc[idx, sel] - sub.loc[idx, 'oracle_51']).mean())
            gap_rows.append({
                'obs_idx': obs_idx, 'target_g': g,
                'selector': sel, 'mean_gap_vs_oracle': round(gap, 6),
                'n': int(idx.sum()),
            })
    df_gap = pd.DataFrame(gap_rows)
    _save_csv(df_gap, out_dir, 'phase5a_oracle_gap.csv')

    # ── Capped block ───────────────────────────────────────────────────────────
    df_cap = capped_block(df, keys=['regime', 'is_holdout', 'obs_idx', 'target_g', 'method'],
                          value_cols=['error', 'skill'])
    _save_csv(df_cap, out_dir, 'phase5a_capped.csv')

    return df_ens, df_comp, df_sigma, df_regime, df_abl, wgt_agg, df_gap, df_cap


# =============================================================================
# 4.  FIGURES
# =============================================================================

def fig_p5a_01_comparison(df_comp, out_dir):
    """Horizontal bar chart of mean error per selector per stratum."""
    if df_comp.empty:
        return ''
    strata = sorted(df_comp['target_g'].unique(), reverse=True)
    fig, axes = plt.subplots(1, len(strata), figsize=(6*len(strata), 10), sharey=False)
    if len(strata) == 1:
        axes = [axes]

    for ax, g in zip(axes, strata):
        sub  = df_comp[df_comp['target_g'] == g].set_index('selector')
        sels = [s for s in SELECTORS if s in sub.index]
        vals = [float(sub.loc[s, 'mean_error']) for s in sels]
        cols = [SELECTOR_COLOURS.get(s, '#999') for s in sels]

        ax.barh(range(len(sels)), vals, color=cols, edgecolor='white', lw=0.4, height=0.65)
        ax.set_yticks(range(len(sels)))
        ax.set_yticklabels([s.replace('_','\n') for s in sels], fontsize=6.5)
        ax.set_xlabel('Mean absolute error', fontsize=9)
        ax.set_title(f'g = {g:g}', fontsize=10, fontweight='bold')

        finite_vals = [v for v in vals if math.isfinite(v)]
        mx = max(finite_vals) if finite_vals else 1.0
        for i, v in enumerate(vals):
            if math.isfinite(v):
                ax.text(v + mx*0.01, i, f'{v:.5f}', va='center', fontsize=6.5)

    fig.suptitle(
        'Figure P5A-1 — Full-Pool Ensemble: Selector Comparison (core, capped excluded)\n'
        'Dark green = threshold ensembles;  Black = oracle upper bounds;  '
        'greys = trivial reference selectors',
        fontsize=10, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(out_dir, 'figure_p5a_01_comparison.png')
    _save(fig, path)
    return path


def fig_p5a_02_by_sigma(df_sigma, out_dir, default_g=None):
    """Mean error by sigma level for key selectors at the headline stratum."""
    if df_sigma.empty:
        return ''
    key_sels = ['oracle_51', 'threshold_ens_010', 'capped_diag_51',
                'equal_ensemble_51', 'fixed_rational', 'fixed_richardson',
                'constant_assumed']
    key_sels = [s for s in key_sels if s in df_sigma['selector'].unique()]
    g        = _headline(df_sigma, default_g)
    sub      = df_sigma[df_sigma['target_g'] == g]

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
        f'Figure P5A-2 — Selector Performance by Noise Level  (g = {g:g})\n'
        'Key question: does the diagnostic add value at higher sigma?',
        fontsize=10, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')
    fig.tight_layout()
    path = os.path.join(out_dir, 'figure_p5a_02_by_sigma.png')
    _save(fig, path)
    return path


def fig_p5a_03_method_weights(wgt_agg, out_dir):
    """Method reliability ranking by mean perturb_IQR."""
    if wgt_agg.empty:
        return ''
    df  = wgt_agg.copy().reset_index(drop=True)
    max_iqr = df['mean_piqr'].max(skipna=True)
    plot_val = df['mean_piqr'].fillna(max_iqr * 1.1 if math.isfinite(max_iqr) else 1.0)
    colours  = ['#c62828' if row['is_dangerous'] else '#1565c0'
                for _, row in df.iterrows()]

    fig, ax = plt.subplots(figsize=(14, max(8, len(df) * 0.22)))
    ax.barh(range(len(df)), plot_val, color=colours, edgecolor='white', lw=0.3, height=0.75)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['method'], fontsize=6.5)
    ax.set_xlabel(
        'Mean perturb_IQR  (lower = more stable = higher diagnostic weight)\n'
        'NaN plotted at max+10%',
        fontsize=9)
    ax.set_title(
        'Figure P5A-3 — Method Reliability Ranking\n'
        'Red = dangerous per the Phase-1 artifact.  '
        'Validation: dangerous methods should sit at the right (high IQR).',
        fontsize=10, fontweight='bold')
    ax.legend(handles=[
        mpatches.Patch(color='#c62828', label='Dangerous (Phase-1 artifact)'),
        mpatches.Patch(color='#1565c0', label='Safe'),
    ], fontsize=9, loc='lower right')
    fig.tight_layout()
    path = os.path.join(out_dir, 'figure_p5a_03_method_weights.png')
    _save(fig, path)
    return path


def fig_p5a_04_obs_gap(df_gap, out_dir, default_g=None):
    """Oracle gap vs obs_idx for key selectors."""
    if df_gap.empty:
        return ''
    obs_vals = sorted(df_gap['obs_idx'].unique())
    key_sels = ['threshold_ens_010', 'threshold_ens_safe',
                'equal_ensemble_51', 'fixed_rational', 'constant_assumed']
    key_sels = [s for s in key_sels if s in df_gap['selector'].unique()]
    g        = _headline(df_gap, default_g)
    sub      = df_gap[df_gap['target_g'] == g]

    fig, ax = plt.subplots(figsize=(10, 6))
    for sel in key_sels:
        sv = sub[sub['selector'] == sel].sort_values('obs_idx')
        if sv.empty:
            continue
        ax.plot(sv['obs_idx'], sv['mean_gap_vs_oracle'],
                'o-', label=sel.replace('_',' '),
                color=SELECTOR_COLOURS.get(sel, '#999'),
                lw=2, markersize=6, alpha=0.9)

    ax.axhline(0, color='black', lw=0.8, ls='--', alpha=0.4, label='Oracle (gap = 0)')
    ax.set_xlabel('obs_idx  (observation depth)', fontsize=10)
    ax.set_ylabel('Mean error gap vs oracle_51', fontsize=10)
    ax.set_title(f'Figure P5A-4 — Oracle Gap vs Observation Depth  (g = {g:g})',
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

def run_all(obs_idx_list, noise_list, gap_fractions, n_seeds,
            window_len, perturb_trials, perturb_scale,
            out_dir, core_regimes=None, holdout_regimes=None,
            default_g=None, verbose=True):
    os.makedirs(out_dir, exist_ok=True)

    # Ordering guard: the dangerous set comes from the Phase-1 artifact.
    dangerous = load_dangerous()
    print(f'  Dangerous set (Phase-1 artifact): {sorted(dangerous)}')

    df = run_phase5a(obs_idx_list, noise_list, gap_fractions, n_seeds,
                     window_len, perturb_trials, perturb_scale, out_dir,
                     dangerous, core_regimes, holdout_regimes, verbose)

    print('\n  Aggregating ensemble results ...')
    (df_ens, df_comp, df_sigma, df_regime, df_abl,
     wgt_agg, df_gap, df_cap) = aggregate_results(df, out_dir, dangerous, default_g)
    g_head = _headline(df, default_g)

    print('\n  Generating figures ...')
    paths = [
        fig_p5a_01_comparison(df_comp, out_dir),
        fig_p5a_02_by_sigma(df_sigma, out_dir, g_head),
        fig_p5a_03_method_weights(wgt_agg, out_dir),
        fig_p5a_04_obs_gap(df_gap, out_dir, g_head),
    ]

    return {
        'raw': df, 'ensemble': df_ens, 'comparison': df_comp,
        'by_sigma': df_sigma, 'regime': df_regime,
        'ablation': df_abl, 'weights': wgt_agg, 'gap': df_gap,
        'capped': df_cap, 'dangerous': dangerous, 'default_g': g_head,
        'figures': [p for p in paths if p],
    }
