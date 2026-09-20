"""
phase5b.py
==========
Phase 5B — Sensitivity Analysis (redesign v2).

Three sweeps test whether the key findings are robust to the main modelling
assumptions.  All sweeps evaluate at the gap strata in
config.PHASE5B_GAP_FRACTIONS (g = 0.5 and 0.1), with n_obs = 90.

Sweep 1 — assumed-asymptote (L_hat) sensitivity
-----------------------------------------------
Question: Does the Phase 2 two-rule cascade still work when what the
methods are told about the asymptote changes?  L_true is hidden and per
(regime, seed); the ASSUMED value is swept over config.ASSUMED_L_MODES
= {zero, half, oracle, double, winmin} (oracle = L_true, labelled).

For each mode, trajectory features (log_log_slope, richardson_r2) are
recomputed with that L_hat and the cascade is applied with fixed Phase 2
thresholds (slope > -0.1, R^2 < 0.5).

Sweep 2 — Window length sensitivity  (window_len in {20, 40, 60, 80, 100})
Sweep 3 — CAT_MULT sensitivity       (CAT_MULT in {2, 5, 10})

Redesign v2
-----------
  * Records carry target_g, achieved_g, n_f, capped and is_holdout.  Global
    rows are pooled over the core regimes with capped cells excluded
    (regime_set = 'core'); held-out regimes get their own pooled rows
    (regime_set = 'holdout'); per-regime rows flag capped cells.
  * Sweep 3 ranks the 51 accelerators plus the four non-oracle trivial
    comparators; the oracle never enters a ranking.  A method takes a rank
    only with valid_rate >= config.RANK_MIN_VALID (rank / rank_eligible
    columns); below-floor methods are listed in phase5b_sweep3_unranked.csv
    and the concordance is computed over ranked methods.  The dangerous flag
    comes from the Phase-1 artifact.

Output files
------------
phase5b_sweep1_global.csv      Cascade metrics vs assumed-asymptote mode (pooled)
phase5b_sweep1_regime.csv      Cascade metrics vs assumed-asymptote mode (per regime)
phase5b_sweep2_global.csv      Cascade metrics vs window_len (pooled)
phase5b_sweep2_regime.csv      Cascade metrics vs window_len (per regime)
phase5b_sweep3_champions.csv   Regime champions at each CAT_MULT
phase5b_sweep3_global.csv      Global stability rankings at each CAT_MULT
phase5b_sweep3_unranked.csv    Below-floor methods (unranked, valid_rate shown)
phase5b_sweep3_concordance.csv Kendall tau between CAT_MULT rankings
figure_p5b_01_linf.png / figure_p5b_02_window.png / figure_p5b_03_catmult.png

Author : Kian Maleki
Date   : 2026-05-24 (v1), 2026-09-19 (redesign v2)
"""

import os, math, warnings
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
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
from src.dangerous    import load_dangerous
from src.pipeline     import (ACCEL_METHODS, TRIVIAL_NON_ORACLE, assign_ranks,
                              exclude_capped, horizon_meta, is_holdout,
                              method_flags, resolve_regimes, unranked_block)

# ── Method sets ────────────────────────────────────────────────────────────────
CASCADE_METHODS = [
    'current_value', 'richardson_1', 'richardson_a10',
    'single_exp_fit', 'rational_fit', 'pade_22',
    'log_linear', 'weniger_d2', 'anderson_1',
]

# Sweep 3 ranks accelerators and the deployable trivial comparators; no oracle.
ALL_METHODS = list(ACCEL_METHODS) + list(TRIVIAL_NON_ORACLE)

FIG_DPI = 150


# ── Helpers ────────────────────────────────────────────────────────────────────
def _cfg(fid, L_hat, cat_mult=None):
    """L_hat is the ASSUMED asymptote (src.asymptote), never L_true."""
    return {
        'L_inf':     float(L_hat),
        'ridge':     CFG_MOD.RIDGE,
        'min_valid': CFG_MOD.MIN_VALID,
        'max_valid': CFG_MOD.MAX_VALID,
        'denom_tol': CFG_MOD.DENOM_TOL,
        'W_CAT':     CFG_MOD.W_CAT,
        'W_BEATS':   CFG_MOD.W_BEATS,
        'CAT_MULT':  cat_mult if cat_mult is not None else CFG_MOD.CAT_MULT,
    }

def _valid(v, cfg):
    return bool(math.isfinite(v) and
                cfg['min_valid'] <= v <= cfg['max_valid'])

def _stability(vr, cr, br, cfg):
    return vr - cfg['W_CAT'] * cr + cfg['W_BEATS'] * br

def _save(fig, path):
    fig.savefig(path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')

def _save_csv(df, out_dir, fname):
    p = os.path.join(out_dir, fname)
    df.to_csv(p, index=False)
    print(f'  Saved: {p}  ({len(df)} rows)')

def _headline(df, default_g=None):
    gs = set(df['target_g'].unique())
    g = CFG_MOD.HEADLINE_G if default_g is None else float(default_g)
    return g if g in gs else float(max(gs))


# ── Feature extraction (inline with configurable L_hat) ───────────────────────
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


# ── Core evaluation helper ────────────────────────────────────────────────────
def _eval_methods(seq_win, idx_win, fid, cfg, methods):
    """Return dict method→estimate for a single sequence."""
    out = {}
    for m in methods:
        fn = METHODS[m]
        try:
            v = fn(seq_win, idx_win, float(fid), cfg)
        except Exception:
            v = float('nan')
        out[m] = v if _valid(v, cfg) else float('nan')
    return out


def _cascade_metrics(sub: pd.DataFrame) -> Dict[str, float]:
    fire    = sub['cascade_fired'] == 1
    correct = fire & (sub['rat_better'] == 1)
    precision = (float(correct.sum() / fire.sum()) if fire.sum() > 0 else float('nan'))
    recall    = (float(correct.sum() / sub['rat_better'].sum())
                 if sub['rat_better'].sum() > 0 else float('nan'))
    mask = sub['rich_err'].notna() & sub['chosen_err'].notna()
    gain = (float((sub.loc[mask, 'rich_err'] - sub.loc[mask, 'chosen_err']).mean())
            if mask.sum() > 0 else float('nan'))
    return {
        'fire_rate': round(float(fire.mean()), 4),
        'precision': round(precision, 4) if math.isfinite(precision) else float('nan'),
        'recall':    round(recall, 4) if math.isfinite(recall) else float('nan'),
        'mean_gain': round(gain, 6) if math.isfinite(gain) else float('nan'),
        'n':         int(len(sub)),
    }


def _cascade_cell_record(regime, sigma, seed, hm, L_true, L_hat, slope, r2,
                         chosen, cascade_fired, ests, true_val, curr_err):
    rich_err = (abs(ests['richardson_1'] - true_val)
                if math.isfinite(ests['richardson_1']) else float('nan'))
    rat_err  = (abs(ests['rational_fit'] - true_val)
                if math.isfinite(ests['rational_fit']) else float('nan'))
    chosen_est = ests.get(chosen, float('nan'))
    chosen_err = (abs(chosen_est - true_val) if math.isfinite(chosen_est) else float('nan'))
    rat_better = (math.isfinite(rat_err) and math.isfinite(rich_err) and rat_err < rich_err)
    rec = {
        'regime':        regime,
        'is_holdout':    is_holdout(regime),
        'noise':         sigma,
        'seed':          seed,
        'L_true':        L_true,
        'L_hat':         L_hat,
        'cascade_fired': int(cascade_fired),
        'rat_better':    int(rat_better),
        'rich_err':      rich_err,
        'rat_err':       rat_err,
        'chosen_err':    chosen_err,
        'curr_err':      curr_err,
        'slope':         slope,
        'r2':            r2,
    }
    rec.update(hm)
    return rec


def _aggregate_cascade(df: pd.DataFrame, sweep_key: str, sweep_values,
                       gap_fractions, noise_list, regimes):
    """Pooled rows (core / holdout, capped excluded) and per-regime rows."""
    global_rows, regime_rows = [], []
    for v in sweep_values:
        for g in gap_fractions:
            for sigma in noise_list:
                sub_all = df[(df[sweep_key] == v) & (df['target_g'] == g)
                             & (df['noise'] == sigma)]
                if sub_all.empty:
                    continue
                for label, flag in (('core', 0), ('holdout', 1)):
                    sub = exclude_capped(sub_all[sub_all['is_holdout'] == flag])
                    if sub.empty:
                        continue
                    row = {sweep_key: v, 'target_g': g, 'noise': sigma, 'regime_set': label,
                           'n_capped_excluded': int((sub_all['is_holdout'] == flag).sum()) - len(sub)}
                    row.update(_cascade_metrics(sub))
                    global_rows.append(row)
                for regime in regimes:
                    rsub = sub_all[sub_all['regime'] == regime]
                    if rsub.empty:
                        continue
                    row = {sweep_key: v, 'regime': regime,
                           'is_holdout': int(rsub['is_holdout'].iloc[0]),
                           'target_g': g, 'noise': sigma,
                           'n_f': float(rsub['n_f'].median()),
                           'achieved_g': float(rsub['achieved_g'].median()),
                           'capped': int(rsub['capped'].max())}
                    m = _cascade_metrics(rsub)
                    row.update({'precision': m['precision'], 'mean_gain': m['mean_gain'],
                                'n': m['n']})
                    regime_rows.append(row)
    return pd.DataFrame(global_rows), pd.DataFrame(regime_rows)


# =============================================================================
# SWEEP 1 — ASSUMED-ASYMPTOTE SENSITIVITY
# =============================================================================

def sweep1_linf(assumed_modes, obs_idx, window_len, noise_list,
                gap_fractions, n_seeds, out_dir, core_regimes=None,
                holdout_regimes=None, verbose=True):
    """
    Test Phase 2 cascade robustness to the assumed asymptote.
    L_true is hidden and per (regime, seed); the ASSUMED value L_hat is
    swept over the config.ASSUMED_L_MODES listed in assumed_modes
    (oracle = L_true, labelled as such).
    """
    regimes = resolve_regimes(core_regimes, holdout_regimes, include_holdout=True)
    gap_fractions = [float(g) for g in gap_fractions]
    n_arr   = np.arange(obs_idx + 1, dtype=float)
    wl      = min(window_len, obs_idx)
    records = []

    total = len(assumed_modes) * len(noise_list) * n_seeds * len(regimes)
    done  = 0

    print(f'  Sweep 1: {len(assumed_modes)} assumed-L modes × '
          f'{len(noise_list)} noise × {n_seeds} seeds × '
          f'{len(regimes)} regimes × {len(gap_fractions)} strata')

    for assumed_mode in assumed_modes:
        for sigma in noise_list:
            for seed in range(n_seeds):
                rng = np.random.RandomState(seed * 137 + int(sigma*1e6) % 9973)

                for regime in regimes:
                    # Hidden per-(regime, seed) asymptote; methods never see L_true.
                    gen, truth_fn, L_true = regime_functions(regime, seed)
                    seq_full = gen(n_arr, rng, sigma)

                    w_start  = max(0, obs_idx - wl + 1)
                    seq_win  = list(seq_full[w_start : obs_idx + 1])
                    idx_win  = list(range(w_start, obs_idx + 1))
                    curr_val = float(seq_full[obs_idx])

                    L_hat     = assumed_asymptote(L_true, seq_win, assumed_mode)
                    slope, r2 = _cascade_features(seq_win, idx_win, L_hat)
                    chosen    = _phase2_cascade(slope, r2)
                    cascade_fired = (chosen == 'rational_fit')

                    for g in gap_fractions:
                        hm       = horizon_meta(regime, obs_idx, g, seed)
                        n_f      = hm['n_f']
                        true_val = float(truth_fn(n_f))
                        curr_err = abs(curr_val - true_val)
                        cfg      = _cfg(n_f, L_hat)
                        ests = _eval_methods(seq_win, idx_win, n_f, cfg,
                                             ['richardson_1', 'rational_fit'])
                        rec = _cascade_cell_record(regime, sigma, seed, hm, L_true, L_hat,
                                                   slope, r2, chosen, cascade_fired,
                                                   ests, true_val, curr_err)
                        rec['assumed_mode'] = assumed_mode
                        records.append(rec)

                    done += 1
                    if verbose and done % max(1, total // 10) == 0:
                        print(f'    [{done:>5}/{total}]  {100*done/total:5.1f}%'
                              f'  mode={assumed_mode}  sigma={sigma:.3f}',
                              flush=True)

    df = pd.DataFrame(records)
    df_global, df_regime = _aggregate_cascade(df, 'assumed_mode', assumed_modes,
                                              gap_fractions, noise_list, regimes)
    _save_csv(df_global, out_dir, 'phase5b_sweep1_global.csv')
    _save_csv(df_regime, out_dir, 'phase5b_sweep1_regime.csv')
    return df_global, df_regime


# =============================================================================
# SWEEP 2 — WINDOW LENGTH SENSITIVITY
# =============================================================================

def sweep2_window(window_lengths, obs_idx, noise_list, gap_fractions,
                  n_seeds, out_dir, core_regimes=None, holdout_regimes=None,
                  verbose=True):
    """
    Test Phase 2 cascade robustness to window length variation.
    Fixed Phase 2 thresholds applied without re-fitting; mode = config default.
    """
    regimes = resolve_regimes(core_regimes, holdout_regimes, include_holdout=True)
    gap_fractions = [float(g) for g in gap_fractions]
    n_arr   = np.arange(obs_idx + 1, dtype=float)
    records = []

    total = len(window_lengths) * len(noise_list) * n_seeds * len(regimes)
    done  = 0

    print(f'  Sweep 2: {len(window_lengths)} window lengths × '
          f'{len(noise_list)} noise × {n_seeds} seeds × '
          f'{len(regimes)} regimes × {len(gap_fractions)} strata')

    for wl in window_lengths:
        actual_wl = min(wl, obs_idx)

        for sigma in noise_list:
            for seed in range(n_seeds):
                rng = np.random.RandomState(seed * 137 + int(sigma*1e6) % 9973
                                            + wl * 11)

                for regime in regimes:
                    # Hidden per-(regime, seed) asymptote; methods never see L_true.
                    gen, truth_fn, L_true = regime_functions(regime, seed)
                    seq_full = gen(n_arr, rng, sigma)

                    w_start  = max(0, obs_idx - actual_wl + 1)
                    seq_win  = list(seq_full[w_start : obs_idx + 1])
                    idx_win  = list(range(w_start, obs_idx + 1))
                    curr_val = float(seq_full[obs_idx])

                    L_hat     = assumed_asymptote(L_true, seq_win)
                    slope, r2 = _cascade_features(seq_win, idx_win, L_hat)
                    chosen    = _phase2_cascade(slope, r2)
                    cascade_fired = (chosen == 'rational_fit')

                    for g in gap_fractions:
                        hm       = horizon_meta(regime, obs_idx, g, seed)
                        n_f      = hm['n_f']
                        true_val = float(truth_fn(n_f))
                        curr_err = abs(curr_val - true_val)
                        cfg      = _cfg(n_f, L_hat)
                        ests = _eval_methods(seq_win, idx_win, n_f, cfg,
                                             ['richardson_1', 'rational_fit'])
                        rec = _cascade_cell_record(regime, sigma, seed, hm, L_true, L_hat,
                                                   slope, r2, chosen, cascade_fired,
                                                   ests, true_val, curr_err)
                        rec['window_len'] = wl
                        records.append(rec)

                    done += 1
                    if verbose and done % max(1, total // 10) == 0:
                        print(f'    [{done:>5}/{total}]  {100*done/total:5.1f}%'
                              f'  win={wl}  sigma={sigma:.3f}',
                              flush=True)

    df = pd.DataFrame(records)
    df_global, df_regime = _aggregate_cascade(df, 'window_len', window_lengths,
                                              gap_fractions, noise_list, regimes)
    _save_csv(df_global, out_dir, 'phase5b_sweep2_global.csv')
    _save_csv(df_regime, out_dir, 'phase5b_sweep2_regime.csv')
    return df_global, df_regime


# =============================================================================
# SWEEP 3 — CAT_MULT SENSITIVITY
# =============================================================================

def sweep3_catmult(catmult_values, obs_idx, window_len, noise_list,
                   gap_fractions, n_seeds, out_dir, dangerous,
                   core_regimes=None, holdout_regimes=None, verbose=True):
    """
    Test whether regime champions and global rankings change when the
    catastrophic threshold is varied.  Rankings are pooled over the core
    regimes with capped cells excluded; champions are per regime (capped
    cells excluded; flagged).
    """
    regimes = resolve_regimes(core_regimes, holdout_regimes, include_holdout=True)
    gap_fractions = [float(g) for g in gap_fractions]
    dangerous = set(dangerous)
    n_arr = np.arange(obs_idx + 1, dtype=float)
    wl    = min(window_len, obs_idx)
    recs  = []

    total = (len(catmult_values) * len(noise_list) * n_seeds * len(regimes))
    done  = 0

    print(f'  Sweep 3: {len(catmult_values)} CAT_MULT × '
          f'{len(noise_list)} noise × {n_seeds} seeds × '
          f'{len(regimes)} regimes × {len(gap_fractions)} strata  '
          f'({len(ALL_METHODS)} methods, oracle excluded)')

    for cat_mult in catmult_values:
        for sigma in noise_list:
            for seed in range(n_seeds):
                rng = np.random.RandomState(seed * 137 + int(sigma*1e6) % 9973)

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
                        cfg      = _cfg(n_f, L_hat, cat_mult=cat_mult)

                        for method in ALL_METHODS:
                            fn = METHODS[method]
                            try:
                                est = fn(seq_win, idx_win, float(n_f), cfg)
                            except Exception:
                                est = float('nan')

                            valid = _valid(est, cfg)
                            err   = abs(est - true_val) if valid else float('nan')
                            cat   = (not valid) or (
                                valid and curr_err > 1e-12
                                and err > cat_mult * curr_err)
                            beats = valid and curr_err > 1e-12 and err < curr_err

                            rec = {
                                'cat_mult':    cat_mult,
                                'regime':      regime,
                                'is_holdout':  hold,
                                'noise':       sigma,
                                'seed':        seed,
                                'L_true':      L_true,
                                'L_hat':       L_hat,
                                'method':      method,
                                'valid':       int(valid),
                                'catastrophic':int(cat),
                                'beats':       int(beats),
                                'error':       err,
                            }
                            rec.update(hm)
                            rec.update(method_flags(method))
                            recs.append(rec)

                    done += 1
                    if verbose and done % max(1, total // 10) == 0:
                        print(f'    [{done:>5}/{total}]  {100*done/total:5.1f}%'
                              f'  CAT_MULT={cat_mult:.0f}  sigma={sigma:.3f}',
                              flush=True)

    df = pd.DataFrame(recs)

    # ── Aggregate stability scores ─────────────────────────────────────────────
    champ_rows, global_rows, concord_rows = [], [], []
    cfg_ref = _cfg(0, 0.0)   # weights only

    core_pool = exclude_capped(df[df['is_holdout'] == 0])
    for cat_mult in catmult_values:
        for g in gap_fractions:
            sub = core_pool[(core_pool['cat_mult'] == cat_mult) & (core_pool['target_g'] == g)]
            for method, mgrp in sub.groupby('method'):
                vr = mgrp['valid'].mean()
                cr = mgrp['catastrophic'].mean()
                br = mgrp['beats'].mean()
                sc = _stability(vr, cr, br, cfg_ref)
                global_rows.append({
                    'cat_mult':   cat_mult,
                    'target_g':   g,
                    'method':     method,
                    'is_trivial': int(mgrp['is_trivial'].iloc[0]),
                    'stability':  round(sc, 4),
                    'valid_rate': round(float(vr), 4),
                    'cat_rate':   round(float(cr), 4),
                    'beats_rate': round(float(br), 4),
                    'med_error':  float(mgrp['error'].median()),
                    'n_cells':    int(len(mgrp)),
                })

            sub_all = df[(df['cat_mult'] == cat_mult) & (df['target_g'] == g)]
            for regime in regimes:
                rsub = exclude_capped(sub_all[sub_all['regime'] == regime])
                capped_n = int(((sub_all['regime'] == regime) & (sub_all['capped'] == 1)).sum())
                if rsub.empty:
                    continue
                per_m = (rsub.groupby('method')
                             .agg(valid_rate=('valid','mean'),
                                  cat_rate=('catastrophic','mean'),
                                  beats_rate=('beats','mean'),
                                  med_error=('error', 'median'))
                             .reset_index())
                per_m['stability'] = [
                    _stability(r.valid_rate, r.cat_rate, r.beats_rate, cfg_ref)
                    for r in per_m.itertuples()]
                best = per_m.sort_values(['stability', 'med_error'],
                                         ascending=[False, True]).iloc[0]
                champ_rows.append({
                    'cat_mult':     cat_mult,
                    'regime':       regime,
                    'is_holdout':   is_holdout(regime),
                    'target_g':     g,
                    'capped_cells_excluded': capped_n,
                    'champion':     best['method'],
                    'stability':    round(float(best['stability']), 4),
                    'is_dangerous': int(best['method'] in dangerous),
                })

    df_champ  = pd.DataFrame(champ_rows)
    # Global ranking per (cat_mult, g) by stability, descending; the validity
    # floor applies (src.pipeline.assign_ranks): below-floor methods are shown
    # unranked and collected in the unranked block.
    df_global = assign_ranks(pd.DataFrame(global_rows), 'stability',
                             group_cols=['cat_mult', 'target_g'], ascending=False)
    df_global = (df_global.sort_values(['cat_mult', 'target_g', 'rank', 'method'],
                                       na_position='last')
                          .reset_index(drop=True))
    df_unranked = unranked_block(df_global, ['cat_mult', 'target_g', 'method', 'is_trivial',
                                             'valid_rate', 'cat_rate', 'stability',
                                             'med_error', 'n_cells'])

    # ── Concordance between CAT_MULT settings (ranked methods only) ────────────
    for g in gap_fractions:
        sub_g = df_global[(df_global['target_g'] == g) & (df_global['rank_eligible'] == 1)]
        pairs = [(catmult_values[i], catmult_values[j])
                 for i in range(len(catmult_values))
                 for j in range(i+1, len(catmult_values))]
        for cm_a, cm_b in pairs:
            a_rank = (sub_g[sub_g['cat_mult'] == cm_a]
                      .sort_values('rank')
                      .reset_index()['method'])
            b_rank = (sub_g[sub_g['cat_mult'] == cm_b]
                      .sort_values('rank')
                      .reset_index()['method'])
            common = list(set(a_rank) & set(b_rank))
            if len(common) < 5:
                continue
            a_pos = {m: i for i, m in enumerate(a_rank)}
            b_pos = {m: i for i, m in enumerate(b_rank)}
            tau, _ = kendalltau([a_pos[m] for m in common], [b_pos[m] for m in common])

            ca = df_champ[(df_champ['cat_mult'] == cm_a) & (df_champ['target_g'] == g)
                          & (df_champ['is_holdout'] == 0)]
            cb = df_champ[(df_champ['cat_mult'] == cm_b) & (df_champ['target_g'] == g)
                          & (df_champ['is_holdout'] == 0)]
            merged = ca.merge(cb, on=['regime', 'target_g'], suffixes=('_a','_b'))
            agree  = (float((merged['champion_a'] == merged['champion_b']).mean())
                      if len(merged) else float('nan'))

            concord_rows.append({
                'cat_mult_a':   cm_a,
                'cat_mult_b':   cm_b,
                'target_g':     g,
                'kendall_tau':  round(float(tau), 4),
                'champion_agreement': round(agree, 4) if math.isfinite(agree) else float('nan'),
                'n_methods':    len(common),
                'n_regimes':    int(len(merged)),
            })

    df_concord = pd.DataFrame(concord_rows, columns=[
        'cat_mult_a', 'cat_mult_b', 'target_g', 'kendall_tau', 'champion_agreement',
        'n_methods', 'n_regimes'])

    _save_csv(df_champ,  out_dir, 'phase5b_sweep3_champions.csv')
    _save_csv(df_global, out_dir, 'phase5b_sweep3_global.csv')
    _save_csv(df_unranked, out_dir, 'phase5b_sweep3_unranked.csv')
    _save_csv(df_concord,out_dir, 'phase5b_sweep3_concordance.csv')
    return df_champ, df_global, df_concord


# =============================================================================
# FIGURES
# =============================================================================

def _cascade_fig(df_global, key, xlabel, title, fname, out_dir, default_g=None,
                 categorical=False, vline=None, vline_label=''):
    core = df_global[df_global['regime_set'] == 'core'] if 'regime_set' in df_global else df_global
    if core.empty:
        return ''
    g     = _headline(core, default_g)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    noise = sorted(core['noise'].unique())
    cols  = {n: c for n, c in zip(noise, ['#1565c0', '#e65100', '#2e7d32'])}
    sub   = core[core['target_g'] == g]

    if categorical:
        present = set(sub[key])
        xs = [m for m in CFG_MOD.ASSUMED_L_MODES if m in present]
        xpos = {m: i for i, m in enumerate(xs)}
    else:
        xs = sorted(sub[key].unique())
        xpos = {v: v for v in xs}

    for ax, metric, ylabel, ttl in zip(
            axes,
            ['precision', 'recall', 'mean_gain'],
            ['Precision', 'Recall', 'Mean gain (rich_err - chosen_err)'],
            ['Cascade precision', 'Cascade recall', 'Cascade mean gain']):
        for sigma in noise:
            sv = sub[sub['noise'] == sigma].set_index(key).reindex(xs)
            ax.plot([xpos[v] for v in xs], sv[metric].values, 'o-',
                    color=cols.get(sigma, '#999'), label=f'σ={sigma}', lw=2, markersize=6)
        if vline is not None and vline in xpos:
            ax.axvline(xpos[vline], color='black', lw=1, ls='--', alpha=0.5, label=vline_label)
        if metric == 'precision':
            ax.axhline(0.80, color='red', lw=0.8, ls=':', alpha=0.6, label='0.80 target')
        if metric == 'mean_gain':
            ax.axhline(0, color='black', lw=0.7, alpha=0.4)
        if categorical:
            ax.set_xticks(range(len(xs)))
            ax.set_xticklabels(xs, fontsize=9)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(ttl, fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)

    fig.suptitle(f'{title}\n(g = {g:g}; core regimes, capped cells excluded)',
                 fontsize=10, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(out_dir, fname)
    _save(fig, path)
    return path


def fig_p5b_01_linf(df_global: pd.DataFrame, out_dir: str, default_g=None) -> str:
    """Cascade precision / recall / gain vs the assumed-asymptote mode."""
    return _cascade_fig(df_global, 'assumed_mode', 'Assumed asymptote mode (L_hat)',
                        'Figure P5B-1 — Cascade Robustness to the Assumed Asymptote '
                        '(L_true hidden per regime x seed; oracle labelled)',
                        'figure_p5b_01_linf.png', out_dir, default_g,
                        categorical=True, vline='oracle', vline_label='oracle (L_hat = L_true)')


def fig_p5b_02_window(df_global: pd.DataFrame, out_dir: str, default_g=None) -> str:
    """Cascade precision, recall, gain vs window_len."""
    return _cascade_fig(df_global, 'window_len', 'Window length',
                        'Figure P5B-2 — Cascade Robustness to Window Length Variation '
                        '(obs_idx fixed at 90)',
                        'figure_p5b_02_window.png', out_dir, default_g,
                        categorical=False, vline=60, vline_label='Phase 2 default (60)')


def fig_p5b_03_catmult(df_champ: pd.DataFrame,
                       df_concord: pd.DataFrame,
                       out_dir: str, default_g=None) -> str:
    """Champion agreement and ranking concordance vs CAT_MULT."""
    if df_champ.empty or df_concord.empty:
        print('  Fig P5B-3 skipped: not enough CAT_MULT values or methods.')
        return ''
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    g      = _headline(df_champ, default_g)
    cmults = sorted(df_champ['cat_mult'].unique())

    ax = axes[0]
    mat = np.zeros((len(cmults), len(cmults)))
    for _, row in df_concord[df_concord['target_g'] == g].iterrows():
        i = cmults.index(row['cat_mult_a'])
        j = cmults.index(row['cat_mult_b'])
        mat[i, j] = mat[j, i] = row['champion_agreement']
    np.fill_diagonal(mat, 1.0)

    im = ax.imshow(mat, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(len(cmults)))
    ax.set_xticklabels([f'CAT={c:.0f}' for c in cmults])
    ax.set_yticks(range(len(cmults)))
    ax.set_yticklabels([f'CAT={c:.0f}' for c in cmults])
    for i in range(len(cmults)):
        for j in range(len(cmults)):
            ax.text(j, i, f'{mat[i,j]:.2f}', ha='center', va='center',
                    fontsize=11, fontweight='bold',
                    color='white' if mat[i,j] < 0.5 else '#333')
    plt.colorbar(im, ax=ax, label='Fraction of core regime champions agreeing', shrink=0.7)
    ax.set_title('Regime Champion Agreement\n(1.0 = all core regimes agree)',
                 fontsize=10, fontweight='bold')

    ax2 = axes[1]
    rows = df_concord[df_concord['target_g'] == g]
    labels = [f'CAT={r["cat_mult_a"]:.0f} vs CAT={r["cat_mult_b"]:.0f}' for _, r in rows.iterrows()]
    taus   = [r['kendall_tau'] for _, r in rows.iterrows()]
    colours= ['#2e7d32' if t >= 0.9 else '#f57f17' if t >= 0.7 else '#c62828' for t in taus]
    ax2.bar(range(len(labels)), taus, color=colours, edgecolor='white')
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("Kendall's τ (method ranking concordance)", fontsize=9)
    ax2.axhline(0.9, color='#2e7d32', lw=1.2, ls='--', alpha=0.7, label='τ = 0.90 (high concordance)')
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Method Ranking Concordance\n(1.0 = identical ranking)", fontsize=10, fontweight='bold')
    ax2.legend(fontsize=9)
    for i, t in enumerate(taus):
        ax2.text(i, t + 0.01, f'{t:.3f}', ha='center', va='bottom', fontsize=9)

    fig.suptitle(f'Figure P5B-3 — CAT_MULT Sensitivity\n(g = {g:g}; core regimes, capped excluded)',
                 fontsize=10, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(out_dir, 'figure_p5b_03_catmult.png')
    _save(fig, path)
    return path


# =============================================================================
# MASTER RUN FUNCTION
# =============================================================================

def run_all(assumed_modes, window_lengths, catmult_values,
            obs_idx, window_len_default, noise_list, gap_fractions,
            n_seeds, out_dir, core_regimes=None, holdout_regimes=None,
            default_g=None, verbose=True):
    os.makedirs(out_dir, exist_ok=True)

    # Ordering guard: the dangerous flag comes from the Phase-1 artifact.
    dangerous = load_dangerous()
    print(f'  Dangerous set (Phase-1 artifact): {sorted(dangerous)}')

    print('\n  === SWEEP 1: Assumed-Asymptote (L_hat) Sensitivity ===')
    df1g, df1r = sweep1_linf(assumed_modes, obs_idx, window_len_default,
                             noise_list, gap_fractions, n_seeds, out_dir,
                             core_regimes, holdout_regimes, verbose)

    print('\n  === SWEEP 2: Window Length Sensitivity ===')
    df2g, df2r = sweep2_window(window_lengths, obs_idx, noise_list,
                               gap_fractions, n_seeds, out_dir,
                               core_regimes, holdout_regimes, verbose)

    print('\n  === SWEEP 3: CAT_MULT Sensitivity ===')
    df3c, df3g, df3cd = sweep3_catmult(catmult_values, obs_idx,
                                       window_len_default, noise_list,
                                       gap_fractions, n_seeds, out_dir, dangerous,
                                       core_regimes, holdout_regimes, verbose)

    print('\n  Generating figures ...')
    paths = [
        fig_p5b_01_linf(df1g, out_dir, default_g),
        fig_p5b_02_window(df2g, out_dir, default_g),
        fig_p5b_03_catmult(df3c, df3cd, out_dir, default_g),
    ]

    return {
        'sweep1_global': df1g, 'sweep1_regime': df1r,
        'sweep2_global': df2g, 'sweep2_regime': df2r,
        'sweep3_champions': df3c, 'sweep3_global': df3g,
        'sweep3_concordance': df3cd,
        'dangerous': dangerous,
        'figures': [p for p in paths if p],
    }
