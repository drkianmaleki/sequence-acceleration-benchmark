"""
evaluation.py
=============
Main benchmark loop (Phase 1) under redesign v2.

Grid:  56 methods (51 accelerators + 5 trivial comparators)
       x (18 core + 6 held-out) regimes x noise levels x seeds
       x gap-stratified horizons g in config.HORIZON_GAP_FRACTIONS

What changed relative to the rejected design
--------------------------------------------
* Each (regime, seed) has a hidden asymptote L_true (src.generators).  The
  prediction target is truth(n_f) = L_true + gap(n_f).
* Methods receive L_hat = assumed_asymptote(L_true, window, ASSUMED_L_MODE)
  under cfg['L_inf'], never L_true.  cfg['L_true'] exists solely for the
  constant_oracle comparator (tests enforce that nothing else reads it).
* Horizons are per regime: n_f(regime, g) is the first n > obs_idx with
  gap(n) <= g * gap(obs_idx), capped at 50,000 (src.horizons).  Every record
  carries (target_g, achieved_g, n_f, capped).
* The trivial comparators are methods.  Every record carries two kinds of
  skill (src.trivial): ``skill`` = err(method) / err(best of
  {constant_assumed, last_value, window_mean, window_min}) -- the HINDSIGHT
  best-of-four (strict) bar, aggregated as med_skill -- and the
  fixed-reference columns skill_vs_{assumed,last,wmean,wmin} /
  win_vs_{...} against each deployable trivial separately, aggregated as
  med_skill_vs_* / win_rate_vs_*.  The oracle is never in a denominator.
* Held-out regimes are evaluated but flagged is_holdout = 1 and pooled
  separately.  Pooled cross-regime tables exclude capped cells (reported in
  a separate capped block), sort by median error, and rank non-oracle
  methods with valid_rate >= config.RANK_MIN_VALID only (below-floor methods
  are shown unranked and listed in a separate unranked block); every table
  still shows the oracle rows.
* The per-regime recommendation column is best_by_skill.

Outputs (out_dir)
-----------------
    phase1_records.csv          one row per (regime, noise, seed, g, method)
    phase1_aggregated.csv       per (method, regime, noise, g)
    phase1_global.csv           pooled over the core regimes, per (method, g),
                                capped cells excluded, sorted by med_error
    phase1_global_holdout.csv   the same over the held-out regimes
    phase1_capped.csv           the capped block: capped (regime, g) cells
                                with achieved_g and per-method medians
    phase1_unranked.csv         the unranked block: methods below the
                                validity floor (valid_rate < RANK_MIN_VALID)
                                in the pooled tables, with valid_rate
    phase1_regime_best.csv      best_by_skill (primary) and best_by_stability
                                per (regime, g), oracle excluded
    phase1_horizons.csv         n_f / achieved_g per (regime, g), seed 0 for
                                seed-dependent shapes
    phase1_heatmap_g{g}.csv     method x regime stability per stratum

Author : Kian Maleki
Date   : 2026-05-24 (v1), 2026-09-19 (redesign v2)
"""

import math
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import src.config as CFG_MOD
from src.accelerators import METHODS, METHOD_NAMES
from src.asymptote import assumed_asymptote, resolve_mode
from src.generators import HOLDOUT, regime_functions
from src.horizons import horizon_for_gap, horizon_table
from src.pipeline import (assign_ranks, capped_block, exclude_capped, median_skill,
                          unranked_block,
                          method_flags, resolve_regimes)
from src.trivial import (MED_SKILL_VS_COLS, ORACLE_METHODS, SKILL_VS_AGG_COLS,
                         WIN_RATE_VS_COLS, aggregate_skill_vs, best_reference_error,
                         skill_score, skill_vs_table)

# ── Method metadata ────────────────────────────────────────────────────────────

# Family label for each method (used in plots)
FAMILY = {
    'current_value': 'baseline',  'linear': 'baseline',
    'log_linear': 'baseline',     'geom_avg_diff': 'baseline',
    'richardson_1': 'richardson',  'richardson_2': 'richardson',
    'richardson_3': 'richardson',  'richardson_a05': 'richardson',
    'richardson_a10': 'richardson','richardson_a20': 'richardson',
    'single_exp_fit': 'parametric','double_exp_fit': 'parametric',
    'rational_fit': 'parametric',  'log_fit': 'parametric',
    'shanks_1': 'shanks',   'shanks_2': 'shanks',
    'shanks_3': 'shanks',   'shanks_4': 'shanks',
    'wynn_eps_1': 'wynn_eps','wynn_eps_2': 'wynn_eps',
    'wynn_eps_3': 'wynn_eps',
    'wynn_rho_1': 'wynn_rho','wynn_rho_2': 'wynn_rho',
    'wynn_rho_3': 'wynn_rho',
    'pade_11': 'pade', 'pade_12': 'pade', 'pade_13': 'pade',
    'pade_21': 'pade', 'pade_22': 'pade', 'pade_23': 'pade',
    'pade_31': 'pade', 'pade_32': 'pade',
    'levin_t1': 'levin',  'levin_t2': 'levin',
    'levin_u1': 'levin',  'levin_u2': 'levin',
    'levin_v1': 'levin',  'levin_v2': 'levin',
    'weniger_d1': 'weniger',   'weniger_d2': 'weniger',
    'brezinski_theta1': 'brezinski', 'brezinski_theta2': 'brezinski',
    'neville_2': 'neville', 'neville_3': 'neville', 'neville_4': 'neville',
    'anderson_1': 'anderson','anderson_2': 'anderson',
    'anderson_3': 'anderson',
    'median_ensemble': 'ensemble',   'stability_weighted': 'ensemble',
    'best_shanks_wynn': 'ensemble',
    # Redesign v2 — trivial comparators
    'constant_assumed': 'trivial', 'constant_oracle': 'trivial',
    'window_mean': 'trivial',      'window_min': 'trivial',
    'last_value': 'trivial',
}

# Whether the method explicitly evaluates at future_x (trajectory extrapolator)
# versus estimating the sequence limit (limit estimator).
USES_FUTURE_X: Dict[str, bool] = {m: False for m in METHOD_NAMES}
for _m in [
    'linear', 'log_linear',
    'richardson_1', 'richardson_2', 'richardson_3',
    'richardson_a05', 'richardson_a10', 'richardson_a20',
    'single_exp_fit', 'double_exp_fit', 'rational_fit', 'log_fit',
    'pade_11', 'pade_12', 'pade_13', 'pade_21', 'pade_22',
    'pade_23', 'pade_31', 'pade_32',
    'neville_2', 'neville_3', 'neville_4',
    'median_ensemble', 'stability_weighted',   # pool contains Richardson
]:
    USES_FUTURE_X[_m] = True

METHOD_TYPE = {m: ('trajectory' if USES_FUTURE_X[m] else 'limit')
               for m in METHOD_NAMES}

IS_ORACLE = {m: (m in ORACLE_METHODS) for m in METHOD_NAMES}
FLAGS = {m: method_flags(m) for m in METHOD_NAMES}


# ── Shared accelerator config ──────────────────────────────────────────────────

def build_cfg(future_idx: int, L_hat: float, L_true: Optional[float] = None) -> dict:
    """
    Accelerator config for one evaluation cell.

    L_hat  : the ASSUMED asymptote (src.asymptote.assumed_asymptote); this is
             what every method sees under cfg['L_inf'].
    L_true : the hidden asymptote; stored under cfg['L_true'] for the
             constant_oracle comparator only.  No other method reads it.
    """
    if L_hat is None:
        raise ValueError("build_cfg needs L_hat; compute it with "
                         "src.asymptote.assumed_asymptote()")
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


def is_valid(v: float, cfg: dict) -> bool:
    return bool(math.isfinite(v)
                and cfg['min_valid'] <= v <= cfg['max_valid'])


def stability_score(valid_r: float, cat_r: float,
                    beats_r: float, cfg: dict) -> float:
    return (valid_r
            - cfg['W_CAT']   * cat_r
            + cfg['W_BEATS'] * beats_r)


def _median(values) -> float:
    """Median ignoring NaN (inf is kept: an infinite skill is a real value)."""
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    return float(np.median(arr)) if arr.size else float('nan')


# ── Core evaluation loop ───────────────────────────────────────────────────────

def run_phase1(n_seeds:        int,
               noise_levels:   List[float],
               gap_fractions:  List[float],
               obs_idx:        int,
               window_len:     int,
               out_dir:        str,
               assumed_mode:   Optional[str] = None,
               asymptote_mode: Optional[str] = None,
               include_holdout: bool = True,
               regimes:        Optional[List[str]] = None,
               core_regimes:   Optional[List[str]] = None,
               holdout_regimes: Optional[List[str]] = None,
               verbose:        bool = True) -> Dict[str, pd.DataFrame]:
    """
    Run the Phase 1 grid and return a dict of result DataFrames.

    Parameters
    ----------
    n_seeds         : replicates per (regime, noise) pair
    noise_levels    : list of sigma values
    gap_fractions   : target remaining-gap fractions g (horizons per regime)
    obs_idx         : last observed index (0-based) = observation depth n_obs
    window_len      : terms fed to each accelerator
    out_dir         : directory for output CSV files
    assumed_mode    : config.ASSUMED_L_MODES entry (default config.ASSUMED_L_MODE)
    asymptote_mode  : "hetero" | "legacy" (default config.ASYMPTOTE_MODE)
    include_holdout : evaluate the held-out regimes as well (flagged)
    regimes         : explicit regime list (overrides everything else)
    core_regimes / holdout_regimes : restrict either group (None = all)
    verbose         : print progress

    Returns
    -------
    dict with keys 'records', 'aggregated', 'global', 'global_holdout',
    'capped', 'regime_best', 'horizons', 'heatmaps'
    """
    os.makedirs(out_dir, exist_ok=True)

    assumed_mode   = resolve_mode(assumed_mode)
    asymptote_mode = (CFG_MOD.ASYMPTOTE_MODE if asymptote_mode is None
                      else asymptote_mode)
    if regimes is None:
        regimes = resolve_regimes(core_regimes, holdout_regimes, include_holdout)
    gap_fractions = [float(g) for g in gap_fractions]

    # Only the observed prefix is ever needed: methods see the window and the
    # target comes from the noiseless truth at n_f.
    n_arr   = np.arange(obs_idx + 1, dtype=float)
    w_start = max(0, obs_idx - window_len + 1)
    idx_win = list(range(w_start, obs_idx + 1))

    total_seq = len(regimes) * n_seeds * len(noise_levels)
    done      = 0
    records   = []

    for regime in regimes:
        holdout = int(regime in HOLDOUT)

        for sigma in noise_levels:
            for seed in range(n_seeds):
                gen, truth, L_true = regime_functions(regime, seed, asymptote_mode)
                rng      = np.random.RandomState(seed * 137 + int(sigma * 1e6) % 9973)
                seq_full = gen(n_arr, rng, sigma)

                seq_win  = list(seq_full[w_start : obs_idx + 1])
                curr_val = float(seq_full[obs_idx])

                # What the methods are told about the asymptote.
                L_hat = assumed_asymptote(L_true, seq_win, assumed_mode)

                for g in gap_fractions:
                    hz       = horizon_for_gap(regime, obs_idx, g, seed=seed)
                    n_f      = hz.n_f
                    true_val = float(truth(n_f))
                    curr_err = abs(curr_val - true_val)
                    cfg      = build_cfg(n_f, L_hat, L_true)

                    ests: Dict[str, float] = {}
                    errs: Dict[str, float] = {}
                    for method in METHOD_NAMES:
                        try:
                            est = float(METHODS[method](seq_win, idx_win, float(n_f), cfg))
                        except Exception:
                            est = float('nan')
                        ests[method] = est
                        errs[method] = (abs(est - true_val) if is_valid(est, cfg)
                                        else float('nan'))
                    ref_err = best_reference_error(errs)

                    for method in METHOD_NAMES:
                        est, err = ests[method], errs[method]
                        valid = math.isfinite(err)
                        cat   = (not valid) or (
                                    curr_err > 1e-12
                                    and err > CFG_MOD.CAT_MULT * curr_err)
                        beats = valid and curr_err > 1e-12 and err < curr_err
                        impv  = ((curr_err / err) if (valid and err > 1e-12)
                                 else (1.0 if valid else float('nan')))
                        rec = {
                            'regime':        regime,
                            'is_holdout':    holdout,
                            'noise':         sigma,
                            'seed':          seed,
                            'L_true':        L_true,
                            'L_hat':         L_hat,
                            'assumed_mode':  assumed_mode,
                            'target_g':      g,
                            'achieved_g':    hz.achieved_g,
                            'n_f':           n_f,
                            'capped':        int(hz.capped),
                            'method':        method,
                            'family':        FAMILY.get(method, 'unknown'),
                            'method_type':   METHOD_TYPE[method],
                            'is_trivial':    FLAGS[method]['is_trivial'],
                            'is_oracle':     FLAGS[method]['is_oracle'],
                            'true_val':      true_val,
                            'curr_val':      curr_val,
                            'estimate':      est if valid else float('nan'),
                            'error':         err if valid else float('nan'),
                            'ref_error':     ref_err,
                            'skill':         skill_score(err, ref_err) if valid else float('nan'),
                            'valid':         int(valid),
                            'catastrophic':  int(cat),
                            'beats_current': int(beats),
                            'improve_ratio': impv if math.isfinite(impv) else float('nan'),
                        }
                        # fixed-reference skill: err / err(each deployable trivial) + win flag
                        rec.update(skill_vs_table(err if valid else float('nan'), errs))
                        records.append(rec)

                done += 1
                if verbose and done % max(1, total_seq // 20) == 0:
                    pct = 100 * done / total_seq
                    print(f'  [{done:>5}/{total_seq}]  {pct:5.1f}%  '
                          f'{regime:<20}  sigma={sigma:.3f}')

    if verbose:
        print(f'  [{total_seq}/{total_seq}] 100.0%  Done.\n')

    df_rec  = pd.DataFrame(records)
    cfg_ref = build_cfg(0, 0.0)

    # ── Aggregate per (method, regime, noise, g) ───────────────────────────────
    agg_records = []
    for (method, regime, sigma, g), grp in df_rec.groupby(
            ['method', 'regime', 'noise', 'target_g'], sort=False):
        vr = float(grp['valid'].mean())
        cr = float(grp['catastrophic'].mean())
        br = float(grp['beats_current'].mean())
        agg_records.append({
            'method':       method,
            'family':       FAMILY.get(method, 'unknown'),
            'method_type':  METHOD_TYPE[method],
            'is_trivial':   FLAGS[method]['is_trivial'],
            'is_oracle':    FLAGS[method]['is_oracle'],
            'regime':       regime,
            'is_holdout':   int(regime in HOLDOUT),
            'noise':        sigma,
            'target_g':     g,
            'n_f':          float(grp['n_f'].median()),
            'achieved_g':   _median(grp['achieved_g']),
            'capped':       int(grp['capped'].max()),
            'L_true':       _median(grp['L_true']),
            'L_hat':        _median(grp['L_hat']),
            'valid_rate':   round(vr, 4),
            'cat_rate':     round(cr, 4),
            'beats_rate':   round(br, 4),
            'med_error':    _median(grp['error']),
            'med_improve':  _median(grp['improve_ratio']),
            'med_skill':    _median(grp['skill']),
            'stability':    round(stability_score(vr, cr, br, cfg_ref), 4),
            'n_seeds':      int(len(grp)),
            **aggregate_skill_vs(grp),       # med_skill_vs_* / win_rate_vs_* over seeds
        })
    df_agg = pd.DataFrame(agg_records)

    # ── Global tables (pooled across regimes) ──────────────────────────────────
    # Capped cells are excluded from every pooled statistic; ranks are by the
    # configured metric (median error, ascending) over non-oracle methods with
    # valid_rate >= RANK_MIN_VALID (src.pipeline.assign_ranks); below-floor
    # methods stay in the table unranked and form the unranked block.
    def _pool(df: pd.DataFrame, label: str) -> pd.DataFrame:
        cols = ['method', 'family', 'method_type', 'is_trivial', 'is_oracle',
                'regime_set', 'target_g', 'valid_rate', 'cat_rate', 'beats_rate',
                'med_error', 'med_improve', 'med_skill', 'stability',
                'n_regimes', 'n_cells', 'n_capped_excluded', 'rank', 'rank_eligible',
                *SKILL_VS_AGG_COLS]
        if df.empty:
            return pd.DataFrame(columns=cols)
        pooled = exclude_capped(df)
        rows = []
        for (method, g), grp_all in df.groupby(['method', 'target_g'], sort=False):
            grp = pooled[(pooled['method'] == method) & (pooled['target_g'] == g)]
            n_excl = int(len(grp_all) - len(grp))
            if grp.empty:
                rows.append({
                    'method': method, 'family': FAMILY.get(method, 'unknown'),
                    'method_type': METHOD_TYPE[method],
                    'is_trivial': FLAGS[method]['is_trivial'],
                    'is_oracle': FLAGS[method]['is_oracle'],
                    'regime_set': label, 'target_g': g,
                    'valid_rate': float('nan'), 'cat_rate': float('nan'),
                    'beats_rate': float('nan'), 'med_error': float('nan'),
                    'med_improve': float('nan'), 'med_skill': float('nan'),
                    'stability': float('nan'), 'n_regimes': 0, 'n_cells': 0,
                    'n_capped_excluded': n_excl,
                    **{c: float('nan') for c in SKILL_VS_AGG_COLS},
                })
                continue
            vr = float(grp['valid_rate'].mean())
            cr = float(grp['cat_rate'].mean())
            br = float(grp['beats_rate'].mean())
            rows.append({
                'method':      method,
                'family':      FAMILY.get(method, 'unknown'),
                'method_type': METHOD_TYPE[method],
                'is_trivial':  FLAGS[method]['is_trivial'],
                'is_oracle':   FLAGS[method]['is_oracle'],
                'regime_set':  label,
                'target_g':    g,
                'valid_rate':  round(vr, 4),
                'cat_rate':    round(cr, 4),
                'beats_rate':  round(br, 4),
                'med_error':   _median(grp['med_error']),
                'med_improve': _median(grp['med_improve']),
                'med_skill':   _median(grp['med_skill']),
                'stability':   round(stability_score(vr, cr, br, cfg_ref), 4),
                'n_regimes':   int(grp['regime'].nunique()),
                'n_cells':     int(len(grp)),
                'n_capped_excluded': n_excl,
                # fixed-reference skill pooled over cells: median of the per-cell
                # medians, mean of the per-cell win rates
                **{c: _median(grp[c]) for c in MED_SKILL_VS_COLS},
                **{c: round(float(grp[c].mean()), 4) for c in WIN_RATE_VS_COLS},
            })
        metric = CFG_MOD.RANK_METRIC
        out = assign_ranks(pd.DataFrame(rows), metric, group_cols=['target_g'])
        return (out.sort_values(['target_g', metric, 'method'],
                                ascending=[True, True, True], na_position='last')
                   .reset_index(drop=True)[cols])

    df_global         = _pool(df_agg[df_agg['is_holdout'] == 0], 'core')
    df_global_holdout = _pool(df_agg[df_agg['is_holdout'] == 1], 'holdout')

    # ── Unranked block: below the validity floor (shown, never ranked) ────────
    unranked_cols = ['regime_set', 'target_g', 'method', 'method_type', 'valid_rate',
                     'cat_rate', 'med_error', 'med_skill', 'stability', 'n_cells']
    df_unranked = pd.concat([unranked_block(df_global, unranked_cols),
                             unranked_block(df_global_holdout, unranked_cols)],
                            ignore_index=True)

    # ── Capped block: the cells the pooled tables left out ─────────────────────
    df_capped = capped_block(df_agg, keys=['regime', 'is_holdout', 'target_g', 'method'],
                             value_cols=['med_error', 'med_skill', 'valid_rate'])

    # ── Per-regime recommendation: best_by_skill (primary), best_by_stability ──
    best_rows = []
    for (regime, g), grp in df_agg.groupby(['regime', 'target_g'], sort=False):
        pool = grp[grp['is_oracle'] == 0]
        per_method = (pool.groupby('method')
                          .agg(valid_rate=('valid_rate', 'mean'),
                               cat_rate=('cat_rate', 'mean'),
                               beats_rate=('beats_rate', 'mean'),
                               med_error=('med_error', 'median'),
                               med_improve=('med_improve', 'median'),
                               med_skill=('med_skill', 'median'))
                          .reset_index())
        per_method['stability'] = [
            stability_score(r.valid_rate, r.cat_rate, r.beats_rate, cfg_ref)
            for r in per_method.itertuples()]
        with_skill = per_method[per_method['med_skill'].notna()]
        by_skill = (with_skill.sort_values(['med_skill', 'med_error']).iloc[0]
                    if len(with_skill) else None)
        by_stab = per_method.sort_values(['stability', 'med_error'],
                                         ascending=[False, True]).iloc[0]
        oracle = grp[grp['is_oracle'] == 1]
        bs_name = by_skill['method'] if by_skill is not None else ''
        best_rows.append({
            'regime':                 regime,
            'is_holdout':             int(regime in HOLDOUT),
            'target_g':               g,
            'n_f':                    float(grp['n_f'].median()),
            'achieved_g':             _median(grp['achieved_g']),
            'capped':                 int(grp['capped'].max()),
            'best_by_skill':          bs_name,
            'best_skill':             (float(by_skill['med_skill'])
                                       if by_skill is not None else float('nan')),
            'family':                 FAMILY.get(bs_name, 'unknown'),
            'method_type':            METHOD_TYPE.get(bs_name, ''),
            'skill_best_med_error':   (float(by_skill['med_error'])
                                       if by_skill is not None else float('nan')),
            'skill_best_valid_rate':  (round(float(by_skill['valid_rate']), 4)
                                       if by_skill is not None else float('nan')),
            'skill_best_cat_rate':    (round(float(by_skill['cat_rate']), 4)
                                       if by_skill is not None else float('nan')),
            'best_by_stability':      by_stab['method'],
            'stab_best_stability':    round(float(by_stab['stability']), 4),
            'stab_best_med_error':    float(by_stab['med_error']),
            'oracle_med_error':       _median(oracle['med_error']) if len(oracle) else float('nan'),
        })
    df_best = pd.DataFrame(best_rows)

    # ── Horizon table (seed 0 for seed-dependent shapes) ───────────────────────
    df_hz = horizon_table(regimes, obs_idx, gap_fractions, seed=0)

    # ── Heatmaps (one per stratum) ─────────────────────────────────────────────
    heatmaps = {}
    for g in gap_fractions:
        sub = df_agg[df_agg['target_g'] == g]
        pooled = (sub.groupby(['method', 'regime'])
                     .agg(vr=('valid_rate', 'mean'),
                          cr=('cat_rate', 'mean'),
                          br=('beats_rate', 'mean'))
                     .reset_index())
        pooled['stability'] = (pooled['vr']
                               - CFG_MOD.W_CAT * pooled['cr']
                               + CFG_MOD.W_BEATS * pooled['br'])
        heatmaps[g] = pooled.pivot(index='method', columns='regime',
                                   values='stability')

    # ── Save ───────────────────────────────────────────────────────────────────
    def _save(df: pd.DataFrame, name: str, **kw):
        p = os.path.join(out_dir, name)
        df.to_csv(p, **kw)
        if verbose:
            print(f'  Saved: {p}  ({len(df)} rows)')

    _save(df_rec,            'phase1_records.csv',        index=False)
    _save(df_agg,            'phase1_aggregated.csv',     index=False)
    _save(df_global,         'phase1_global.csv',         index=False)
    _save(df_global_holdout, 'phase1_global_holdout.csv', index=False)
    _save(df_capped,         'phase1_capped.csv',         index=False)
    _save(df_unranked,       'phase1_unranked.csv',       index=False)
    _save(df_best,           'phase1_regime_best.csv',    index=False)
    _save(df_hz,             'phase1_horizons.csv',       index=False)
    for g, hm in heatmaps.items():
        _save(hm.round(4), f'phase1_heatmap_g{g:g}.csv')

    return {
        'records':        df_rec,
        'aggregated':     df_agg,
        'global':         df_global,
        'global_holdout': df_global_holdout,
        'capped':         df_capped,
        'unranked':       df_unranked,
        'regime_best':    df_best,
        'horizons':       df_hz,
        'heatmaps':       heatmaps,
    }
