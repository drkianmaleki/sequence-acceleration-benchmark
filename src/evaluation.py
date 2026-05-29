"""
evaluation.py
=============
Phase 1 evaluation loop for the sequence acceleration study.

Runs the full grid:
    51 methods  x  18 regimes  x  noise levels  x  seeds  x  horizons

Produces two output artefacts:
    phase1_aggregated.csv   Per (method, regime, noise, horizon) statistics.
                            ~8 K rows in full mode; the primary data file.
    phase1_raw_sample.csv   Full per-seed records for the noiseless case only.
                            Used for diagnostic plots and error distributions.

Usage (from within sequence_accel/):
    python run_phase1.py --quick    # 5 seeds, sigma=0, horizon 5000 only
    python run_phase1.py --full     # 30 seeds, 3 noise levels, 3 horizons

Author : Kian Maleki
Date   : 2026-05-24
"""

import os
import math
import numpy as np
import pandas as pd
from typing import List, Dict

import config as CFG_MOD
from accelerators import METHODS, METHOD_NAMES
from generators   import GENERATORS, REGIME_NAMES, TRUTH


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


# ── Shared accelerator config ──────────────────────────────────────────────────

def build_cfg(future_idx: int) -> dict:
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
        'future_idx':     future_idx,
    }


def is_valid(v: float, cfg: dict) -> bool:
    return bool(math.isfinite(v)
                and cfg['min_valid'] <= v <= cfg['max_valid'])


def stability_score(valid_r: float, cat_r: float,
                    beats_r: float, cfg: dict) -> float:
    return (valid_r
            - cfg['W_CAT']   * cat_r
            + cfg['W_BEATS'] * beats_r)


# ── Core evaluation loop ───────────────────────────────────────────────────────

def run_phase1(n_seeds:      int,
               noise_levels: List[float],
               future_idxs:  List[int],
               obs_idx:      int,
               window_len:   int,
               out_dir:      str,
               verbose:      bool = True) -> Dict[str, pd.DataFrame]:
    """
    Run the full Phase 1 grid and return a dict of result DataFrames.

    Parameters
    ----------
    n_seeds      : replicates per (regime, noise) pair
    noise_levels : list of sigma values
    future_idxs  : list of prediction horizons
    obs_idx      : last observed index (0-based)
    window_len   : terms fed to each accelerator
    out_dir      : directory for output CSV files
    verbose      : print progress

    Returns
    -------
    dict with keys 'aggregated', 'global', 'regime_best', 'raw_sample'
    """
    os.makedirs(out_dir, exist_ok=True)

    n_arr     = np.arange(max(future_idxs) + 100, dtype=float)
    total_seq = len(REGIME_NAMES) * n_seeds * len(noise_levels)
    done      = 0

    raw_records  = []   # noiseless only, for diagnostic plots
    agg_records  = []   # per (method, regime, noise, horizon) aggregated

    # ── Per-(regime, noise, horizon) accumulators ──────────────────────────────
    # Key: (regime, noise, future_idx, method)
    # Value: dict of lists accumulating per-seed stats
    from collections import defaultdict
    buckets: Dict = defaultdict(lambda: defaultdict(list))

    # ── Main loop ──────────────────────────────────────────────────────────────
    for regime in REGIME_NAMES:
        gen   = GENERATORS[regime]
        truth = TRUTH[regime]

        for sigma in noise_levels:
            for seed in range(n_seeds):
                rng      = np.random.RandomState(seed * 137 + int(sigma * 1e6) % 9973)
                seq_full = gen(n_arr, rng, sigma)

                # Window
                w_start  = max(0, obs_idx - window_len + 1)
                seq_win  = list(seq_full[w_start : obs_idx + 1])
                idx_win  = list(range(w_start, obs_idx + 1))
                curr_val = float(seq_full[obs_idx])

                for future_idx in future_idxs:
                    true_val  = float(truth(future_idx))
                    curr_err  = abs(curr_val - true_val)
                    cfg       = build_cfg(future_idx)

                    for method in METHOD_NAMES:
                        fn = METHODS[method]
                        try:
                            est = fn(seq_win, idx_win, float(future_idx), cfg)
                        except Exception:
                            est = float('nan')

                        valid = is_valid(est, cfg)
                        err   = abs(est - true_val) if valid else float('nan')
                        cat   = (not valid) or (
                                    valid and curr_err > 1e-12
                                    and err > CFG_MOD.CAT_MULT * curr_err)
                        beats = valid and curr_err > 1e-12 and err < curr_err
                        impv  = ((curr_err / err) if (valid and err > 1e-12)
                                 else (1.0 if valid else float('nan')))

                        key = (regime, sigma, future_idx, method)
                        b   = buckets[key]
                        b['valid'].append(int(valid))
                        b['cat'].append(int(cat))
                        b['beats'].append(int(beats))
                        if valid:
                            b['error'].append(err)
                            b['impv'].append(impv if math.isfinite(impv)
                                             else float('nan'))

                        # Raw sample: noiseless + first seed only
                        if sigma == 0.0 and seed == 0:
                            raw_records.append({
                                'regime':        regime,
                                'future_idx':    future_idx,
                                'method':        method,
                                'family':        FAMILY.get(method, 'unknown'),
                                'method_type':   METHOD_TYPE[method],
                                'true_val':      round(true_val, 8),
                                'curr_val':      round(curr_val, 8),
                                'estimate':      round(est, 8) if valid else float('nan'),
                                'error':         round(err, 8) if valid else float('nan'),
                                'valid':         int(valid),
                                'catastrophic':  int(cat),
                                'beats_current': int(beats),
                                'improve_ratio': round(impv, 6)
                                                 if math.isfinite(impv) else float('nan'),
                            })

                done += 1
                if verbose and done % max(1, total_seq // 20) == 0:
                    pct = 100 * done / total_seq
                    print(f'  [{done:>5}/{total_seq}]  {pct:5.1f}%  '
                          f'{regime:<20}  sigma={sigma:.3f}')

    if verbose:
        print(f'  [{total_seq}/{total_seq}] 100.0%  Done.\n')

    # ── Aggregate ──────────────────────────────────────────────────────────────
    cfg_ref = build_cfg(future_idxs[0])
    for (regime, sigma, future_idx, method), b in buckets.items():
        n       = len(b['valid'])
        vr      = float(np.mean(b['valid']))
        cr      = float(np.mean(b['cat']))
        br      = float(np.mean(b['beats']))
        errors  = [e for e in b.get('error', []) if math.isfinite(e)]
        impvs   = [v for v in b.get('impv',  []) if math.isfinite(v)]
        me      = float(np.median(errors)) if errors else float('nan')
        mi      = float(np.median(impvs))  if impvs  else float('nan')
        sc      = stability_score(vr, cr, br, cfg_ref)

        agg_records.append({
            'method':       method,
            'family':       FAMILY.get(method, 'unknown'),
            'method_type':  METHOD_TYPE[method],
            'regime':       regime,
            'noise':        sigma,
            'future_idx':   future_idx,
            'valid_rate':   round(vr, 4),
            'cat_rate':     round(cr, 4),
            'beats_rate':   round(br, 4),
            'med_error':    me,
            'med_improve':  mi,
            'stability':    round(sc, 4),
            'n_seeds':      n,
        })

    df_agg    = pd.DataFrame(agg_records)
    df_raw    = pd.DataFrame(raw_records)

    # ── Global table (pooled across all regimes) ───────────────────────────────
    global_rows = []
    for (method, future_idx), grp in df_agg.groupby(['method', 'future_idx']):
        vr = grp['valid_rate'].mean()
        cr = grp['cat_rate'].mean()
        br = grp['beats_rate'].mean()
        me = grp['med_error'].median()
        mi = grp['med_improve'].median()
        sc = stability_score(vr, cr, br, cfg_ref)
        global_rows.append({
            'method':      method,
            'family':      FAMILY.get(method, 'unknown'),
            'method_type': METHOD_TYPE[method],
            'future_idx':  future_idx,
            'valid_rate':  round(vr, 4),
            'cat_rate':    round(cr, 4),
            'beats_rate':  round(br, 4),
            'med_error':   me,
            'med_improve': mi,
            'stability':   round(sc, 4),
        })
    df_global = (pd.DataFrame(global_rows)
                   .sort_values(['future_idx', 'stability'], ascending=[True, False])
                   .reset_index(drop=True))

    # ── Per-regime best method ─────────────────────────────────────────────────
    best_rows = []
    for (regime, future_idx), grp in df_agg.groupby(['regime', 'future_idx']):
        # Pool across noise levels for regime recommendation
        per_method = (grp.groupby('method')
                        .agg(valid_rate=('valid_rate','mean'),
                             cat_rate=('cat_rate','mean'),
                             beats_rate=('beats_rate','mean'),
                             med_error=('med_error','median'),
                             med_improve=('med_improve','median'))
                        .reset_index())
        per_method['stability'] = per_method.apply(
            lambda r: stability_score(r.valid_rate, r.cat_rate,
                                      r.beats_rate, cfg_ref), axis=1)
        best = per_method.sort_values('stability', ascending=False).iloc[0]
        best_rows.append({
            'regime':      regime,
            'future_idx':  future_idx,
            'best_method': best['method'],
            'family':      FAMILY.get(best['method'], 'unknown'),
            'method_type': METHOD_TYPE[best['method']],
            'stability':   round(best['stability'], 4),
            'valid_rate':  round(best['valid_rate'], 4),
            'cat_rate':    round(best['cat_rate'], 4),
            'med_error':   best['med_error'],
            'med_improve': best['med_improve'],
        })
    df_best = pd.DataFrame(best_rows)

    # ── Heatmaps (one per horizon) ─────────────────────────────────────────────
    heatmaps = {}
    for fid in future_idxs:
        sub = df_agg[df_agg['future_idx'] == fid]
        hm  = (sub.groupby(['method', 'regime'])
                  .apply(lambda g: stability_score(
                      g['valid_rate'].mean(),
                      g['cat_rate'].mean(),
                      g['beats_rate'].mean(), cfg_ref))
                  .unstack(fill_value=float('nan')))
        heatmaps[fid] = hm

    # ── Save ───────────────────────────────────────────────────────────────────
    paths = {}

    p = os.path.join(out_dir, 'phase1_aggregated.csv')
    df_agg.to_csv(p, index=False);  paths['aggregated'] = p
    print(f'  Saved: {p}  ({len(df_agg)} rows)')

    p = os.path.join(out_dir, 'phase1_global.csv')
    df_global.to_csv(p, index=False);  paths['global'] = p
    print(f'  Saved: {p}  ({len(df_global)} rows)')

    p = os.path.join(out_dir, 'phase1_regime_best.csv')
    df_best.to_csv(p, index=False);  paths['regime_best'] = p
    print(f'  Saved: {p}  ({len(df_best)} rows)')

    p = os.path.join(out_dir, 'phase1_raw_sample.csv')
    df_raw.to_csv(p, index=False);  paths['raw_sample'] = p
    print(f'  Saved: {p}  ({len(df_raw)} rows)')

    for fid, hm in heatmaps.items():
        p = os.path.join(out_dir, f'phase1_heatmap_n{fid}.csv')
        hm.round(4).to_csv(p);  paths[f'heatmap_{fid}'] = p
        print(f'  Saved: {p}')

    return {
        'aggregated':  df_agg,
        'global':      df_global,
        'regime_best': df_best,
        'raw_sample':  df_raw,
        'heatmaps':    heatmaps,
    }
