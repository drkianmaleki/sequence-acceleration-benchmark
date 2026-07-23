"""
phase5b.py
==========
Phase 5B — Sensitivity Analysis.

Three sweeps test whether the key Phase 1-4 findings are robust to the
main modelling assumptions.

Sweep 1 — L_inf sensitivity
----------------------------
Question: Does the Phase 2 two-rule cascade still work when the assumed
asymptotic value L_inf differs from the true value?

In real gradient boosting L_inf is unknown and must be guessed.
The true synthetic L_inf is always 0.01.  We vary the ASSUMED L_inf
to simulate practitioner misspecification:

  assumed_L_inf in {0.001, 0.005, 0.010, 0.020, 0.050}

For each assumed value, trajectory features (log_log_slope, richardson_r2)
are recomputed with that L_inf, and the cascade is applied with fixed
Phase 2 thresholds (slope > -0.1, R² < 0.5).

Metric: cascade precision (P(rational_fit better | rule fires)),
recall (P(rule fires | richardson fails)), and mean gain.

Sweep 2 — Window length sensitivity
-------------------------------------
Question: Do the fixed Phase 2 thresholds generalise to different window
lengths?  All Phase 1-4 work used window_len = 60.

  window_len in {20, 40, 60, 80, 100}  (obs_idx fixed at 90)

For each window length, the cascade thresholds are applied WITHOUT
re-fitting.  Measures whether practitioners can use Phase 2 rules
directly without re-tuning for their window length.

Metric: same as Sweep 1, plus optimal threshold at each window_len
(what threshold would maximise precision × recall?).

Sweep 3 — CAT_MULT sensitivity
---------------------------------
Question: Do the Phase 1 regime champions change when the catastrophic
threshold is varied?

  CAT_MULT in {2, 5, 10}  (currently 5 throughout)

For each value, stability scores are recomputed for all 51 methods
and regime champions are identified.

Metric: Kendall's tau between method rankings at different CAT_MULT;
fraction of 54 regime-horizon champion slots that change.

Output files
------------
phase5b_sweep1_global.csv      Cascade metrics vs assumed L_inf (global)
phase5b_sweep1_regime.csv      Cascade metrics vs assumed L_inf (per regime)
phase5b_sweep2_global.csv      Cascade metrics vs window_len (global)
phase5b_sweep2_regime.csv      Cascade metrics vs window_len (per regime)
phase5b_sweep3_champions.csv   Regime champions at each CAT_MULT
phase5b_sweep3_global.csv      Global stability rankings at each CAT_MULT
phase5b_sweep3_concordance.csv Kendall tau between CAT_MULT rankings
figure_p5b_01_linf.png
figure_p5b_02_window.png
figure_p5b_03_catmult.png

Author : Kian Maleki
Date   : 2026-05-24
"""

import os, math, warnings
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from scipy.optimize import curve_fit
from typing import List, Dict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

import src.config as CFG_MOD
from src.accelerators import METHODS, METHOD_NAMES
from src.generators   import GENERATORS, REGIME_NAMES, TRUTH

# ── Method sets ────────────────────────────────────────────────────────────────
CASCADE_METHODS = [
    'current_value', 'richardson_1', 'richardson_a10',
    'single_exp_fit', 'rational_fit', 'pade_22',
    'log_linear', 'weniger_d2', 'anderson_1',
]

ALL_METHODS = list(METHOD_NAMES)
# Single definition lives in src/config.py; see the note there on re-deriving
# it after any change that alters Phase 1 results.
DANGEROUS   = set(CFG_MOD.DANGEROUS_METHODS)

FIG_DPI = 150


# ── Helpers ────────────────────────────────────────────────────────────────────
def _cfg(fid, L_inf=None, cat_mult=None):
    return {
        'L_inf':     L_inf     if L_inf     is not None else CFG_MOD.L_INF,
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


# ── Feature extraction (inline with configurable L_inf) ───────────────────────
def _cascade_features(seq_win, idx_win, L_inf):
    s  = np.asarray(seq_win, dtype=float)
    x  = np.asarray(idx_win, dtype=float)
    L0 = max(0.0, min(L_inf, float(np.min(s)) * 0.5))

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


# =============================================================================
# SWEEP 1 — L_inf SENSITIVITY
# =============================================================================

def sweep1_linf(l_inf_values, obs_idx, window_len, noise_list,
                future_list, n_seeds, out_dir, verbose=True):
    """
    Test Phase 2 cascade robustness to L_inf misspecification.
    True L_inf = 0.01 for all synthetic regimes.
    Assumed L_inf varied in l_inf_values.
    """
    n_arr   = np.arange(max(future_list) + 200, dtype=float)
    wl      = min(window_len, obs_idx)
    records = []

    total = len(l_inf_values) * len(noise_list) * n_seeds * len(REGIME_NAMES)
    done  = 0

    print(f'  Sweep 1: {len(l_inf_values)} L_inf values × '
          f'{len(noise_list)} noise × {n_seeds} seeds × '
          f'{len(REGIME_NAMES)} regimes')

    for L_inf_assumed in l_inf_values:
        for sigma in noise_list:
            for seed in range(n_seeds):
                rng = np.random.RandomState(seed * 137 + int(sigma*1e6) % 9973)

                for regime in REGIME_NAMES:
                    seq_full = GENERATORS[regime](n_arr, rng, sigma)
                    truth_fn = TRUTH[regime]

                    w_start  = max(0, obs_idx - wl + 1)
                    seq_win  = list(seq_full[w_start : obs_idx + 1])
                    idx_win  = list(range(w_start, obs_idx + 1))

                    slope, r2 = _cascade_features(seq_win, idx_win, L_inf_assumed)
                    chosen    = _phase2_cascade(slope, r2)
                    cascade_fired = (chosen == 'rational_fit')

                    for fid in future_list:
                        true_val = float(truth_fn(fid))
                        curr_val = float(seq_full[obs_idx])
                        curr_err = abs(curr_val - true_val)
                        cfg      = _cfg(fid, L_inf=L_inf_assumed)

                        ests = _eval_methods(seq_win, idx_win, fid, cfg,
                                             ['richardson_1','rational_fit'])
                        rich_err = (abs(ests['richardson_1'] - true_val)
                                    if math.isfinite(ests['richardson_1'])
                                    else float('nan'))
                        rat_err  = (abs(ests['rational_fit'] - true_val)
                                    if math.isfinite(ests['rational_fit'])
                                    else float('nan'))

                        # Cascade outcome
                        chosen_est = ests.get(chosen, float('nan'))
                        chosen_err = (abs(chosen_est - true_val)
                                      if math.isfinite(chosen_est)
                                      else float('nan'))

                        # Is Richardson the best we can do?
                        rich_is_best = (math.isfinite(rich_err)
                                        and (not math.isfinite(rat_err)
                                             or rich_err <= rat_err))
                        rat_better   = (math.isfinite(rat_err)
                                        and math.isfinite(rich_err)
                                        and rat_err < rich_err)

                        records.append({
                            'L_inf_assumed': L_inf_assumed,
                            'regime':        regime,
                            'noise':         sigma,
                            'seed':          seed,
                            'future_idx':    fid,
                            'cascade_fired': int(cascade_fired),
                            'rat_better':    int(rat_better),
                            'rich_err':      rich_err,
                            'rat_err':       rat_err,
                            'chosen_err':    chosen_err,
                            'curr_err':      curr_err,
                            'slope':         slope,
                            'r2':            r2,
                        })

                done += 1
                if verbose and done % max(1, total // 10) == 0:
                    print(f'    [{done:>5}/{total}]  {100*done/total:5.1f}%'
                          f'  L_inf={L_inf_assumed:.3f}  sigma={sigma:.3f}',
                          flush=True)

    df = pd.DataFrame(records)

    # ── Aggregate globally and per regime ─────────────────────────────────────
    global_rows, regime_rows = [], []

    for L_inf_v in l_inf_values:
        for fid in future_list:
            for sigma in noise_list:
                sub = df[(df['L_inf_assumed'] == L_inf_v)
                         & (df['future_idx']  == fid)
                         & (df['noise']       == sigma)]
                if sub.empty:
                    continue

                fire    = sub['cascade_fired'] == 1
                correct = fire & (sub['rat_better'] == 1)
                missed  = (~fire) & (sub['rat_better'] == 1)

                precision = (float(correct.sum() / fire.sum())
                             if fire.sum() > 0 else float('nan'))
                recall    = (float(correct.sum() / sub['rat_better'].sum())
                             if sub['rat_better'].sum() > 0 else float('nan'))

                mask = sub['rich_err'].notna() & sub['chosen_err'].notna()
                gain = float((sub.loc[mask,'rich_err']
                              - sub.loc[mask,'chosen_err']).mean()) if mask.sum() > 0 \
                       else float('nan')

                global_rows.append({
                    'L_inf_assumed': L_inf_v,
                    'future_idx':    fid,
                    'noise':         sigma,
                    'fire_rate':     round(float(fire.mean()), 4),
                    'precision':     round(precision, 4) if math.isfinite(precision) else float('nan'),
                    'recall':        round(recall, 4) if math.isfinite(recall) else float('nan'),
                    'mean_gain':     round(gain, 6) if math.isfinite(gain) else float('nan'),
                    'n':             len(sub),
                })

                # Per regime
                for regime in REGIME_NAMES:
                    rsub = sub[sub['regime'] == regime]
                    if rsub.empty:
                        continue
                    rfire = rsub['cascade_fired'] == 1
                    rcorr = rfire & (rsub['rat_better'] == 1)
                    rp    = (float(rcorr.sum() / rfire.sum())
                             if rfire.sum() > 0 else float('nan'))
                    rmask = rsub['rich_err'].notna() & rsub['chosen_err'].notna()
                    rgain = float((rsub.loc[rmask,'rich_err']
                                   - rsub.loc[rmask,'chosen_err']).mean()) \
                            if rmask.sum() > 0 else float('nan')
                    regime_rows.append({
                        'L_inf_assumed': L_inf_v,
                        'regime': regime, 'future_idx': fid, 'noise': sigma,
                        'precision': round(rp, 4) if math.isfinite(rp) else float('nan'),
                        'mean_gain': round(rgain, 6) if math.isfinite(rgain) else float('nan'),
                        'n': len(rsub),
                    })

    df_global = pd.DataFrame(global_rows)
    df_regime = pd.DataFrame(regime_rows)
    _save_csv(df_global, out_dir, 'phase5b_sweep1_global.csv')
    _save_csv(df_regime, out_dir, 'phase5b_sweep1_regime.csv')
    return df_global, df_regime


# =============================================================================
# SWEEP 2 — WINDOW LENGTH SENSITIVITY
# =============================================================================

def sweep2_window(window_lengths, obs_idx, noise_list, future_list,
                  n_seeds, out_dir, verbose=True):
    """
    Test Phase 2 cascade robustness to window length variation.
    Fixed Phase 2 thresholds applied without re-fitting.
    """
    n_arr   = np.arange(max(future_list) + 200, dtype=float)
    L_inf   = CFG_MOD.L_INF
    records = []

    total = len(window_lengths) * len(noise_list) * n_seeds * len(REGIME_NAMES)
    done  = 0

    print(f'  Sweep 2: {len(window_lengths)} window lengths × '
          f'{len(noise_list)} noise × {n_seeds} seeds × '
          f'{len(REGIME_NAMES)} regimes')

    for wl in window_lengths:
        actual_wl = min(wl, obs_idx)

        for sigma in noise_list:
            for seed in range(n_seeds):
                rng = np.random.RandomState(seed * 137 + int(sigma*1e6) % 9973
                                            + wl * 11)

                for regime in REGIME_NAMES:
                    seq_full = GENERATORS[regime](n_arr, rng, sigma)
                    truth_fn = TRUTH[regime]

                    w_start  = max(0, obs_idx - actual_wl + 1)
                    seq_win  = list(seq_full[w_start : obs_idx + 1])
                    idx_win  = list(range(w_start, obs_idx + 1))

                    slope, r2 = _cascade_features(seq_win, idx_win, L_inf)
                    chosen    = _phase2_cascade(slope, r2)
                    cascade_fired = (chosen == 'rational_fit')

                    for fid in future_list:
                        true_val = float(truth_fn(fid))
                        curr_val = float(seq_full[obs_idx])
                        curr_err = abs(curr_val - true_val)
                        cfg      = _cfg(fid)

                        ests = _eval_methods(seq_win, idx_win, fid, cfg,
                                             ['richardson_1','rational_fit'])
                        rich_err = (abs(ests['richardson_1'] - true_val)
                                    if math.isfinite(ests['richardson_1'])
                                    else float('nan'))
                        rat_err  = (abs(ests['rational_fit'] - true_val)
                                    if math.isfinite(ests['rational_fit'])
                                    else float('nan'))
                        chosen_est = ests.get(chosen, float('nan'))
                        chosen_err = (abs(chosen_est - true_val)
                                      if math.isfinite(chosen_est)
                                      else float('nan'))
                        rat_better = (math.isfinite(rat_err)
                                      and math.isfinite(rich_err)
                                      and rat_err < rich_err)

                        records.append({
                            'window_len':  wl,
                            'regime':      regime,
                            'noise':       sigma,
                            'seed':        seed,
                            'future_idx':  fid,
                            'cascade_fired': int(cascade_fired),
                            'rat_better':  int(rat_better),
                            'rich_err':    rich_err,
                            'rat_err':     rat_err,
                            'chosen_err':  chosen_err,
                            'curr_err':    curr_err,
                            'slope':       slope,
                            'r2':          r2,
                        })

                done += 1
                if verbose and done % max(1, total // 10) == 0:
                    print(f'    [{done:>5}/{total}]  {100*done/total:5.1f}%'
                          f'  win={wl}  sigma={sigma:.3f}',
                          flush=True)

    df = pd.DataFrame(records)

    # ── Aggregate ──────────────────────────────────────────────────────────────
    global_rows, regime_rows = [], []

    for wl in window_lengths:
        for fid in future_list:
            for sigma in noise_list:
                sub = df[(df['window_len']  == wl)
                         & (df['future_idx'] == fid)
                         & (df['noise']      == sigma)]
                if sub.empty:
                    continue

                fire    = sub['cascade_fired'] == 1
                correct = fire & (sub['rat_better'] == 1)
                precision = (float(correct.sum() / fire.sum())
                             if fire.sum() > 0 else float('nan'))
                recall    = (float(correct.sum() / sub['rat_better'].sum())
                             if sub['rat_better'].sum() > 0 else float('nan'))
                mask = sub['rich_err'].notna() & sub['chosen_err'].notna()
                gain = float((sub.loc[mask,'rich_err']
                              - sub.loc[mask,'chosen_err']).mean()) \
                       if mask.sum() > 0 else float('nan')

                global_rows.append({
                    'window_len':  wl,
                    'future_idx':  fid,
                    'noise':       sigma,
                    'fire_rate':   round(float(fire.mean()), 4),
                    'precision':   round(precision, 4) if math.isfinite(precision) else float('nan'),
                    'recall':      round(recall, 4) if math.isfinite(recall) else float('nan'),
                    'mean_gain':   round(gain, 6) if math.isfinite(gain) else float('nan'),
                    'n':           len(sub),
                })

                for regime in REGIME_NAMES:
                    rsub = sub[sub['regime'] == regime]
                    if rsub.empty:
                        continue
                    rfire = rsub['cascade_fired'] == 1
                    rcorr = rfire & (rsub['rat_better'] == 1)
                    rp    = (float(rcorr.sum() / rfire.sum())
                             if rfire.sum() > 0 else float('nan'))
                    rmask = rsub['rich_err'].notna() & rsub['chosen_err'].notna()
                    rgain = float((rsub.loc[rmask,'rich_err']
                                   - rsub.loc[rmask,'chosen_err']).mean()) \
                            if rmask.sum() > 0 else float('nan')
                    regime_rows.append({
                        'window_len': wl, 'regime': regime,
                        'future_idx': fid, 'noise': sigma,
                        'precision':  round(rp, 4) if math.isfinite(rp) else float('nan'),
                        'mean_gain':  round(rgain, 6) if math.isfinite(rgain) else float('nan'),
                        'n':          len(rsub),
                    })

    df_global = pd.DataFrame(global_rows)
    df_regime = pd.DataFrame(regime_rows)
    _save_csv(df_global, out_dir, 'phase5b_sweep2_global.csv')
    _save_csv(df_regime, out_dir, 'phase5b_sweep2_regime.csv')
    return df_global, df_regime


# =============================================================================
# SWEEP 3 — CAT_MULT SENSITIVITY
# =============================================================================

def sweep3_catmult(catmult_values, obs_idx, window_len, noise_list,
                   future_list, n_seeds, out_dir, verbose=True):
    """
    Test whether Phase 1 regime champions and global rankings change
    when the catastrophic threshold is varied.
    Uses all 51 methods (no perturb_IQR needed — just central estimates).
    """
    n_arr = np.arange(max(future_list) + 200, dtype=float)
    wl    = min(window_len, obs_idx)
    recs  = []

    total = (len(catmult_values) * len(noise_list)
             * n_seeds * len(REGIME_NAMES))
    done  = 0

    print(f'  Sweep 3: {len(catmult_values)} CAT_MULT × '
          f'{len(noise_list)} noise × {n_seeds} seeds × '
          f'{len(REGIME_NAMES)} regimes  (all 51 methods)')

    for cat_mult in catmult_values:
        for sigma in noise_list:
            for seed in range(n_seeds):
                rng = np.random.RandomState(seed * 137 + int(sigma*1e6) % 9973)

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
                        cfg      = _cfg(fid, cat_mult=cat_mult)

                        for method in ALL_METHODS:
                            fn = METHODS[method]
                            try:
                                est = fn(seq_win, idx_win, float(fid), cfg)
                            except Exception:
                                est = float('nan')

                            valid = _valid(est, cfg)
                            err   = abs(est - true_val) if valid else float('nan')
                            cat   = (not valid) or (
                                valid and curr_err > 1e-12
                                and err > cat_mult * curr_err)
                            beats = valid and curr_err > 1e-12 and err < curr_err

                            recs.append({
                                'cat_mult':    cat_mult,
                                'regime':      regime,
                                'noise':       sigma,
                                'seed':        seed,
                                'method':      method,
                                'future_idx':  fid,
                                'valid':       int(valid),
                                'catastrophic':int(cat),
                                'beats':       int(beats),
                                'error':       err,
                            })

                done += 1
                if verbose and done % max(1, total // 10) == 0:
                    print(f'    [{done:>5}/{total}]  {100*done/total:5.1f}%'
                          f'  CAT_MULT={cat_mult:.0f}  sigma={sigma:.3f}',
                          flush=True)

    df = pd.DataFrame(recs)

    # ── Aggregate stability scores ─────────────────────────────────────────────
    champ_rows, global_rows, concord_rows = [], [], []

    # Reference cfg for W_CAT/W_BEATS
    cfg_ref = _cfg(future_list[0])

    for cat_mult in catmult_values:
        for fid in future_list:
            sub = df[(df['cat_mult'] == cat_mult) & (df['future_idx'] == fid)]

            # Global ranking
            for method, mgrp in sub.groupby('method'):
                vr = mgrp['valid'].mean()
                cr = mgrp['catastrophic'].mean()
                br = mgrp['beats'].mean()
                sc = _stability(vr, cr, br, cfg_ref)
                global_rows.append({
                    'cat_mult':   cat_mult,
                    'future_idx': fid,
                    'method':     method,
                    'stability':  round(sc, 4),
                    'valid_rate': round(float(vr), 4),
                    'cat_rate':   round(float(cr), 4),
                    'beats_rate': round(float(br), 4),
                })

            # Per-regime champion
            for regime in REGIME_NAMES:
                rsub = sub[sub['regime'] == regime]
                per_m = (rsub.groupby('method')
                             .agg(valid_rate=('valid','mean'),
                                  cat_rate=('catastrophic','mean'),
                                  beats_rate=('beats','mean'))
                             .reset_index())
                per_m['stability'] = per_m.apply(
                    lambda r: _stability(r.valid_rate, r.cat_rate,
                                         r.beats_rate, cfg_ref),
                    axis=1)
                best = per_m.sort_values('stability', ascending=False).iloc[0]
                champ_rows.append({
                    'cat_mult':     cat_mult,
                    'regime':       regime,
                    'future_idx':   fid,
                    'champion':     best['method'],
                    'stability':    round(float(best['stability']), 4),
                    'is_dangerous': int(best['method'] in DANGEROUS),
                })

    df_champ  = pd.DataFrame(champ_rows)
    df_global = pd.DataFrame(global_rows)

    # ── Concordance between CAT_MULT settings ─────────────────────────────────
    for fid in future_list:
        sub_g = df_global[df_global['future_idx'] == fid]
        pairs = [(catmult_values[i], catmult_values[j])
                 for i in range(len(catmult_values))
                 for j in range(i+1, len(catmult_values))]
        for cm_a, cm_b in pairs:
            a_rank = (sub_g[sub_g['cat_mult'] == cm_a]
                      .sort_values('stability', ascending=False)
                      .reset_index()['method'])
            b_rank = (sub_g[sub_g['cat_mult'] == cm_b]
                      .sort_values('stability', ascending=False)
                      .reset_index()['method'])
            # Align on common methods
            common = list(set(a_rank) & set(b_rank))
            if len(common) < 5:
                continue
            a_pos = {m: i for i, m in enumerate(a_rank)}
            b_pos = {m: i for i, m in enumerate(b_rank)}
            a_v   = [a_pos[m] for m in common]
            b_v   = [b_pos[m] for m in common]
            tau, _ = kendalltau(a_v, b_v)

            # Fraction of regime champions that agree
            ca = df_champ[(df_champ['cat_mult'] == cm_a)
                          & (df_champ['future_idx'] == fid)]
            cb = df_champ[(df_champ['cat_mult'] == cm_b)
                          & (df_champ['future_idx'] == fid)]
            merged = ca.merge(cb, on=['regime', 'future_idx'],
                              suffixes=('_a','_b'))
            agree  = float((merged['champion_a'] == merged['champion_b']).mean())

            concord_rows.append({
                'cat_mult_a':   cm_a,
                'cat_mult_b':   cm_b,
                'future_idx':   fid,
                'kendall_tau':  round(float(tau), 4),
                'champion_agreement': round(agree, 4),
                'n_methods':    len(common),
            })

    df_concord = pd.DataFrame(concord_rows)

    _save_csv(df_champ,  out_dir, 'phase5b_sweep3_champions.csv')
    _save_csv(df_global, out_dir, 'phase5b_sweep3_global.csv')
    _save_csv(df_concord,out_dir, 'phase5b_sweep3_concordance.csv')
    return df_champ, df_global, df_concord


# =============================================================================
# FIGURES
# =============================================================================

def fig_p5b_01_linf(df_global: pd.DataFrame, out_dir: str) -> str:
    """Cascade precision and recall vs assumed L_inf."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    fid   = df_global['future_idx'].max()
    noise = sorted(df_global['noise'].unique())
    cols  = {n: c for n, c in zip(noise, ['#1565c0','#e65100','#2e7d32'])}

    true_L = 0.01  # true synthetic L_inf

    for ax, metric, ylabel, title in zip(
            axes,
            ['precision', 'recall', 'mean_gain'],
            ['Precision', 'Recall', 'Mean gain (rich_err - chosen_err)'],
            ['Cascade precision', 'Cascade recall', 'Cascade mean gain']):

        sub = df_global[df_global['future_idx'] == fid]
        for sigma in noise:
            sv = sub[sub['noise'] == sigma].sort_values('L_inf_assumed')
            ax.plot(sv['L_inf_assumed'], sv[metric],
                    'o-', color=cols.get(sigma, '#999'),
                    label=f'σ={sigma}', lw=2, markersize=6)

        ax.axvline(true_L, color='black', lw=1, ls='--', alpha=0.5,
                   label='True L_inf')
        if metric == 'precision':
            ax.axhline(0.80, color='red', lw=0.8, ls=':', alpha=0.6,
                       label='0.80 target')
        if metric == 'mean_gain':
            ax.axhline(0, color='black', lw=0.7, alpha=0.4)
        ax.set_xlabel('Assumed L_inf', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xscale('log')
        ax.legend(fontsize=8)

    fig.suptitle(
        f'Figure P5B-1 — Cascade Robustness to L_inf Misspecification\n'
        f'(horizon = {fid};  true synthetic L_inf = {true_L})',
        fontsize=10, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(out_dir, 'figure_p5b_01_linf.png')
    _save(fig, path)
    return path


def fig_p5b_02_window(df_global: pd.DataFrame, out_dir: str) -> str:
    """Cascade precision, recall, gain vs window_len."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fid   = df_global['future_idx'].max()
    noise = sorted(df_global['noise'].unique())
    cols  = {n: c for n, c in zip(noise, ['#1565c0','#e65100','#2e7d32'])}

    for ax, metric, ylabel, title in zip(
            axes,
            ['precision', 'recall', 'mean_gain'],
            ['Precision', 'Recall', 'Mean gain (rich_err - chosen_err)'],
            ['Cascade precision', 'Cascade recall', 'Cascade mean gain']):

        sub = df_global[df_global['future_idx'] == fid]
        for sigma in noise:
            sv = sub[sub['noise'] == sigma].sort_values('window_len')
            ax.plot(sv['window_len'], sv[metric],
                    'o-', color=cols.get(sigma, '#999'),
                    label=f'σ={sigma}', lw=2, markersize=6)

        ax.axvline(60, color='black', lw=1, ls='--', alpha=0.5,
                   label='Phase 2 default (60)')
        if metric == 'precision':
            ax.axhline(0.80, color='red', lw=0.8, ls=':', alpha=0.6,
                       label='0.80 target')
        if metric == 'mean_gain':
            ax.axhline(0, color='black', lw=0.7, alpha=0.4)
        ax.set_xlabel('Window length', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)

    fig.suptitle(
        f'Figure P5B-2 — Cascade Robustness to Window Length Variation\n'
        f'(horizon = {fid};  obs_idx fixed at 90)',
        fontsize=10, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(out_dir, 'figure_p5b_02_window.png')
    _save(fig, path)
    return path


def fig_p5b_03_catmult(df_champ: pd.DataFrame,
                        df_concord: pd.DataFrame,
                        out_dir: str) -> str:
    """Champion stability and ranking concordance vs CAT_MULT."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # Left: champion agreement heatmap
    fid    = df_champ['future_idx'].max()
    cmults = sorted(df_champ['cat_mult'].unique())

    ax = axes[0]
    mat = np.zeros((len(cmults), len(cmults)))
    for _, row in df_concord[df_concord['future_idx'] == fid].iterrows():
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
    plt.colorbar(im, ax=ax, label='Fraction of regime champions agreeing',
                 shrink=0.7)
    ax.set_title('Regime Champion Agreement\n(1.0 = all 18 regimes agree)',
                 fontsize=10, fontweight='bold')

    # Right: Kendall tau
    ax2 = axes[1]
    rows = df_concord[df_concord['future_idx'] == fid]
    labels = [f'CAT={r["cat_mult_a"]:.0f} vs CAT={r["cat_mult_b"]:.0f}'
              for _, r in rows.iterrows()]
    taus   = [r['kendall_tau'] for _, r in rows.iterrows()]
    colours= ['#2e7d32' if t >= 0.9 else '#f57f17' if t >= 0.7 else '#c62828'
              for t in taus]
    ax2.bar(range(len(labels)), taus, color=colours, edgecolor='white')
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("Kendall's τ (method ranking concordance)", fontsize=9)
    ax2.axhline(0.9, color='#2e7d32', lw=1.2, ls='--', alpha=0.7,
                label='τ = 0.90 (high concordance)')
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Method Ranking Concordance\n(1.0 = identical ranking)",
                  fontsize=10, fontweight='bold')
    ax2.legend(fontsize=9)
    for i, (v, t) in enumerate(zip(range(len(taus)), taus)):
        ax2.text(i, t + 0.01, f'{t:.3f}', ha='center', va='bottom', fontsize=9)

    fig.suptitle(
        f'Figure P5B-3 — CAT_MULT Sensitivity\n'
        f'(horizon = {fid})',
        fontsize=10, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(out_dir, 'figure_p5b_03_catmult.png')
    _save(fig, path)
    return path


# =============================================================================
# MASTER RUN FUNCTION
# =============================================================================

def run_all(l_inf_values, window_lengths, catmult_values,
            obs_idx, window_len_default, noise_list, future_list,
            n_seeds, out_dir, verbose=True):
    os.makedirs(out_dir, exist_ok=True)

    print('\n  === SWEEP 1: L_inf Sensitivity ===')
    df1g, df1r = sweep1_linf(l_inf_values, obs_idx, window_len_default,
                              noise_list, future_list, n_seeds, out_dir, verbose)

    print('\n  === SWEEP 2: Window Length Sensitivity ===')
    df2g, df2r = sweep2_window(window_lengths, obs_idx, noise_list,
                                future_list, n_seeds, out_dir, verbose)

    print('\n  === SWEEP 3: CAT_MULT Sensitivity ===')
    df3c, df3g, df3cd = sweep3_catmult(catmult_values, obs_idx,
                                        window_len_default, noise_list,
                                        future_list, n_seeds, out_dir, verbose)

    print('\n  Generating figures ...')
    paths = [
        fig_p5b_01_linf(df1g, out_dir),
        fig_p5b_02_window(df2g, out_dir),
        fig_p5b_03_catmult(df3c, df3cd, out_dir),
    ]

    return {
        'sweep1_global': df1g, 'sweep1_regime': df1r,
        'sweep2_global': df2g, 'sweep2_regime': df2r,
        'sweep3_champions': df3c, 'sweep3_global': df3g,
        'sweep3_concordance': df3cd,
        'figures': [p for p in paths if p],
    }
