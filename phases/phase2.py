"""
phase2.py
=========
Phase 2 — Richardson failure characterisation (redesign v2, descriptive).

Phase 2 characterises richardson_1's own failure behaviour by its raw error
and by its error normalised by the last-value error, over a grid of
observation depths and noise levels on the core regimes.  The other members
of the pool (src.pipeline.PHASE2_POOL: eight accelerators plus the last-value
trivial as the floor option, with constant_assumed and constant_oracle
reported alongside) are reported descriptively -- the same panel for every
method, no winner, no rank, no composite score.  Candidate threshold rules
are evaluated by what happened when they fired: the descriptive panels of
richardson_1 and of the routed method over the records of the fired cells,
the fraction of records and of cells where the routed method's error was
lower, and the median relative change (src.panels.rule_panel).

Components
----------
  1. Sweep evaluation   -- every pool method on every (regime, obs_idx, noise,
                           seed, gap stratum) cell; per-record output
                           (phase2_records.csv, git-ignored) and the per-cell
                           descriptive panel (phase2_sweep_aggregated.csv).
  2. Feature extraction -- six trajectory features per window (dynamic L0
                           baseline so near-converged windows give no NaN).
  3. Richardson targets -- per cell: R_R_med, the median over valid records
                           of R_R = E_R / E_last (src.trivial.skill_score
                           rule: if E_last <= SKILL_EPS the ratio is 1.0 when
                           E_R <= SKILL_EPS and +inf otherwise; NaN when
                           either error is invalid), and log_med_error_R, the
                           natural log of Richardson's cell-median error
                           (conditional on validity).  The records and cells
                           where the E_last <= SKILL_EPS branch fired are
                           counted per horizon (phase2_denominator_counts.csv).
  4. Correlation        -- Spearman rank correlation between each cell-level
                           window feature (features averaged over the cell's
                           seeds) and (i) R_R_med, (ii) log_med_error_R; per
                           regime (MIN_OBS finite cells required) and pooled
                           over regimes; cells where a target is NaN are
                           dropped and counted.  +inf ranks largest.
  5. Rule evaluation    -- every candidate rule through src.panels.rule_panel.

Redesign v2
-----------
  * Targets are gap-stratified: for every (regime, obs_idx) and every g in
    config.HORIZON_GAP_FRACTIONS the target index is n_f = first n > obs_idx
    with gap(n) <= g * gap(obs_idx), capped at config.HORIZON_N_CAP with the
    achieved fraction recorded.  Records and aggregations are keyed by
    target_g; capped cells are flagged and excluded from every pooled
    cross-regime statistic (targets, correlations, rules).
  * Core regimes only (this phase feeds selector / cascade training).
  * Methods receive L_hat (ASSUMED_L_MODE "zero"); L_true is hidden.
  * Every record carries L_true, L_hat, target_g, achieved_g, n_f, capped,
    E_last (the last-value error of that seed), skill (hindsight best-of-four,
    strict) and skill_vs_* / win_vs_* against each deployable trivial.

Author : Kian Maleki
Date   : 2026-05-24 (v1), 2026-09-19 (redesign v2), 2026-09-27 (descriptive reporting)
"""

import os, math, warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.optimize import curve_fit
from typing import List, Dict, Optional, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

import src.config as CFG_MOD
from src.accelerators import METHODS
from src.generators   import regime_functions
from src.asymptote    import assumed_asymptote
from src.panels       import (PANEL_COLS, RULE_COLS, error_panel, rule_panel, threshold_rule,
                              zero_denominator_flags)
from src.pipeline     import (PHASE2_POOL, REFERENCE_METHODS, capped_block, exclude_capped,
                              horizon_meta, method_flags, resolve_regimes)
from src.trivial      import (ORACLE_METHODS, aggregate_skill_vs, best_reference_error,
                              skill_score, skill_vs_table)

# ── Method pool ────────────────────────────────────────────────────────────────
PHASE2_BASE_METHODS = [
    *PHASE2_POOL,        # src.pipeline: one definition for Phases 2, 3, 4 and 5a
]
# Redesign v2: the trivial constant predictors are first-class comparators.
PHASE2_METHODS = PHASE2_BASE_METHODS + ['constant_assumed', 'constant_oracle']
# Selector candidates (Phase 3) are the pool; constant_assumed gets the oracle's
# treatment: evaluated, reported alongside, never a candidate (Report-2 review,
# decision 4).
RANK_POOL = list(PHASE2_BASE_METHODS)
UNRANKED_COMPARATORS = [m for m in PHASE2_METHODS if m not in RANK_POOL]
assert set(UNRANKED_COMPARATORS) == {'constant_assumed', 'constant_oracle'}
assert not set(RANK_POOL) & ORACLE_METHODS
RICHARDSON = 'richardson_1'

METHOD_COLOURS = {
    'last_value':       '#888888',
    'richardson_1':     '#f4a261',
    'richardson_a10':   '#e76f51',
    'single_exp_fit':   '#2196f3',
    'rational_fit':     '#1565c0',
    'pade_22':          '#e91e63',
    'log_linear':       '#00897b',
    'levin_t2':         '#9c27b0',
    'anderson_1':       '#795548',
    'constant_assumed': '#212121',
    'constant_oracle':  '#000000',
}

FIG_DPI = 150
CELL_KEYS = ['regime', 'obs_idx', 'noise', 'target_g']
WINDOW_KEYS = ['regime', 'obs_idx', 'noise']          # a cell at a fixed stratum
RECORD_COLS = ['method', 'regime', 'obs_idx', 'noise', 'seed', 'target_g',
               'estimate', 'error', 'valid', 'catastrophic', 'E_last', 'capped',
               'L_true', 'L_hat', 'n_f', 'achieved_g']
MIN_OBS = 15                                          # finite cells per correlation


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
        'future_idx':     future_idx,
    }
    if L_true is not None:
        cfg['L_true'] = float(L_true)
    return cfg


def _valid(v: float, cfg: dict) -> bool:
    return bool(math.isfinite(v) and
                cfg['min_valid'] <= v <= cfg['max_valid'])


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
    ('richardson_r2', '<', 0.50, 'levin_t2'),
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
    ('oscillation_idx', '>', 0.05, 'levin_t2'),
    # Curvature-based rules
    ('curvature_idx', '>', 0.10, 'rational_fit'),
    ('curvature_idx', '>', 0.20, 'pade_22'),
    # Ratio-consistency rules — targeting staircase/plateau
    ('diff_ratio_cv', '<', 0.01, 'levin_t2'),
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
              verbose:      bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Evaluate PHASE2_METHODS across all (obs_idx, noise, gap stratum) grids on
    the core regimes.  Returns (df_agg, df_feat, df_rec): the per-cell
    descriptive panels, the per-window features and the per-record frame.
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
                            rec = {
                                'method':       method,
                                'regime':       regime,
                                'obs_idx':      obs_idx,
                                'noise':        sigma,
                                'seed':         seed,
                                'L_true':       L_true,
                                'L_hat':        L_hat,
                                'estimate':     est if valid else float('nan'),
                                'error':        err if valid else float('nan'),
                                'valid':        int(valid),
                                'catastrophic': int(cat),
                                'E_last':       errs['last_value'],
                                'ref_error':    ref_err,
                                'skill':        skill_score(err, ref_err) if valid else float('nan'),
                            }
                            rec.update(skill_vs_table(err if valid else float('nan'), errs))
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

    df_rec  = pd.DataFrame(sweep_records)
    df_feat = pd.DataFrame(feature_records)

    # ── Aggregate: the descriptive panel of the seeds per (method, cell) ───────
    agg = []
    for keys, grp in df_rec.groupby(['method'] + CELL_KEYS, sort=False):
        method, regime, obs_idx, sigma, g = keys
        panel = error_panel(grp['error'], grp['valid'], grp['catastrophic'], grp['E_last'])
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
            'n_total':    panel['n_total'],
            'n_valid':    panel['n_valid'],
            'valid_rate': round(panel['valid_rate'], 4),
            'cat_rate':   round(panel['cat_rate'], 4),
            'mean_error': panel['mean_error'],
            'sd_error':   panel['sd_error'],
            'med_error':  panel['med_error'],
            'q25_error':  panel['q25_error'],
            'q75_error':  panel['q75_error'],
            'p90_error':  panel['p90_error'],
            'med_skill':  _med(grp['skill']),
            'n_seeds':    int(len(grp)),
            **aggregate_skill_vs(grp),
            'win_rate_vs_last': round(panel['win_rate_vs_last'], 4),
        })

    df_agg = pd.DataFrame(agg)

    # ── Save ───────────────────────────────────────────────────────────────────
    p = os.path.join(out_dir, 'phase2_records.csv')
    df_rec[RECORD_COLS].to_csv(p, index=False)
    print(f'  Saved: {p}  ({len(df_rec)} rows; git-ignored)')

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

    return df_agg, df_feat, df_rec


# =============================================================================
# 2.  RICHARDSON TARGETS
# =============================================================================

def richardson_targets(df_rec: pd.DataFrame, g: float, out_dir: str) -> pd.DataFrame:
    """
    Per cell (regime, obs_idx, noise) at stratum g, richardson_1's two targets:

        R_R_med          median over valid records of R_R = E_R / E_last
                         (src.trivial.skill_score: if E_last <= SKILL_EPS the
                         ratio is 1.0 when E_R <= SKILL_EPS and +inf otherwise;
                         NaN when either error is invalid)
        log_med_error_R  natural log of the cell-median error over valid records

    plus med_error_R, n_valid_R, n_total, n_zero_denominator (records where
    the E_last <= SKILL_EPS branch fired) and zero_denominator_cell (any such
    record).  Every cell is written with its capped flag; the summaries that
    follow apply the capped-exclusion rule.  File: phase2_richardson_targets_g{g}.csv.
    """
    sub = df_rec[(df_rec['target_g'] == g) & (df_rec['method'] == RICHARDSON)]
    rows = []
    for cell, grp in sub.groupby(WINDOW_KEYS, sort=False):
        regime, obs_idx, sigma = cell
        e = grp['error'].to_numpy(dtype=float)
        el = grp['E_last'].to_numpy(dtype=float)
        ok = grp['valid'].to_numpy().astype(bool)
        ratios = np.array([skill_score(a, b) for a, b in zip(e[ok], el[ok])], dtype=float)
        ratios = ratios[~np.isnan(ratios)]
        med_err = float(np.median(e[ok])) if ok.any() else float('nan')
        zero = zero_denominator_flags(e, el)
        with np.errstate(divide='ignore'):
            log_med = float(np.log(med_err)) if math.isfinite(med_err) else float('nan')
        rows.append({
            'regime':               regime,
            'obs_idx':              obs_idx,
            'noise':                sigma,
            'target_g':             g,
            'n_f':                  float(grp['n_f'].median()),
            'achieved_g':           _med(grp['achieved_g']),
            'capped':               int(grp['capped'].max()),
            'R_R_med':              float(np.median(ratios)) if ratios.size else float('nan'),
            'log_med_error_R':      log_med,
            'med_error_R':          med_err,
            'n_valid_R':            int(ok.sum()),
            'n_total':              int(len(grp)),
            'n_zero_denominator':   int(zero.sum()),
            'zero_denominator_cell': int(zero.any()),
        })
    df_t = pd.DataFrame(rows)
    p = os.path.join(out_dir, f'phase2_richardson_targets_g{g:g}.csv')
    df_t.to_csv(p, index=False)
    print(f'  Saved: {p}  ({len(df_t)} rows)')
    return df_t


def denominator_counts(targets_by_g: Dict[float, pd.DataFrame], out_dir: str) -> pd.DataFrame:
    """
    Per horizon, over the uncapped cells (the cells that enter the
    correlations): the richardson_1 records and the cells where the
    E_last <= SKILL_EPS branch of the normalised error fired, with the record
    and cell totals.  File: phase2_denominator_counts.csv (zeros when none).
    """
    rows = []
    for g in sorted(targets_by_g):
        t = exclude_capped(targets_by_g[g])
        rows.append({
            'target_g':                   g,
            'n_records_zero_denominator': int(t['n_zero_denominator'].sum()),
            'n_cells_affected':           int(t['zero_denominator_cell'].sum()),
            'n_records_total':            int(t['n_total'].sum()),
            'n_cells_total':              int(len(t)),
        })
    df = pd.DataFrame(rows, columns=['target_g', 'n_records_zero_denominator', 'n_cells_affected',
                                     'n_records_total', 'n_cells_total'])
    p = os.path.join(out_dir, 'phase2_denominator_counts.csv')
    df.to_csv(p, index=False)
    print(f'  Saved: {p}  ({len(df)} rows)')
    return df


# =============================================================================
# 3.  CORRELATION ANALYSIS
# =============================================================================

def _feature_means(df_feat: pd.DataFrame) -> pd.DataFrame:
    """Cell-level features: the window features averaged over the cell's seeds."""
    return (df_feat.drop(columns=['seed'])
                   .groupby(WINDOW_KEYS)
                   .mean()
                   .reset_index())


def run_correlation_analysis(df_feat:    pd.DataFrame,
                             df_targets: pd.DataFrame,
                             g:          float,
                             out_dir:    str,
                             suffix:     str = '') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Spearman rank correlation between each cell-level window feature and
    (i) R_R_med, (ii) log_med_error_R at stratum g; per regime and pooled
    over regimes ('ALL').  Capped cells are excluded.  A cell is dropped from
    BOTH correlations when the feature or either target is NaN (in practice
    the two targets are NaN together: no valid richardson_1 record); the
    dropped cells are counted in n_dropped_nan.  +inf (the E_last <= SKILL_EPS
    branch) is a value and ranks largest; -inf (a conditional median error of
    exactly 0, log_med_error_R) is kept and ranks smallest.  A row needs
    MIN_OBS kept cells.  Returns (df_corr, base) where base is the merged
    cell table.
    """
    tgt = exclude_capped(df_targets)
    base = tgt.merge(_feature_means(df_feat), on=WINDOW_KEYS, how='left')
    base['target_g'] = g

    def _rows(label: str, sub: pd.DataFrame) -> List[dict]:
        out = []
        for feat in FEATURE_COLS:
            mask = sub[feat].notna() & sub['R_R_med'].notna() & sub['log_med_error_R'].notna()
            n = int(mask.sum())
            row = {'regime': label, 'target_g': g, 'feature': feat,
                   'spearman_vs_RR': float('nan'), 'p_vs_RR': float('nan'),
                   'spearman_vs_log_err': float('nan'), 'p_vs_log_err': float('nan'),
                   'n_cells': n, 'n_dropped_nan': int(len(sub) - n)}
            if n >= MIN_OBS:
                r1, p1 = spearmanr(sub.loc[mask, feat], sub.loc[mask, 'R_R_med'])
                r2, p2 = spearmanr(sub.loc[mask, feat], sub.loc[mask, 'log_med_error_R'])
                row.update({'spearman_vs_RR': round(float(r1), 4), 'p_vs_RR': round(float(p1), 4),
                            'spearman_vs_log_err': round(float(r2), 4), 'p_vs_log_err': round(float(p2), 4)})
            out.append(row)
        return out

    corr_rows = _rows('ALL', base)
    for regime in sorted(base['regime'].unique()):
        corr_rows += _rows(regime, base[base['regime'] == regime])

    df_corr = pd.DataFrame(corr_rows)
    p = os.path.join(out_dir, f'phase2_correlations{suffix}.csv')
    df_corr.to_csv(p, index=False)
    print(f'  Saved: {p}  ({len(df_corr)} rows)')
    return df_corr, base


# =============================================================================
# 4.  RULE EVALUATION
# =============================================================================

def evaluate_rules(df_rec:  pd.DataFrame,
                   df_feat: pd.DataFrame,
                   df_agg:  pd.DataFrame,
                   g:       float,
                   out_dir: str,
                   suffix:  str = '') -> pd.DataFrame:
    """
    Every candidate rule (feature, operator, threshold, alternative) through
    src.panels.rule_panel at stratum g: the eligible cells are the uncapped
    (regime, obs_idx, noise) cells whose cell-level feature is finite; the
    records are richardson_1's and the routed method's records of those
    cells.  One row per rule, the rule_panel columns after the rule's
    definition; rules are listed in CANDIDATE_RULES order (no ranking).
    """
    cells = (exclude_capped(df_agg[(df_agg['target_g'] == g) & (df_agg['method'] == RICHARDSON)])
             [WINDOW_KEYS].drop_duplicates()
             .merge(_feature_means(df_feat), on=WINDOW_KEYS, how='left'))
    rec_g = df_rec[df_rec['target_g'] == g]
    rec_r1 = rec_g[rec_g['method'] == RICHARDSON]

    rule_rows = []
    for feat, op, thresh, alt in CANDIDATE_RULES:
        if feat not in cells.columns:
            continue
        elig, fired = threshold_rule(cells, feat, op, thresh)
        rec_alt = rec_g[rec_g['method'] == alt]
        row = {'target_g': g, 'feature': feat, 'operator': op, 'threshold': thresh,
               'alternative': alt}
        row.update(rule_panel(elig, fired, rec_r1, rec_alt, WINDOW_KEYS))
        rule_rows.append(row)

    cols = ['target_g', 'feature', 'operator', 'threshold', 'alternative', *RULE_COLS]
    df_rules = pd.DataFrame(rule_rows, columns=cols)
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


# ── Figure 1 — Richardson's normalised error over depth × noise ───────────────

def fig_p2_01_phase_diagram(df_targets: pd.DataFrame,
                            out_dir:    str) -> str:
    """2-D (obs_idx x noise) map of richardson_1's R_R_med (log10), per key
    regime and pooled over regimes (median over regimes, capped excluded)."""
    key_regimes = [
        'rational_decay', 'osc_exp', 'staircase',
        'broken_power_law', 'single_exp', 'power_law',
    ]
    obs_vals   = sorted(df_targets['obs_idx'].unique())
    noise_vals = sorted(df_targets['noise'].unique())
    g          = float(df_targets['target_g'].iloc[0]) if len(df_targets) else float('nan')

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes_flat  = axes.flatten()

    def _draw(ax, df_sub, title):
        mat = np.full((len(noise_vals), len(obs_vals)), np.nan)
        for _, row in df_sub.iterrows():
            ni = _nearest_idx(noise_vals, row['noise'])
            oi = _nearest_idx(obs_vals,   row['obs_idx'])
            mat[ni, oi] = row['R_R_med']
        with np.errstate(divide='ignore', invalid='ignore'):
            lmat = np.log10(mat)
        finite = lmat[np.isfinite(lmat)]
        lim = max(1.0, float(np.abs(finite).max())) if finite.size else 1.0
        im = ax.imshow(np.clip(lmat, -lim, lim), cmap='RdBu_r', aspect='auto',
                       vmin=-lim, vmax=lim, interpolation='nearest')
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
                    ax.text(oi, ni, f'{v:.2g}', ha='center', va='center', fontsize=6)
                elif np.isinf(v):
                    ax.text(oi, ni, 'inf', ha='center', va='center', fontsize=6)
        return im

    pooled = (exclude_capped(df_targets).groupby(['obs_idx', 'noise'])['R_R_med']
              .median().reset_index())
    im = _draw(axes_flat[0], pooled, 'ALL REGIMES (median over regimes, capped excluded)')
    plt.colorbar(im, ax=axes_flat[0], label='log10 R_R_med', shrink=0.8)

    for k, regime in enumerate(key_regimes):
        ax  = axes_flat[k + 1]
        sub = df_targets[df_targets['regime'] == regime]
        cap = ' [CAP]' if (len(sub) and sub['capped'].max() == 1) else ''
        im  = _draw(ax, sub, regime.replace('_', '\n') + cap)
        plt.colorbar(im, ax=ax, label='log10 R_R_med', shrink=0.8)

    axes_flat[7].axis('off')
    axes_flat[7].text(0.5, 0.5,
        'R_R_med = median over valid seeds of\n'
        'richardson_1 error / last-value error\n\n'
        'Blue  (< 1) = Richardson below the last value\n'
        'Red   (> 1) = Richardson above the last value\n'
        'inf = last-value error at zero (documented branch)\n'
        'Blank = no valid richardson_1 record\n\n'
        'Rows    = noise level\n'
        'Columns = obs_idx\n'
        '[CAP] = some cells hit the horizon cap',
        ha='center', va='center', fontsize=10,
        transform=axes_flat[7].transAxes)

    fig.suptitle(
        f"Figure P2-1 — richardson_1's last-value-normalised error over depth × noise\n"
        f'Gap stratum g = {g:g}; conditional on validity',
        fontsize=11, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(out_dir, 'figure_p2_01_phase_diagram.png')
    _save(fig, path)
    return path


# ── Figure 2 — R_R_med by obs_idx ─────────────────────────────────────────────

def fig_p2_02_crossover(df_targets: pd.DataFrame,
                        out_dir:    str) -> str:
    """richardson_1's R_R_med vs obs_idx for each regime (lowest noise level)."""
    sub = df_targets[df_targets['noise'] < 1e-9].copy()
    if sub.empty:
        min_noise = df_targets['noise'].min()
        sub = df_targets[df_targets['noise'] == min_noise].copy()
    obs_vals = sorted(sub['obs_idx'].unique())

    fig, ax = plt.subplots(figsize=(13, 7))
    for regime in sorted(sub['regime'].unique()):
        rsub = sub[sub['regime'] == regime].sort_values('obs_idx')
        if rsub.empty:
            continue
        below = bool((rsub['R_R_med'] < 1).any())
        colour = '#2e7d32' if below else '#c62828'
        lw     = 2.0       if below else 1.0
        y = rsub['R_R_med'].replace([np.inf], np.nan)
        ax.plot(rsub['obs_idx'], y, lw=lw, color=colour, alpha=0.75)
        last = rsub.iloc[-1]
        if np.isfinite(last['R_R_med']):
            ax.text(last['obs_idx'] + 1, last['R_R_med'], regime[:8], fontsize=6,
                    va='center', color=colour)

    ax.axhline(1, color='#555', lw=0.8, ls='--', alpha=0.5, label='parity with the last value')
    ax.set_yscale('log')
    ax.set_xlabel('obs_idx  (observation depth)', fontsize=10)
    ax.set_ylabel('R_R_med  (richardson_1 error / last-value error, median over valid seeds)', fontsize=9)
    ax.set_title(
        "Figure P2-2 — richardson_1's normalised error vs observation depth\n"
        'Green = below the last value at some depth; Red = never  (lowest noise level; +inf not drawn)',
        fontsize=10, fontweight='bold')
    ax.set_xticks(obs_vals)
    ax.set_xticklabels(obs_vals, fontsize=8)
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = os.path.join(out_dir, 'figure_p2_02_crossover.png')
    _save(fig, path)
    return path


# ── Figure 3 — Per-regime correlation heatmap ─────────────────────────────────

def fig_p2_03_correlations(df_corr: pd.DataFrame,
                            out_dir: str) -> str:
    """Spearman r vs R_R_med per (feature x regime), excluding ALL and NaN-only regimes."""
    sub = df_corr[df_corr['regime'] != 'ALL'].copy()
    sub = sub[sub['spearman_vs_RR'].notna()]
    if sub.empty:
        print('  Fig P2-3 skipped: no per-regime correlations with enough data.')
        return ''

    pivot   = sub.pivot_table(index='feature', columns='regime',
                               values='spearman_vs_RR')
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
                 label='Spearman r  (feature vs R_R_med)')
    ax.set_title(
        'Figure P2-3 — Feature × Regime Correlation with richardson_1\'s normalised error\n'
        'Red = feature↑ → larger R_R_med (Richardson worse vs the last value);  '
        'Blue = feature↑ → smaller R_R_med',
        fontsize=10, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(out_dir, 'figure_p2_03_correlations.png')
    _save(fig, path)
    return path


# ── Figure 4 — Global feature correlations bar chart ─────────────────────────

def fig_p2_04_global_correlations(df_corr: pd.DataFrame,
                                   out_dir: str) -> str:
    """Horizontal bars of the pooled Spearman r for both targets, with significance flags."""
    sub = df_corr[df_corr['regime'] == 'ALL'].copy()
    if sub['spearman_vs_RR'].isna().all() and sub['spearman_vs_log_err'].isna().all():
        print('  Fig P2-4 skipped.')
        return ''

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, (rcol, pcol, label) in zip(axes, [
            ('spearman_vs_RR', 'p_vs_RR', 'R_R_med (error / last-value error)'),
            ('spearman_vs_log_err', 'p_vs_log_err', 'log median error')]):
        s = sub.dropna(subset=[rcol]).sort_values(rcol, key=abs, ascending=True)
        colours = ['#c62828' if r > 0 else '#1565c0' for r in s[rcol]]
        ax.barh(range(len(s)), s[rcol], color=colours, edgecolor='white', lw=0.5, height=0.6)
        ax.set_yticks(range(len(s)))
        ax.set_yticklabels(s['feature'], fontsize=10)
        ax.axvline(0, color='black', lw=0.7)
        ax.axvline( 0.30, color='#43a047', lw=1.2, ls='--', alpha=0.8, label='|r| = 0.30')
        ax.axvline(-0.30, color='#43a047', lw=1.2, ls='--', alpha=0.8)
        for i, (_, row) in enumerate(s.iterrows()):
            r, p = row[rcol], row[pcol]
            sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
            ax.text(r + (0.01 if r >= 0 else -0.01), i, f'{r:+.3f} {sig}  (n={int(row["n_cells"])})',
                    va='center', ha='left' if r >= 0 else 'right', fontsize=8.5,
                    fontweight='bold' if abs(r) >= 0.3 else 'normal')
        ax.set_xlabel(f'Spearman r vs {label}', fontsize=10)
        ax.set_title(f'vs {label}', fontsize=10, fontweight='bold')
        ax.legend(fontsize=9)

    fig.suptitle(
        "Figure P2-4 — Pooled feature correlations with richardson_1's error (all regimes, capped excluded)\n"
        'Red = feature↑ → larger error;  Blue = feature↑ → smaller error',
        fontsize=10, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(out_dir, 'figure_p2_04_global_correlations.png')
    _save(fig, path)
    return path


# ── Figure 5 — Rules: what happened when they fired ──────────────────────────

def fig_p2_05_rules(df_rules: pd.DataFrame,
                    out_dir:  str) -> str:
    """Scatter of every rule: fire rate (x) vs the fraction of fired-cell records
    where the routed method's error was lower (y); annotation = median relative change."""
    valid = df_rules.dropna(subset=['fire_rate', 'lower_error_frac_records']).copy()
    if valid.empty:
        print('  Fig P2-5 skipped.')
        return ''

    colours = [METHOD_COLOURS.get(m, '#999') for m in valid['alternative']]
    sizes   = np.clip(valid['n_cells_fired'] * 2, 20, 400)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(valid['fire_rate'], valid['lower_error_frac_records'],
               c=colours, s=sizes, alpha=0.8, edgecolors='black', linewidths=0.4)

    for _, row in valid.iterrows():
        rel = row['median_rel_change']
        rel_str = f'{rel:+.2f}' if np.isfinite(rel) else 'n/a'
        ax.annotate(
            f"{row['feature'][:9]}{row['operator']}{row['threshold']}\n"
            f"→ {row['alternative'][:10]}  (Δmed {rel_str}; alt V={row['alt_valid_rate']:.2f})",
            (row['fire_rate'], row['lower_error_frac_records']),
            textcoords='offset points', xytext=(5, 3), fontsize=5.5, alpha=0.85)

    ax.axhline(0.5, color='grey', lw=0.7, ls='--', alpha=0.4)
    ax.set_xlabel('Fire rate  (fraction of eligible cells where the rule fired)', fontsize=10)
    ax.set_ylabel('Fraction of fired-cell records where the routed method\'s error was lower\n'
                  '(both valid; an invalid record never counts as lower)', fontsize=9)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(
        'Figure P2-5 — Threshold rules: what happened when they fired\n'
        'Bubble size ∝ fired cells; Δmed = median over fired cells of (alt − r1) / r1 cell-median error; '
        'V = routed method\'s valid rate on the fired records',
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

def make_all_figures(df_targets: pd.DataFrame,
                     df_corr:    pd.DataFrame,
                     df_rules:   pd.DataFrame,
                     g:          float,
                     out_dir:    str) -> list:
    print('\n  Generating figures ...')
    paths = [
        fig_p2_01_phase_diagram(df_targets, out_dir),
        fig_p2_02_crossover(df_targets, out_dir),
        fig_p2_03_correlations(df_corr, out_dir),
        fig_p2_04_global_correlations(df_corr, out_dir),
        fig_p2_05_rules(df_rules, out_dir),
    ]
    return [p for p in paths if p]
