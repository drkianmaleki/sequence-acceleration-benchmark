"""
phase3.py
=========
Phase 3 — Adaptive Selector Pipeline (redesign v2).

Loads Phase 2 data (no new simulations) and builds, evaluates, and
validates a complete adaptive method selector.

Three components
----------------
1. Selector comparison
   Seven selectors ranging from fixed baselines to a full adaptive
   cascade are evaluated on all Phase 2 grid points.
   Metric: mean achieved stability across all (regime, obs_idx, noise,
   gap stratum) combinations.

2. Regime fingerprinting classifier
   A simple decision-tree classifier predicts the convergence regime
   from the six trajectory features.  Accuracy is reported per regime.

3. Leave-one-regime-out cross-validation
   The Phase 2 two-rule cascade and the Phase 3 enhanced cascade are
   applied with fixed thresholds to each held-out regime.

Redesign v2
-----------
  * Grid points are keyed by target_g (gap stratum) instead of a fixed
    horizon; each carries n_f, achieved_g and capped.
  * Capped cells are excluded from every pooled comparison and from the
    cross-validation; they are listed in phase3_capped_cells.csv.
  * Selector candidates are the Phase-2 RANK_POOL (the original 9 methods;
    Report-2 review, decision 4).  constant_assumed and constant_oracle get
    the same treatment: evaluated in Phase 2, never a candidate, never in
    the "oracle" selector (which is the best available *candidate* per cell).
  * The enhanced cascade's horizon-aware default uses the cell's actual
    n_f (rational_fit when n_f <= 300, single_exp_fit otherwise).
  * Training data are the core 18 regimes only (Phase 2 output).

Selectors evaluated
-------------------
fixed_current     always use current_value  (floor)
fixed_richardson  always use richardson_1   (Phase 2 reference)
fixed_rational    always use rational_fit   (short-horizon Phase 1 winner)
fixed_single_exp  always use single_exp_fit (long-horizon Phase 1 winner)
phase2_cascade    Rules 1+2 from Phase 2, default = richardson_1
enhanced_cascade  Rules 1-4, horizon-aware default (Phase 3 contribution)
oracle            best available candidate at each grid point (upper bound)

Input files (from results/phase2/)
-----------------------------------
    phase2_sweep_aggregated.csv
    phase2_features.csv

Output files (to results/phase3/)
----------------------------------
    phase3_selector_comparison.csv  selector x stratum mean stability
    phase3_regime_results.csv       per-regime achieved stability
    phase3_cv_results.csv           leave-one-regime-out CV
    phase3_regime_classifier.csv    regime classification accuracy
    phase3_capped_cells.csv         capped grid cells (excluded above)
    figure_p3_01 ... figure_p3_05

Author : Kian Maleki
Date   : 2026-05-24 (v1), 2026-09-19 (redesign v2)
"""

import os, math, warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

import src.config as CFG_MOD
from src.pipeline import exclude_capped
from phases.phase2 import FEATURE_COLS, RANK_POOL

# ── Constants ──────────────────────────────────────────────────────────────────

CANDIDATES = list(RANK_POOL)          # oracle comparator never a candidate
GRID_KEYS  = ['regime', 'obs_idx', 'noise', 'target_g']

METHOD_COLOURS = {
    'current_value':    '#888888',
    'richardson_1':     '#f4a261',
    'richardson_a10':   '#e76f51',
    'single_exp_fit':   '#2196f3',
    'rational_fit':     '#1565c0',
    'pade_22':          '#e91e63',
    'log_linear':       '#00897b',
    'levin_t2':         '#9c27b0',
    'anderson_1':       '#795548',
    'constant_assumed': '#212121',
}

SELECTOR_COLOURS = {
    'fixed_current':    '#bbbbbb',
    'fixed_richardson': '#f4a261',
    'fixed_rational':   '#1565c0',
    'fixed_single_exp': '#2196f3',
    'phase2_cascade':   '#ff9800',
    'enhanced_cascade': '#2e7d32',
    'oracle':           '#000000',
}

FIG_DPI = 150


# =============================================================================
# 1.  DATA LOADING
# =============================================================================

def load_phase2(phase2_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load Phase 2 sweep_aggregated and features CSVs."""
    agg  = pd.read_csv(os.path.join(phase2_dir, 'phase2_sweep_aggregated.csv'))
    feat = pd.read_csv(os.path.join(phase2_dir, 'phase2_features.csv'))
    if 'target_g' not in agg.columns:
        raise ValueError('phase2_sweep_aggregated.csv predates redesign v2 '
                         '(no target_g column); re-run scripts/run_phase2.py')
    return agg, feat


def build_grid(df_agg: pd.DataFrame,
               df_feat: pd.DataFrame) -> pd.DataFrame:
    """
    Build the evaluation grid: one row per (regime, obs_idx, noise, target_g).
    Each row contains:
      - stability of every candidate method (as columns)
      - n_f, achieved_g, capped for the cell
      - mean feature values (averaged across seeds)
      - best_method  : candidate with highest stability at this grid point
      - best_stability: that method's stability
    """
    sub = df_agg[df_agg['method'].isin(CANDIDATES)]
    pivot = (sub.pivot_table(index=GRID_KEYS, columns='method', values='stability')
                .reset_index())
    pivot.columns.name = None

    meta = (sub.groupby(GRID_KEYS)
               .agg(n_f=('n_f', 'first'), achieved_g=('achieved_g', 'first'),
                    capped=('capped', 'max'))
               .reset_index())
    pivot = pivot.merge(meta, on=GRID_KEYS, how='left')

    for m in CANDIDATES:
        if m not in pivot.columns:
            pivot[m] = float('nan')

    method_cols = [m for m in CANDIDATES if m in pivot.columns]
    pivot['best_stability'] = pivot[method_cols].max(axis=1)
    pivot['best_method']    = pivot[method_cols].idxmax(axis=1)

    feat_avg = (df_feat.drop(columns=['seed'])
                        .groupby(['regime', 'obs_idx', 'noise'])
                        .mean()
                        .reset_index())

    grid = pivot.merge(feat_avg, on=['regime', 'obs_idx', 'noise'], how='left')
    return grid


# =============================================================================
# 2.  SELECTOR DEFINITIONS
# =============================================================================

def _apply_phase2_cascade(row: pd.Series) -> str:
    """
    Phase 2 two-rule cascade.
    Default: richardson_1.
    Rule A: log_log_slope > -0.1  ->  rational_fit
    Rule B: richardson_r2  < 0.50 ->  rational_fit
    """
    slope = row.get('log_log_slope', float('nan'))
    r2    = row.get('richardson_r2', float('nan'))

    if math.isfinite(slope) and slope > -0.10:
        return 'rational_fit'
    if math.isfinite(r2) and r2 < 0.50:
        return 'rational_fit'
    return 'richardson_1'


def _apply_enhanced_cascade(row: pd.Series) -> str:
    """
    Phase 3 enhanced cascade.
    Step 1: horizon-aware default (cell n_f <= 300 -> rational_fit,
            otherwise single_exp_fit).
    Step 2: Rule A  (flat trajectory)
    Step 3: Rule B  (poor power-law fit)
    Step 4: Rule C  (staircase signature)
    Step 5: Rule D  (oscillatory + high ratio variation)
    Step 6: horizon-aware default.
    """
    n_f = row.get('n_f', float('nan'))
    try:
        n_f = float(n_f)
    except (TypeError, ValueError):
        n_f = float('nan')
    default = 'rational_fit' if (math.isfinite(n_f) and n_f <= 300) else 'single_exp_fit'

    slope   = row.get('log_log_slope',   float('nan'))
    r2      = row.get('richardson_r2',   float('nan'))
    curv    = row.get('curvature_idx',   float('nan'))
    osc     = row.get('oscillation_idx', float('nan'))
    d_cv    = row.get('diff_ratio_cv',   float('nan'))

    if math.isfinite(slope) and slope > -0.10:
        return 'rational_fit'
    if math.isfinite(r2) and r2 < 0.50:
        return 'rational_fit'
    if (math.isfinite(curv) and curv > 0.50
            and math.isfinite(d_cv) and d_cv < 0.05):
        return 'levin_t2'
    if (math.isfinite(osc) and osc > 0.30
            and math.isfinite(d_cv) and d_cv > 0.20):
        return 'log_linear'
    return default


SELECTORS = {
    'fixed_current':    lambda row: 'current_value',
    'fixed_richardson': lambda row: 'richardson_1',
    'fixed_rational':   lambda row: 'rational_fit',
    'fixed_single_exp': lambda row: 'single_exp_fit',
    'phase2_cascade':   _apply_phase2_cascade,
    'enhanced_cascade': _apply_enhanced_cascade,
    'oracle':           lambda row: row.get('best_method', 'richardson_1'),
}


def _achieved(row, chosen_col: str) -> float:
    m = row[chosen_col]
    v = row.get(m, float('nan'))
    try:
        v = float(v)
    except (TypeError, ValueError):
        return float('nan')
    return v if math.isfinite(v) else float('nan')


# =============================================================================
# 3.  SELECTOR EVALUATION
# =============================================================================

def evaluate_selectors(grid: pd.DataFrame,
                       out_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply every selector to every grid point.  Pooled means use the
    non-capped cells only; capped cells are written to a separate file.
    Returns:
      df_comparison : mean stability per (selector, target_g)
      df_per_regime : mean stability per (selector, regime, target_g)
    """
    pooled = exclude_capped(grid)
    comp_rows, regime_rows, capped_rows = [], [], []

    for sel_name, sel_fn in SELECTORS.items():
        gc = grid.copy()
        gc['chosen_method'] = gc.apply(sel_fn, axis=1)
        gc['achieved_stability'] = gc.apply(lambda r: _achieved(r, 'chosen_method'), axis=1)
        pc = gc.loc[pooled.index]

        for g, grp in pc.groupby('target_g'):
            n_all = int((grid['target_g'] == g).sum())
            comp_rows.append({
                'selector':          sel_name,
                'target_g':          g,
                'mean_stability':    round(float(grp['achieved_stability'].mean()), 4),
                'n':                 int(len(grp)),
                'n_capped_excluded': n_all - int(len(grp)),
            })

        for (regime, g), grp in pc.groupby(['regime', 'target_g']):
            n_all = int(((grid['regime'] == regime) & (grid['target_g'] == g)).sum())
            regime_rows.append({
                'selector':          sel_name,
                'regime':            regime,
                'target_g':          g,
                'mean_stability':    round(float(grp['achieved_stability'].mean()), 4),
                'n':                 int(len(grp)),
                'n_capped_excluded': n_all - int(len(grp)),
            })

        cap = gc[gc['capped'] == 1]
        for _, r in cap.iterrows():
            capped_rows.append({
                'selector': sel_name, 'regime': r['regime'], 'obs_idx': r['obs_idx'],
                'noise': r['noise'], 'target_g': r['target_g'], 'n_f': r['n_f'],
                'achieved_g': r['achieved_g'], 'chosen_method': r['chosen_method'],
                'achieved_stability': r['achieved_stability'],
            })

    df_comp   = pd.DataFrame(comp_rows)
    df_regime = pd.DataFrame(regime_rows)
    df_capped = pd.DataFrame(capped_rows, columns=[
        'selector', 'regime', 'obs_idx', 'noise', 'target_g', 'n_f', 'achieved_g',
        'chosen_method', 'achieved_stability'])

    for df, name in ((df_comp, 'phase3_selector_comparison.csv'),
                     (df_regime, 'phase3_regime_results.csv'),
                     (df_capped, 'phase3_capped_cells.csv')):
        p = os.path.join(out_dir, name)
        df.to_csv(p, index=False)
        print(f'  Saved: {p}  ({len(df)} rows)')

    return df_comp, df_regime


# =============================================================================
# 4.  LEAVE-ONE-REGIME-OUT CROSS-VALIDATION
# =============================================================================

def cross_validate(grid: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    """
    Leave-one-regime-out cross-validation for the Phase 2 cascade and the
    enhanced cascade, on non-capped cells.  The cascade rules use fixed
    thresholds, so this measures out-of-regime generalisation.
    """
    pooled  = exclude_capped(grid)
    regimes = sorted(pooled['regime'].unique())
    cv_rows = []

    for held_out in regimes:
        test_grid = pooled[pooled['regime'] == held_out].copy()
        for sel_name in ['phase2_cascade', 'enhanced_cascade', 'oracle',
                         'fixed_richardson', 'fixed_single_exp']:
            sel_fn = SELECTORS[sel_name]
            test_grid['chosen'] = test_grid.apply(sel_fn, axis=1)
            achieved = test_grid.apply(lambda r: _achieved(r, 'chosen'), axis=1).mean()
            cv_rows.append({
                'held_out_regime': held_out,
                'selector':        sel_name,
                'mean_stability':  round(float(achieved), 4),
                'n':               int(len(test_grid)),
            })

    df_cv = pd.DataFrame(cv_rows)
    p = os.path.join(out_dir, 'phase3_cv_results.csv')
    df_cv.to_csv(p, index=False)
    print(f'  Saved: {p}  ({len(df_cv)} rows)')
    return df_cv


# =============================================================================
# 5.  REGIME FINGERPRINTING CLASSIFIER
# =============================================================================

def regime_classifier(df_feat: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    """
    Predict convergence regime from trajectory features using a decision tree.
    Falls back to a k-nearest-neighbours implementation if scikit-learn is
    unavailable.  Reports per-regime accuracy and the top confusable pairs.
    """
    feat_avg = (df_feat.drop(columns=['seed'])
                        .groupby(['regime', 'obs_idx', 'noise'])
                        .mean()
                        .reset_index())

    valid = feat_avg.dropna(subset=FEATURE_COLS).copy()
    if len(valid) < 20 or valid['regime'].nunique() < 2:
        print('  Regime classifier: insufficient valid data, skipped.')
        return pd.DataFrame()

    X = valid[FEATURE_COLS].values
    y = valid['regime'].values
    regimes = sorted(set(y))

    clf_name = 'unknown'
    try:
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.model_selection import StratifiedKFold

        clf = DecisionTreeClassifier(max_depth=6, min_samples_leaf=3,
                                     random_state=42)
        n_splits = min(5, int(pd.Series(y).value_counts().min()))
        skf  = StratifiedKFold(n_splits=max(2, n_splits), shuffle=True, random_state=42)
        preds = np.empty(len(y), dtype=object)
        for train_idx, test_idx in skf.split(X, y):
            clf.fit(X[train_idx], y[train_idx])
            preds[test_idx] = clf.predict(X[test_idx])
        clf_name = f'DecisionTree(depth=6, {max(2, n_splits)}-fold CV)'

    except ImportError:
        from scipy.spatial.distance import cdist
        preds = np.empty(len(y), dtype=object)
        n = len(X)
        for i in range(n):
            X_train = np.delete(X, i, axis=0)
            y_train = np.delete(y, i, axis=0)
            dists   = cdist(X[[i]], X_train, metric='euclidean')[0]
            preds[i] = y_train[np.argmin(dists)]
        clf_name = '1-NN (leave-one-out, scipy fallback)'

    rows = []
    for regime in regimes:
        mask   = y == regime
        n_tot  = int(mask.sum())
        n_corr = int((preds[mask] == regime).sum())
        acc    = n_corr / n_tot if n_tot > 0 else float('nan')
        wrong  = preds[mask & (preds != y)]
        top_wrong = (pd.Series(wrong).value_counts().index[0]
                     if len(wrong) > 0 else 'none')
        rows.append({
            'regime':        regime,
            'n_samples':     n_tot,
            'n_correct':     n_corr,
            'accuracy':      round(acc, 3),
            'top_confusion': top_wrong,
        })

    overall_acc = float((preds == y).mean())
    rows.append({
        'regime':    '__OVERALL__',
        'n_samples': len(y),
        'n_correct': int((preds == y).sum()),
        'accuracy':  round(overall_acc, 3),
        'top_confusion': '-',
    })

    df_clf = pd.DataFrame(rows).sort_values('accuracy', ascending=True)
    p = os.path.join(out_dir, 'phase3_regime_classifier.csv')
    df_clf.to_csv(p, index=False)
    print(f'  Saved: {p}  (classifier: {clf_name})')
    print(f'  Overall regime classification accuracy: {overall_acc:.3f}')
    return df_clf


# =============================================================================
# 6.  FIGURES
# =============================================================================

def _save(fig, path: str):
    fig.savefig(path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')


def fig_p3_01_selector_comparison(df_comp: pd.DataFrame,
                                   out_dir: str) -> str:
    """Grouped bar chart: mean achieved stability per selector per stratum."""
    strata     = sorted(df_comp['target_g'].unique(), reverse=True)
    selectors  = list(SELECTORS.keys())
    n_sel      = len(selectors)
    n_hor      = len(strata)

    fig, axes = plt.subplots(1, n_hor, figsize=(5 * n_hor, 7), sharey=True)
    if n_hor == 1:
        axes = [axes]

    for ax, g in zip(axes, strata):
        sub = df_comp[df_comp['target_g'] == g].set_index('selector')
        vals = [float(sub.loc[s, 'mean_stability'])
                if s in sub.index else float('nan')
                for s in selectors]
        colours = [SELECTOR_COLOURS.get(s, '#999') for s in selectors]

        bars = ax.bar(range(n_sel), vals, color=colours,
                      edgecolor='white', lw=0.5)
        ax.set_xticks(range(n_sel))
        ax.set_xticklabels([s.replace('_', '\n') for s in selectors],
                           fontsize=7.5, rotation=0)
        ax.set_title(f'g = {g:g}', fontsize=10, fontweight='bold')
        ax.set_ylabel('Mean achieved stability', fontsize=9)
        ax.axhline(1.0, color='grey', lw=0.8, ls='--', alpha=0.5,
                   label='Current-value floor')

        for bar, v in zip(bars, vals):
            if math.isfinite(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        v + 0.005, f'{v:.3f}',
                        ha='center', va='bottom', fontsize=7.5,
                        fontweight='bold')

    fig.suptitle(
        'Figure P3-1 — Mean Achieved Stability by Selector and Gap Stratum\n'
        '(capped cells excluded)  Green = enhanced_cascade; Black = oracle upper bound',
        fontsize=10, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(out_dir, 'figure_p3_01_selector_comparison.png')
    _save(fig, path)
    return path


def fig_p3_02_improvement_map(df_regime: pd.DataFrame,
                               out_dir: str) -> str:
    """Heatmap: stability gain of enhanced_cascade over fixed_richardson,
    per (regime, stratum)."""
    rich = (df_regime[df_regime['selector'] == 'fixed_richardson']
            .set_index(['regime', 'target_g'])['mean_stability'])
    enh  = (df_regime[df_regime['selector'] == 'enhanced_cascade']
            .set_index(['regime', 'target_g'])['mean_stability'])

    gain = (enh - rich).reset_index()
    gain.columns = ['regime', 'target_g', 'gain']

    strata  = sorted(gain['target_g'].unique(), reverse=True)
    regimes = sorted(gain['regime'].unique())

    mat = np.full((len(regimes), len(strata)), np.nan)
    for _, row in gain.iterrows():
        i = regimes.index(row['regime'])
        j = strata.index(row['target_g'])
        mat[i, j] = row['gain']

    finite = mat[np.isfinite(mat)]
    vmax = max(abs(finite.max()), abs(finite.min()), 0.05) if finite.size else 0.05
    fig, ax = plt.subplots(figsize=(max(7, len(strata) * 2.5), max(4, 0.5 * len(regimes))))
    im = ax.imshow(mat, cmap='RdYlGn', aspect='auto', vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(strata)))
    ax.set_xticklabels([f'g={h:g}' for h in strata], fontsize=9)
    ax.set_yticks(range(len(regimes)))
    ax.set_yticklabels([r.replace('_', '\n') for r in regimes], fontsize=8)

    for i in range(len(regimes)):
        for j in range(len(strata)):
            v = mat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f'{v:+.3f}', ha='center', va='center',
                        fontsize=7.5,
                        color='white' if abs(v) > 0.5 * vmax else '#333')

    plt.colorbar(im, ax=ax, label='Stability gain (enhanced − fixed_richardson)',
                 shrink=0.5)
    ax.set_title(
        'Figure P3-2 — Enhanced Cascade Gain over Fixed Richardson\n'
        'Green = cascade improves; Red = cascade hurts (capped cells excluded)',
        fontsize=10, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(out_dir, 'figure_p3_02_improvement_map.png')
    _save(fig, path)
    return path


def fig_p3_03_classifier_accuracy(df_clf: pd.DataFrame,
                                   out_dir: str) -> str:
    """Horizontal bar chart of regime classification accuracy."""
    if df_clf.empty:
        return ''

    sub = df_clf[df_clf['regime'] != '__OVERALL__'].sort_values('accuracy')
    overall = df_clf[df_clf['regime'] == '__OVERALL__']['accuracy'].values
    overall_acc = float(overall[0]) if len(overall) > 0 else float('nan')

    colours = ['#2e7d32' if a >= 0.7 else
               '#f57f17' if a >= 0.4 else
               '#c62828' for a in sub['accuracy']]

    fig, ax = plt.subplots(figsize=(9, max(6, len(sub) * 0.35)))
    ax.barh(range(len(sub)), sub['accuracy'], color=colours,
            edgecolor='white', lw=0.4, height=0.7)
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels([r.replace('_', '\n') for r in sub['regime']],
                       fontsize=8)

    for i, (acc, wrong) in enumerate(zip(sub['accuracy'], sub['top_confusion'])):
        ax.text(acc + 0.01, i, f'{acc:.2f}  ← {wrong}',
                va='center', fontsize=7.5)

    ax.axvline(overall_acc, color='black', lw=1.5, ls='--',
               label=f'Overall accuracy = {overall_acc:.3f}')
    ax.set_xlabel('Classification accuracy (stratified CV)', fontsize=9)
    ax.set_title(
        'Figure P3-3 — Regime Classification Accuracy from Trajectory Features\n'
        'Arrow shows most common misclassification target',
        fontsize=10, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1.15)
    fig.tight_layout()
    path = os.path.join(out_dir, 'figure_p3_03_classifier_accuracy.png')
    _save(fig, path)
    return path


def fig_p3_04_obs_depth(df_regime: pd.DataFrame,
                         grid: pd.DataFrame,
                         default_g: float,
                         out_dir: str) -> str:
    """Selector mean achieved stability vs obs_idx at the headline stratum
    (capped cells excluded)."""
    sub  = exclude_capped(grid[grid['target_g'] == default_g]).copy()
    if sub.empty:
        print('  Fig P3-4 skipped: no non-capped cells at the headline stratum.')
        return ''
    obs_vals = sorted(sub['obs_idx'].unique())

    result_rows = []
    for sel_name, sel_fn in SELECTORS.items():
        sub['chosen'] = sub.apply(sel_fn, axis=1)
        sub['achieved'] = sub.apply(lambda r: _achieved(r, 'chosen'), axis=1)
        for obs in obs_vals:
            ms = sub[sub['obs_idx'] == obs]['achieved'].mean()
            result_rows.append({'selector': sel_name, 'obs_idx': obs,
                                'mean_stability': ms})

    df_obs = pd.DataFrame(result_rows)

    fig, ax = plt.subplots(figsize=(10, 6))
    for sel_name in SELECTORS:
        s = df_obs[df_obs['selector'] == sel_name].sort_values('obs_idx')
        if s.empty:
            continue
        lw  = 2.5 if sel_name in ('enhanced_cascade', 'oracle') else 1.2
        ls  = '--' if sel_name == 'oracle' else '-'
        ax.plot(s['obs_idx'], s['mean_stability'],
                label=sel_name.replace('_', ' '),
                color=SELECTOR_COLOURS.get(sel_name, '#999'),
                lw=lw, ls=ls, alpha=0.9)

    ax.set_xlabel('obs_idx (observation depth)', fontsize=10)
    ax.set_ylabel('Mean achieved stability', fontsize=10)
    ax.set_title(
        f'Figure P3-4 — Selector Performance vs Observation Depth\n'
        f'(gap stratum g = {default_g:g}, all regimes, all noise levels, capped excluded)',
        fontsize=10, fontweight='bold')
    ax.legend(fontsize=8, loc='lower right')
    ax.set_xticks(obs_vals)
    ax.set_xticklabels(obs_vals, fontsize=8)
    ax.axhline(1.0, color='grey', lw=0.7, ls=':', alpha=0.5)
    fig.tight_layout()
    path = os.path.join(out_dir, 'figure_p3_04_obs_depth.png')
    _save(fig, path)
    return path


def fig_p3_05_cv_summary(df_cv: pd.DataFrame, out_dir: str) -> str:
    """Box plot: leave-one-regime-out mean stability distribution."""
    sel_order = ['fixed_richardson', 'fixed_single_exp',
                 'phase2_cascade', 'enhanced_cascade', 'oracle']
    sel_order = [s for s in sel_order if s in df_cv['selector'].unique()]
    if not sel_order:
        return ''

    data    = [df_cv[df_cv['selector'] == s]['mean_stability'].dropna().values
               for s in sel_order]
    colours = [SELECTOR_COLOURS.get(s, '#999') for s in sel_order]
    labels  = [s.replace('_', '\n') for s in sel_order]
    n_reg   = df_cv['held_out_regime'].nunique()

    fig, ax = plt.subplots(figsize=(9, 6))
    bps = ax.boxplot(data, positions=range(len(sel_order)), widths=0.5,
                     patch_artist=True,
                     medianprops=dict(color='white', lw=2),
                     flierprops=dict(marker='.', markersize=4, alpha=0.5))
    for patch, colour in zip(bps['boxes'], colours):
        patch.set_facecolor(colour)
        patch.set_alpha(0.8)

    ax.set_xticks(range(len(sel_order)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Mean achieved stability (held-out regime)', fontsize=10)
    ax.set_title(
        'Figure P3-5 — Leave-One-Regime-Out Cross-Validation\n'
        f'Box shows distribution across {n_reg} held-out regimes; '
        'median line = overall CV performance',
        fontsize=10, fontweight='bold')
    ax.axhline(1.0, color='grey', lw=0.7, ls=':', alpha=0.5,
               label='Current-value floor')
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = os.path.join(out_dir, 'figure_p3_05_cv_summary.png')
    _save(fig, path)
    return path


# =============================================================================
# MASTER RUN FUNCTION
# =============================================================================

def run_phase3(phase2_dir: str, out_dir: str,
               default_g: Optional[float] = None) -> dict:
    """
    Full Phase 3 pipeline.

    Parameters
    ----------
    phase2_dir : path to results/phase2/
    out_dir    : path to results/phase3/
    default_g  : headline stratum for per-grid analyses (config.HEADLINE_G)
    """
    os.makedirs(out_dir, exist_ok=True)
    default_g = CFG_MOD.HEADLINE_G if default_g is None else float(default_g)

    print('  Loading Phase 2 data ...')
    df_agg, df_feat = load_phase2(phase2_dir)
    grid = build_grid(df_agg, df_feat)
    if default_g not in set(grid['target_g'].unique()):
        default_g = float(sorted(grid['target_g'].unique())[-1])
    print(f'  Grid: {len(grid)} rows  '
          f'({grid["regime"].nunique()} regimes × '
          f'{grid["obs_idx"].nunique()} obs_idx × '
          f'{grid["noise"].nunique()} noise × '
          f'{grid["target_g"].nunique()} strata; '
          f'{int(grid["capped"].sum())} capped cells)\n')

    print('  Evaluating selectors ...')
    df_comp, df_regime = evaluate_selectors(grid, out_dir)

    print('\n  Running leave-one-regime-out cross-validation ...')
    df_cv = cross_validate(grid, out_dir)

    print('\n  Building regime fingerprinting classifier ...')
    df_clf = regime_classifier(df_feat, out_dir)

    print('\n  Generating figures ...')
    paths = [
        fig_p3_01_selector_comparison(df_comp, out_dir),
        fig_p3_02_improvement_map(df_regime, out_dir),
        fig_p3_03_classifier_accuracy(df_clf, out_dir),
        fig_p3_04_obs_depth(df_regime, grid, default_g, out_dir),
        fig_p3_05_cv_summary(df_cv, out_dir),
    ]

    return {
        'grid':       grid,
        'comparison': df_comp,
        'regime':     df_regime,
        'cv':         df_cv,
        'classifier': df_clf,
        'default_g':  default_g,
        'figures':    [p for p in paths if p],
    }
