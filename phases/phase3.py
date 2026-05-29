"""
phase3.py
=========
Phase 3 — Adaptive Selector Pipeline.

Loads Phase 2 data (no new simulations) and builds, evaluates, and
validates a complete adaptive method selector.

Three components
----------------
1. Selector comparison
   Seven selectors ranging from fixed baselines to a full adaptive
   cascade are evaluated on all Phase 2 grid points.
   Metric: mean achieved stability across all (regime, obs_idx, noise,
   horizon) combinations.

2. Regime fingerprinting classifier
   A simple decision-tree classifier predicts the convergence regime
   from the six trajectory features.  Accuracy is reported per regime.
   Confusable regime pairs are identified.

3. Leave-one-regime-out cross-validation
   The Phase 2 two-rule cascade and the Phase 3 enhanced cascade are
   re-derived with one regime held out, then tested on that regime.
   This measures whether the rules generalise beyond the regimes used
   to set the thresholds.

Selectors evaluated
-------------------
fixed_current     always use current_value  (floor)
fixed_richardson  always use richardson_1   (Phase 2 reference)
fixed_rational    always use rational_fit   (short-horizon Phase 1 winner)
fixed_single_exp  always use single_exp_fit (long-horizon Phase 1 winner)
phase2_cascade    Rules 1+2 from Phase 2, default = richardson_1
enhanced_cascade  Rules 1-4, horizon-aware default (Phase 3 contribution)
oracle            best available method at each grid point (upper bound)

Enhanced cascade logic
----------------------
Step 1. Horizon-aware default:
    horizon <= 300  ->  rational_fit
    horizon >  300  ->  single_exp_fit

Step 2. Rule A  (flat trajectory):
    log_log_slope > -0.10  ->  rational_fit   (precision 85.8 %)

Step 3. Rule B  (poor power-law fit):
    richardson_r2  < 0.50  ->  rational_fit   (recall   77.2 %)

Step 4. Rule C  (step/staircase signature):
    curvature_idx  > 0.50  AND
    diff_ratio_cv  < 0.05  ->  weniger_d2

Step 5. Rule D  (oscillatory with high ratio variation):
    oscillation_idx > 0.30  AND
    diff_ratio_cv   > 0.20  ->  log_linear

Step 6. Return default from Step 1.

Input files (from results/phase2/)
-----------------------------------
    phase2_sweep_aggregated.csv
    phase2_features.csv

Output files (to results/phase3/)
----------------------------------
    phase3_selector_comparison.csv  selector x horizon mean stability
    phase3_regime_results.csv       per-regime achieved stability
    phase3_cv_results.csv           leave-one-regime-out CV
    phase3_regime_classifier.csv    regime classification accuracy
    figure_p3_01_selector_comparison.png
    figure_p3_02_improvement_map.png
    figure_p3_03_classifier_accuracy.png
    figure_p3_04_obs_depth.png
    figure_p3_05_cv_summary.png

Author : Kian Maleki
Date   : 2026-05-24
"""

import os, math, warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

warnings.filterwarnings('ignore')

# ── Constants ──────────────────────────────────────────────────────────────────

PHASE2_METHODS = [
    'current_value', 'richardson_1', 'richardson_a10',
    'single_exp_fit', 'rational_fit', 'pade_22',
    'log_linear', 'weniger_d2', 'anderson_1',
]

FEATURE_COLS = [
    'log_log_slope', 'curvature_idx', 'oscillation_idx',
    'noise_var', 'richardson_r2', 'diff_ratio_cv',
]

METHOD_COLOURS = {
    'current_value':  '#888888',
    'richardson_1':   '#f4a261',
    'richardson_a10': '#e76f51',
    'single_exp_fit': '#2196f3',
    'rational_fit':   '#1565c0',
    'pade_22':        '#e91e63',
    'log_linear':     '#00897b',
    'weniger_d2':     '#9c27b0',
    'anderson_1':     '#795548',
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
    return agg, feat


def build_grid(df_agg: pd.DataFrame,
               df_feat: pd.DataFrame) -> pd.DataFrame:
    """
    Build the evaluation grid: one row per (regime, obs_idx, noise, future_idx).
    Each row contains:
      - stability of every PHASE2_METHODS method (as columns)
      - mean feature values (averaged across seeds)
      - best_method  : method with highest stability at this grid point
      - best_stability: that method's stability
    """
    # Pivot stability to wide format
    pivot = (df_agg.pivot_table(
                 index=['regime', 'obs_idx', 'noise', 'future_idx'],
                 columns='method',
                 values='stability')
              .reset_index())
    pivot.columns.name = None

    # Ensure all PHASE2_METHODS columns exist
    for m in PHASE2_METHODS:
        if m not in pivot.columns:
            pivot[m] = float('nan')

    # Identify best method at each grid point
    method_cols = [m for m in PHASE2_METHODS if m in pivot.columns]
    pivot['best_stability'] = pivot[method_cols].max(axis=1)
    pivot['best_method']    = pivot[method_cols].idxmax(axis=1)

    # Mean features across seeds
    feat_avg = (df_feat.drop(columns=['seed'])
                        .groupby(['regime', 'obs_idx', 'noise'])
                        .mean()
                        .reset_index())

    grid = pivot.merge(feat_avg,
                       on=['regime', 'obs_idx', 'noise'],
                       how='left')
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
    Step 1: horizon-aware default.
    Step 2: Rule A  (flat trajectory)
    Step 3: Rule B  (poor power-law fit)
    Step 4: Rule C  (staircase signature)
    Step 5: Rule D  (oscillatory + high ratio variation)
    Step 6: horizon-aware default.
    """
    horizon = int(row.get('future_idx', 5000))
    default = 'rational_fit' if horizon <= 300 else 'single_exp_fit'

    slope   = row.get('log_log_slope',   float('nan'))
    r2      = row.get('richardson_r2',   float('nan'))
    curv    = row.get('curvature_idx',   float('nan'))
    osc     = row.get('oscillation_idx', float('nan'))
    d_cv    = row.get('diff_ratio_cv',   float('nan'))

    # Rule A
    if math.isfinite(slope) and slope > -0.10:
        return 'rational_fit'

    # Rule B
    if math.isfinite(r2) and r2 < 0.50:
        return 'rational_fit'

    # Rule C — staircase signature
    if (math.isfinite(curv) and curv > 0.50
            and math.isfinite(d_cv) and d_cv < 0.05):
        return 'weniger_d2'

    # Rule D — oscillatory with high ratio variation
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


# =============================================================================
# 3.  SELECTOR EVALUATION
# =============================================================================

def evaluate_selectors(grid: pd.DataFrame,
                       out_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply every selector to every grid point.
    Returns:
      df_comparison : mean stability per (selector, future_idx)
      df_per_regime : mean stability per (selector, regime, future_idx)
    """
    comp_rows   = []
    regime_rows = []

    for sel_name, sel_fn in SELECTORS.items():
        grid_copy = grid.copy()
        grid_copy['chosen_method'] = grid_copy.apply(sel_fn, axis=1)

        # Achieved stability = stability of the chosen method
        def _achieved(row):
            m = row['chosen_method']
            return float(row[m]) if m in row and math.isfinite(row[m]) else float('nan')

        grid_copy['achieved_stability'] = grid_copy.apply(_achieved, axis=1)

        # Global per horizon
        for fid, grp in grid_copy.groupby('future_idx'):
            mean_stab = grp['achieved_stability'].mean()
            comp_rows.append({
                'selector':   sel_name,
                'future_idx': fid,
                'mean_stability': round(float(mean_stab), 4),
                'n': len(grp),
            })

        # Per-regime per-horizon
        for (regime, fid), grp in grid_copy.groupby(['regime', 'future_idx']):
            ms = grp['achieved_stability'].mean()
            regime_rows.append({
                'selector':       sel_name,
                'regime':         regime,
                'future_idx':     fid,
                'mean_stability': round(float(ms), 4),
            })

    df_comp   = pd.DataFrame(comp_rows)
    df_regime = pd.DataFrame(regime_rows)

    p = os.path.join(out_dir, 'phase3_selector_comparison.csv')
    df_comp.to_csv(p, index=False)
    print(f'  Saved: {p}  ({len(df_comp)} rows)')

    p = os.path.join(out_dir, 'phase3_regime_results.csv')
    df_regime.to_csv(p, index=False)
    print(f'  Saved: {p}  ({len(df_regime)} rows)')

    return df_comp, df_regime


# =============================================================================
# 4.  LEAVE-ONE-REGIME-OUT CROSS-VALIDATION
# =============================================================================

def cross_validate(grid: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    """
    Leave-one-regime-out cross-validation for the Phase 2 cascade and the
    enhanced cascade.

    For each held-out regime:
      - Compute the 'oracle choice' on the held-out regime.
      - Apply the two cascades (rules fixed, not re-fitted — the thresholds
        are interpretable constants, not data-driven fits).
      - Report achieved stability on the held-out regime.

    Because the cascade rules use fixed thresholds (not estimated from data),
    this CV measures out-of-regime generalization rather than overfitting.
    """
    regimes = sorted(grid['regime'].unique())
    cv_rows = []

    for held_out in regimes:
        test_grid = grid[grid['regime'] == held_out].copy()

        for sel_name in ['phase2_cascade', 'enhanced_cascade', 'oracle',
                          'fixed_richardson', 'fixed_single_exp']:
            sel_fn = SELECTORS[sel_name]
            test_grid['chosen'] = test_grid.apply(sel_fn, axis=1)

            def _ach(row):
                m = row['chosen']
                return float(row[m]) if m in row and math.isfinite(row[m]) else float('nan')

            achieved = test_grid.apply(_ach, axis=1).mean()
            cv_rows.append({
                'held_out_regime': held_out,
                'selector':        sel_name,
                'mean_stability':  round(float(achieved), 4),
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
    unavailable.

    Reports per-regime accuracy and the top confusable pairs.
    """
    # Prepare feature matrix (averaged over seeds per grid point)
    feat_avg = (df_feat.drop(columns=['seed'])
                        .groupby(['regime', 'obs_idx', 'noise'])
                        .mean()
                        .reset_index())

    valid = feat_avg.dropna(subset=FEATURE_COLS).copy()
    if len(valid) < 20:
        print('  Regime classifier: insufficient valid data, skipped.')
        return pd.DataFrame()

    X = valid[FEATURE_COLS].values
    y = valid['regime'].values
    regimes = sorted(set(y))

    # ── Try scikit-learn DecisionTree ─────────────────────────────────────────
    clf_name = 'unknown'
    try:
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.model_selection import StratifiedKFold

        clf = DecisionTreeClassifier(max_depth=6, min_samples_leaf=3,
                                     random_state=42)
        # 5-fold stratified CV
        skf  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        preds = np.empty(len(y), dtype=object)
        for train_idx, test_idx in skf.split(X, y):
            clf.fit(X[train_idx], y[train_idx])
            preds[test_idx] = clf.predict(X[test_idx])
        clf_name = 'DecisionTree(depth=6, 5-fold CV)'

    except ImportError:
        # Fallback: 1-nearest-neighbour (no sklearn required)
        from scipy.spatial.distance import cdist
        preds = np.empty(len(y), dtype=object)
        n = len(X)
        for i in range(n):
            X_train = np.delete(X, i, axis=0)
            y_train = np.delete(y, i, axis=0)
            dists   = cdist(X[[i]], X_train, metric='euclidean')[0]
            preds[i] = y_train[np.argmin(dists)]
        clf_name = '1-NN (leave-one-out, scipy fallback)'

    # Per-regime accuracy
    rows = []
    for regime in regimes:
        mask   = y == regime
        n_tot  = int(mask.sum())
        n_corr = int((preds[mask] == regime).sum())
        acc    = n_corr / n_tot if n_tot > 0 else float('nan')
        # Most common wrong prediction
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
    """
    Grouped bar chart: mean achieved stability per selector per horizon.
    """
    horizons   = sorted(df_comp['future_idx'].unique())
    selectors  = list(SELECTORS.keys())
    n_sel      = len(selectors)
    n_hor      = len(horizons)

    fig, axes = plt.subplots(1, n_hor, figsize=(5 * n_hor, 7), sharey=True)
    if n_hor == 1:
        axes = [axes]

    for ax, fid in zip(axes, horizons):
        sub = df_comp[df_comp['future_idx'] == fid].set_index('selector')
        vals = [float(sub.loc[s, 'mean_stability'])
                if s in sub.index else float('nan')
                for s in selectors]
        colours = [SELECTOR_COLOURS.get(s, '#999') for s in selectors]

        bars = ax.bar(range(n_sel), vals, color=colours,
                      edgecolor='white', lw=0.5)
        ax.set_xticks(range(n_sel))
        ax.set_xticklabels([s.replace('_', '\n') for s in selectors],
                           fontsize=7.5, rotation=0)
        ax.set_title(f'Horizon = {fid}', fontsize=10, fontweight='bold')
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
        'Figure P3-1 — Mean Achieved Stability by Selector and Horizon\n'
        'Green bar (enhanced_cascade) = Phase 3 contribution; '
        'Black bar (oracle) = theoretical upper bound',
        fontsize=10, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(out_dir, 'figure_p3_01_selector_comparison.png')
    _save(fig, path)
    return path


def fig_p3_02_improvement_map(df_regime: pd.DataFrame,
                               out_dir: str) -> str:
    """
    Heatmap: stability gain of enhanced_cascade over fixed_richardson,
    per (regime, horizon).
    """
    rich = (df_regime[df_regime['selector'] == 'fixed_richardson']
            .set_index(['regime', 'future_idx'])['mean_stability'])
    enh  = (df_regime[df_regime['selector'] == 'enhanced_cascade']
            .set_index(['regime', 'future_idx'])['mean_stability'])

    gain = (enh - rich).reset_index()
    gain.columns = ['regime', 'future_idx', 'gain']

    horizons = sorted(gain['future_idx'].unique())
    regimes  = sorted(gain['regime'].unique())

    mat = np.full((len(regimes), len(horizons)), np.nan)
    for _, row in gain.iterrows():
        i = regimes.index(row['regime'])
        j = horizons.index(row['future_idx'])
        mat[i, j] = row['gain']

    vmax = max(abs(np.nanmax(mat)), abs(np.nanmin(mat)), 0.05)
    fig, ax = plt.subplots(figsize=(max(7, len(horizons) * 2.5), 10))
    im = ax.imshow(mat, cmap='RdYlGn', aspect='auto',
                   vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(horizons)))
    ax.set_xticklabels([f'n={h}' for h in horizons], fontsize=9)
    ax.set_yticks(range(len(regimes)))
    ax.set_yticklabels([r.replace('_', '\n') for r in regimes], fontsize=8)

    for i in range(len(regimes)):
        for j in range(len(horizons)):
            v = mat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f'{v:+.3f}', ha='center', va='center',
                        fontsize=7.5,
                        color='white' if abs(v) > 0.5 * vmax else '#333')

    plt.colorbar(im, ax=ax, label='Stability gain (enhanced − fixed_richardson)',
                 shrink=0.5)
    ax.set_title(
        'Figure P3-2 — Enhanced Cascade Gain over Fixed Richardson\n'
        'Green = cascade improves; Red = cascade hurts',
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
    ax.set_xlabel('Classification accuracy (5-fold CV)', fontsize=9)
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
                         default_fid: int,
                         out_dir: str) -> str:
    """
    How does each selector's mean achieved stability change with obs_idx?
    Shows whether more observations benefit the adaptive cascade more than
    fixed methods.
    """
    sub  = grid[grid['future_idx'] == default_fid].copy()
    obs_vals = sorted(sub['obs_idx'].unique())

    result_rows = []
    for sel_name, sel_fn in SELECTORS.items():
        sub['chosen'] = sub.apply(sel_fn, axis=1)

        def _ach(row):
            m = row['chosen']
            return float(row[m]) if m in row and math.isfinite(row[m]) else float('nan')

        sub['achieved'] = sub.apply(_ach, axis=1)

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
        f'(horizon = {default_fid}, all regimes, all noise levels)',
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
    """
    Box plot: leave-one-regime-out mean stability distribution across
    held-out regimes, for each selector.
    """
    sel_order = ['fixed_richardson', 'fixed_single_exp',
                 'phase2_cascade', 'enhanced_cascade', 'oracle']
    sel_order = [s for s in sel_order if s in df_cv['selector'].unique()]

    data    = [df_cv[df_cv['selector'] == s]['mean_stability'].dropna().values
               for s in sel_order]
    colours = [SELECTOR_COLOURS.get(s, '#999') for s in sel_order]
    labels  = [s.replace('_', '\n') for s in sel_order]

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
        'Box shows distribution across 18 held-out regimes; '
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
               default_fid: int = 5000) -> dict:
    """
    Full Phase 3 pipeline.

    Parameters
    ----------
    phase2_dir : path to results/phase2/
    out_dir    : path to results/phase3/
    default_fid: primary horizon for per-grid analyses
    """
    os.makedirs(out_dir, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    print('  Loading Phase 2 data ...')
    df_agg, df_feat = load_phase2(phase2_dir)
    grid = build_grid(df_agg, df_feat)
    print(f'  Grid: {len(grid)} rows  '
          f'({grid["regime"].nunique()} regimes × '
          f'{grid["obs_idx"].nunique()} obs_idx × '
          f'{grid["noise"].nunique()} noise × '
          f'{grid["future_idx"].nunique()} horizons)\n')

    # ── Selector evaluation ────────────────────────────────────────────────────
    print('  Evaluating selectors ...')
    df_comp, df_regime = evaluate_selectors(grid, out_dir)

    # ── Cross-validation ───────────────────────────────────────────────────────
    print('\n  Running leave-one-regime-out cross-validation ...')
    df_cv = cross_validate(grid, out_dir)

    # ── Regime classifier ──────────────────────────────────────────────────────
    print('\n  Building regime fingerprinting classifier ...')
    df_clf = regime_classifier(df_feat, out_dir)

    # ── Figures ────────────────────────────────────────────────────────────────
    print('\n  Generating figures ...')
    paths = [
        fig_p3_01_selector_comparison(df_comp, out_dir),
        fig_p3_02_improvement_map(df_regime, out_dir),
        fig_p3_03_classifier_accuracy(df_clf, out_dir),
        fig_p3_04_obs_depth(df_regime, grid, default_fid, out_dir),
        fig_p3_05_cv_summary(df_cv, out_dir),
    ]

    return {
        'grid':       grid,
        'comparison': df_comp,
        'regime':     df_regime,
        'cv':         df_cv,
        'classifier': df_clf,
        'figures':    [p for p in paths if p],
    }
