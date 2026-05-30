"""
run_real_data.py
================
Real-data experiment entry point — Section 9 of the paper.

Downloads 6 OpenML CC-18 tabular classification datasets, trains XGBoost
for 500 rounds on each, extracts Phase 2 features from the validation loss
curves at obs in {30, 60, 90}, applies the two-rule cascade selector,
calls the selected accelerator to predict the final loss, and compares
cascade prediction error against the current_value baseline.

Usage
-----
    python scripts/run_real_data.py

Prerequisites
-------------
    pip install xgboost openml scikit-learn

    Phase 2 full run should have completed first for regime mapping:
        results/phase2/phase2_features.csv
    (If absent, regime mapping is skipped but all else runs fine.)

Output directory: results/real_data/

Key output files
----------------
    real_data_curves.csv            Raw loss curves (500 rounds x 6 datasets)
    real_data_results.csv           Per (dataset, obs_depth): features,
                                    selected method, predicted_val,
                                    cascade_err, current_err
    figure_rd_01_comparison.png     Cascade vs current_value error by dataset
    figure_rd_02_regimes.png        Real curves in synthetic feature space
    figure_rd_03_improvement.png    Per-dataset improvement ratio heatmap
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from src.datasets     import run_real_data_experiment, DATASET_IDS
from src.trajectories import process_curves, FEATURE_COLS

# ── Configuration ──────────────────────────────────────────────────────────────

OUT_DIR      = os.path.join('results', 'real_data')
PHASE2_FEATS = os.path.join('results', 'phase2', 'phase2_features.csv')
OBS_DEPTHS   = [30, 60, 90]
WINDOW_LEN   = 60
L_INF        = 0.01
N_ROUNDS     = 500

METHOD_COLOURS = {
    'richardson_1': '#f4a261',
    'rational_fit': '#e91e63',
}


# ── Figures ────────────────────────────────────────────────────────────────────

def fig_comparison(df: pd.DataFrame, out_dir: str):
    """
    Figure 1: cascade prediction error vs current_value error,
    grouped by observation depth. One panel per obs depth.
    """
    obs_depths = sorted(df['obs_depth'].unique())
    datasets   = sorted(df['dataset'].unique())
    n_obs      = len(obs_depths)
    x          = np.arange(len(datasets))
    width      = 0.35

    fig, axes = plt.subplots(1, n_obs, figsize=(5 * n_obs, 5), sharey=True)
    if n_obs == 1:
        axes = [axes]

    for ax, obs in zip(axes, obs_depths):
        sub = df[df['obs_depth'] == obs].set_index('dataset')

        curr_errs = [sub.loc[d, 'current_err']  if d in sub.index else np.nan
                     for d in datasets]
        casc_errs = [sub.loc[d, 'cascade_err']  if d in sub.index else np.nan
                     for d in datasets]

        bars1 = ax.bar(x - width/2, curr_errs, width,
                       label='current_value', color='#888888', alpha=0.8)
        bars2 = ax.bar(x + width/2, casc_errs, width,
                       label='cascade', color='#2196f3', alpha=0.8)

        # Colour cascade bars by selected method
        for i, d in enumerate(datasets):
            if d in sub.index:
                method = sub.loc[d, 'selected_method']
                bars2[i].set_color(METHOD_COLOURS.get(method, '#2196f3'))

        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=30, ha='right', fontsize=8)
        ax.set_title(f'obs = {obs}', fontsize=10)
        ax.set_xlabel('Dataset')
        if ax is axes[0]:
            ax.set_ylabel('|predicted − true final loss|')
        ax.axhline(0, color='black', linewidth=0.5)

    # Legend
    from matplotlib.patches import Patch
    legend_els = [
        Patch(facecolor='#888888', alpha=0.8, label='current_value (baseline)'),
        Patch(facecolor=METHOD_COLOURS['richardson_1'], alpha=0.8,
              label='cascade → richardson_1'),
        Patch(facecolor=METHOD_COLOURS['rational_fit'],  alpha=0.8,
              label='cascade → rational_fit'),
    ]
    axes[-1].legend(handles=legend_els, fontsize=7, loc='upper right')

    fig.suptitle(
        'Real-Data Validation: Cascade vs Current-Value Error\n'
        'by Dataset and Observation Depth',
        fontsize=11)
    fig.tight_layout()
    path = os.path.join(out_dir, 'figure_rd_01_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')


def fig_regime_scatter(df: pd.DataFrame, out_dir: str):
    """
    Figure 2: Real curves overlaid on synthetic regime centroids
    in (log_log_slope, richardson_r2) feature space.
    """
    if not os.path.exists(PHASE2_FEATS):
        print('  Skipping regime scatter — phase2_features.csv not found.')
        return

    df_synth  = pd.read_csv(PHASE2_FEATS)
    centroids = df_synth.groupby('regime')[FEATURE_COLS].mean().dropna()
    fx, fy    = 'log_log_slope', 'richardson_r2'

    fig, ax = plt.subplots(figsize=(9, 6))

    # Synthetic regime centroids
    cmap   = plt.get_cmap('tab20', len(centroids))
    for i, (regime, row) in enumerate(centroids.iterrows()):
        ax.scatter(row[fx], row[fy],
                   marker='s', s=100, color=cmap(i), alpha=0.55, zorder=2)
        ax.annotate(regime, (row[fx], row[fy]),
                    fontsize=6, alpha=0.75,
                    xytext=(3, 3), textcoords='offset points')

    # Real curves — one point per (dataset, obs_depth)
    markers = ['*', 'D', 'P', '^', 'v', 'X']
    datasets = sorted(df['dataset'].unique())
    for j, dataset in enumerate(datasets):
        sub = df[df['dataset'] == dataset]
        ax.scatter(sub[fx], sub[fy],
                   marker=markers[j % len(markers)],
                   s=160, zorder=5, edgecolors='black', linewidths=0.5,
                   label=dataset)
        for _, row in sub.iterrows():
            ax.annotate(
                f"obs={int(row['obs_depth'])}",
                (row[fx], row[fy]),
                fontsize=6, color='black',
                xytext=(4, -8), textcoords='offset points')

    ax.set_xlabel('log_log_slope', fontsize=10)
    ax.set_ylabel('richardson_r2', fontsize=10)
    ax.set_title('Real XGBoost Curves in Synthetic Feature Space', fontsize=11)
    ax.legend(fontsize=7, ncol=2, loc='lower right',
              title='Real datasets', title_fontsize=7)
    fig.tight_layout()
    path = os.path.join(out_dir, 'figure_rd_02_regimes.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')


def fig_improvement(df: pd.DataFrame, out_dir: str):
    """
    Figure 3: Improvement ratio heatmap.
    improvement = (current_err - cascade_err) / current_err
    Positive = cascade beats baseline. Negative = cascade worse.
    """
    datasets   = sorted(df['dataset'].unique())
    obs_depths = sorted(df['obs_depth'].unique())

    matrix = np.full((len(datasets), len(obs_depths)), np.nan)
    for i, d in enumerate(datasets):
        for j, obs in enumerate(obs_depths):
            row = df[(df['dataset'] == d) & (df['obs_depth'] == obs)]
            if row.empty:
                continue
            c_err = float(row['current_err'].values[0])
            p_err = float(row['cascade_err'].values[0])
            if np.isfinite(p_err) and c_err > 1e-10:
                matrix[i, j] = (c_err - p_err) / c_err

    fig, ax = plt.subplots(figsize=(6, 5))
    vmax = max(abs(np.nanmax(matrix)), abs(np.nanmin(matrix)), 0.01)
    im = ax.imshow(matrix, aspect='auto',
                   cmap='RdYlGn', vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(obs_depths)))
    ax.set_xticklabels([f'obs={o}' for o in obs_depths], fontsize=9)
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels(datasets, fontsize=9)
    ax.set_title('Cascade Improvement over Current-Value\n'
                 '(green = cascade better, red = cascade worse)', fontsize=10)

    # Annotate cells
    for i in range(len(datasets)):
        for j in range(len(obs_depths)):
            v = matrix[i, j]
            if np.isfinite(v):
                ax.text(j, i, f'{v:+.1%}', ha='center', va='center',
                        fontsize=8, color='black' if abs(v) < 0.5 else 'white')
            else:
                ax.text(j, i, 'N/A', ha='center', va='center',
                        fontsize=8, color='grey')

    plt.colorbar(im, ax=ax, label='Relative improvement')
    fig.tight_layout()
    path = os.path.join(out_dir, 'figure_rd_03_improvement.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')


# ── Console summary ────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame):
    print('\n' + '=' * 80)
    print('  REAL-DATA EXPERIMENT SUMMARY')
    print('=' * 80)

    print(f'\n  Datasets  : {sorted(df["dataset"].unique())}')
    print(f'  Obs depths: {sorted(df["obs_depth"].unique())}')

    # Per-row table
    print(f'\n  {"Dataset":<18} {"Obs":>5} {"Method":<16} '
          f'{"Predicted":>11} {"True final":>11} '
          f'{"Cascade err":>12} {"Curr err":>10} {"Improv":>8}')
    print('  ' + '─' * 95)

    for _, row in df.sort_values(['dataset', 'obs_depth']).iterrows():
        improv = ((row['current_err'] - row['cascade_err']) / row['current_err']
                  if np.isfinite(row['cascade_err']) and row['current_err'] > 1e-10
                  else float('nan'))
        pred_str = (f"{row['predicted_val']:>11.6f}"
                    if np.isfinite(row['predicted_val']) else '        NaN')
        casc_str = (f"{row['cascade_err']:>12.6f}"
                    if np.isfinite(row['cascade_err']) else '         NaN')
        imp_str  = (f"{improv:>+8.1%}"
                    if np.isfinite(improv) else '     N/A')
        print(f"  {row['dataset']:<18} {int(row['obs_depth']):>5} "
              f"{row['selected_method']:<16} "
              f"{pred_str} {row['true_final']:>11.6f} "
              f"{casc_str} {row['current_err']:>10.6f} {imp_str}")

    # Aggregate by obs_depth
    print(f'\n  Mean errors by observation depth:')
    print(f"  {'Obs':>5} {'Cascade err':>13} {'Current err':>13} "
          f"{'Mean improvement':>18}")
    print('  ' + '─' * 55)
    for obs in sorted(df['obs_depth'].unique()):
        sub = df[df['obs_depth'] == obs]
        c_mean = sub['cascade_err'].mean()
        b_mean = sub['current_err'].mean()
        improv = (b_mean - c_mean) / b_mean if b_mean > 1e-10 else float('nan')
        print(f"  {obs:>5} {c_mean:>13.6f} {b_mean:>13.6f} {improv:>+18.1%}")

    # Method selection
    print(f'\n  Method selection:')
    counts = df['selected_method'].value_counts()
    for method, count in counts.items():
        sub    = df[df['selected_method'] == method]
        c_mean = sub['cascade_err'].mean()
        pct    = 100 * count / len(df)
        print(f"    {method:<20} {count:>4} ({pct:.1f}%)  "
              f"mean cascade err = {c_mean:.6f}")

    # Regime mapping
    print(f'\n  Nearest synthetic regime:')
    for regime, count in df['nearest_regime'].value_counts().items():
        pct = 100 * count / len(df)
        print(f"    {regime:<28} {count:>4} ({pct:.1f}%)")

    print()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print('=' * 72)
    print('  REAL-DATA EXPERIMENT  (Section 9)')
    print('=' * 72)
    print(f'  Datasets   : {list(DATASET_IDS.keys())}')
    print(f'  Rounds     : {N_ROUNDS}')
    print(f'  Obs depths : {OBS_DEPTHS}')
    print(f'  Window len : {WINDOW_LEN}')
    print(f'  L_inf      : {L_INF}')
    print(f'  Output dir : {OUT_DIR}')

    if not os.path.exists(PHASE2_FEATS):
        print(f'\n  WARNING: {PHASE2_FEATS} not found.')
        print('  Regime mapping skipped.')
        print('  Run  python scripts/run_phase2.py --full  first.\n')

    print('=' * 72 + '\n')

    # ── Step 1: Download datasets and train XGBoost ────────────────────────────
    print('STEP 1 — Downloading datasets and training XGBoost ...\n')
    curves = run_real_data_experiment(
        out_dir  = OUT_DIR,
        n_rounds = N_ROUNDS,
    )

    # ── Step 2: Features + cascade + accelerator predictions ──────────────────
    print('\nSTEP 2 — Extracting features, selecting methods, predicting ...\n')
    df = process_curves(
        curves               = curves,
        obs_depths           = OBS_DEPTHS,
        window_len           = WINDOW_LEN,
        L_inf                = L_INF,
        phase2_features_path = PHASE2_FEATS,
        future_x             = N_ROUNDS,
    )

    # Save full results table
    results_path = os.path.join(OUT_DIR, 'real_data_results.csv')
    df.to_csv(results_path, index=False)
    print(f'  Results saved to {results_path}')

    # ── Step 3: Figures ────────────────────────────────────────────────────────
    print('\nSTEP 3 — Generating figures ...\n')
    fig_comparison(df, OUT_DIR)
    fig_regime_scatter(df, OUT_DIR)
    fig_improvement(df, OUT_DIR)

    # ── Step 4: Console summary ────────────────────────────────────────────────
    print_summary(df)

    print('=' * 72)
    print('  Output files:')
    print(f'    {OUT_DIR}/real_data_curves.csv')
    print(f'    {OUT_DIR}/real_data_results.csv')
    print(f'    {OUT_DIR}/figure_rd_01_comparison.png')
    print(f'    {OUT_DIR}/figure_rd_02_regimes.png')
    print(f'    {OUT_DIR}/figure_rd_03_improvement.png')
    print('=' * 72 + '\n')


if __name__ == '__main__':
    main()
