"""
plots.py
========
Phase 1 figure generation.

Six publication-quality figures:
    Fig 1  Global stability ranking (default horizon)
    Fig 2  Method x Regime heatmap  (default horizon)
    Fig 3  Horizon sensitivity — top-15 method ranks across horizons
    Fig 4  Limit estimator vs Trajectory extrapolator split
    Fig 5  Per-regime best method summary
    Fig 6  Richardson failure profile across regimes

Author : Kian Maleki
Date   : 2026-05-24
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

# ── Family colour palette ──────────────────────────────────────────────────────
FAMILY_COLOURS = {
    'baseline':   '#888888',
    'richardson': '#f4a261',
    'parametric': '#e9c46a',
    'shanks':     '#2196f3',
    'wynn_eps':   '#4caf50',
    'wynn_rho':   '#8bc34a',
    'pade':       '#e91e63',
    'levin':      '#9c27b0',
    'weniger':    '#673ab7',
    'brezinski':  '#ff9800',
    'neville':    '#00bcd4',
    'anderson':   '#795548',
    'ensemble':   '#607d8b',
    'unknown':    '#cccccc',
}

METHOD_TYPE_COLOURS = {
    'trajectory': '#1565c0',
    'limit':      '#b71c1c',
}

FIG_DPI = 150


def _save(fig, path):
    fig.savefig(path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Global stability ranking
# ─────────────────────────────────────────────────────────────────────────────

def fig1_stability_ranking(df_global: pd.DataFrame,
                            default_horizon: int,
                            out_dir: str) -> str:
    sub = (df_global[df_global['future_idx'] == default_horizon]
             .sort_values('stability', ascending=True))

    labels   = sub['method'].tolist()
    scores   = sub['stability'].tolist()
    families = sub['family'].tolist()
    types    = sub['method_type'].tolist()
    colours  = [FAMILY_COLOURS.get(f, '#cccccc') for f in families]

    fig, ax = plt.subplots(figsize=(12, max(8, len(labels) * 0.22)))
    bars = ax.barh(range(len(scores)), scores, color=colours,
                   edgecolor='white', linewidth=0.4, height=0.75)

    # Annotate each bar
    for i, (sc, vr, cr, mtype) in enumerate(
            zip(scores, sub['valid_rate'], sub['cat_rate'], types)):
        t_col = METHOD_TYPE_COLOURS.get(mtype, '#333333')
        ax.text(max(sc, 0) + 0.01, i,
                f'{sc:.3f}  V={vr:.2f} C={cr:.2f}',
                va='center', fontsize=6.5, color='#333333')
        ax.text(min(sc, 0) - 0.01, i, '●',
                va='center', ha='right', fontsize=5, color=t_col)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.axvline(0, color='black', linewidth=0.7)
    ax.set_xlabel(
        'Stability Score  =  valid_rate − 2·cat_rate + 0.4·beats_rate',
        fontsize=9)
    ax.set_title(
        f'Figure 1 — Global Stability Ranking  (horizon = {default_horizon})\n'
        'V = valid rate, C = catastrophic rate; '
        '● blue = trajectory extrapolator, ● red = limit estimator',
        fontsize=10, fontweight='bold')

    # Family legend
    handles = [mpatches.Patch(color=c, label=f)
               for f, c in FAMILY_COLOURS.items()
               if f in set(families)]
    ax.legend(handles=handles, loc='lower right', fontsize=7,
              framealpha=0.9, ncol=2)

    ax.set_xlim(min(scores) - 0.25, max(scores) + 0.55)
    fig.tight_layout()
    path = os.path.join(out_dir, 'figure_01_stability_ranking.png')
    _save(fig, path)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Method × Regime heatmap
# ─────────────────────────────────────────────────────────────────────────────

def fig2_heatmap(heatmap_df: pd.DataFrame,
                 default_horizon: int,
                 out_dir: str) -> str:
    mat  = heatmap_df.values.astype(float)
    rows = list(heatmap_df.index)
    cols = [c.replace('_', '\n') for c in heatmap_df.columns]

    # Sort rows by mean stability descending
    row_means  = np.nanmean(mat, axis=1)
    sort_order = np.argsort(row_means)[::-1]
    mat   = mat[sort_order]
    rows  = [rows[i] for i in sort_order]

    vmax = max(abs(np.nanmax(mat)), abs(np.nanmin(mat)), 0.1)

    fig, ax = plt.subplots(figsize=(max(16, len(cols) * 0.9),
                                    max(10, len(rows) * 0.25)))
    im = ax.imshow(mat, cmap='RdYlGn', aspect='auto',
                   vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=7.5, ha='center')
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=7.5)

    for i in range(len(rows)):
        for j in range(len(heatmap_df.columns)):
            v = mat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                        fontsize=5.5,
                        color='white' if abs(v) > 0.5 * vmax else '#333333')

    plt.colorbar(im, ax=ax, label='Stability Score', shrink=0.5)
    ax.set_title(
        f'Figure 2 — Method × Regime Stability Heatmap  '
        f'(horizon = {default_horizon})\n'
        'Green = stable & accurate, Red = dangerous or invalid',
        fontsize=10, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(out_dir, 'figure_02_heatmap.png')
    _save(fig, path)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Horizon sensitivity
# ─────────────────────────────────────────────────────────────────────────────

def fig3_horizon_sensitivity(df_global: pd.DataFrame,
                              horizons: list,
                              out_dir: str,
                              top_n: int = 15) -> str:
    if len(horizons) < 2:
        print('  Fig 3 skipped: only one horizon.')
        return ''

    # Select top_n methods by mean stability across all horizons
    mean_stab = (df_global.groupby('method')['stability']
                           .mean().sort_values(ascending=False))
    top_methods = list(mean_stab.index[:top_n])

    fig, axes = plt.subplots(1, len(horizons), figsize=(5 * len(horizons), 9),
                             sharey=False)
    if len(horizons) == 1:
        axes = [axes]

    rank_data = {}
    for fid in horizons:
        sub = (df_global[df_global['future_idx'] == fid]
                 .sort_values('stability', ascending=False)
                 .reset_index(drop=True))
        sub['rank'] = sub.index + 1
        rank_data[fid] = sub.set_index('method')['rank'].to_dict()

    for ax_idx, fid in enumerate(horizons):
        ax  = axes[ax_idx]
        sub = (df_global[df_global['future_idx'] == fid]
                 .set_index('method'))

        for rank_pos, method in enumerate(top_methods):
            if method not in sub.index:
                continue
            row    = sub.loc[method]
            sc     = row['stability']
            colour = FAMILY_COLOURS.get(row['family'], '#cccccc')
            ax.barh(rank_pos, sc, color=colour, edgecolor='white', lw=0.4)
            ax.text(max(sc, 0) + 0.01, rank_pos,
                    f'{sc:.3f}', va='center', fontsize=7)

        ax.set_yticks(range(len(top_methods)))
        ax.set_yticklabels(top_methods if ax_idx == 0 else [],
                           fontsize=7.5)
        ax.set_title(f'Horizon = {fid}', fontsize=9, fontweight='bold')
        ax.axvline(0, color='black', lw=0.6)
        ax.set_xlabel('Stability', fontsize=8)
        ax.invert_yaxis()

    fig.suptitle(
        f'Figure 3 — Horizon Sensitivity: Top-{top_n} Methods\n'
        'Rankings can shift substantially across prediction horizons',
        fontsize=10, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(out_dir, 'figure_03_horizon_sensitivity.png')
    _save(fig, path)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Limit estimator vs Trajectory extrapolator
# ─────────────────────────────────────────────────────────────────────────────

def fig4_method_type(df_global: pd.DataFrame,
                     horizons: list,
                     out_dir: str) -> str:
    fig, axes = plt.subplots(1, len(horizons),
                              figsize=(5 * len(horizons), 6), sharey=True)
    if len(horizons) == 1:
        axes = [axes]

    for ax, fid in zip(axes, horizons):
        sub = df_global[df_global['future_idx'] == fid]
        for mtype, colour in METHOD_TYPE_COLOURS.items():
            vals = sub[sub['method_type'] == mtype]['stability'].dropna()
            pos  = 0 if mtype == 'limit' else 1
            parts = ax.violinplot([vals], positions=[pos],
                                  showmedians=True, showextrema=True)
            for pc in parts['bodies']:
                pc.set_facecolor(colour)
                pc.set_alpha(0.6)
            parts['cmedians'].set_color(colour)
            parts['cbars'].set_color(colour)
            parts['cmaxes'].set_color(colour)
            parts['cmins'].set_color(colour)

            med = vals.median()
            ax.text(pos, med, f'  {med:.3f}', va='bottom',
                    fontsize=8, color=colour, fontweight='bold')

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Limit\nEstimator', 'Trajectory\nExtrapolator'],
                           fontsize=9)
        ax.set_title(f'Horizon = {fid}', fontsize=9, fontweight='bold')
        ax.set_ylabel('Stability Score', fontsize=8)
        ax.axhline(0, color='grey', lw=0.5, ls='--')

    fig.suptitle(
        'Figure 4 — Stability by Method Type Across Horizons\n'
        'Limit estimators predict the mathematical limit;\n'
        'trajectory extrapolators predict s(n) at the target index',
        fontsize=10, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(out_dir, 'figure_04_method_type.png')
    _save(fig, path)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — Per-regime best method summary
# ─────────────────────────────────────────────────────────────────────────────

def fig5_regime_recommendations(df_best: pd.DataFrame,
                                 default_horizon: int,
                                 out_dir: str) -> str:
    sub = (df_best[df_best['future_idx'] == default_horizon]
             .copy()
             .reset_index(drop=True))

    fig, ax = plt.subplots(figsize=(12, 7))
    regime_labels = [r.replace('_', '\n') for r in sub['regime']]
    colours = [FAMILY_COLOURS.get(f, '#cccccc') for f in sub['family']]

    bars = ax.bar(range(len(sub)), sub['stability'],
                  color=colours, edgecolor='white', lw=0.5)

    for i, (bm, sc, vr, cr) in enumerate(zip(
            sub['best_method'], sub['stability'],
            sub['valid_rate'],  sub['cat_rate'])):
        ax.text(i, sc + 0.01, bm.replace('_', '\n'),
                ha='center', va='bottom', fontsize=6.5, rotation=0)
        ax.text(i, -0.05, f'V={vr:.2f}\nC={cr:.2f}',
                ha='center', va='top', fontsize=5.5, color='#555555')

    ax.set_xticks(range(len(sub)))
    ax.set_xticklabels(regime_labels, fontsize=7.5)
    ax.set_ylabel('Best Stability Score', fontsize=9)
    ax.set_title(
        f'Figure 5 — Per-Regime Best Method  (horizon = {default_horizon})\n'
        'Method name shown above each bar; V = valid rate, C = cat rate',
        fontsize=10, fontweight='bold')
    ax.axhline(0, color='black', lw=0.6)

    handles = [mpatches.Patch(color=c, label=f)
               for f, c in FAMILY_COLOURS.items()
               if f in set(sub['family'])]
    ax.legend(handles=handles, fontsize=7, framealpha=0.9,
              loc='upper right', ncol=2)

    fig.tight_layout()
    path = os.path.join(out_dir, 'figure_05_regime_recommendations.png')
    _save(fig, path)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 — Richardson failure profile
# ─────────────────────────────────────────────────────────────────────────────

def fig6_richardson_profile(df_agg: pd.DataFrame,
                              df_global: pd.DataFrame,
                              default_horizon: int,
                              out_dir: str) -> str:
    """
    For each regime, show Richardson_1 stability vs the best-method stability.
    Highlights where Richardson is weak and which family fills the gap.
    """
    rich_rows  = []
    best_rows  = []
    regime_order = []

    for regime in sorted(df_agg['regime'].unique()):
        sub = df_agg[(df_agg['regime'] == regime)
                     & (df_agg['future_idx'] == default_horizon)]
        if sub.empty:
            continue

        per_m = (sub.groupby('method')
                    .agg(valid_rate=('valid_rate','mean'),
                         cat_rate=('cat_rate','mean'),
                         beats_rate=('beats_rate','mean'))
                    .reset_index())
        per_m['stability'] = (per_m['valid_rate']
                               - 2.0 * per_m['cat_rate']
                               + 0.4 * per_m['beats_rate'])

        r1 = per_m[per_m['method'] == 'richardson_1']
        best = per_m.sort_values('stability', ascending=False).iloc[0]

        r1_sc   = float(r1['stability'].values[0]) if not r1.empty else float('nan')
        best_sc = float(best['stability'])
        best_m  = best['method']

        rich_rows.append(r1_sc)
        best_rows.append(best_sc)
        regime_order.append((regime, best_m,
                             FAMILY_COLOURS.get(
                                 best['method'].rsplit('_',1)[0], '#aaa')))

    reg_labels  = [r[0].replace('_', '\n') for r in regime_order]
    best_cols   = [r[2] for r in regime_order]
    best_names  = [r[1] for r in regime_order]
    x           = np.arange(len(regime_order))
    w           = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - w/2, rich_rows,  w, label='Richardson 1',
           color=FAMILY_COLOURS['richardson'], alpha=0.85, edgecolor='white')
    ax.bar(x + w/2, best_rows, w, label='Best method',
           color=best_cols, alpha=0.85, edgecolor='white')

    for i, (bs, bn) in enumerate(zip(best_rows, best_names)):
        ax.text(i + w/2, bs + 0.02, bn.replace('_', '\n'),
                ha='center', va='bottom', fontsize=5.5)

    ax.set_xticks(x)
    ax.set_xticklabels(reg_labels, fontsize=7.5)
    ax.axhline(0, color='black', lw=0.6)
    ax.set_ylabel('Stability Score', fontsize=9)
    ax.set_title(
        f'Figure 6 — Richardson vs Best-Method Per Regime  '
        f'(horizon = {default_horizon})\n'
        'Regimes where bars differ most are Richardson failure conditions',
        fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(out_dir, 'figure_06_richardson_profile.png')
    _save(fig, path)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Master call
# ─────────────────────────────────────────────────────────────────────────────

def make_all_figures(results: dict,
                     horizons: list,
                     default_horizon: int,
                     out_dir: str) -> list:
    """Generate all six Phase 1 figures and return list of paths."""
    df_agg    = results['aggregated']
    df_global = results['global']
    df_best   = results['regime_best']
    heatmaps  = results['heatmaps']

    paths = []
    print('\n  Generating figures ...')

    paths.append(fig1_stability_ranking(df_global, default_horizon, out_dir))
    if default_horizon in heatmaps:
        paths.append(fig2_heatmap(heatmaps[default_horizon],
                                  default_horizon, out_dir))
    paths.append(fig3_horizon_sensitivity(df_global, horizons, out_dir))
    paths.append(fig4_method_type(df_global, horizons, out_dir))
    paths.append(fig5_regime_recommendations(df_best, default_horizon, out_dir))
    paths.append(fig6_richardson_profile(df_agg, df_global,
                                          default_horizon, out_dir))
    return [p for p in paths if p]
