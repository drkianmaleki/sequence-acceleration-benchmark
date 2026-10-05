"""
plots.py
========
Phase 1 figure generation.

Six figures, all descriptive: errors are medians over valid records
(conditional on validity) and the validity rate is shown alongside; the rank
rule of the pooled tables (valid_rate >= config.RANK_MIN_VALID, sorted by
median error) decides who is ranked and who is shown in the unranked block.
No composite score is drawn anywhere.

    Fig 1  Global ranking by median error (headline stratum), unranked block below
    Fig 2  Method x Regime median-error heatmap (headline stratum), below-floor cells marked
    Fig 3  Stratum sensitivity: the top-15 methods' median error across gap strata
    Fig 4  Limit estimator vs Trajectory extrapolator split (median error)
    Fig 5  Per-regime best method by skill
    Fig 6  Richardson_1 vs the best method by median error, per regime

Author : Kian Maleki
Date   : 2026-05-24 (v1), 2026-09-27 (descriptive reporting)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import src.config as CFG_MOD

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
    'brezinski':  '#ff9800',
    'neville':    '#00bcd4',
    'anderson':   '#795548',
    'ensemble':   '#607d8b',
    'trivial':    '#212121',
    'unknown':    '#cccccc',
}

METHOD_TYPE_COLOURS = {
    'trajectory': '#1565c0',
    'limit':      '#b71c1c',
}

FIG_DPI = 150

# Redesign v2: horizons are gap strata (column 'target_g'), not fixed indices.
HORIZON_COL = 'target_g'


def _no_oracle(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the oracle reference rows from ranking figures (tables keep them)."""
    return df[df['is_oracle'] == 0] if 'is_oracle' in df.columns else df


def _save(fig, path):
    fig.savefig(path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')


def _floor_note() -> str:
    return (f'validity floor: valid_rate >= {CFG_MOD.RANK_MIN_VALID:g} to be ranked; '
            f'errors are medians over valid records (conditional on validity)')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Global ranking by median error
# ─────────────────────────────────────────────────────────────────────────────

def fig1_error_ranking(df_global: pd.DataFrame,
                       default_horizon: float,
                       out_dir: str) -> str:
    df_global = _no_oracle(df_global)
    sub = df_global[df_global[HORIZON_COL] == default_horizon]
    ranked = sub[sub['rank_eligible'] == 1].sort_values('med_error', ascending=False)
    below = (sub[(sub['rank_eligible'] == 0) & sub['valid_rate'].notna()]
             .sort_values('valid_rate', ascending=True))
    rows = pd.concat([below, ranked], ignore_index=True)   # best at the top of the axis
    if rows.empty:
        print('  Fig 1 skipped: no rows at the headline stratum.')
        return ''

    labels   = rows['method'].tolist()
    errs     = rows['med_error'].to_numpy(dtype=float)
    colours  = [FAMILY_COLOURS.get(f, '#cccccc') for f in rows['family']]
    hatches  = ['//' if e == 0 else '' for e in rows['rank_eligible']]

    fig, ax = plt.subplots(figsize=(12, max(8, len(labels) * 0.22)))
    plot_err = np.where(np.isfinite(errs), errs, np.nan)
    bars = ax.barh(range(len(labels)), plot_err, color=colours,
                   edgecolor='white', linewidth=0.4, height=0.75)
    for b, h in zip(bars, hatches):
        b.set_hatch(h)
    finite = plot_err[np.isfinite(plot_err)]
    x_text = float(np.nanmax(finite)) * 1.05 if finite.size else 1.0
    for i, r in rows.iterrows():
        t_col = METHOD_TYPE_COLOURS.get(r['method_type'], '#333333')
        tag = (f"med.err={r['med_error']:.4f}  V={r['valid_rate']:.2f}  C={r['cat_rate']:.2f}"
               if np.isfinite(r['med_error']) else f"no valid record  V={r['valid_rate']:.2f}")
        if not r['rank_eligible']:
            tag += '  (below the floor, unranked)'
        ax.text(x_text, i, tag, va='center', fontsize=6.5, color='#333333')
        ax.text(0, i, '●', va='center', ha='right', fontsize=5, color=t_col)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xscale('log')
    ax.set_xlabel('Median absolute error over valid records (log scale)', fontsize=9)
    ax.set_title(
        f'Figure 1 — Global Ranking by Median Error  (gap stratum g = {default_horizon:g}, core regimes)\n'
        f'{_floor_note()}; hatched = below the floor (shown, unranked); '
        'V = valid rate, C = catastrophic rate; ● blue = trajectory extrapolator, ● red = limit estimator',
        fontsize=9.5, fontweight='bold')
    if len(below):
        ax.axhline(len(below) - 0.5, color='black', lw=0.8, ls='--')

    handles = [mpatches.Patch(color=c, label=f)
               for f, c in FAMILY_COLOURS.items()
               if f in set(rows['family'])]
    ax.legend(handles=handles, loc='lower right', fontsize=7,
              framealpha=0.9, ncol=2)
    fig.tight_layout()
    path = os.path.join(out_dir, 'figure_01_error_ranking.png')
    _save(fig, path)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Method × Regime median-error heatmap
# ─────────────────────────────────────────────────────────────────────────────

def fig2_heatmap(heatmap_df: pd.DataFrame,
                 valid_df: pd.DataFrame,
                 default_horizon: float,
                 out_dir: str) -> str:
    """heatmap_df: method x regime median error; valid_df: the matching valid rate."""
    mat  = heatmap_df.values.astype(float)
    vmat = valid_df.reindex(index=heatmap_df.index, columns=heatmap_df.columns).values.astype(float)
    rows = list(heatmap_df.index)
    cols = [c.replace('_', '\n') for c in heatmap_df.columns]

    with np.errstate(divide='ignore', invalid='ignore'):
        lmat = np.log10(mat)
    # Sort rows by mean log10 median error ascending (best at the top)
    row_means  = np.nanmean(np.where(np.isfinite(lmat), lmat, np.nan), axis=1)
    sort_order = np.argsort(np.where(np.isfinite(row_means), row_means, np.inf))
    lmat, vmat = lmat[sort_order], vmat[sort_order]
    rows = [rows[i] for i in sort_order]

    fig, ax = plt.subplots(figsize=(max(16, len(cols) * 0.9),
                                    max(10, len(rows) * 0.25)))
    im = ax.imshow(lmat, cmap='RdYlGn_r', aspect='auto')

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=7.5, ha='center')
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=7.5)

    for i in range(len(rows)):
        for j in range(len(heatmap_df.columns)):
            v, vr = lmat[i, j], vmat[i, j]
            if np.isfinite(v):
                txt = f'{v:.1f}' + ('*' if (np.isfinite(vr) and vr < CFG_MOD.RANK_MIN_VALID) else '')
                ax.text(j, i, txt, ha='center', va='center', fontsize=5.5, color='#333333')
            elif np.isfinite(vr):
                ax.text(j, i, 'inv', ha='center', va='center', fontsize=5.5, color='#333333')

    plt.colorbar(im, ax=ax, label='log10 median absolute error (valid records)', shrink=0.5)
    ax.set_title(
        f'Figure 2 — Method × Regime Median-Error Heatmap  (gap stratum g = {default_horizon:g})\n'
        f'median over noise levels of the cell median error, conditional on validity; '
        f'* = valid rate below {CFG_MOD.RANK_MIN_VALID:g}; inv = no valid record',
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
                              default_horizon: float,
                              out_dir: str,
                              top_n: int = 15) -> str:
    df_global = _no_oracle(df_global)
    if len(horizons) < 2:
        print('  Fig 3 skipped: only one horizon.')
        return ''

    # top_n by median error among the rank-eligible methods at the headline stratum
    head = df_global[(df_global[HORIZON_COL] == default_horizon) & (df_global['rank_eligible'] == 1)]
    top_methods = list(head.sort_values(['med_error', 'method'])['method'][:top_n])
    if not top_methods:
        print('  Fig 3 skipped: no rank-eligible method at the headline stratum.')
        return ''

    fig, axes = plt.subplots(1, len(horizons), figsize=(5 * len(horizons), 9),
                             sharey=False)
    if len(horizons) == 1:
        axes = [axes]

    for ax_idx, fid in enumerate(horizons):
        ax  = axes[ax_idx]
        sub = df_global[df_global[HORIZON_COL] == fid].set_index('method')
        for rank_pos, method in enumerate(top_methods):
            if method not in sub.index:
                continue
            row    = sub.loc[method]
            err    = float(row['med_error'])
            colour = FAMILY_COLOURS.get(row['family'], '#cccccc')
            if np.isfinite(err):
                ax.barh(rank_pos, err, color=colour, edgecolor='white', lw=0.4,
                        hatch='' if row['rank_eligible'] else '//')
                ax.text(err * 1.05, rank_pos,
                        f"{err:.4f}  V={row['valid_rate']:.2f}" + ('' if row['rank_eligible'] else '  unranked'),
                        va='center', fontsize=7)
            else:
                ax.text(1e-6, rank_pos, f"no valid record  V={row['valid_rate']:.2f}", va='center', fontsize=7)

        ax.set_yticks(range(len(top_methods)))
        ax.set_yticklabels(top_methods if ax_idx == 0 else [], fontsize=7.5)
        ax.set_title(f'g = {fid:g}', fontsize=9, fontweight='bold')
        ax.set_xscale('log')
        ax.set_xlabel('Median absolute error (valid records, log scale)', fontsize=8)
        ax.invert_yaxis()

    fig.suptitle(
        f'Figure 3 — Stratum Sensitivity: the Top-{len(top_methods)} Methods by Median Error at g = {default_horizon:g}\n'
        f'{_floor_note()}; hatched = below the floor at that stratum; V = valid rate',
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
    df_global = _no_oracle(df_global)
    fig, axes = plt.subplots(1, len(horizons),
                              figsize=(5 * len(horizons), 6), sharey=True)
    if len(horizons) == 1:
        axes = [axes]

    for ax, fid in zip(axes, horizons):
        sub = df_global[df_global[HORIZON_COL] == fid]
        for mtype, colour in METHOD_TYPE_COLOURS.items():
            s = sub[sub['method_type'] == mtype]
            vals = np.log10(s['med_error'].dropna().to_numpy(dtype=float))
            vals = vals[np.isfinite(vals)]
            pos  = 0 if mtype == 'limit' else 1
            if vals.size == 0:
                continue
            parts = ax.violinplot([vals], positions=[pos],
                                  showmedians=True, showextrema=True)
            for pc in parts['bodies']:
                pc.set_facecolor(colour)
                pc.set_alpha(0.6)
            for key in ('cmedians', 'cbars', 'cmaxes', 'cmins'):
                parts[key].set_color(colour)
            med = float(np.median(vals))
            n_below = int((s['rank_eligible'] == 0).sum())
            ax.text(pos, med, f'  {10 ** med:.4f}\n  ({len(s)} methods, {n_below} below floor)',
                    va='bottom', fontsize=7.5, color=colour, fontweight='bold')

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Limit\nEstimator', 'Trajectory\nExtrapolator'],
                           fontsize=9)
        ax.set_title(f'g = {fid:g}', fontsize=9, fontweight='bold')
        ax.set_ylabel('log10 median absolute error (valid records)', fontsize=8)

    fig.suptitle(
        'Figure 4 — Median Error by Method Type Across Gap Strata\n'
        'Limit estimators predict the mathematical limit; trajectory extrapolators predict s(n) at the target index;\n'
        'every method with a valid record is drawn, the count below the validity floor is noted',
        fontsize=10, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(out_dir, 'figure_04_method_type.png')
    _save(fig, path)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — Per-regime best method summary
# ─────────────────────────────────────────────────────────────────────────────

def fig5_regime_recommendations(df_best: pd.DataFrame,
                                 default_horizon: float,
                                 out_dir: str) -> str:
    sub = (df_best[df_best[HORIZON_COL] == default_horizon]
             .copy()
             .reset_index(drop=True))

    # Redesign v2: the primary per-regime recommendation is best_by_skill
    # (median skill vs the best-of-four trivial reference; lower is better,
    # 1.0 = trivial parity).  Capped cells are marked.
    fig, ax = plt.subplots(figsize=(12, 7))
    regime_labels = [r.replace('_', '\n') + (' [CAP]' if c else '')
                     for r, c in zip(sub['regime'], sub['capped'])]
    colours = [FAMILY_COLOURS.get(f, '#cccccc') for f in sub['family']]
    heights = sub['best_skill'].fillna(0.0)

    ax.bar(range(len(sub)), heights, color=colours, edgecolor='white', lw=0.5)

    for i, (bm, sk, vr, cr, be) in enumerate(zip(
            sub['best_by_skill'], sub['best_skill'],
            sub['skill_best_valid_rate'], sub['skill_best_cat_rate'], sub['best_by_error'])):
        ax.text(i, (sk if np.isfinite(sk) else 0.0) + 0.01, str(bm).replace('_', '\n'),
                ha='center', va='bottom', fontsize=6.5, rotation=0)
        be_txt = str(be) if isinstance(be, str) and be else 'none at floor'
        ax.text(i, -0.02, f'V={vr:.2f}\nC={cr:.2f}\nby err:\n{be_txt.replace("_", chr(10))}',
                ha='center', va='top', fontsize=5.5, color='#555555')

    ax.set_xticks(range(len(sub)))
    ax.set_xticklabels(regime_labels, fontsize=7.5)
    ax.set_ylabel('Median skill of best-by-skill method  (lower = better)', fontsize=9)
    ax.set_title(
        f'Figure 5 — Per-Regime Best Method by Skill  (gap stratum g = {default_horizon:g})\n'
        'Method name above each bar; skill 1.0 = parity with the best trivial predictor; '
        'V = valid rate, C = cat rate; "by err" = best by median error at or above the validity floor',
        fontsize=10, fontweight='bold')
    ax.axhline(1.0, color='black', lw=0.8, ls='--', alpha=0.6)
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
                              default_horizon: float,
                              out_dir: str) -> str:
    """
    For each regime, richardson_1's median error next to the best method's
    (lowest median error among the methods at or above the validity floor,
    pooled over the noise levels).  Highlights where Richardson is weak and
    which family fills the gap.
    """
    df_agg = _no_oracle(df_agg)
    rich_rows, rich_vr = [], []
    best_rows  = []
    regime_order = []

    for regime in sorted(df_agg['regime'].unique()):
        sub = df_agg[(df_agg['regime'] == regime)
                     & (df_agg[HORIZON_COL] == default_horizon)]
        if sub.empty:
            continue
        per_m = (sub.groupby('method')
                    .agg(valid_rate=('valid_rate', 'mean'),
                         med_error=('med_error', 'median'),
                         family=('family', 'first'))
                    .reset_index())
        r1 = per_m[per_m['method'] == 'richardson_1']
        elig = per_m[(per_m['valid_rate'] >= CFG_MOD.RANK_MIN_VALID) & per_m['med_error'].notna()]
        best = elig.sort_values(['med_error', 'method']).iloc[0] if len(elig) else None

        rich_rows.append(float(r1['med_error'].values[0]) if not r1.empty else float('nan'))
        rich_vr.append(float(r1['valid_rate'].values[0]) if not r1.empty else float('nan'))
        best_rows.append(float(best['med_error']) if best is not None else float('nan'))
        regime_order.append((regime,
                             best['method'] if best is not None else 'none at floor',
                             FAMILY_COLOURS.get(best['family'], '#aaa') if best is not None else '#aaa'))

    reg_labels  = [r[0].replace('_', '\n') for r in regime_order]
    best_cols   = [r[2] for r in regime_order]
    best_names  = [r[1] for r in regime_order]
    x           = np.arange(len(regime_order))
    w           = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - w/2, rich_rows,  w, label='richardson_1',
           color=FAMILY_COLOURS['richardson'], alpha=0.85, edgecolor='white')
    ax.bar(x + w/2, best_rows, w, label='best method by median error (at the floor)',
           color=best_cols, alpha=0.85, edgecolor='white')

    for i, (bs, bn, rv) in enumerate(zip(best_rows, best_names, rich_vr)):
        if np.isfinite(bs):
            ax.text(i + w/2, bs * 1.05, bn.replace('_', '\n'),
                    ha='center', va='bottom', fontsize=5.5)
        ax.text(i - w/2, (rich_rows[i] if np.isfinite(rich_rows[i]) else 1e-6) * 1.05,
                f'V={rv:.2f}', ha='center', va='bottom', fontsize=5.5, color='#555555')

    ax.set_xticks(x)
    ax.set_xticklabels(reg_labels, fontsize=7.5)
    ax.set_yscale('log')
    ax.set_ylabel('Median absolute error (valid records, log scale)', fontsize=9)
    ax.set_title(
        f'Figure 6 — richardson_1 vs the Best Method by Median Error, per Regime  '
        f'(gap stratum g = {default_horizon:g})\n'
        f'{_floor_note()}; V = richardson_1 valid rate; regimes where the bars differ most are Richardson failure conditions',
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
                     default_horizon: float,
                     out_dir: str) -> list:
    """Generate all six Phase 1 figures and return list of paths."""
    df_agg    = results['aggregated']
    df_global = results['global']
    df_best   = results['regime_best']
    heatmaps  = results['heatmaps']
    heatmaps_valid = results['heatmaps_valid']

    paths = []
    print('\n  Generating figures ...')

    paths.append(fig1_error_ranking(df_global, default_horizon, out_dir))
    if default_horizon in heatmaps:
        paths.append(fig2_heatmap(heatmaps[default_horizon], heatmaps_valid[default_horizon],
                                  default_horizon, out_dir))
    paths.append(fig3_horizon_sensitivity(df_global, horizons, default_horizon, out_dir))
    paths.append(fig4_method_type(df_global, horizons, out_dir))
    paths.append(fig5_regime_recommendations(df_best, default_horizon, out_dir))
    paths.append(fig6_richardson_profile(df_agg, df_global,
                                          default_horizon, out_dir))
    return [p for p in paths if p]
