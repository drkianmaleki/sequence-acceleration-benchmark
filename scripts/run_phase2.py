"""
run_phase2.py
=============
Phase 2 entry point — Richardson Failure Condition Mapping (redesign v2).

    python scripts/run_phase2.py --quick     config.PHASE2["quick"]
    python scripts/run_phase2.py --full      config.PHASE2["full"]

Grid: 13 observation depths x noise levels x seeds x core regimes x three
gap strata (g in config.HORIZON_GAP_FRACTIONS) x 11 methods (the 9-method
pool + constant_assumed + constant_oracle).  Targets are per-depth
gap-stratified horizons; capped cells are flagged and excluded from pooled
statistics.

Output directory: results/phase2/

Key output files
----------------
    phase2_sweep_aggregated.csv    Per (method, regime, obs_idx, noise, target_g)
                                   with n_f, achieved_g, capped, skill, L_true, L_hat.
    phase2_features.csv            Six trajectory features per sequence (+ L_true, L_hat).
    phase2_capped.csv              The capped block.
    phase2_phase_diagram_g{g}.csv  2-D Richardson rank grid per stratum.
    phase2_correlations[_g{g}].csv Spearman feature–failure correlations
                                   (headline stratum under the legacy name).
    phase2_rules[_g{g}].csv        Threshold rule performance per stratum.
    figure_p2_01 ... figure_p2_05  Five figures at the headline stratum.
"""

import os, sys, argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import src.config as CFG_MOD
from src.pipeline import resolve_regimes
from phases.phase2 import (
    run_sweep, build_phase_diagrams,
    run_correlation_analysis, test_simple_rules,
    make_all_figures, PHASE2_METHODS, RANK_POOL,
)


def n_evaluations(cfg: dict) -> int:
    """Planned pool evaluations (skill references add two cheap calls per cell)."""
    regimes = resolve_regimes(cfg['core_regimes'], include_holdout=False)
    return (len(cfg['obs_idx_list']) * len(cfg['noise_list']) * cfg['n_seeds']
            * len(regimes) * len(cfg['gap_fractions']) * len(PHASE2_METHODS))


def parse_args():
    p = argparse.ArgumentParser(description='Phase 2 — Richardson failure mapping (v2)')
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--quick', action='store_true')
    mode.add_argument('--full',  action='store_true')
    p.add_argument('--out-dir', default=os.path.join('results', 'phase2'))
    return p.parse_args()


def main():
    args    = parse_args()
    mode    = 'quick' if args.quick else 'full'
    cfg     = dict(CFG_MOD.PHASE2[mode])
    out_dir = args.out_dir
    gs      = [float(g) for g in cfg['gap_fractions']]
    g_head  = CFG_MOD.HEADLINE_G if CFG_MOD.HEADLINE_G in gs else gs[-1]
    regimes = resolve_regimes(cfg['core_regimes'], include_holdout=False)

    print('=' * 72)
    print(f'  PHASE 2 — Richardson Failure Condition Mapping  [{mode.upper()}, redesign v2]')
    print('=' * 72)
    print(f'  obs_idx sweep : {cfg["obs_idx_list"]}')
    print(f'  noise levels  : {cfg["noise_list"]}')
    print(f'  gap strata    : {gs}  (headline g = {g_head})')
    print(f'  seeds         : {cfg["n_seeds"]}')
    print(f'  regimes       : {len(regimes)} core (selector training data; no holdout)')
    print(f'  methods       : {len(PHASE2_METHODS)}  {PHASE2_METHODS}')
    print(f'  rank pool     : {len(RANK_POOL)} (oracle excluded from ranks)')
    print(f'  L_hat mode    : {CFG_MOD.ASSUMED_L_MODE}')
    print(f'  total evals   : {n_evaluations(cfg):,}')
    print(f'  output dir    : {out_dir}')
    print('=' * 72 + '\n')

    # ── 1. Sweep ───────────────────────────────────────────────────────────────
    df_agg, df_feat = run_sweep(
        obs_idx_list  = cfg['obs_idx_list'],
        noise_list    = cfg['noise_list'],
        gap_fractions = gs,
        n_seeds       = cfg['n_seeds'],
        window_len    = cfg['window_len'],
        out_dir       = out_dir,
        core_regimes  = cfg['core_regimes'],
    )

    # ── 2-4. Phase diagrams, correlations and rules for every stratum ──────────
    df_pd_head = df_corr_head = df_rules_head = None
    for g in gs:
        suffix = f'_g{g:g}'
        print(f'\n  Stratum g = {g:g}: phase diagram, correlations, rules ...')
        df_pd = build_phase_diagrams(df_agg, g, out_dir)
        df_corr, df_merged = run_correlation_analysis(df_feat, df_agg, g, out_dir, suffix)
        df_rules = test_simple_rules(df_merged, df_feat, df_agg, g, out_dir, suffix)
        if g == g_head:
            df_pd_head, df_corr_head, df_rules_head = df_pd, df_corr, df_rules
            # headline stratum also under the legacy file names
            df_corr.to_csv(os.path.join(out_dir, 'phase2_correlations.csv'), index=False)
            df_rules.to_csv(os.path.join(out_dir, 'phase2_rules.csv'), index=False)

    # ── 5. Figures (headline stratum) ──────────────────────────────────────────
    make_all_figures(df_pd_head, df_corr_head, df_rules_head, g_head, out_dir)

    # ── Console summary ────────────────────────────────────────────────────────
    print('\n' + '=' * 72)
    print('  PHASE 2 SUMMARY  (headline stratum g = %g)' % g_head)
    print('=' * 72)

    print('\n  Richardson rank by regime (mean across obs_idx and noise; capped cells flagged):')
    print(f"  {'Regime':<22} {'Mean rank':>10} {'Min rank':>10} {'Wins':>8} {'Capped':>8}")
    print('  ' + '─' * 65)
    for regime in sorted(df_pd_head['regime'].unique()):
        sub = df_pd_head[df_pd_head['regime'] == regime]
        mr = sub['richardson_rank'].mean()
        mn = sub['richardson_rank'].min()
        wins = (sub['richardson_rank'] == 1).sum()
        capn = int(sub['capped'].sum())
        print(f"  {regime:<22} {mr:>10.2f} {mn:>10d} {wins:>8d} {capn:>8d}")

    print('\n  Top feature correlations (all regimes, vs Richardson losing margin):')
    all_c = df_corr_head[df_corr_head['regime'] == 'ALL'].sort_values(
        'spearman_vs_margin', key=abs, ascending=False)
    for _, row in all_c.iterrows():
        flag = '  *** candidate' if abs(row['spearman_vs_margin']) >= 0.3 else ''
        print(f"  {row['feature']:<22}  r = {row['spearman_vs_margin']:>7.4f}"
              f"  (p = {row['p_vs_margin']:.3f}){flag}")

    print('\n  Top 5 threshold rules (by precision):')
    print(f"  {'Rule':<40} {'Prec':>6} {'Recall':>7} {'Gain':>7}")
    print('  ' + '─' * 65)
    for _, row in df_rules_head.head(5).iterrows():
        rule = f"{row['feature']} {row['operator']} {row['threshold']} → {row['alternative']}"
        print(f"  {rule:<40} {row['precision']:>6.3f} {row['recall']:>7.3f} "
              f"{row['mean_gain']:>7.4f}")

    print(f'\n  All files saved to: {out_dir}/')
    print('=' * 72)


if __name__ == '__main__':
    main()
