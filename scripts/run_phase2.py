"""
run_phase2.py
=============
Phase 2 entry point — Richardson Failure Condition Mapping.

Usage (from inside sequence_accel/)
------------------------------------
    python run_phase2.py --quick     Fast sanity-check
                                     ~5 min on a modern laptop

    python run_phase2.py --full      Full publication run
                                     ~25-40 min depending on hardware

Quick mode configuration
-------------------------
    obs_idx     : [30, 60, 90, 120]
    noise       : [0.0, 0.005]
    future_idx  : [5000]
    seeds       : 10
    window_len  : 60 (capped to obs_idx)

Full mode configuration
-----------------------
    obs_idx     : [20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140]
    noise       : [0.0, 0.003, 0.005, 0.010, 0.020]
    future_idx  : [150, 1000, 5000]
    seeds       : 20
    window_len  : 60 (capped to obs_idx)

Output directory: results/phase2/

Key output files
----------------
    phase2_sweep_aggregated.csv   Per (method, regime, obs_idx, noise, horizon).
    phase2_features.csv           Six trajectory features per sequence.
    phase2_phase_diagram_n5000.csv  2-D Richardson rank grid.
    phase2_correlations.csv       Spearman feature–failure correlations.
    phase2_rules.csv              Threshold rule performance table.
    figure_p2_01 ... figure_p2_05 Five publication figures.
"""

import os, sys, argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from phases.phase2 import (
    run_sweep, build_phase_diagrams,
    run_correlation_analysis, test_simple_rules,
    make_all_figures, PHASE2_METHODS,
)
# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Phase 2 — Richardson failure mapping')
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--quick', action='store_true')
    mode.add_argument('--full',  action='store_true')
    return p.parse_args()

# ── Configurations ─────────────────────────────────────────────────────────────

QUICK = dict(
    obs_idx_list = [30, 60, 90, 120],
    noise_list   = [0.0, 0.005],
    future_list  = [5000],
    n_seeds      = 10,
    window_len   = 60,
)

FULL = dict(
    obs_idx_list = [20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140],
    noise_list   = [0.0, 0.003, 0.005, 0.010, 0.020],
    future_list  = [150, 1000, 5000],
    n_seeds      = 20,
    window_len   = 60,
)

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args    = parse_args()
    cfg     = QUICK if args.quick else FULL
    mode    = 'QUICK' if args.quick else 'FULL'
    out_dir = os.path.join('results', 'phase2')
    default_fid = cfg['future_list'][-1]

    n_evals = (len(cfg['obs_idx_list']) * len(cfg['noise_list'])
               * cfg['n_seeds'] * 18 * len(PHASE2_METHODS)
               * len(cfg['future_list']))

    print('=' * 72)
    print(f'  PHASE 2 — Richardson Failure Condition Mapping  [{mode}]')
    print('=' * 72)
    print(f'  obs_idx sweep : {cfg["obs_idx_list"]}')
    print(f'  noise levels  : {cfg["noise_list"]}')
    print(f'  horizons      : {cfg["future_list"]}')
    print(f'  seeds         : {cfg["n_seeds"]}')
    print(f'  methods       : {len(PHASE2_METHODS)}  {PHASE2_METHODS}')
    print(f'  total evals   : {n_evals:,}')
    print(f'  output dir    : {out_dir}')
    print('=' * 72 + '\n')

    # ── 1. Sweep ───────────────────────────────────────────────────────────────
    df_agg, df_feat = run_sweep(
        obs_idx_list = cfg['obs_idx_list'],
        noise_list   = cfg['noise_list'],
        future_list  = cfg['future_list'],
        n_seeds      = cfg['n_seeds'],
        window_len   = cfg['window_len'],
        out_dir      = out_dir,
    )

    # ── 2. Phase diagrams ──────────────────────────────────────────────────────
    print('\n  Building phase diagrams ...')
    df_pd = build_phase_diagrams(df_agg, default_fid, out_dir)

    # ── 3. Correlation analysis ────────────────────────────────────────────────
    print('\n  Running correlation analysis ...')
    df_corr, df_merged = run_correlation_analysis(
        df_feat, df_agg, default_fid, out_dir)

    # ── 4. Rule testing ────────────────────────────────────────────────────────
    print('\n  Testing simple threshold rules ...')
    df_rules = test_simple_rules(
        df_merged, df_feat, df_agg, default_fid, out_dir)

    # ── 5. Figures ─────────────────────────────────────────────────────────────
    make_all_figures(df_pd, df_corr, df_rules, default_fid, out_dir)

    # ── Console summary ────────────────────────────────────────────────────────
    print('\n' + '=' * 72)
    print('  PHASE 2 SUMMARY')
    print('=' * 72)

    # Richardson win rate by regime
    print('\n  Richardson rank by regime (mean across obs_idx and noise):')
    print(f"  {'Regime':<22} {'Mean rank':>10} {'Min rank':>10} {'Wins':>8}")
    print('  ' + '─' * 55)
    for regime in sorted(df_pd['regime'].unique()):
        sub = df_pd[df_pd['regime'] == regime]
        mr = sub['richardson_rank'].mean()
        mn = sub['richardson_rank'].min()
        wins = (sub['richardson_rank'] == 1).sum()
        print(f"  {regime:<22} {mr:>10.2f} {mn:>10d} {wins:>8d}")

    # Top correlations
    print('\n  Top feature correlations (all regimes, vs Richardson losing margin):')
    all_c = df_corr[df_corr['regime'] == 'ALL'].sort_values(
        'spearman_vs_margin', key=abs, ascending=False)
    for _, row in all_c.iterrows():
        flag = '  *** candidate' if abs(row['spearman_vs_margin']) >= 0.3 else ''
        print(f"  {row['feature']:<22}  r = {row['spearman_vs_margin']:>7.4f}"
              f"  (p = {row['p_vs_margin']:.3f}){flag}")

    # Top rules
    print('\n  Top 5 threshold rules (by precision):')
    print(f"  {'Rule':<40} {'Prec':>6} {'Recall':>7} {'Gain':>7}")
    print('  ' + '─' * 65)
    for _, row in df_rules.head(5).iterrows():
        rule = f"{row['feature']} {row['operator']} {row['threshold']} → {row['alternative']}"
        print(f"  {rule:<40} {row['precision']:>6.3f} {row['recall']:>7.3f} "
              f"{row['mean_gain']:>7.4f}")

    print(f'\n  All files saved to: {out_dir}/')
    print('=' * 72)


if __name__ == '__main__':
    main()
