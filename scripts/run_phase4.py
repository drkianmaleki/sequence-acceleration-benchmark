"""
run_phase4.py
=============
Phase 4 entry point — Stability Diagnostic Stress Testing.

Usage (from inside sequence_accel/)
------------------------------------
    python run_phase4.py --quick     ~5 min on a modern laptop
    python run_phase4.py --full      ~25-35 min

Quick mode
----------
    obs_idx  : [60, 90]
    noise    : [0.0, 0.005]
    horizons : [5000]
    seeds    : 5
    shifts   : [-1, 0, +1]
    perturbs : 3

Full mode
---------
    obs_idx  : [30, 60, 90, 120]
    noise    : [0.0, 0.005, 0.020]
    horizons : [1000, 5000]
    seeds    : 20
    shifts   : [-2, -1, 0, +1, +2]
    perturbs : 5

Output directory: results/phase4/

Key output files
----------------
    phase4_diagnostic_correlations.csv   Spearman r (both diagnostics)
    phase4_rejection_rules.csv           Precision / recall (both diagnostics)
    phase4_cascade_filter.csv            Cascade + perturb_IQR filter
    phase4_ensemble.csv                  Ensemble vs fixed selectors
    phase4_obs_reliability.csv           Diagnostic r vs obs_idx
    figure_p4_01 ... figure_p4_05        Five publication figures

What to paste back
------------------
After running, paste:
    1. Console summary
    2. phase4_diagnostic_correlations.csv  (small)
    3. phase4_rejection_rules.csv          (medium — filter to precision >= 0.3)
    4. phase4_obs_reliability.csv          (small)
"""

import os, sys, argparse, math
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from phases.phase4 import run_all

def parse_args():
    p = argparse.ArgumentParser(description='Phase 4 — Stability diagnostics')
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--quick', action='store_true')
    g.add_argument('--full',  action='store_true')
    return p.parse_args()

QUICK = dict(
    obs_idx_list   = [60, 90],
    noise_list     = [0.0, 0.005],
    future_list    = [5000],
    n_seeds        = 5,
    window_len     = 60,
    shifts         = [-1, 0, 1],
    perturb_trials = 3,
    perturb_scale  = 0.02,
)
FULL = dict(
    obs_idx_list   = [30, 60, 90, 120],
    noise_list     = [0.0, 0.005, 0.020],
    future_list    = [1000, 5000],
    n_seeds        = 20,
    window_len     = 60,
    shifts         = [-2, -1, 0, 1, 2],
    perturb_trials = 5,
    perturb_scale  = 0.02,
)

def main():
    args    = parse_args()
    cfg     = QUICK if args.quick else FULL
    mode    = 'QUICK' if args.quick else 'FULL'
    out_dir = os.path.join('results', 'phase4')
    feat_p  = os.path.join('results', 'phase2', 'phase2_features.csv')

    print('=' * 72)
    print(f'  PHASE 4 — Stability Diagnostic Stress Testing  [{mode}]')
    print('=' * 72)
    print(f'  obs_idx    : {cfg["obs_idx_list"]}')
    print(f'  noise      : {cfg["noise_list"]}')
    print(f'  horizons   : {cfg["future_list"]}')
    print(f'  seeds      : {cfg["n_seeds"]}')
    print(f'  shifts     : {cfg["shifts"]}')
    print(f'  perturbs   : {cfg["perturb_trials"]}')
    print(f'  output dir : {out_dir}')
    print('=' * 72 + '\n')

    results = run_all(
        **cfg,
        out_dir          = out_dir,
        phase2_feat_path = feat_p,
    )

    # ── Console summary ────────────────────────────────────────────────────────
    df_corr  = results['correlations']
    df_rules = results['rules']
    df_rel   = results['reliability']
    df_ens   = results['ensemble']

    print('\n' + '=' * 72)
    print('  PHASE 4 SUMMARY')
    print('=' * 72)

    # Q1 — Global correlations
    print('\n  Global diagnostic correlations (vs |error|):')
    print(f"  {'Diagnostic':<16} {'r':>8} {'p':>10}  Status")
    print('  ' + '─' * 55)
    for _, row in df_corr[df_corr['method'] == 'ALL'].iterrows():
        r = row['spearman_r']
        flag = '*** candidate' if abs(r) >= 0.30 else \
               '** moderate'   if abs(r) >= 0.20 else ''
        print(f"  {row['diagnostic']:<16} {r:>8.4f} "
              f"{row['p_value']:>10.4f}  {flag}")

    # Q2 — Best rejection rules (both diagnostics)
    print('\n  Best rejection rules (precision >= 0.30):')
    print(f"  {'Diagnostic':<14} {'Method':<20} "
          f"{'Thresh':>8} {'Prec':>7} {'Recall':>7}")
    print('  ' + '─' * 62)
    if not df_rules.empty:
        good = (df_rules[df_rules['precision'] >= 0.30]
                .sort_values('precision', ascending=False)
                .drop_duplicates(['diagnostic', 'method']))
        for _, row in good.head(12).iterrows():
            print(f"  {row['diagnostic']:<14} {row['method']:<20} "
                  f"{row['threshold']:>8.3f} {row['precision']:>7.3f} "
                  f"{row['recall']:>7.3f}")

    # Q3 — Cascade filter
    df_filt = results['filter']
    if not df_filt.empty:
        print('\n  Cascade + perturb_IQR filter (mean error):')
        print(f"  {'Filter':<25} {'Mean error':>12}")
        print('  ' + '─' * 40)
        for _, row in df_filt.iterrows():
            print(f"  {row['filter']:<25} {row['mean_error']:>12.6f}")

    # Q4 — Ensemble
    if not df_ens.empty:
        print('\n  Selector mean error comparison:')
        print(f"  {'Selector':<22} {'Mean error':>12} {'Median':>10}")
        print('  ' + '─' * 48)
        for _, row in df_ens.sort_values('mean_error').iterrows():
            print(f"  {row['selector']:<22} {row['mean_error']:>12.6f} "
                  f"{row['median_error']:>10.6f}")

    # Reliability by obs_idx for perturb_iqr + richardson_1
    print('\n  perturb_IQR vs error correlation (richardson_1, by obs_idx):')
    r1_piqr = (df_rel[(df_rel['method']     == 'richardson_1')
                       & (df_rel['diagnostic'] == 'perturb_iqr')]
               .sort_values('obs_idx'))
    print(f"  {'obs_idx':>8} {'r':>8} {'p':>10}")
    print('  ' + '─' * 32)
    for _, row in r1_piqr.iterrows():
        print(f"  {int(row['obs_idx']):>8} {row['spearman_r']:>8.4f} "
              f"{row['p_value']:>10.4f}")

    print(f'\n  All files saved to: {out_dir}/')
    print('=' * 72)
    print('\n  NEXT STEP:')
    print('  Paste the console output above plus:')
    print('    results/phase4/phase4_diagnostic_correlations.csv')
    print('    results/phase4/phase4_rejection_rules.csv')
    print('    results/phase4/phase4_obs_reliability.csv')

if __name__ == '__main__':
    main()
