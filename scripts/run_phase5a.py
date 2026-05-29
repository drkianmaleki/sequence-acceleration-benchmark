"""
run_phase5a.py  (v3)
====================
Phase 5A entry point — Full 51-Method Ensemble with Ablation.

    python run_phase5a.py --quick     ~20 min
    python run_phase5a.py --full      ~3-4 hrs

New in v3
---------
  * threshold_ensemble variants added (filter high-IQR, equal weight rest)
  * capped_diag ensemble added (weight capped at 5x median to prevent
    any single method from dominating)
  * phase5a_by_sigma.csv added — shows whether diagnostic adds value at
    sigma=0.005 and sigma=0.020 vs sigma=0.0

What to paste back
------------------
    1. Console summary
    2. results/phase5a/phase5a_ensemble.csv       (small)
    3. results/phase5a/phase5a_ablation.csv       (small)
    4. results/phase5a/phase5a_by_sigma.csv       (medium — key analysis)
    5. results/phase5a/phase5a_method_weights.csv (medium)
"""

import os, sys, argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from phase5a import run_all, SELECTORS, DANGEROUS, EPS
from accelerators import METHOD_NAMES

def parse_args():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--quick', action='store_true')
    g.add_argument('--full',  action='store_true')
    return p.parse_args()

QUICK = dict(
    obs_idx_list   = [60, 90],
    noise_list     = [0.0, 0.005],    # include sigma>0 even in quick
    future_list    = [5000],
    n_seeds        = 5,
    window_len     = 60,
    perturb_trials = 3,
    perturb_scale  = 0.02,
)

FULL = dict(
    obs_idx_list   = [30, 60, 90, 120],
    noise_list     = [0.0, 0.005, 0.020],
    future_list    = [1000, 5000],
    n_seeds        = 20,
    window_len     = 60,
    perturb_trials = 5,
    perturb_scale  = 0.02,
)

def _print_comparison(df_comp, fid, n_show=12):
    sub = (df_comp[df_comp['future_idx'] == fid]
           .sort_values('mean_error').head(n_show))
    print(f'\n  Global mean error  (horizon = {fid},'
          f' top {n_show}):')
    print(f"  {'Selector':<26} {'Mean err':>10} {'Median':>10}")
    print('  ' + '-' * 52)
    for _, row in sub.iterrows():
        marker = ''
        if row['selector'] == 'threshold_ens_010':
            marker = '  <-- Phase 5A best?'
        elif row['selector'] == 'oracle_51':
            marker = '  <-- upper bound'
        elif row['selector'] == 'fixed_rational':
            marker = '  <-- fixed baseline'
        print(f"  {row['selector']:<26} "
              f"{row['mean_error']:>10.6f} "
              f"{row['median_error']:>10.6f}{marker}")

def main():
    args    = parse_args()
    cfg     = QUICK if args.quick else FULL
    mode    = 'QUICK' if args.quick else 'FULL'
    out_dir = os.path.join('results', 'phase5a')

    n_methods = len(list(METHOD_NAMES))
    n_calls   = (len(cfg['obs_idx_list']) * len(cfg['noise_list'])
                 * cfg['n_seeds'] * 18 * n_methods
                 * (1 + cfg['perturb_trials'])
                 * len(cfg['future_list']))

    print('=' * 72)
    print(f'  PHASE 5A — Full 51-Method Ensemble  [{mode}]')
    print('=' * 72)
    print(f'  Methods     : {n_methods}  ({len(DANGEROUS)} dangerous)')
    print(f'  obs_idx     : {cfg["obs_idx_list"]}')
    print(f'  noise       : {cfg["noise_list"]}')
    print(f'  horizons    : {cfg["future_list"]}')
    print(f'  seeds       : {cfg["n_seeds"]}')
    print(f'  perturbs    : {cfg["perturb_trials"]}')
    print(f'  EPS         : {EPS}')
    print(f'  Method calls: ~{n_calls:,}')
    if not args.quick:
        print(f'  Est. runtime: 3-4 hours')
        print(f'  Progress    : prints every ~5% (~5-15 min intervals)')
    print('=' * 72 + '\n')

    results = run_all(**cfg, out_dir=out_dir)

    df_comp  = results['comparison']
    df_sigma = results['by_sigma']
    df_abl   = results['ablation']
    wgt_agg  = results['weights']
    df_gap   = results['gap']

    print('\n' + '=' * 72)
    print('  PHASE 5A SUMMARY')
    print('=' * 72)

    for fid in sorted(df_comp['future_idx'].unique()):
        _print_comparison(df_comp, fid)

    # Per-sigma (key analysis)
    fid_def = df_sigma['future_idx'].max()
    print(f'\n  Performance by sigma level  (horizon = {fid_def}):')
    key_sels = ['oracle_51', 'threshold_ens_010', 'capped_diag_51',
                 'equal_ensemble_51', 'fixed_rational']
    key_sels = [s for s in key_sels if s in df_sigma['selector'].unique()]
    sigmas   = sorted(df_sigma['noise'].unique())

    header = f"  {'Selector':<26}" + ''.join(
        f"  {'sigma='+str(s)[:8]:>12}" for s in sigmas)
    print(header)
    print('  ' + '-' * (28 + 14*len(sigmas)))
    for sel in key_sels:
        row_str = f"  {sel:<26}"
        for s in sigmas:
            v = df_sigma[(df_sigma['selector']==sel)
                         & (df_sigma['noise']==s)
                         & (df_sigma['future_idx']==fid_def)]
            val = float(v['mean_error'].values[0]) if len(v) else float('nan')
            row_str += f"  {val:>12.6f}"
        print(row_str)

    # Ablation
    print('\n  Ablation (positive = first selector is better):')
    print(f"  {'Comparison':<35} {'Horizon':>8} {'Improvement':>13}")
    print('  ' + '-' * 60)
    for _, row in df_abl.sort_values(['comparison','future_idx']).iterrows():
        print(f"  {row['comparison']:<35} {int(row['future_idx']):>8} "
              f"{row['mean_improvement']:>13.6f}")

    # Method reliability
    print('\n  10 most reliable (lowest mean perturb_IQR):')
    print(f"  {'Method':<24} {'Mean IQR':>10} {'Mean err':>10}  Dangerous?")
    print('  ' + '-' * 60)
    for _, row in wgt_agg.head(10).iterrows():
        d = 'YES' if row['is_dangerous'] else ''
        print(f"  {row['method']:<24} {row['mean_piqr']:>10.4f} "
              f"{row['mean_err']:>10.4f}  {d}")

    print('\n  7 least reliable (highest mean perturb_IQR):')
    print(f"  {'Method':<24} {'Mean IQR':>10} {'Mean err':>10}  Dangerous?")
    print('  ' + '-' * 60)
    for _, row in wgt_agg.tail(7).iloc[::-1].iterrows():
        d = 'YES' if row['is_dangerous'] else ''
        iqr = (f"{row['mean_piqr']:>10.4f}" if row['mean_piqr']==row['mean_piqr']
               else '       NaN')
        print(f"  {row['method']:<24} {iqr} {row['mean_err']:>10.4f}  {d}")

    print(f'\n  All files saved to: {out_dir}/')
    print('=' * 72)
    print('\n  NEXT STEP: paste console + ensemble.csv + ablation.csv + by_sigma.csv')

if __name__ == '__main__':
    main()
