"""
run_phase3.py
=============
Phase 3 entry point — Adaptive Selector Pipeline (redesign v2).

    python scripts/run_phase3.py [--quick | --full]

Phase 3 loads Phase 2 data and performs analysis only; --quick / --full are
accepted for a uniform pipeline interface and only label the run.

Prerequisites
-------------
    results/phase2/phase2_sweep_aggregated.csv   (redesign v2, keyed by target_g)
    results/phase2/phase2_features.csv

Output directory: results/phase3/
"""

import os, sys, argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import src.config as CFG_MOD
from phases.phase3 import run_phase3


def parse_args():
    p = argparse.ArgumentParser(description='Phase 3 — Adaptive selector pipeline (v2)')
    mode = p.add_mutually_exclusive_group(required=False)
    mode.add_argument('--quick', action='store_true')
    mode.add_argument('--full',  action='store_true')
    p.add_argument('--phase2-dir', default=os.path.join('results', 'phase2'))
    p.add_argument('--out-dir', default=os.path.join('results', 'phase3'))
    return p.parse_args()


def main():
    args = parse_args()
    mode = 'QUICK' if args.quick else 'FULL'

    for fname in ['phase2_sweep_aggregated.csv', 'phase2_features.csv']:
        path = os.path.join(args.phase2_dir, fname)
        if not os.path.exists(path):
            print(f'ERROR: {path} not found.')
            print('       Run  python scripts/run_phase2.py --full  first.')
            sys.exit(1)

    print('=' * 72)
    print(f'  PHASE 3 — Adaptive Selector Pipeline  [{mode}, redesign v2]')
    print('=' * 72)
    print(f'  Phase 2 data : {args.phase2_dir}')
    print(f'  Output dir   : {args.out_dir}')
    print(f'  Headline stratum for per-grid figures: g = {CFG_MOD.HEADLINE_G}')
    print(f'  Evaluations  : 0 (analysis of Phase 2 output; core regimes only)')
    print('=' * 72 + '\n')

    results = run_phase3(args.phase2_dir, args.out_dir, CFG_MOD.HEADLINE_G)

    df_comp = results['comparison']
    df_cv   = results['cv']
    df_clf  = results['classifier']

    print('\n' + '=' * 72)
    print('  PHASE 3 SUMMARY')
    print('=' * 72)

    for g in sorted(df_comp['target_g'].unique(), reverse=True):
        sub = (df_comp[df_comp['target_g'] == g]
               .sort_values('mean_stability', ascending=False))
        n_excl = int(sub['n_capped_excluded'].max()) if len(sub) else 0
        print(f'\n  Global mean stability  (g = {g:g}; {n_excl} capped cells excluded):')
        print(f"  {'Selector':<22} {'Stability':>10} {'n':>6}")
        print('  ' + '─' * 42)
        for _, row in sub.iterrows():
            marker = ' <-- Phase 3' if row['selector'] == 'enhanced_cascade' else \
                     ' <-- Phase 2' if row['selector'] == 'phase2_cascade'   else \
                     ' <-- oracle'  if row['selector'] == 'oracle'           else ''
            print(f"  {row['selector']:<22} {row['mean_stability']:>10.4f} {int(row['n']):>6}{marker}")

    print('\n  Leave-one-regime-out CV  (median stability across held-out regimes):')
    print(f"  {'Selector':<22} {'Median':>8} {'Min':>8} {'Max':>8}")
    print('  ' + '─' * 50)
    for sel in ['fixed_richardson', 'fixed_single_exp',
                'phase2_cascade', 'enhanced_cascade', 'oracle']:
        sub = df_cv[df_cv['selector'] == sel]['mean_stability'].dropna()
        if sub.empty:
            continue
        print(f"  {sel:<22} {sub.median():>8.4f} {sub.min():>8.4f} {sub.max():>8.4f}")

    if not df_clf.empty:
        ov = df_clf[df_clf['regime'] == '__OVERALL__']
        if not ov.empty:
            print(f'\n  Regime classifier overall accuracy: '
                  f'{float(ov["accuracy"].values[0]):.3f}')
        worst = (df_clf[df_clf['regime'] != '__OVERALL__']
                 .sort_values('accuracy').head(3))
        print('  Hardest regimes to classify:')
        for _, r in worst.iterrows():
            print(f"    {r['regime']:<22}  acc={r['accuracy']:.3f}  "
                  f"confused with {r['top_confusion']}")

    print(f'\n  Files saved to: {args.out_dir}/')
    print('=' * 72)


if __name__ == '__main__':
    main()
