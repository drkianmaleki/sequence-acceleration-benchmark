"""
run_phase3.py
=============
Phase 3 entry point — Adaptive Selector Pipeline (redesign v2, descriptive).

    python scripts/run_phase3.py [--quick | --full]

Phase 3 loads Phase 2 data and performs analysis only; --quick / --full are
accepted for a uniform pipeline interface and only label the run.

Prerequisites
-------------
    results/phase2/phase2_sweep_aggregated.csv   (redesign v2, keyed by target_g)
    results/phase2/phase2_features.csv
    results/phase2/phase2_records.csv            (git-ignored; written by Phase 2)

Output directory: results/phase3/
"""

import os, sys, argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import src.config as CFG_MOD
from phases.phase3 import run_phase3, CASCADES


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

    for fname in ['phase2_sweep_aggregated.csv', 'phase2_features.csv', 'phase2_records.csv']:
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
    print(f'  Reporting    : per-record descriptive panels of the chosen records '
          f'(conditional on validity; validity rate alongside; no fallback)')
    print('=' * 72 + '\n')

    results = run_phase3(args.phase2_dir, args.out_dir, CFG_MOD.HEADLINE_G)

    df_comp = results['comparison']
    df_cv   = results['cv']
    df_clf  = results['classifier']

    print('\n' + '=' * 72)
    print('  PHASE 3 SUMMARY')
    print('=' * 72)

    for g in sorted(df_comp['target_g'].unique(), reverse=True):
        sub = df_comp[df_comp['target_g'] == g]
        n_excl = int(sub['n_capped_excluded'].max()) if len(sub) else 0
        print(f'\n  Selectors at g = {g:g}  ({n_excl} capped cells excluded; the error columns are '
              f'conditional on validity, read with the validity rate):')
        print(f"  {'Selector':<18} {'valid':>7} {'n_valid/n_total':>16} {'cat':>6} {'med err':>10} "
              f"{'q25':>9} {'q75':>9} {'p90':>9} {'win/last':>9}")
        print('  ' + '─' * 100)
        for _, row in sub.iterrows():
            marker = ('  <-- Phase 3' if row['selector'] == 'enhanced_cascade' else
                      '  <-- Phase 2' if row['selector'] == 'phase2_cascade' else
                      '  <-- hindsight reference' if row['selector'] == 'oracle' else '')
            print(f"  {row['selector']:<18} {row['valid_rate']:>7.4f} "
                  f"{f'{int(row['n_valid'])}/{int(row['n_total'])}':>16} {row['cat_rate']:>6.3f} "
                  f"{row['med_error']:>10.5f} {row['q25_error']:>9.5f} {row['q75_error']:>9.5f} "
                  f"{row['p90_error']:>9.5f} {row['win_rate_vs_last']:>9.3f}{marker}")
    print(f"\n  Validity warning fired: {'YES' if results['validity_warned'] else 'no'} "
          f"(spread of valid_rate across selectors above 0.001 at some horizon)")

    print('\n  Leave-one-regime-out (the cascades; per held-out regime, median over regimes of the panel):')
    print(f"  {'Selector':<18} {'g':>5} {'med of med err':>15} {'min':>10} {'max':>10} {'med valid':>10}")
    print('  ' + '─' * 74)
    for sel in CASCADES:
        for g in sorted(df_cv['target_g'].unique(), reverse=True):
            sub = df_cv[(df_cv['selector'] == sel) & (df_cv['target_g'] == g)]
            e = sub['med_error'].dropna()
            if e.empty:
                continue
            print(f"  {sel:<18} {g:>5g} {e.median():>15.5f} {e.min():>10.5f} {e.max():>10.5f} "
                  f"{sub['valid_rate'].median():>10.4f}")

    if not df_clf.empty:
        ov = df_clf[df_clf['regime'] == '__OVERALL__']
        if 'protocol' in df_clf.columns:
            print('\n  Regime classifier overall accuracy by protocol (never combined):')
            for _, r in ov.iterrows():
                print(f"    {r['protocol']:<24} {float(r['accuracy']):.3f}   ({r['split_unit']})")
        elif not ov.empty:
            print(f'\n  Regime classifier overall accuracy: {float(ov["accuracy"].values[0]):.3f}')
        worst = (df_clf[df_clf['regime'] != '__OVERALL__'].sort_values('accuracy').head(3))
        print('  Hardest regimes to classify:')
        for _, r in worst.iterrows():
            proto = f"[{r['protocol']}] " if 'protocol' in df_clf.columns else ''
            print(f"    {proto}{r['regime']:<22}  acc={r['accuracy']:.3f}  confused with {r['top_confusion']}")

    print(f'\n  Files saved to: {args.out_dir}/')
    print('=' * 72)


if __name__ == '__main__':
    main()
