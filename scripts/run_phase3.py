"""
run_phase3.py
=============
Phase 3 entry point — Adaptive Selector Pipeline.

Usage (from inside sequence_accel/)
------------------------------------
    python run_phase3.py

No --quick / --full modes needed: Phase 3 loads Phase 2 data and
performs analysis only.  Runtime is 2-5 minutes.

Prerequisites
-------------
Phase 2 full run must have completed.  The following files must exist:
    results/phase2/phase2_sweep_aggregated.csv
    results/phase2/phase2_features.csv

Optional: scikit-learn for a decision-tree regime classifier.
If not installed, a scipy-based 1-NN fallback is used automatically.

Output directory: results/phase3/

Key output files
----------------
    phase3_selector_comparison.csv   Mean stability per selector per horizon
    phase3_regime_results.csv        Per-regime stability per selector
    phase3_cv_results.csv            Leave-one-regime-out CV results
    phase3_regime_classifier.csv     Regime classification accuracy
    figure_p3_01 ... figure_p3_05    Five publication figures

What to paste back
------------------
After running, paste:
  1. The console summary printed below.
  2. phase3_selector_comparison.csv  (small)
  3. phase3_cv_results.csv           (small)
  4. phase3_regime_classifier.csv    (small)
"""

import os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from phases.phase3 import run_phase3

PHASE2_DIR  = os.path.join('results', 'phase2')
PHASE3_DIR  = os.path.join('results', 'phase3')
DEFAULT_FID = 5000


def main():
    # Sanity check
    for fname in ['phase2_sweep_aggregated.csv', 'phase2_features.csv']:
        path = os.path.join(PHASE2_DIR, fname)
        if not os.path.exists(path):
            print(f'ERROR: {path} not found.')
            print('       Run  python run_phase2.py --full  first.')
            sys.exit(1)

    print('=' * 72)
    print('  PHASE 3 — Adaptive Selector Pipeline')
    print('=' * 72)
    print(f'  Phase 2 data : {PHASE2_DIR}')
    print(f'  Output dir   : {PHASE3_DIR}')
    print(f'  Default horizon for per-grid figures: {DEFAULT_FID}')
    print('=' * 72 + '\n')

    results = run_phase3(PHASE2_DIR, PHASE3_DIR, DEFAULT_FID)

    # ── Console summary ────────────────────────────────────────────────────────
    df_comp = results['comparison']
    df_cv   = results['cv']
    df_clf  = results['classifier']

    print('\n' + '=' * 72)
    print('  PHASE 3 SUMMARY')
    print('=' * 72)

    # Selector comparison table
    for fid in sorted(df_comp['future_idx'].unique()):
        sub = (df_comp[df_comp['future_idx'] == fid]
               .sort_values('mean_stability', ascending=False))
        print(f'\n  Global mean stability  (horizon = {fid}):')
        print(f"  {'Selector':<22} {'Stability':>10}")
        print('  ' + '─' * 35)
        for _, row in sub.iterrows():
            marker = ' <-- Phase 3' if row['selector'] == 'enhanced_cascade' else \
                     ' <-- Phase 2' if row['selector'] == 'phase2_cascade'   else \
                     ' <-- oracle'  if row['selector'] == 'oracle'           else ''
            print(f"  {row['selector']:<22} {row['mean_stability']:>10.4f}{marker}")

    # CV summary
    print('\n  Leave-one-regime-out CV  (median stability across held-out regimes):')
    print(f"  {'Selector':<22} {'Median':>8} {'Min':>8} {'Max':>8}")
    print('  ' + '─' * 50)
    for sel in ['fixed_richardson', 'fixed_single_exp',
                'phase2_cascade', 'enhanced_cascade', 'oracle']:
        sub = df_cv[df_cv['selector'] == sel]['mean_stability'].dropna()
        if sub.empty:
            continue
        print(f"  {sel:<22} {sub.median():>8.4f} {sub.min():>8.4f} {sub.max():>8.4f}")

    # Classifier summary
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

    print(f'\n  Files saved to: {PHASE3_DIR}/')
    print('=' * 72)
    print('\n  NEXT STEP:')
    print('  Paste the console output above plus:')
    print('    results/phase3/phase3_selector_comparison.csv')
    print('    results/phase3/phase3_cv_results.csv')
    print('    results/phase3/phase3_regime_classifier.csv')


if __name__ == '__main__':
    main()
