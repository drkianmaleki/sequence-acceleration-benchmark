"""
run_phase5b.py
==============
Phase 5B entry point — Sensitivity Analysis.

Usage (from inside sequence_accel/)
------------------------------------
    python run_phase5b.py --quick     ~15 min
    python run_phase5b.py --full      ~50 min

Quick mode
----------
    L_inf values   : [0.001, 0.010, 0.050]  (3 of 5)
    window lengths : [20, 60, 100]           (3 of 5)
    CAT_MULT       : [2, 10]                 (2 of 3)
    noise          : [0.0, 0.005]
    seeds          : 5
    horizons       : [5000]

Full mode
---------
    L_inf values   : [0.001, 0.005, 0.010, 0.020, 0.050]
    window lengths : [20, 40, 60, 80, 100]
    CAT_MULT       : [2, 5, 10]
    noise          : [0.0, 0.005, 0.020]
    seeds          : 20
    horizons       : [1000, 5000]

Output directory: results/phase5b/

Key output files
----------------
    phase5b_sweep1_global.csv    Cascade precision/recall vs assumed L_inf
    phase5b_sweep2_global.csv    Cascade precision/recall vs window length
    phase5b_sweep3_champions.csv Regime champions at each CAT_MULT
    phase5b_sweep3_concordance.csv  Ranking concordance between CAT_MULT
    figure_p5b_01_linf.png
    figure_p5b_02_window.png
    figure_p5b_03_catmult.png

What to paste back
------------------
After running, paste:
    1. Console summary
    2. results/phase5b/phase5b_sweep1_global.csv   (small)
    3. results/phase5b/phase5b_sweep2_global.csv   (small)
    4. results/phase5b/phase5b_sweep3_concordance.csv  (tiny)
"""

import os, sys, argparse, math
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from phases.phase5b import run_all
import src.config as CFG_MOD

def parse_args():
    p = argparse.ArgumentParser(
        description='Phase 5B — Sensitivity Analysis')
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--quick', action='store_true')
    g.add_argument('--full',  action='store_true')
    return p.parse_args()


QUICK = dict(
    l_inf_values       = [0.001, 0.010, 0.050],
    window_lengths     = [20, 60, 100],
    catmult_values     = [2.0, 10.0],
    obs_idx            = 90,
    window_len_default = 60,
    noise_list         = [0.0, 0.005],
    future_list        = [5000],
    n_seeds            = 5,
)

FULL = dict(
    l_inf_values       = [0.001, 0.005, 0.010, 0.020, 0.050],
    window_lengths     = [20, 40, 60, 80, 100],
    catmult_values     = [2.0, 5.0, 10.0],
    obs_idx            = 90,
    window_len_default = 60,
    noise_list         = [0.0, 0.005, 0.020],
    future_list        = [1000, 5000],
    n_seeds            = 20,
)


def _n_evals(cfg):
    from src.accelerators import METHOD_NAMES
    n1  = (len(cfg['l_inf_values'])   * len(cfg['noise_list'])
           * cfg['n_seeds'] * 18 * 2 * len(cfg['future_list']))
    n2  = (len(cfg['window_lengths']) * len(cfg['noise_list'])
           * cfg['n_seeds'] * 18 * 2 * len(cfg['future_list']))
    n3  = (len(cfg['catmult_values']) * len(cfg['noise_list'])
           * cfg['n_seeds'] * 18 * len(list(METHOD_NAMES))
           * len(cfg['future_list']))
    return n1, n2, n3


def main():
    args    = parse_args()
    cfg     = QUICK if args.quick else FULL
    mode    = 'QUICK' if args.quick else 'FULL'
    out_dir = os.path.join('results', 'phase5b')

    n1, n2, n3 = _n_evals(cfg)

    print('=' * 72)
    print(f'  PHASE 5B — Sensitivity Analysis  [{mode}]')
    print('=' * 72)
    print(f'  Sweep 1 (L_inf):    {cfg["l_inf_values"]}')
    print(f'  Sweep 2 (window):   {cfg["window_lengths"]}')
    print(f'  Sweep 3 (CAT_MULT): {cfg["catmult_values"]}')
    print(f'  noise:   {cfg["noise_list"]}')
    print(f'  seeds:   {cfg["n_seeds"]}')
    print(f'  Sweep 1 evals: ~{n1:,}')
    print(f'  Sweep 2 evals: ~{n2:,}')
    print(f'  Sweep 3 evals: ~{n3:,}')
    print(f'  Output dir: {out_dir}')
    print('=' * 72 + '\n')

    results = run_all(**cfg, out_dir=out_dir)

    # ── Console summary ────────────────────────────────────────────────────────
    print('\n' + '=' * 72)
    print('  PHASE 5B SUMMARY')
    print('=' * 72)

    df1g = results['sweep1_global']
    df2g = results['sweep2_global']
    df3c = results['sweep3_concordance']
    df3ch = results['sweep3_champions']

    # Sweep 1 — L_inf
    fid  = df1g['future_idx'].max()
    sig0 = df1g['noise'].min()
    sub1 = df1g[(df1g['future_idx'] == fid) & (df1g['noise'] == sig0)]
    print(f'\n  SWEEP 1 — Cascade metrics vs assumed L_inf '
          f'(horizon={fid}, sigma={sig0}):')
    print(f"  {'L_inf':>8} {'Precision':>10} {'Recall':>8} {'Gain':>10}  Robust?")
    print('  ' + '-' * 55)
    for _, row in sub1.sort_values('L_inf_assumed').iterrows():
        prec = row['precision']
        robust = 'YES' if (math.isfinite(prec) and prec >= 0.70) else 'NO'
        print(f"  {row['L_inf_assumed']:>8.3f} {prec:>10.3f} "
              f"{row['recall']:>8.3f} {row['mean_gain']:>10.4f}  {robust}")

    # Sweep 2 — window length
    sub2 = df2g[(df2g['future_idx'] == fid) & (df2g['noise'] == sig0)]
    print(f'\n  SWEEP 2 — Cascade metrics vs window length '
          f'(horizon={fid}, sigma={sig0}):')
    print(f"  {'Window':>8} {'Precision':>10} {'Recall':>8} {'Gain':>10}  Robust?")
    print('  ' + '-' * 55)
    for _, row in sub2.sort_values('window_len').iterrows():
        prec = row['precision']
        robust = 'YES' if (math.isfinite(prec) and prec >= 0.70) else 'NO'
        marker = ' <-- Phase 2 default' if row['window_len'] == 60 else ''
        print(f"  {int(row['window_len']):>8} {prec:>10.3f} "
              f"{row['recall']:>8.3f} {row['mean_gain']:>10.4f}  {robust}{marker}")

    # Sweep 3 — CAT_MULT concordance
    sub3 = df3c[df3c['future_idx'] == fid]
    print(f'\n  SWEEP 3 — CAT_MULT concordance (horizon={fid}):')
    print(f"  {'Comparison':<25} {'Kendall tau':>12} {'Champ agree':>13}")
    print('  ' + '-' * 55)
    for _, row in sub3.iterrows():
        print(f"  CAT={row['cat_mult_a']:.0f} vs CAT={row['cat_mult_b']:.0f}"
              f"{'':>10} {row['kendall_tau']:>12.4f} "
              f"{row['champion_agreement']:>13.4f}")

    # Which regime champions change across CAT_MULT?
    fid_def = df3ch['future_idx'].max()
    sub3ch  = df3ch[df3ch['future_idx'] == fid_def]
    pivot   = sub3ch.pivot_table(
        index='regime', columns='cat_mult', values='champion', aggfunc='first')
    pivot.columns = [f'CAT={c:.0f}' for c in pivot.columns]
    changed = pivot.apply(lambda r: len(set(r.dropna())) > 1, axis=1)

    print(f'\n  Regime champions: {int(changed.sum())}/{len(pivot)} regimes '
          f'change champion across CAT_MULT values.')
    if changed.any():
        print('  Changed regimes:')
        for regime in pivot[changed].index:
            vals = '  |  '.join(f'{c}={pivot.loc[regime,c]}' for c in pivot.columns)
            print(f'    {regime:<22}: {vals}')
    else:
        print('  ALL regime champions are robust to CAT_MULT variation.')

    print(f'\n  All files saved to: {out_dir}/')
    print('=' * 72)
    print('\n  NEXT STEP: paste console + sweep1_global + sweep2_global '
          '+ sweep3_concordance')


if __name__ == '__main__':
    main()
