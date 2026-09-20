"""
run_phase5a.py  (v4, redesign v2)
=================================
Phase 5A entry point — Full-Pool Ensemble with Ablation.

    python scripts/run_phase5a.py --quick     config.PHASE5A["quick"]
    python scripts/run_phase5a.py --full      config.PHASE5A["full"]
    python scripts/run_phase5a.py --full --jobs 4     process pool over
                                                      (obs_idx x noise) blocks

--jobs N (default cpu_count() - 1) evaluates the (obs_idx x noise) blocks of
the grid in N worker processes; every block writes a shard and the shards are
concatenated in serial order, so phase5a_raw.csv is byte-identical for any N.
The first completed block prints its timing and a projection of the
evaluation wall time at the chosen job count.  --keep-shards retains
results/phase5a/shards/ after the concatenation.

Requires the dangerous-method artifact written by scripts/derive_dangerous.py
(the phase stops with instructions if it is missing).
"""

import os, sys, argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import src.config as CFG_MOD
from src.dangerous import load_dangerous
from src.pipeline import resolve_regimes
from phases.phase5a import run_all, SELECTORS, EPS, POOL, EVAL_METHODS, default_jobs


def n_evaluations(cfg: dict) -> dict:
    regimes = resolve_regimes(cfg['core_regimes'], cfg['holdout_regimes'], True)
    cells = (len(cfg['obs_idx_list']) * len(cfg['noise_list']) * cfg['n_seeds']
             * len(regimes) * len(cfg['gap_fractions']))
    central = cells * len(EVAL_METHODS)
    perturb = cells * len(POOL) * cfg['perturb_trials']
    return {'central': central, 'diagnostic_calls': perturb, 'total_calls': central + perturb}


def parse_args():
    p = argparse.ArgumentParser(description='Phase 5A — ensemble ablation (v2)')
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--quick', action='store_true')
    g.add_argument('--full',  action='store_true')
    p.add_argument('--out-dir', default=os.path.join('results', 'phase5a'))
    p.add_argument('--jobs', type=int, default=default_jobs(),
                   help='worker processes over (obs_idx x noise) blocks '
                        f'(default cpu_count() - 1 = {default_jobs()}; 1 = serial)')
    p.add_argument('--keep-shards', action='store_true',
                   help='keep the per-block shards under <out-dir>/shards/')
    return p.parse_args()


def _print_comparison(df_comp, g, n_show=14):
    sub = (df_comp[df_comp['target_g'] == g].sort_values('mean_error').head(n_show))
    print(f'\n  Global mean error  (g = {g:g}, core, capped excluded, top {n_show}):')
    print(f"  {'Selector':<26} {'Mean err':>10} {'Median':>10} {'Skill':>8}")
    print('  ' + '-' * 60)
    for _, row in sub.iterrows():
        marker = ''
        if row['selector'] == 'threshold_ens_010':
            marker = '  <-- Phase 5A best?'
        elif row['selector'] == 'oracle_51':
            marker = '  <-- upper bound'
        elif row['selector'] == 'fixed_rational':
            marker = '  <-- fixed baseline'
        elif row['selector'] == 'constant_oracle':
            marker = '  <-- oracle comparator (reference)'
        print(f"  {row['selector']:<26} {row['mean_error']:>10.6f} "
              f"{row['median_error']:>10.6f} {row['med_skill']:>8.3f}{marker}")


def main():
    args    = parse_args()
    mode    = 'quick' if args.quick else 'full'
    cfg     = dict(CFG_MOD.PHASE5A[mode])
    out_dir = args.out_dir
    counts  = n_evaluations(cfg)
    regimes = resolve_regimes(cfg['core_regimes'], cfg['holdout_regimes'], True)

    try:
        dangerous = load_dangerous()
    except FileNotFoundError as exc:
        print(f'ERROR: {exc}')
        sys.exit(1)

    print('=' * 72)
    print(f'  PHASE 5A — Full-Pool Ensemble  [{mode.upper()}, redesign v2]')
    print('=' * 72)
    print(f'  Pool        : {len(POOL)} accelerators  ({len(dangerous)} dangerous per artifact)')
    print(f'  Reference   : {len(EVAL_METHODS) - len(POOL)} trivial comparators (oracle labelled)')
    print(f'  obs_idx     : {cfg["obs_idx_list"]}')
    print(f'  noise       : {cfg["noise_list"]}')
    print(f'  gap strata  : {cfg["gap_fractions"]}  (headline g = {CFG_MOD.HEADLINE_G})')
    print(f'  seeds       : {cfg["n_seeds"]}')
    print(f'  regimes     : {len(regimes)} (core + held-out)')
    print(f'  perturbs    : {cfg["perturb_trials"]}')
    print(f'  EPS         : {EPS}')
    print(f'  evals       : {counts["central"]:,} central + {counts["diagnostic_calls"]:,} perturbation calls')
    n_blocks = len(cfg['obs_idx_list']) * len(cfg['noise_list'])
    print(f'  jobs        : {args.jobs}  over {n_blocks} (obs_idx x noise) blocks'
          + ('  (serial)' if args.jobs <= 1 else ''))
    print(f'  output dir  : {out_dir}')
    print('=' * 72 + '\n')

    results = run_all(
        obs_idx_list    = cfg['obs_idx_list'],
        noise_list      = cfg['noise_list'],
        gap_fractions   = cfg['gap_fractions'],
        n_seeds         = cfg['n_seeds'],
        window_len      = cfg['window_len'],
        perturb_trials  = cfg['perturb_trials'],
        perturb_scale   = cfg['perturb_scale'],
        out_dir         = out_dir,
        core_regimes    = cfg['core_regimes'],
        holdout_regimes = cfg['holdout_regimes'],
        default_g       = CFG_MOD.HEADLINE_G,
        jobs            = args.jobs,
        keep_shards     = args.keep_shards,
    )

    df_comp  = results['comparison']
    df_sigma = results['by_sigma']
    df_abl   = results['ablation']
    wgt_agg  = results['weights']
    g_head   = results['default_g']

    print('\n' + '=' * 72)
    print('  PHASE 5A SUMMARY')
    print('=' * 72)

    for g in sorted(df_comp['target_g'].unique(), reverse=True):
        _print_comparison(df_comp, g)

    if not df_sigma.empty:
        print(f'\n  Performance by sigma level  (g = {g_head:g}):')
        key_sels = ['oracle_51', 'threshold_ens_010', 'capped_diag_51',
                    'equal_ensemble_51', 'fixed_rational', 'constant_assumed']
        key_sels = [s for s in key_sels if s in df_sigma['selector'].unique()]
        sigmas   = sorted(df_sigma['noise'].unique())
        header = f"  {'Selector':<26}" + ''.join(f"  {'sigma='+str(s)[:8]:>12}" for s in sigmas)
        print(header)
        print('  ' + '-' * (28 + 14*len(sigmas)))
        for sel in key_sels:
            row_str = f"  {sel:<26}"
            for s in sigmas:
                v = df_sigma[(df_sigma['selector']==sel) & (df_sigma['noise']==s)
                             & (df_sigma['target_g']==g_head)]
                val = float(v['mean_error'].values[0]) if len(v) else float('nan')
                row_str += f"  {val:>12.6f}"
            print(row_str)

    print('\n  Ablation (positive = first selector is better):')
    print(f"  {'Comparison':<32} {'g':>6} {'Improvement':>13}")
    print('  ' + '-' * 56)
    for _, row in df_abl.sort_values(['comparison','target_g']).iterrows():
        print(f"  {row['comparison']:<32} {row['target_g']:>6g} {row['mean_improvement']:>13.6f}")

    if not wgt_agg.empty:
        print(f'\n  10 most reliable (lowest mean perturb_IQR, g = {g_head:g}):')
        print(f"  {'Method':<24} {'Mean IQR':>10} {'Mean err':>10} {'Skill':>8}  Dangerous?")
        print('  ' + '-' * 68)
        for _, row in wgt_agg.head(10).iterrows():
            d = 'YES' if row['is_dangerous'] else ''
            print(f"  {row['method']:<24} {row['mean_piqr']:>10.4f} "
                  f"{row['mean_err']:>10.4f} {row['med_skill']:>8.3f}  {d}")

    print(f'\n  All files saved to: {out_dir}/')
    print('=' * 72)


if __name__ == '__main__':
    main()
