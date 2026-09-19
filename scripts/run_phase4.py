"""
run_phase4.py
=============
Phase 4 entry point — Stability Diagnostic Stress Testing (redesign v2).

    python scripts/run_phase4.py --quick     config.PHASE4["quick"]
    python scripts/run_phase4.py --full      config.PHASE4["full"]

Evaluation points are the three gap strata per (regime, obs_idx); core and
held-out regimes are both evaluated, pooled analyses use the core regimes
with capped cells excluded.

Output directory: results/phase4/
"""

import os, sys, argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import src.config as CFG_MOD
from src.pipeline import resolve_regimes
from phases.phase4 import run_all, PHASE4_METHODS, EVAL_METHODS


def n_evaluations(cfg: dict) -> dict:
    """Central evaluations and diagnostic method calls for a Phase-4 config."""
    regimes = resolve_regimes(cfg['core_regimes'], cfg['holdout_regimes'], True)
    cells = (len(cfg['obs_idx_list']) * len(cfg['noise_list']) * cfg['n_seeds']
             * len(regimes) * len(cfg['gap_fractions']))
    central = cells * len(EVAL_METHODS)
    diag = cells * len(PHASE4_METHODS) * (len(cfg['shifts']) + cfg['perturb_trials'])
    return {'central': central, 'diagnostic_calls': diag, 'total_calls': central + diag}


def parse_args():
    p = argparse.ArgumentParser(description='Phase 4 — Stability diagnostics (v2)')
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--quick', action='store_true')
    g.add_argument('--full',  action='store_true')
    p.add_argument('--out-dir', default=os.path.join('results', 'phase4'))
    return p.parse_args()


def main():
    args    = parse_args()
    mode    = 'quick' if args.quick else 'full'
    cfg     = dict(CFG_MOD.PHASE4[mode])
    out_dir = args.out_dir
    feat_p  = os.path.join('results', 'phase2', 'phase2_features.csv')
    counts  = n_evaluations(cfg)
    regimes = resolve_regimes(cfg['core_regimes'], cfg['holdout_regimes'], True)

    print('=' * 72)
    print(f'  PHASE 4 — Stability Diagnostic Stress Testing  [{mode.upper()}, redesign v2]')
    print('=' * 72)
    print(f'  obs_idx    : {cfg["obs_idx_list"]}')
    print(f'  noise      : {cfg["noise_list"]}')
    print(f'  gap strata : {cfg["gap_fractions"]}  (headline g = {CFG_MOD.HEADLINE_G})')
    print(f'  seeds      : {cfg["n_seeds"]}')
    print(f'  regimes    : {len(regimes)} (core + held-out; pooled stats core only)')
    print(f'  methods    : {len(PHASE4_METHODS)} diagnostic pool + '
          f'{len(EVAL_METHODS) - len(PHASE4_METHODS)} trivial references')
    print(f'  shifts     : {cfg["shifts"]}')
    print(f'  perturbs   : {cfg["perturb_trials"]}')
    print(f'  evals      : {counts["central"]:,} central + {counts["diagnostic_calls"]:,} diagnostic calls')
    print(f'  output dir : {out_dir}')
    print('=' * 72 + '\n')

    results = run_all(
        obs_idx_list     = cfg['obs_idx_list'],
        noise_list       = cfg['noise_list'],
        gap_fractions    = cfg['gap_fractions'],
        n_seeds          = cfg['n_seeds'],
        window_len       = cfg['window_len'],
        shifts           = cfg['shifts'],
        perturb_trials   = cfg['perturb_trials'],
        perturb_scale    = cfg['perturb_scale'],
        out_dir          = out_dir,
        phase2_feat_path = feat_p,
        core_regimes     = cfg['core_regimes'],
        holdout_regimes  = cfg['holdout_regimes'],
        default_g        = CFG_MOD.HEADLINE_G,
    )

    df_corr  = results['correlations']
    df_rules = results['rules']
    df_rel   = results['reliability']
    df_ens   = results['ensemble']
    g_head   = results['default_g']

    print('\n' + '=' * 72)
    print(f'  PHASE 4 SUMMARY  (core regimes, capped excluded, headline g = {g_head:g})')
    print('=' * 72)

    print('\n  Global diagnostic correlations (vs |error|):')
    print(f"  {'Diagnostic':<16} {'r':>8} {'p':>10}  Status")
    print('  ' + '─' * 55)
    for _, row in df_corr[df_corr['method'] == 'ALL'].iterrows():
        r = row['spearman_r']
        flag = '*** candidate' if abs(r) >= 0.30 else \
               '** moderate'   if abs(r) >= 0.20 else ''
        print(f"  {row['diagnostic']:<16} {r:>8.4f} {row['p_value']:>10.4f}  {flag}")

    print('\n  Best rejection rules (precision >= 0.30):')
    print(f"  {'Diagnostic':<14} {'Method':<20} {'Thresh':>8} {'Prec':>7} {'Recall':>7}")
    print('  ' + '─' * 62)
    if not df_rules.empty:
        good = (df_rules[df_rules['precision'] >= 0.30]
                .sort_values('precision', ascending=False)
                .drop_duplicates(['diagnostic', 'method']))
        for _, row in good.head(12).iterrows():
            print(f"  {row['diagnostic']:<14} {row['method']:<20} "
                  f"{row['threshold']:>8.3f} {row['precision']:>7.3f} {row['recall']:>7.3f}")

    df_filt = results['filter']
    if not df_filt.empty:
        print('\n  Cascade + perturb_IQR filter (mean error):')
        print(f"  {'Filter':<25} {'Mean error':>12}")
        print('  ' + '─' * 40)
        for _, row in df_filt.iterrows():
            print(f"  {row['filter']:<25} {row['mean_error']:>12.6f}")

    if not df_ens.empty:
        print('\n  Selector mean error and skill (vs best-of-four trivial reference):')
        print(f"  {'Selector':<22} {'Mean error':>12} {'Median':>10} {'Skill':>8}")
        print('  ' + '─' * 58)
        for _, row in df_ens.sort_values('mean_error').iterrows():
            print(f"  {row['selector']:<22} {row['mean_error']:>12.6f} "
                  f"{row['median_error']:>10.6f} {row['med_skill']:>8.3f}")

    print('\n  perturb_IQR vs error correlation (richardson_1, by obs_idx):')
    r1_piqr = (df_rel[(df_rel['method'] == 'richardson_1')
                       & (df_rel['diagnostic'] == 'perturb_iqr')]
               .sort_values('obs_idx'))
    print(f"  {'obs_idx':>8} {'r':>8} {'p':>10}")
    print('  ' + '─' * 32)
    for _, row in r1_piqr.iterrows():
        print(f"  {int(row['obs_idx']):>8} {row['spearman_r']:>8.4f} {row['p_value']:>10.4f}")

    print(f'\n  All files saved to: {out_dir}/')
    print('=' * 72)


if __name__ == '__main__':
    main()
