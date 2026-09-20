"""
run_phase5b.py
==============
Phase 5B entry point — Sensitivity Analysis (redesign v2).

    python scripts/run_phase5b.py --quick     config.PHASE5B["quick"]
    python scripts/run_phase5b.py --full      config.PHASE5B["full"]

Sweeps run at g in config.PHASE5B_GAP_FRACTIONS = [0.5, 0.1].
Sweep 1 = assumed-asymptote mode {zero, half, oracle, double, winmin};
sweeps 2 (window length) and 3 (CAT_MULT) keep their structure.
Requires the dangerous-method artifact (scripts/derive_dangerous.py).

Output directory: results/phase5b/
"""

import os, sys, argparse, math

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import src.config as CFG_MOD
from src.dangerous import load_dangerous
from src.pipeline import resolve_regimes
from phases.phase5b import run_all, ALL_METHODS


def n_evaluations(cfg: dict) -> dict:
    regimes = resolve_regimes(cfg['core_regimes'], cfg['holdout_regimes'], True)
    base = len(cfg['noise_list']) * cfg['n_seeds'] * len(regimes) * len(cfg['gap_fractions'])
    n1 = len(cfg['assumed_modes']) * base * 2
    n2 = len(cfg['window_lengths']) * base * 2
    n3 = len(cfg['catmult_values']) * base * len(ALL_METHODS)
    return {'sweep1': n1, 'sweep2': n2, 'sweep3': n3, 'total': n1 + n2 + n3}


def parse_args():
    p = argparse.ArgumentParser(description='Phase 5B — Sensitivity Analysis (v2)')
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--quick', action='store_true')
    g.add_argument('--full',  action='store_true')
    p.add_argument('--out-dir', default=os.path.join('results', 'phase5b'))
    return p.parse_args()


def main():
    args    = parse_args()
    mode    = 'quick' if args.quick else 'full'
    cfg     = dict(CFG_MOD.PHASE5B[mode])
    out_dir = args.out_dir
    counts  = n_evaluations(cfg)
    regimes = resolve_regimes(cfg['core_regimes'], cfg['holdout_regimes'], True)

    try:
        dangerous = load_dangerous()
    except FileNotFoundError as exc:
        print(f'ERROR: {exc}')
        sys.exit(1)

    print('=' * 72)
    print(f'  PHASE 5B — Sensitivity Analysis  [{mode.upper()}, redesign v2]')
    print('=' * 72)
    print(f'  Sweep 1 (L_hat mode): {cfg["assumed_modes"]}')
    print(f'  Sweep 2 (window):     {cfg["window_lengths"]}')
    print(f'  Sweep 3 (CAT_MULT):   {cfg["catmult_values"]}  ({len(ALL_METHODS)} methods)')
    print(f'  gap strata: {cfg["gap_fractions"]}  (headline g = {CFG_MOD.HEADLINE_G})')
    print(f'  noise:      {cfg["noise_list"]}')
    print(f'  seeds:      {cfg["n_seeds"]}')
    print(f'  regimes:    {len(regimes)} (core + held-out)')
    print(f'  dangerous:  {len(dangerous)} per artifact')
    print(f'  Sweep 1 evals: {counts["sweep1"]:,}')
    print(f'  Sweep 2 evals: {counts["sweep2"]:,}')
    print(f'  Sweep 3 evals: {counts["sweep3"]:,}')
    print(f'  Output dir: {out_dir}')
    print('=' * 72 + '\n')

    results = run_all(
        assumed_modes      = cfg['assumed_modes'],
        window_lengths     = cfg['window_lengths'],
        catmult_values     = cfg['catmult_values'],
        obs_idx            = cfg['obs_idx'],
        window_len_default = cfg['window_len_default'],
        noise_list         = cfg['noise_list'],
        gap_fractions      = cfg['gap_fractions'],
        n_seeds            = cfg['n_seeds'],
        out_dir            = out_dir,
        core_regimes       = cfg['core_regimes'],
        holdout_regimes    = cfg['holdout_regimes'],
        default_g          = CFG_MOD.HEADLINE_G,
    )

    print('\n' + '=' * 72)
    print('  PHASE 5B SUMMARY')
    print('=' * 72)

    df1g = results['sweep1_global']
    df2g = results['sweep2_global']
    df3c = results['sweep3_concordance']
    df3ch = results['sweep3_champions']

    gs = sorted(df1g['target_g'].unique()) if len(df1g) else []
    g_head = CFG_MOD.HEADLINE_G if CFG_MOD.HEADLINE_G in gs else (gs[-1] if gs else float('nan'))
    sig0 = df1g['noise'].min() if len(df1g) else float('nan')

    sub1 = df1g[(df1g['target_g'] == g_head) & (df1g['noise'] == sig0)
                & (df1g['regime_set'] == 'core')]
    print(f'\n  SWEEP 1 — Cascade metrics vs assumed-asymptote mode '
          f'(g={g_head:g}, sigma={sig0}, core, capped excluded):')
    print(f"  {'mode':>8} {'Precision':>10} {'Recall':>8} {'Gain':>10}  Robust?")
    print('  ' + '-' * 55)
    order = {m: i for i, m in enumerate(CFG_MOD.ASSUMED_L_MODES)}
    for _, row in sub1.assign(_o=sub1['assumed_mode'].map(order)).sort_values('_o').iterrows():
        prec = row['precision']
        robust = 'YES' if (math.isfinite(prec) and prec >= 0.70) else 'NO'
        tag = ' (oracle)' if row['assumed_mode'] == 'oracle' else ''
        print(f"  {row['assumed_mode']:>8} {prec:>10.3f} "
              f"{row['recall']:>8.3f} {row['mean_gain']:>10.4f}  {robust}{tag}")

    sub2 = df2g[(df2g['target_g'] == g_head) & (df2g['noise'] == sig0)
                & (df2g['regime_set'] == 'core')]
    print(f'\n  SWEEP 2 — Cascade metrics vs window length (g={g_head:g}, sigma={sig0}):')
    print(f"  {'Window':>8} {'Precision':>10} {'Recall':>8} {'Gain':>10}  Robust?")
    print('  ' + '-' * 55)
    for _, row in sub2.sort_values('window_len').iterrows():
        prec = row['precision']
        robust = 'YES' if (math.isfinite(prec) and prec >= 0.70) else 'NO'
        marker = ' <-- Phase 2 default' if row['window_len'] == 60 else ''
        print(f"  {int(row['window_len']):>8} {prec:>10.3f} "
              f"{row['recall']:>8.3f} {row['mean_gain']:>10.4f}  {robust}{marker}")

    sub3 = df3c[df3c['target_g'] == g_head] if len(df3c) else df3c
    print(f'\n  SWEEP 3 — CAT_MULT concordance (g={g_head:g}, core):')
    print(f"  {'Comparison':<25} {'Kendall tau':>12} {'Champ agree':>13}")
    print('  ' + '-' * 55)
    for _, row in sub3.iterrows():
        print(f"  CAT={row['cat_mult_a']:.0f} vs CAT={row['cat_mult_b']:.0f}"
              f"{'':>10} {row['kendall_tau']:>12.4f} {row['champion_agreement']:>13.4f}")

    df3g = results['sweep3_global']
    if len(df3g):
        cm0 = float(sorted(df3g['cat_mult'].unique())[0])
        sub3g = df3g[(df3g['target_g'] == g_head) & (df3g['cat_mult'] == cm0)]
        top = sub3g[sub3g['rank_eligible'] == 1].sort_values('rank').head(5)
        print(f'\n  SWEEP 3 — top-5 global stability ranking (g={g_head:g}, CAT_MULT={cm0:g}, '
              f'core, capped excluded, rank floor valid_rate >= {CFG_MOD.RANK_MIN_VALID}):')
        for _, row in top.iterrows():
            print(f"    {int(row['rank']):>3}  {row['method']:<22} S={row['stability']:.3f}  "
                  f"valid={row['valid_rate']:.3f}")
        unr = sub3g[(sub3g['rank_eligible'] == 0)
                    & (sub3g['valid_rate'] < CFG_MOD.RANK_MIN_VALID)]
        print(f'  SWEEP 3 — unranked, below the validity floor (g={g_head:g}, '
              f'CAT_MULT={cm0:g}; {len(unr)} methods):')
        if len(unr):
            for _, row in unr.sort_values('valid_rate', ascending=False).iterrows():
                print(f"       -  {row['method']:<22} S={row['stability']:.3f}  "
                      f"valid={row['valid_rate']:.3f}")
        else:
            print('       (none)')

    if len(df3ch):
        sub3ch = df3ch[(df3ch['target_g'] == g_head) & (df3ch['is_holdout'] == 0)]
        pivot  = sub3ch.pivot_table(index='regime', columns='cat_mult',
                                    values='champion', aggfunc='first')
        pivot.columns = [f'CAT={c:.0f}' for c in pivot.columns]
        changed = pivot.apply(lambda r: len(set(r.dropna())) > 1, axis=1)
        print(f'\n  Core regime champions: {int(changed.sum())}/{len(pivot)} regimes '
              f'change champion across CAT_MULT values (g={g_head:g}).')
        for regime in pivot[changed].index:
            vals = '  |  '.join(f'{c}={pivot.loc[regime,c]}' for c in pivot.columns)
            print(f'    {regime:<22}: {vals}')

    print(f'\n  All files saved to: {out_dir}/')
    print('=' * 72)


if __name__ == '__main__':
    main()
