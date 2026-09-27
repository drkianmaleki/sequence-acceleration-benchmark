"""
run_phase5b.py
==============
Phase 5B entry point — Sensitivity Analysis (redesign v2).

    python scripts/run_phase5b.py --quick     config.PHASE5B["quick"]
    python scripts/run_phase5b.py --full      config.PHASE5B["full"]

Sweeps run at g in config.PHASE5B_GAP_FRACTIONS = [0.5, 0.1].
Sweep 1 = assumed-asymptote mode (config.ASSUMED_L_MODES); sweep 2 = window
length.  The former CAT_MULT sweep was removed with the retired composite
score S (its outputs remain at commit 842ddb9).  Requires the excluded-method
artifact (scripts/derive_dangerous.py) as the pipeline-ordering guard.

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
from phases.phase5b import run_all, SWEEP1_METHODS, LHAT_CONSUMERS


def n_evaluations(cfg: dict) -> dict:
    regimes = resolve_regimes(cfg['core_regimes'], cfg['holdout_regimes'], True)
    base = len(cfg['noise_list']) * cfg['n_seeds'] * len(regimes) * len(cfg['gap_fractions'])
    n1 = len(cfg['assumed_modes']) * base * 2                          # 1a cascade (2 methods)
    n1b = len(cfg['assumed_modes']) * base * (len(SWEEP1_METHODS) + 3)  # 1b consumers + 3 trivial refs
    n2 = len(cfg['window_lengths']) * base * 2
    return {'sweep1': n1, 'sweep1b': n1b, 'sweep2': n2, 'total': n1 + n1b + n2}


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
    print(f'  gap strata: {cfg["gap_fractions"]}  (headline g = {CFG_MOD.HEADLINE_G})')
    print(f'  noise:      {cfg["noise_list"]}')
    print(f'  seeds:      {cfg["n_seeds"]}')
    print(f'  regimes:    {len(regimes)} (core + held-out)')
    print(f'  excluded:   {len(dangerous)} per artifact (ordering guard; not consumed by this phase)')
    print(f'  Sweep 1 evals: {counts["sweep1"]:,} (1a cascade, clamped features) + '
          f'{counts["sweep1b"]:,} (1b: {len(SWEEP1_METHODS)} L_hat consumers + 3 references)')
    print(f'  Sweep 2 evals: {counts["sweep2"]:,}')
    print(f'  Output dir: {out_dir}')
    print('=' * 72 + '\n')

    results = run_all(
        assumed_modes      = cfg['assumed_modes'],
        window_lengths     = cfg['window_lengths'],
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

    gs = sorted(df1g['target_g'].unique()) if len(df1g) else []
    g_head = CFG_MOD.HEADLINE_G if CFG_MOD.HEADLINE_G in gs else (gs[-1] if gs else float('nan'))
    sig0 = df1g['noise'].min() if len(df1g) else float('nan')

    def _rule_table(sub, key, header):
        print(f"  {header:>8} {'fire':>6} {'lower rec':>10} {'lower cell':>11} {'med Δrel':>9} "
              f"{'r1 V':>6} {'alt V':>6} {'r1 med':>9} {'alt med':>9} {'windows':>8}")
        print('  ' + '-' * 96)
        for _, row in sub.iterrows():
            rel = row['median_rel_change']
            tag = ' (oracle)' if (key == 'assumed_mode' and row[key] == 'oracle') else \
                  (' <-- Phase 2 default' if (key == 'window_len' and row[key] == 60) else '')
            print(f"  {str(row[key]) if key == 'assumed_mode' else int(row[key]):>8} {row['fire_rate']:>6.3f} "
                  f"{row['lower_error_frac_records']:>10.3f} {row['lower_error_frac_cells']:>11.3f} "
                  f"{(f'{rel:+.3f}' if math.isfinite(rel) else 'n/a'):>9} "
                  f"{row['r1_valid_rate']:>6.3f} {row['alt_valid_rate']:>6.3f} "
                  f"{row['r1_med_error']:>9.5f} {row['alt_med_error']:>9.5f} "
                  f"{int(row['n_cells_total']):>8}{tag}")

    sub1 = df1g[(df1g['target_g'] == g_head) & (df1g['noise'] == sig0)
                & (df1g['regime_set'] == 'core')]
    print(f'\n  SWEEP 1 — the cascade by what happened when it fired, vs assumed-asymptote mode '
          f'(g={g_head:g}, sigma={sig0}, core, capped excluded; one window = one cell; '
          f'r1 = richardson_1, alt = routed rational_fit; errors conditional on validity):')
    order = {m: i for i, m in enumerate(CFG_MOD.ASSUMED_L_MODES)}
    _rule_table(sub1.assign(_o=sub1['assumed_mode'].map(order)).sort_values('_o'), 'assumed_mode', 'mode')

    df1c = results['sweep1_consumers']
    subc = df1c[(df1c['target_g'] == g_head) & (df1c['regime_set'] == 'core')]
    print(f'\n  SWEEP 1b — L_hat consumers + constant_assumed vs mode '
          f'(g={g_head:g}, core, pooled over noise, capped excluded): median error / '
          f'median skill (hindsight best-of-four, strict) / win rate vs constant_assumed')
    modes = [m for m in CFG_MOD.ASSUMED_L_MODES if m in set(subc['assumed_mode'])]
    print(f"  {'method':<20}" + ''.join(f"  {m:>21}" for m in modes))
    print('  ' + '-' * (20 + 23 * len(modes)))
    for m in SWEEP1_METHODS:
        cells = []
        for mode in modes:
            r = subc[(subc['method'] == m) & (subc['assumed_mode'] == mode)]
            cells.append(f"{r['med_error'].iloc[0]:.4f}/{r['med_skill'].iloc[0]:.2f}/{r['win_rate_vs_assumed'].iloc[0]:.2f}"
                         if len(r) else '--')
        print(f"  {m:<20}" + ''.join(f"  {c:>21}" for c in cells))
    print(f"  (the 1a cascade rows above are labelled: {df1g['note'].iloc[0] if 'note' in df1g and len(df1g) else ''})")

    sub2 = df2g[(df2g['target_g'] == g_head) & (df2g['noise'] == sig0)
                & (df2g['regime_set'] == 'core')]
    print(f'\n  SWEEP 2 — the cascade by what happened when it fired, vs window length '
          f'(g={g_head:g}, sigma={sig0}, core, capped excluded):')
    _rule_table(sub2.sort_values('window_len'), 'window_len', 'window')

    print(f'\n  All files saved to: {out_dir}/')
    print('=' * 72)


if __name__ == '__main__':
    main()
