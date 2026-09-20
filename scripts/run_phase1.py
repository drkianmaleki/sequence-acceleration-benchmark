"""
run_phase1.py
=============
Phase 1 entry point — main synthetic benchmark (redesign v2).

Usage
-----
    cd sequence-acceleration-benchmark/
    python scripts/run_phase1.py --quick     # config.PHASE1["quick"]: 2 seeds, 2+2 regimes
    python scripts/run_phase1.py --full      # config.PHASE1["full"]:  30 seeds, 18+6 regimes

    optional:  --assumed-mode {zero,half,oracle,double,winmin}   (default: config)
               --no-holdout                                       (core only)
               --out-dir DIR                                      (default results/phase1)

Output directory:  results/phase1/

Key output files
----------------
    phase1_records.csv          Per-seed records with (target_g, achieved_g,
                                n_f, capped, L_true, L_hat, skill, is_*).
    phase1_aggregated.csv       Per (method, regime, noise, g) stats.
    phase1_global.csv           Pooled over the core regimes, capped cells
                                excluded, sorted by med_error; rank column
                                excludes the oracle comparator and methods
                                below the validity floor (rank_eligible = 0).
    phase1_global_holdout.csv   The same over the held-out regimes.
    phase1_capped.csv           The capped block (achieved_g per cell).
    phase1_unranked.csv         The unranked block: methods below the rank
                                validity floor (valid_rate < RANK_MIN_VALID).
    phase1_regime_best.csv      best_by_skill (primary) per (regime, g).
    phase1_horizons.csv         n_f(regime, g) table with achieved-g flags.
    phase1_heatmap_g{g}.csv     Stability score matrix per stratum.
    figure_01 ... figure_06     Six figures (rankings exclude the oracle).
"""

import os
import sys
import argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)   # one level up: the repo root
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import src.config as CFG_MOD
from src.accelerators import METHOD_NAMES
from src.evaluation import run_phase1
from src.horizons import format_horizon_table
from src.pipeline import resolve_regimes
from src.plots      import make_all_figures


def n_evaluations(cfg: dict, include_holdout: bool = True) -> int:
    """Planned method evaluations for a Phase-1 config."""
    regimes = resolve_regimes(cfg['core_regimes'], cfg['holdout_regimes'], include_holdout)
    return (len(regimes) * cfg['n_seeds'] * len(cfg['noise_levels'])
            * len(cfg['gap_fractions']) * len(METHOD_NAMES))


# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Phase 1 — main synthetic benchmark (redesign v2)')
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--quick', action='store_true',
                      help='config.PHASE1["quick"] (2 seeds, 2 regimes per group)')
    mode.add_argument('--full',  action='store_true',
                      help='config.PHASE1["full"] (30 seeds, all regimes)')
    p.add_argument('--assumed-mode', default=None,
                   choices=list(CFG_MOD.ASSUMED_L_MODES),
                   help='assumed-asymptote mode handed to methods '
                        f'(default: config.ASSUMED_L_MODE = {CFG_MOD.ASSUMED_L_MODE})')
    p.add_argument('--no-holdout', action='store_true',
                   help='evaluate the core regimes only')
    p.add_argument('--out-dir', default=os.path.join('results', 'phase1'),
                   help='output directory (default: results/phase1)')
    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args    = parse_args()
    mode    = 'quick' if args.quick else 'full'
    cfg     = dict(CFG_MOD.PHASE1[mode])
    out_dir = args.out_dir
    assumed_mode = args.assumed_mode or CFG_MOD.ASSUMED_L_MODE
    include_holdout = not args.no_holdout
    regimes = resolve_regimes(cfg['core_regimes'], cfg['holdout_regimes'], include_holdout)
    n_core  = len(resolve_regimes(cfg['core_regimes'], None, include_holdout=False))

    print('=' * 72)
    print(f'  PHASE 1 — Main Synthetic Benchmark  [{mode.upper()} mode, redesign v2]')
    print('=' * 72)
    print(f'  Regimes       : {n_core} core'
          + (f' + {len(regimes) - n_core} held-out' if include_holdout else ''))
    print(f'  Methods       : {len(METHOD_NAMES)} (51 accelerators + 5 trivial comparators)')
    print(f'  Seeds         : {cfg["n_seeds"]}')
    print(f'  Noise         : {cfg["noise_levels"]}')
    print(f'  Gap strata g  : {cfg["gap_fractions"]}  (headline g = {CFG_MOD.HEADLINE_G}; '
          f'cap n = {CFG_MOD.HORIZON_N_CAP:,})')
    print(f'  obs_idx       : {cfg["obs_idx"]}')
    print(f'  window_len    : {cfg["window_len"]}')
    print(f'  L_true        : {CFG_MOD.ASYMPTOTE_MODE} '
          f'(log-uniform on {CFG_MOD.L_TRUE_RANGE}, hidden per regime x seed)')
    print(f'  L_hat mode    : {assumed_mode}'
          + ('   <-- ORACLE: methods see L_true' if assumed_mode == 'oracle' else ''))
    print(f'  Total evals   : {n_evaluations(cfg, include_holdout):,}')
    print(f'  Output dir    : {out_dir}')
    print('=' * 72 + '\n')

    results = run_phase1(
        n_seeds         = cfg['n_seeds'],
        noise_levels    = cfg['noise_levels'],
        gap_fractions   = cfg['gap_fractions'],
        obs_idx         = cfg['obs_idx'],
        window_len      = cfg['window_len'],
        out_dir         = out_dir,
        assumed_mode    = assumed_mode,
        include_holdout = include_holdout,
        core_regimes    = cfg['core_regimes'],
        holdout_regimes = cfg['holdout_regimes'],
        verbose         = True,
    )

    gs = cfg['gap_fractions']
    headline_g = CFG_MOD.HEADLINE_G if CFG_MOD.HEADLINE_G in gs else gs[-1]
    make_all_figures(
        results         = results,
        horizons        = gs,
        default_horizon = headline_g,
        out_dir         = out_dir,
    )

    # ── Horizon table ──────────────────────────────────────────────────────────
    print(f'\n  GAP-STRATIFIED HORIZONS  (n_obs = {cfg["obs_idx"]}; '
          f'seed 0 for seed-dependent shapes)')
    print(format_horizon_table(results['horizons']))
    print()

    # ── Top-20 global table (core regimes, sorted by median error) ─────────────
    # Rows below the validity floor are not in this block (they are unranked
    # and listed separately below); the oracle is shown unranked.
    df_g   = results['global']
    df_h   = df_g[df_g['target_g'] == headline_g]
    df_top = df_h[(df_h['rank_eligible'] == 1) | (df_h['is_oracle'] == 1)].head(20)

    print(f'\n  TOP-20 GLOBAL BY MEDIAN ERROR  (core regimes, g = {headline_g}, '
          f'capped cells excluded, L_hat mode = {assumed_mode}, '
          f'rank floor valid_rate >= {CFG_MOD.RANK_MIN_VALID})')
    print('  ' + '─' * 92)
    print(f"  {'Method':<24} {'Type':<12} {'MedErr':>9} {'Skill':>7} {'Valid':>6} "
          f"{'Cat':>6} {'Stab':>7} {'Cells':>6} {'Rank':>5}")
    print('  ' + '─' * 92)
    for _, row in df_top.iterrows():
        rank = '-' if not row['rank_eligible'] else f"{int(row['rank'])}"
        print(f"  {row['method']:<24} {row['method_type']:<12} "
              f"{row['med_error']:>9.5f} {row['med_skill']:>7.3f} "
              f"{row['valid_rate']:>6.3f} {row['cat_rate']:>6.3f} "
              f"{row['stability']:>7.3f} {int(row['n_cells']):>6} {rank:>5}")
    print()

    # ── Unranked block: below the validity floor ───────────────────────────────
    unr = results['unranked']
    unr = unr[(unr['target_g'] == headline_g) & (unr['regime_set'] == 'core')]
    print(f'  UNRANKED — BELOW VALIDITY FLOOR  (g = {headline_g}, core; '
          f'valid_rate < {CFG_MOD.RANK_MIN_VALID}; shown, never ranked; '
          f'{len(unr)} methods)')
    print('  ' + '─' * 92)
    if len(unr):
        print(f"  {'Method':<24} {'Type':<12} {'Valid':>6} {'Cat':>6} {'MedErr':>9} "
              f"{'Skill':>7} {'Stab':>7} {'Cells':>6}")
        for _, row in unr.iterrows():
            print(f"  {row['method']:<24} {row['method_type']:<12} "
                  f"{row['valid_rate']:>6.3f} {row['cat_rate']:>6.3f} "
                  f"{row['med_error']:>9.5f} {row['med_skill']:>7.3f} "
                  f"{row['stability']:>7.3f} {int(row['n_cells']):>6}")
    else:
        print('  (none)')
    print()

    # ── Trivial comparators at the headline stratum ────────────────────────────
    triv = df_g[(df_g['target_g'] == headline_g) & (df_g['is_trivial'] == 1)]
    print(f'  TRIVIAL COMPARATORS  (g = {headline_g}, core, capped excluded)')
    print('  ' + '─' * 60)
    for _, row in triv.iterrows():
        tag = '(oracle, unranked)' if row['is_oracle'] else ''
        print(f"  {row['method']:<24} med_err={row['med_error']:.5f}  "
              f"skill={row['med_skill']:.3f}  stab={row['stability']:.3f} {tag}")
    print()

    # ── Capped block ───────────────────────────────────────────────────────────
    cap = results['capped']
    if len(cap):
        cells = cap[['regime', 'target_g', 'n_f', 'achieved_g']].drop_duplicates()
        print(f'  CAPPED CELLS  (excluded from pooled tables; {len(cells)} regime x g cells)')
        print('  ' + '─' * 60)
        for _, r in cells.iterrows():
            print(f"  {r['regime']:<18} g={r['target_g']:<5g} n_f={int(r['n_f']):>6}  "
                  f"achieved g = {r['achieved_g']:.3f}")
        print()

    # ── Per-regime recommendations ─────────────────────────────────────────────
    df_best = results['regime_best']
    df_b    = df_best[df_best['target_g'] == headline_g]
    print(f'  PER-REGIME BEST BY SKILL  (g = {headline_g}; oracle excluded)')
    print('  ' + '─' * 92)
    for _, row in df_b.iterrows():
        flag = ' [holdout]' if row['is_holdout'] else ''
        cap  = ' CAP' if row['capped'] else ''
        print(f"  {row['regime']:<18}{flag:<10} n_f={int(row['n_f']):>6}{cap:<4} "
              f"{row['best_by_skill']:<20} skill={row['best_skill']:.3f}  "
              f"(by stability: {row['best_by_stability']})")
    print()

    print(f'  All files saved to: {out_dir}/')
    print('=' * 72 + '\n')


if __name__ == '__main__':
    main()
