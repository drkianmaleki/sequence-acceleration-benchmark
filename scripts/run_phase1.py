"""
run_phase1.py
=============
Phase 1 entry point — main synthetic benchmark (redesign v2).

Usage
-----
    cd sequence-acceleration-benchmark/
    python scripts/run_phase1.py --quick     # 5 seeds, sigma=0, all gap strata (~3 min)
    python scripts/run_phase1.py --full      # 30 seeds, 3 noise, 3 gap strata (~30-60 min)

    optional:  --assumed-mode {zero,half,oracle,double,winmin}   (default: config)
               --no-holdout                                       (core 18 only)

Output directory:  results/phase1/

Key output files
----------------
    phase1_records.csv          Per-seed records with (target_g, achieved_g,
                                n_f, capped, L_true, L_hat, skill).
    phase1_aggregated.csv       Per (method, regime, noise, g) stats.
    phase1_global.csv           Pooled over the 18 core regimes; rank column
                                excludes the oracle comparator.
    phase1_global_holdout.csv   Pooled over the 6 held-out regimes.
    phase1_regime_best.csv      Best non-oracle method per (regime, g).
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
from src.generators import HOLDOUT_REGIME_NAMES, REGIME_NAMES
from src.horizons import format_horizon_table
from src.plots      import make_all_figures


# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Phase 1 — main synthetic benchmark (redesign v2)')
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--quick', action='store_true',
                      help='Fast sanity-check run (5 seeds, 1 noise level)')
    mode.add_argument('--full',  action='store_true',
                      help='Full publication run (30 seeds, 3 noise levels)')
    p.add_argument('--assumed-mode', default=None,
                   choices=list(CFG_MOD.ASSUMED_L_MODES),
                   help='assumed-asymptote mode handed to methods '
                        f'(default: config.ASSUMED_L_MODE = {CFG_MOD.ASSUMED_L_MODE})')
    p.add_argument('--no-holdout', action='store_true',
                   help='evaluate the 18 core regimes only')
    p.add_argument('--out-dir', default=os.path.join('results', 'phase1'),
                   help='output directory (default: results/phase1)')
    return p.parse_args()


# ── Mode configurations ────────────────────────────────────────────────────────

QUICK_CFG = dict(
    n_seeds       = 5,
    noise_levels  = [0.0],
    gap_fractions = list(CFG_MOD.HORIZON_GAP_FRACTIONS),
    obs_idx       = CFG_MOD.OBS_IDX,
    window_len    = CFG_MOD.WINDOW_LEN,
)

FULL_CFG = dict(
    n_seeds       = CFG_MOD.N_SEEDS,                           # 30
    noise_levels  = CFG_MOD.NOISE_LEVELS[:3],                  # [0.0, 0.001, 0.005]
    gap_fractions = list(CFG_MOD.HORIZON_GAP_FRACTIONS),       # [0.5, 0.1, 0.02]
    obs_idx       = CFG_MOD.OBS_IDX,
    window_len    = CFG_MOD.WINDOW_LEN,
)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args    = parse_args()
    cfg     = QUICK_CFG if args.quick else FULL_CFG
    out_dir = args.out_dir
    mode    = 'QUICK' if args.quick else 'FULL'
    assumed_mode = args.assumed_mode or CFG_MOD.ASSUMED_L_MODE
    include_holdout = not args.no_holdout
    n_regimes = len(REGIME_NAMES) + (len(HOLDOUT_REGIME_NAMES) if include_holdout else 0)

    print('=' * 72)
    print(f'  PHASE 1 — Main Synthetic Benchmark  [{mode} mode, redesign v2]')
    print('=' * 72)
    print(f'  Regimes       : {len(REGIME_NAMES)} core'
          + (f' + {len(HOLDOUT_REGIME_NAMES)} held-out' if include_holdout else ''))
    print(f'  Methods       : {len(METHOD_NAMES)} (incl. 5 trivial comparators)')
    print(f'  Seeds         : {cfg["n_seeds"]}')
    print(f'  Noise         : {cfg["noise_levels"]}')
    print(f'  Gap strata g  : {cfg["gap_fractions"]}  (cap n = {CFG_MOD.HORIZON_N_CAP:,})')
    print(f'  obs_idx       : {cfg["obs_idx"]}')
    print(f'  window_len    : {cfg["window_len"]}')
    print(f'  L_true        : {CFG_MOD.ASYMPTOTE_MODE} '
          f'(log-uniform on {CFG_MOD.L_TRUE_RANGE}, hidden per regime x seed)')
    print(f'  L_hat mode    : {assumed_mode}'
          + ('   <-- ORACLE: methods see L_true' if assumed_mode == 'oracle' else ''))
    n_evals = (n_regimes * cfg['n_seeds'] * len(cfg['noise_levels'])
               * len(cfg['gap_fractions']) * len(METHOD_NAMES))
    print(f'  Total evals   : {n_evals:,}')
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

    # ── Top-20 global table (core regimes, oracle shown but unranked) ──────────
    df_g   = results['global']
    df_top = (df_g[df_g['target_g'] == headline_g]
                .sort_values('stability', ascending=False)
                .head(20))

    print(f'\n  TOP-20 GLOBAL STABILITY  (core regimes, g = {headline_g}, '
          f'L_hat mode = {assumed_mode})')
    print('  ' + '─' * 84)
    print(f"  {'Method':<24} {'Type':<12} {'Valid':>6} {'Cat':>6} "
          f"{'Beats':>6} {'Stab':>8} {'Skill':>8} {'Rank':>5}")
    print('  ' + '─' * 84)
    for _, row in df_top.iterrows():
        rank = '-' if row['is_oracle'] else f"{int(row['rank'])}"
        print(f"  {row['method']:<24} {row['method_type']:<12} "
              f"{row['valid_rate']:>6.3f} {row['cat_rate']:>6.3f} "
              f"{row['beats_rate']:>6.3f} {row['stability']:>8.4f} "
              f"{row['med_skill']:>8.3f} {rank:>5}")
    print()

    # ── Trivial comparators at the headline stratum ────────────────────────────
    triv = df_g[(df_g['target_g'] == headline_g) & (df_g['family'] == 'trivial')]
    print(f'  TRIVIAL COMPARATORS  (g = {headline_g})')
    print('  ' + '─' * 60)
    for _, row in triv.iterrows():
        tag = '(oracle)' if row['is_oracle'] else ''
        print(f"  {row['method']:<24} med_err={row['med_error']:.5f}  "
              f"stab={row['stability']:.3f}  skill={row['med_skill']:.3f} {tag}")
    print()

    # ── Per-regime recommendations ─────────────────────────────────────────────
    df_best = results['regime_best']
    df_b    = df_best[df_best['target_g'] == headline_g]
    print(f'  PER-REGIME BEST NON-ORACLE METHOD  (g = {headline_g})')
    print('  ' + '─' * 84)
    for _, row in df_b.iterrows():
        flag = ' [holdout]' if row['holdout'] else ''
        cap  = ' CAP' if row['capped'] else ''
        print(f"  {row['regime']:<18}{flag:<10} n_f={int(row['n_f']):>6}{cap:<4} "
              f"{row['best_method']:<20} stab={row['stability']:.3f}  "
              f"skill={row['med_skill']:.3f}")
    print()

    print(f'  All files saved to: {out_dir}/')
    print('=' * 72 + '\n')


if __name__ == '__main__':
    main()
