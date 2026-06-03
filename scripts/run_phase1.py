"""
run_phase1.py
=============
Phase 1 entry point — full synthetic benchmark.

Usage
-----
    cd sequence-acceleration-benchmark/
    python scripts/run_phase1.py --quick     # 5 seeds, sigma=0, horizon 5000 only (~3 min)
    python scripts/run_phase1.py --full      # 30 seeds, 3 noise, 3 horizons (~20-40 min)

Output directory:  results/phase1/

Key output files
----------------
    phase1_aggregated.csv       Per (method, regime, noise, horizon) stats.
                                Primary data file; ~8 K rows in full mode.
    phase1_global.csv           Pooled across regimes; one row per
                                (method, horizon). Source for Table 4 / Table 22.
    phase1_regime_best.csv      Best method per (regime, horizon). Source for Table 5 / Table 23.
    phase1_heatmap_n{H}.csv     Stability score matrix for horizon H.
    phase1_raw_sample.csv       Full per-seed records, noiseless case only.
    figure_01 ... figure_06     Six publication figures.
"""

import os
import sys
import argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)   # one level up: the repo root
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import src.config as CFG_MOD
from src.evaluation import run_phase1
from src.plots      import make_all_figures


# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Phase 1 — Full synthetic benchmark')
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--quick', action='store_true',
                      help='Fast sanity-check run (5 seeds, 1 noise, 1 horizon)')
    mode.add_argument('--full',  action='store_true',
                      help='Full publication run (30 seeds, 3 noise, 3 horizons)')
    return p.parse_args()


# ── Mode configurations ────────────────────────────────────────────────────────

QUICK_CFG = dict(
    n_seeds      = 5,
    noise_levels = [0.0],
    future_idxs  = [5000],
    obs_idx      = CFG_MOD.OBS_IDX,
    window_len   = CFG_MOD.WINDOW_LEN,
)

FULL_CFG = dict(
    n_seeds      = CFG_MOD.N_SEEDS,                           # 30
    noise_levels = CFG_MOD.NOISE_LEVELS[:3],                  # [0.0, 0.001, 0.005]
    future_idxs  = [CFG_MOD.FUTURE_IDX_NEAR,                  # 150
                    CFG_MOD.FUTURE_IDX_MID,                    # 1000
                    CFG_MOD.FUTURE_IDX_DEFAULT],               # 5000
    obs_idx      = CFG_MOD.OBS_IDX,
    window_len   = CFG_MOD.WINDOW_LEN,
)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args    = parse_args()
    cfg     = QUICK_CFG if args.quick else FULL_CFG
    out_dir = os.path.join('results', 'phase1')
    mode    = 'QUICK' if args.quick else 'FULL'

    print('=' * 72)
    print(f'  PHASE 1 — Full Synthetic Benchmark  [{mode} mode]')
    print('=' * 72)
    print(f'  Regimes    : 18')
    print(f'  Methods    : 51')
    print(f'  Seeds      : {cfg["n_seeds"]}')
    print(f'  Noise      : {cfg["noise_levels"]}')
    print(f'  Horizons   : {cfg["future_idxs"]}')
    print(f'  obs_idx    : {cfg["obs_idx"]}')
    print(f'  window_len : {cfg["window_len"]}')
    n_evals = (18 * cfg['n_seeds'] * len(cfg['noise_levels'])
               * len(cfg['future_idxs']) * 51)
    print(f'  Total evals: {n_evals:,}')
    print(f'  Output dir : {out_dir}')
    print('=' * 72 + '\n')

    results = run_phase1(
        n_seeds      = cfg['n_seeds'],
        noise_levels = cfg['noise_levels'],
        future_idxs  = cfg['future_idxs'],
        obs_idx      = cfg['obs_idx'],
        window_len   = cfg['window_len'],
        out_dir      = out_dir,
        verbose      = True,
    )

    default_horizon = cfg['future_idxs'][-1]   # largest horizon
    make_all_figures(
        results         = results,
        horizons        = cfg['future_idxs'],
        default_horizon = default_horizon,
        out_dir         = out_dir,
    )

    # ── Print top-20 global table for immediate reading ────────────────────────
    df_g   = results['global']
    df_top = (df_g[df_g['future_idx'] == default_horizon]
                .sort_values('stability', ascending=False)
                .head(20))

    print(f'\n  TOP-20 GLOBAL STABILITY  (horizon = {default_horizon})')
    print('  ' + '─' * 68)
    print(f"  {'Method':<24} {'Type':<12} {'Valid':>6} {'Cat':>6} "
          f"{'Beats':>6} {'Stab':>8}")
    print('  ' + '─' * 68)
    for _, row in df_top.iterrows():
        print(f"  {row['method']:<24} {row['method_type']:<12} "
              f"{row['valid_rate']:>6.3f} {row['cat_rate']:>6.3f} "
              f"{row['beats_rate']:>6.3f} {row['stability']:>8.4f}")
    print()

    # ── Print per-regime recommendations ──────────────────────────────────────
    df_best = results['regime_best']
    df_b    = df_best[df_best['future_idx'] == default_horizon]
    print(f'  PER-REGIME RECOMMENDATIONS  (horizon = {default_horizon})')
    print('  ' + '─' * 68)
    for _, row in df_b.iterrows():
        print(f"  {row['regime']:<22}  {row['best_method']:<24} "
              f"stab={row['stability']:.3f}")
    print()

    print(f'  All files saved to: {out_dir}/')
    print('=' * 72 + '\n')


if __name__ == '__main__':
    main()
