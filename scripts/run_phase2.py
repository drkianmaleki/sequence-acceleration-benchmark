"""
run_phase2.py
=============
Phase 2 entry point — Richardson failure characterisation (redesign v2, descriptive).

    python scripts/run_phase2.py --quick     config.PHASE2["quick"]
    python scripts/run_phase2.py --full      config.PHASE2["full"]

Grid: the observation depths x noise levels x seeds x core regimes x gap
strata (g in config.HORIZON_GAP_FRACTIONS) of the config, for every member of
phases.phase2.PHASE2_METHODS (the Phase-2 pool plus constant_assumed and
constant_oracle).  Targets are per-depth gap-stratified horizons; capped
cells are flagged and excluded from pooled statistics.

Output directory: results/phase2/

Key output files
----------------
    phase2_records.csv             One row per record (git-ignored): estimate, error,
                                   valid, catastrophic, E_last, capped, L_true, L_hat,
                                   n_f, achieved_g.
    phase2_sweep_aggregated.csv    Per (method, regime, obs_idx, noise, target_g): the
                                   descriptive panel of the seeds (validity and
                                   catastrophe rates; mean / sd / median / q25 / q75 /
                                   p90 error conditional on validity; win rate vs the
                                   last value) plus the skill columns.
    phase2_features.csv            Six trajectory features per window (+ L_true, L_hat).
    phase2_capped.csv              The capped block.
    phase2_richardson_targets_g{g}.csv   Per cell: R_R_med (richardson_1 error / last-value
                                   error, median over valid seeds), log_med_error_R,
                                   n_valid_R, n_total and the zero-denominator flags.
    phase2_denominator_counts.csv  Per horizon: records and cells where the
                                   E_last <= SKILL_EPS branch of the ratio fired.
    phase2_correlations[_g{g}].csv Spearman feature correlations with both targets
                                   (headline stratum under the legacy name).
    phase2_rules[_g{g}].csv        Every candidate rule: what happened when it fired
                                   (src.panels.rule_panel).
    figure_p2_01 ... figure_p2_05  Five figures at the headline stratum.
"""

import os, sys, math, argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

import src.config as CFG_MOD
from src.pipeline import exclude_capped, resolve_regimes
from phases.phase2 import (
    run_sweep, richardson_targets, denominator_counts,
    run_correlation_analysis, evaluate_rules,
    make_all_figures, PHASE2_METHODS, RANK_POOL, RICHARDSON,
)


def n_evaluations(cfg: dict) -> int:
    """Planned pool evaluations (skill references add two cheap calls per cell)."""
    regimes = resolve_regimes(cfg['core_regimes'], include_holdout=False)
    return (len(cfg['obs_idx_list']) * len(cfg['noise_list']) * cfg['n_seeds']
            * len(regimes) * len(cfg['gap_fractions']) * len(PHASE2_METHODS))


def parse_args():
    p = argparse.ArgumentParser(description='Phase 2 — Richardson failure characterisation (v2)')
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--quick', action='store_true')
    mode.add_argument('--full',  action='store_true')
    p.add_argument('--out-dir', default=os.path.join('results', 'phase2'))
    return p.parse_args()


def main():
    args    = parse_args()
    mode    = 'quick' if args.quick else 'full'
    cfg     = dict(CFG_MOD.PHASE2[mode])
    out_dir = args.out_dir
    gs      = [float(g) for g in cfg['gap_fractions']]
    g_head  = CFG_MOD.HEADLINE_G if CFG_MOD.HEADLINE_G in gs else gs[-1]
    regimes = resolve_regimes(cfg['core_regimes'], include_holdout=False)

    print('=' * 72)
    print(f'  PHASE 2 — Richardson Failure Characterisation  [{mode.upper()}, redesign v2]')
    print('=' * 72)
    print(f'  obs_idx sweep : {cfg["obs_idx_list"]}')
    print(f'  noise levels  : {cfg["noise_list"]}')
    print(f'  gap strata    : {gs}  (headline g = {g_head})')
    print(f'  seeds         : {cfg["n_seeds"]}')
    print(f'  regimes       : {len(regimes)} core (selector training data; no holdout)')
    print(f'  methods       : {len(PHASE2_METHODS)}  {PHASE2_METHODS}')
    print(f'  pool          : {len(RANK_POOL)} (constant_assumed and constant_oracle '
          f'reported alongside; never selector candidates)')
    print(f'  reporting     : descriptive panels (conditional on validity, validity rate alongside); '
          f'richardson_1 error normalised by the last-value error; no composite score')
    print(f'  L_hat mode    : {CFG_MOD.ASSUMED_L_MODE}')
    print(f'  total evals   : {n_evaluations(cfg):,}')
    print(f'  output dir    : {out_dir}')
    print('=' * 72 + '\n')

    # ── 1. Sweep ───────────────────────────────────────────────────────────────
    df_agg, df_feat, df_rec = run_sweep(
        obs_idx_list  = cfg['obs_idx_list'],
        noise_list    = cfg['noise_list'],
        gap_fractions = gs,
        n_seeds       = cfg['n_seeds'],
        window_len    = cfg['window_len'],
        out_dir       = out_dir,
        core_regimes  = cfg['core_regimes'],
    )

    # ── 2-4. Targets, correlations and rules for every stratum ─────────────────
    targets, head = {}, {}
    for g in gs:
        suffix = f'_g{g:g}'
        print(f'\n  Stratum g = {g:g}: Richardson targets, correlations, rules ...')
        df_t = richardson_targets(df_rec, g, out_dir)
        targets[g] = df_t
        df_corr, _ = run_correlation_analysis(df_feat, df_t, g, out_dir, suffix)
        df_rules = evaluate_rules(df_rec, df_feat, df_agg, g, out_dir, suffix)
        if g == g_head:
            head = dict(targets=df_t, corr=df_corr, rules=df_rules)
            # headline stratum also under the legacy file names
            df_corr.to_csv(os.path.join(out_dir, 'phase2_correlations.csv'), index=False)
            df_rules.to_csv(os.path.join(out_dir, 'phase2_rules.csv'), index=False)
    df_den = denominator_counts(targets, out_dir)

    # ── 5. Figures (headline stratum) ──────────────────────────────────────────
    make_all_figures(head['targets'], head['corr'], head['rules'], g_head, out_dir)

    # ── Console summary ────────────────────────────────────────────────────────
    print('\n' + '=' * 72)
    print('  PHASE 2 SUMMARY  (headline stratum g = %g)' % g_head)
    print('=' * 72)

    t = exclude_capped(head['targets'])
    print(f'\n  richardson_1 by regime (uncapped cells; R_R_med = error / last-value error, '
          f'median over valid seeds; cells with R_R_med < 1 = Richardson below the last value):')
    print(f"  {'Regime':<22} {'cells':>6} {'med R_R_med':>12} {'cells < 1':>10} {'cells inf':>10} "
          f"{'med valid':>10} {'capped':>7}")
    print('  ' + '─' * 84)
    for regime in sorted(head['targets']['regime'].unique()):
        sub = t[t['regime'] == regime]
        allc = head['targets'][head['targets']['regime'] == regime]
        rr = sub['R_R_med']
        med = float(np.nanmedian(rr.replace([np.inf], np.nan))) if rr.notna().any() else float('nan')
        print(f"  {regime:<22} {len(sub):>6} {med:>12.3f} {int((rr < 1).sum()):>10} "
              f"{int(np.isinf(rr).sum()):>10} "
              f"{float((sub['n_valid_R'] / sub['n_total']).median()) if len(sub) else float('nan'):>10.3f} "
              f"{int(allc['capped'].sum()):>7}")

    print('\n  Zero-denominator branch of the normalised error (E_last <= SKILL_EPS), per horizon '
          '(uncapped cells):')
    for _, r in df_den.iterrows():
        print(f"    g = {r['target_g']:g}: {int(r['n_records_zero_denominator'])} of "
              f"{int(r['n_records_total'])} records, {int(r['n_cells_affected'])} of "
              f"{int(r['n_cells_total'])} cells")

    print('\n  Pooled feature correlations (all regimes, capped excluded):')
    print(f"  {'feature':<18} {'r vs R_R_med':>13} {'p':>7} {'r vs log err':>13} {'p':>7} {'cells':>6} {'dropped':>8}")
    print('  ' + '─' * 78)
    all_c = head['corr'][head['corr']['regime'] == 'ALL']
    for _, row in all_c.iterrows():
        print(f"  {row['feature']:<18} {row['spearman_vs_RR']:>13.4f} {row['p_vs_RR']:>7.3f} "
              f"{row['spearman_vs_log_err']:>13.4f} {row['p_vs_log_err']:>7.3f} "
              f"{int(row['n_cells']):>6} {int(row['n_dropped_nan']):>8}")

    print('\n  Threshold rules: what happened when they fired (fired cells; records of both methods):')
    print(f"  {'Rule':<40} {'fire':>6} {'lower rec':>10} {'lower cell':>11} {'med Δrel':>9} "
          f"{'r1 V':>6} {'alt V':>6}")
    print('  ' + '─' * 94)
    for _, row in head['rules'].iterrows():
        rule = f"{row['feature']} {row['operator']} {row['threshold']} → {row['alternative']}"
        rel = row['median_rel_change']
        print(f"  {rule:<40} {row['fire_rate']:>6.3f} {row['lower_error_frac_records']:>10.3f} "
              f"{row['lower_error_frac_cells']:>11.3f} "
              f"{(f'{rel:+.3f}' if math.isfinite(rel) else 'n/a'):>9} "
              f"{row['r1_valid_rate']:>6.3f} {row['alt_valid_rate']:>6.3f}")

    print(f'\n  All files saved to: {out_dir}/')
    print('=' * 72)


if __name__ == '__main__':
    main()
