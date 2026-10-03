"""
run_real_data.py
================
Real-data experiment entry point (redesign v2).

Default mode re-evaluates the EXISTING recorded XGBoost validation-loss
curves in results/real_data/real_data_curves.csv.  No retraining, no
downloads.  Grid: observation depths {30, 60, 90, 120, 150} x target rounds
{300, 400, 500} with depth < target, assumed-asymptote mode "zero",
trivial comparators and skill included.  The original 18-cell results
(real_data_results.csv and its three figures) are left untouched.

    python scripts/run_real_data.py                 # re-evaluate recorded curves
    python scripts/run_real_data.py --quick          # same (accepted for the pipeline)
    python scripts/run_real_data.py --roster-only    # only the roster evaluation (two files below)
    python scripts/run_real_data.py --retrain        # LEGACY: download + train
                                                     # (writes the 18-cell files)

Output directory: results/real_data/

Roster evaluation (R9d Part B; written after the files below, or alone with
--roster-only)
---------------
    real_data_roster_v2.csv     one row per (dataset, obs_depth, target_round, method) for
                                every method of src.trajectories.ROSTER_REAL_METHODS (every
                                accelerator and the four deployable trivial predictors; no
                                oracle), scored under the benchmark's own configuration and
                                per-record definitions (evaluate_recorded_curves_roster)
    real_data_roster_provenance.json   script, commit, tree state, timing, library versions
                                (and whether they equal the run manifest's), the numerical
                                configuration, datasets, cells, methods, rows

New files
---------
    real_data_results_v2.csv    one row per (dataset, obs_depth, target_round, method);
                                perturb_iqr = perturbation IQR of the routed method
                                for the cell (repeated on every row of the cell)
    real_data_summary_v2.csv    one row per (dataset, obs_depth, target_round) with
                                perturb_iqr and the provenance of the regime
                                centroids (phase2_features_path,
                                phase2_features_rows, git_head).  The columns
                                current_val / current_err are schema names for
                                the last observed value of the window (the
                                value at the observation depth) and its error
                                at the target round, i.e. the error of the
                                last_value trivial comparator.
    figure_rd_v2_01_skill.png   cascade skill heatmaps (dataset x depth, per target)
    real_data_curve_minima_v2.csv   per dataset: argmin round of the recorded curve,
                                its minimum, the value at round 500 and the relative
                                rise from the minimum (Prompt 5A)
    real_data_strata_v2.csv     every summary statistic three ways: all cells,
                                pre-minimum targets, post-minimum targets
                                (post_min_target = target_round > argmin_round),
                                including the perturbation-diagnostic AUC per stratum

Every row of the two per-cell files carries argmin_round and post_min_target;
the summary also carries rise_from_min.  Skill columns: ``skill`` /
``cascade_skill`` = hindsight best-of-four (strict); ``skill_vs_*`` /
``win_vs_*`` (and ``cascade_skill_vs_*`` / ``cascade_win_vs_*``) against each
deployable trivial.

The legacy 18-cell diagnostics (perturb_IQR AUC, z-scored regime mapping) are
verified by scripts/analyze_real_diagnostics_legacy.py against the stored
pre-redesign run; they are not part of the v2 pipeline.
"""

import os
import sys
import argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import src.config as CFG_MOD
from src.trajectories import (FEATURE_COLS, REAL_DATA_METHODS, ROSTER_REAL_METHODS, curve_minimum_table,
                              evaluate_recorded_curves, evaluate_recorded_curves_roster,
                              process_curves, real_data_strata)

OUT_DIR      = os.path.join('results', 'real_data')
PHASE2_FEATS = os.path.join('results', 'phase2', 'phase2_features.csv')
ASSUMED_MODE = CFG_MOD.ASSUMED_L_MODE   # "zero": real curves have no oracle
N_ROUNDS     = 500
ROSTER_CSV   = 'real_data_roster_v2.csv'
ROSTER_PROV  = 'real_data_roster_provenance.json'


def n_evaluations(curves_csv: str = None) -> dict:
    """Planned cells and method evaluations for the re-evaluation grid."""
    cfg = CFG_MOD.REAL_DATA
    csv = curves_csv or os.path.join(_ROOT, cfg['curves_csv'])
    if os.path.exists(csv):
        n_datasets = len([c for c in pd.read_csv(csv, nrows=1).columns if c != 'round'])
    else:
        from src.datasets import DATASET_IDS
        n_datasets = len(DATASET_IDS)
    pairs = [(d, t) for d in cfg['depths'] for t in cfg['targets'] if d < t]
    cells = n_datasets * len(pairs)
    n_methods = len(REAL_DATA_METHODS)
    return {'datasets': n_datasets, 'pairs': len(pairs), 'cells': cells,
            'methods': n_methods, 'evaluations': cells * n_methods}


def load_recorded_curves(path: str) -> dict:
    df = pd.read_csv(path)
    return {c: df[c].to_numpy(dtype=float) for c in df.columns if c != 'round'}


def fig_skill_heatmap(df_sum: pd.DataFrame, out_dir: str) -> str:
    """Cascade skill per (dataset, depth), one panel per target round."""
    targets  = sorted(df_sum['target_round'].unique())
    datasets = sorted(df_sum['dataset'].unique())
    depths   = sorted(df_sum['obs_depth'].unique())
    fig, axes = plt.subplots(1, len(targets), figsize=(4.5 * len(targets), 4.5), sharey=True)
    if len(targets) == 1:
        axes = [axes]
    for ax, t in zip(axes, targets):
        mat = np.full((len(datasets), len(depths)), np.nan)
        sub = df_sum[df_sum['target_round'] == t]
        for _, r in sub.iterrows():
            if r['obs_depth'] in depths:
                mat[datasets.index(r['dataset']), depths.index(r['obs_depth'])] = r['cascade_skill']
        im = ax.imshow(np.log10(np.clip(mat, 1e-3, 1e3)), cmap='RdYlGn_r', vmin=-1, vmax=1,
                       aspect='auto')
        ax.set_xticks(range(len(depths)))
        ax.set_xticklabels([f'd={d}' for d in depths], fontsize=8)
        ax.set_yticks(range(len(datasets)))
        ax.set_yticklabels(datasets, fontsize=8)
        ax.set_title(f'target round {t}', fontsize=10, fontweight='bold')
        for i in range(len(datasets)):
            for j in range(len(depths)):
                v = mat[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=7,
                            color='black' if 0.3 < v < 3 else 'white')
    plt.colorbar(im, ax=axes, label='log10 cascade skill  (<0: beats best trivial)', shrink=0.8)
    fig.suptitle('Recorded real curves: cascade skill vs best-of-four trivial reference '
                 f'(mode = {ASSUMED_MODE})', fontsize=10, fontweight='bold')
    path = os.path.join(out_dir, 'figure_rd_v2_01_skill.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')
    return path


def print_strata_v2(df_min: pd.DataFrame, df_strata: pd.DataFrame, df_sum: pd.DataFrame):
    """Curve minima per dataset and the three-way (all / pre-min / post-min) summary."""
    print('  Recorded-curve minima (a target round beyond the argmin is a post-minimum target):')
    print(f"  {'dataset':<16} {'rounds':>6} {'argmin':>7} {'min':>9} {'@round':>7} {'value':>9} {'rise':>8}")
    print('  ' + '-' * 70)
    for _, r in df_min.iterrows():
        print(f"  {r['dataset']:<16} {int(r['n_rounds']):>6} {int(r['argmin_round']):>7} {r['min_value']:>9.5f} "
              f"{int(r['at_round']):>7} {r['value_at_round']:>9.5f} {100 * r['rise_from_min']:>+7.2f}%")
    print('\n  Three-way summary (failure = cascade skill >= 1; AUC: higher perturb_iqr read as failure):')
    print(f"  {'stratum':<9} {'cells':>5} {'fail':>5} {'rate':>6} {'med err':>9} {'med skill':>10} {'med impr':>9} "
          f"{'win/assumed':>11} {'win/last':>9} {'AUC':>6} {'p':>6}  ordering")
    print('  ' + '-' * 110)
    for _, r in df_strata.iterrows():
        print(f"  {r['stratum']:<9} {int(r['n_cells']):>5} {int(r['n_fail']):>5} {r['fail_rate']:>6.3f} "
              f"{r['med_cascade_err']:>9.5f} {r['med_cascade_skill']:>10.3f} {r['med_improvement']:>+9.3f} "
              f"{r['win_rate_vs_assumed']:>11.3f} {r['win_rate_vs_last']:>9.3f} {r['perturb_auc']:>6.3f} "
              f"{r['perturb_p']:>6.3f}  {r['perturb_ordering']}")
    ct = pd.crosstab(df_sum['post_min_target'].map({0: 'pre-min', 1: 'post-min'}),
                     (df_sum['cascade_skill'] >= 1).map({False: 'ok', True: 'fail'}), margins=True)
    print('\n  Failure crosstab (rows: target vs the curve minimum):')
    for line in ct.to_string().splitlines():
        print('    ' + line)
    print()


def print_summary_v2(df_long: pd.DataFrame, df_sum: pd.DataFrame):
    print('\n' + '=' * 80)
    print('  REAL-DATA RE-EVALUATION SUMMARY  (recorded curves, no retraining)')
    print('=' * 80)
    print(f'\n  Datasets   : {sorted(df_sum["dataset"].unique())}')
    print(f'  Cells      : {len(df_sum)}  (depth < target)')
    print(f'  L_hat mode : {ASSUMED_MODE}')

    print(f'\n  Per (depth, target): mean errors and median cascade skill')
    print(f"  {'depth':>6} {'target':>7} {'cascade':>10} {'richardson':>11} {'rational':>10} "
          f"{'current':>10} {'best-triv':>10} {'med skill':>10} {'beats triv':>11}")
    print('  ' + '─' * 92)
    for (d, t), sub in df_sum.groupby(['obs_depth', 'target_round']):
        beats = float((sub['cascade_skill'] < 1).mean())
        print(f"  {int(d):>6} {int(t):>7} {sub['cascade_err'].mean():>10.5f} "
              f"{sub['richardson_err'].mean():>11.5f} {sub['rational_err'].mean():>10.5f} "
              f"{sub['current_err'].mean():>10.5f} {sub['ref_error'].mean():>10.5f} "
              f"{sub['cascade_skill'].median():>10.3f} {beats:>11.2f}")

    print(f'\n  Method median skill over all cells (hindsight best-of-four, strict; skill < 1 beats the best trivial reference):')
    for m, sub in df_long.groupby('method'):
        print(f"    {m:<18} med skill = {sub['skill'].median():.3f}   "
              f"finite = {int(sub['skill'].notna().sum())}/{len(sub)}")

    print(f'\n  Cascade method selection: {dict(df_sum["selected_method"].value_counts())}')
    print(f'  Best trivial reference:   {dict(df_sum["ref_best_method"].value_counts())}')

    piqr = df_sum['perturb_iqr']
    print(f'\n  perturb_iqr of the routed method ({CFG_MOD.PERTURB_TRIALS} trials, '
          f'{100 * CFG_MOD.PERTURB_SCALE:g}% perturbation, crc32 seed per (dataset, depth)): '
          f'finite = {int(piqr.notna().sum())}/{len(piqr)}, median = {piqr.median():.5f}, '
          f'range = [{piqr.min():.5f}, {piqr.max():.5f}]')
    for m, sub in df_sum.groupby('selected_method'):
        print(f'    routed {m:<14} n = {len(sub):>3}  median perturb_iqr = '
              f'{sub["perturb_iqr"].median():.5f}')

    prov = df_sum.iloc[0]
    print(f'\n  Provenance (recorded on every summary row):')
    print(f'    phase2_features_path = {prov["phase2_features_path"]!r}')
    print(f'    phase2_features_rows = {int(prov["phase2_features_rows"])}'
          + ('   (file absent: regime mapping = unknown)'
             if int(prov["phase2_features_rows"]) == 0 else ''))
    print(f'    git_head             = {prov["git_head"]}')
    print()


def _git_head() -> dict:
    """Full and short hash of the checked-out commit ("unknown" outside a git checkout)."""
    import subprocess
    out = {}
    for key, cmd in (('full', ['rev-parse', 'HEAD']), ('short', ['rev-parse', '--short', 'HEAD'])):
        try:
            out[key] = subprocess.check_output(['git'] + cmd, cwd=_ROOT, stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            out[key] = 'unknown'
    return out


def _tracked_tree_clean():
    """True when `git status --porcelain --untracked-files=no` prints nothing (None outside git)."""
    import subprocess
    try:
        out = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'],
                                      cwd=_ROOT, stderr=subprocess.DEVNULL).decode()
        return out.strip() == ''
    except Exception:
        return None


def run_roster(args, curves: dict) -> tuple:
    """
    The roster evaluation of the recorded curves (R9d Part B): every method of
    ROSTER_REAL_METHODS under the benchmark's own configuration and scoring
    (src.trajectories.evaluate_recorded_curves_roster), written to
    real_data_roster_v2.csv with its own provenance file
    real_data_roster_provenance.json.  Touches nothing else.
    """
    import datetime as _dt
    import json
    import time
    from reproduce_all import _versions
    from scripts.analyze_by_ltrue import load_manifest

    cfg = CFG_MOD.REAL_DATA
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()
    started = _dt.datetime.now().isoformat(timespec='seconds')
    head = _git_head()
    clean = _tracked_tree_clean()
    df = evaluate_recorded_curves_roster(curves, depths=cfg['depths'], targets=cfg['targets'],
                                         window_len=cfg['window_len'], assumed_mode=ASSUMED_MODE)
    finished = _dt.datetime.now().isoformat(timespec='seconds')
    seconds = round(time.time() - t0, 1)

    p_csv = os.path.join(args.out_dir, ROSTER_CSV)
    df.to_csv(p_csv, index=False)
    cells = int(df.drop_duplicates(['dataset', 'obs_depth', 'target_round']).shape[0]) if len(df) else 0
    versions = _versions()
    M = load_manifest(os.path.join(_ROOT, 'results'))
    run_versions = (M or {}).get('versions')
    prov = {
        'script': 'scripts/run_real_data.py' + (' --roster-only' if getattr(args, 'roster_only', False) else ''),
        'output': os.path.relpath(p_csv, _ROOT).replace(os.sep, '/'),
        'git_head': head,
        'tracked_tree_clean': clean,
        'started': started,
        'finished': finished,
        'seconds': seconds,
        'versions': versions,
        'versions_equal_run_manifest': (versions == run_versions) if run_versions is not None else None,
        'configuration': {
            'RIDGE': CFG_MOD.RIDGE, 'DENOM_TOL': CFG_MOD.DENOM_TOL,
            'MIN_VALID': CFG_MOD.MIN_VALID, 'MAX_VALID': CFG_MOD.MAX_VALID,
            'CAT_MULT': CFG_MOD.CAT_MULT,
            'window_len': cfg['window_len'], 'assumed_mode': ASSUMED_MODE,
            'depths': list(cfg['depths']), 'targets': list(cfg['targets']),
            'validity': 'src.evaluation.is_valid: finite and MIN_VALID <= prediction <= MAX_VALID '
                        '(valid_strict = the legacy rule, finite and 0 <= prediction <= 2 * window max, recorded alongside)',
        },
        'datasets': sorted(curves),
        'cells': cells,
        'methods': len(ROSTER_REAL_METHODS),
        'method_names': list(ROSTER_REAL_METHODS),
        'rows': int(len(df)),
        'note': ('This evaluation was added after the full run recorded in results/run_manifest.json'
                 + (f" (started {M.get('started')}, code {M.get('git_head', {}).get('short')})" if M else '')
                 + '; it applies the benchmark\'s own protocol (src.evaluation: build_cfg, is_valid, the per-record '
                   'definitions of run_phase1) to the recorded curves and changes no file of that run.'),
    }
    p_prov = os.path.join(args.out_dir, ROSTER_PROV)
    with open(p_prov, 'w', encoding='utf-8', newline='\n') as fh:
        json.dump(prov, fh, indent=2)
        fh.write('\n')
    print(f'  Saved: {p_csv}  ({len(df)} rows = {cells} cells x {len(ROSTER_REAL_METHODS)} methods; {seconds} s)')
    print(f'  Saved: {p_prov}')
    return p_csv, p_prov


def parse_args():
    p = argparse.ArgumentParser(description='Real-data experiment (redesign v2)')
    p.add_argument('--quick', action='store_true', help='accepted for pipeline uniformity')
    p.add_argument('--full', action='store_true', help='accepted for pipeline uniformity')
    p.add_argument('--retrain', action='store_true',
                   help='LEGACY path: download OpenML data, retrain XGBoost, write the '
                        '18-cell files (requires internet, xgboost, openml)')
    p.add_argument('--roster-only', action='store_true',
                   help='write only real_data_roster_v2.csv and real_data_roster_provenance.json '
                        '(every method under the benchmark protocol); no legacy CSV, no figure')
    p.add_argument('--curves', default=os.path.join(_ROOT, CFG_MOD.REAL_DATA['curves_csv']))
    p.add_argument('--out-dir', default=OUT_DIR)
    return p.parse_args()


def main_reevaluate(args):
    cfg = CFG_MOD.REAL_DATA
    os.makedirs(args.out_dir, exist_ok=True)
    counts = n_evaluations(args.curves)

    print('=' * 72)
    print('  REAL DATA — RE-EVALUATION OF RECORDED CURVES  (redesign v2)')
    print('=' * 72)
    print(f'  Curves     : {args.curves}')
    print(f'  Depths     : {cfg["depths"]}')
    print(f'  Targets    : {cfg["targets"]}  (cells need depth < target)')
    print(f'  Window len : {cfg["window_len"]}')
    print(f'  L_hat mode : {ASSUMED_MODE}  (no oracle on real curves)')
    print(f'  Methods    : {len(REAL_DATA_METHODS)}  {REAL_DATA_METHODS}')
    print(f'  Cells      : {counts["cells"]}  ->  {counts["evaluations"]} evaluations')
    print(f'  Output dir : {args.out_dir}')
    print(f'  Legacy 18-cell results are preserved (real_data_results.csv untouched).')
    print('=' * 72 + '\n')

    if not os.path.exists(args.curves):
        print(f'ERROR: {args.curves} not found; nothing to re-evaluate.')
        return 1

    curves = load_recorded_curves(args.curves)
    df_long, df_sum = evaluate_recorded_curves(
        curves, depths=cfg['depths'], targets=cfg['targets'],
        window_len=cfg['window_len'], assumed_mode=ASSUMED_MODE,
        phase2_features_path=PHASE2_FEATS)

    p1 = os.path.join(args.out_dir, 'real_data_results_v2.csv')
    p2 = os.path.join(args.out_dir, 'real_data_summary_v2.csv')
    df_long.to_csv(p1, index=False)
    df_sum.to_csv(p2, index=False)
    print(f'  Saved: {p1}  ({len(df_long)} rows)')
    print(f'  Saved: {p2}  ({len(df_sum)} rows)')

    df_min = curve_minimum_table(curves, last_round=N_ROUNDS)
    p3 = os.path.join(args.out_dir, 'real_data_curve_minima_v2.csv')
    df_min.to_csv(p3, index=False)
    print(f'  Saved: {p3}  ({len(df_min)} rows)')
    df_strata = real_data_strata(df_sum)
    p4 = os.path.join(args.out_dir, 'real_data_strata_v2.csv')
    df_strata.to_csv(p4, index=False)
    print(f'  Saved: {p4}  ({len(df_strata)} rows)')

    fig_skill_heatmap(df_sum, args.out_dir)
    print_summary_v2(df_long, df_sum)
    print_strata_v2(df_min, df_strata, df_sum)
    # R9d Part B: every method under the benchmark protocol, after the legacy outputs
    print(f'  Roster evaluation: {len(ROSTER_REAL_METHODS)} methods x {counts["cells"]} cells '
          f'under the benchmark configuration (src.evaluation)')
    run_roster(args, curves)
    print('=' * 72 + '\n')
    return 0


def main_roster_only(args):
    """--roster-only: the two roster files and nothing else."""
    if not os.path.exists(args.curves):
        print(f'ERROR: {args.curves} not found; nothing to evaluate.')
        return 1
    curves = load_recorded_curves(args.curves)
    print('=' * 72)
    print('  REAL DATA — ROSTER EVALUATION OF THE RECORDED CURVES  (benchmark protocol)')
    print('=' * 72)
    print(f'  Curves     : {args.curves}')
    print(f'  Methods    : {len(ROSTER_REAL_METHODS)}  (every accelerator + the four deployable trivials)')
    print(f'  L_hat mode : {ASSUMED_MODE}')
    run_roster(args, curves)
    print('=' * 72 + '\n')
    return 0


def main_retrain(args):
    """LEGACY path (not part of reproduce_all): download, retrain, 18-cell outputs."""
    from src.datasets import run_real_data_experiment, DATASET_IDS
    os.makedirs(args.out_dir, exist_ok=True)
    print('=' * 72)
    print('  REAL-DATA EXPERIMENT — LEGACY RETRAIN PATH')
    print('=' * 72)
    print(f'  Datasets   : {list(DATASET_IDS.keys())}')
    print(f'  Rounds     : {N_ROUNDS}')
    print(f'  L_hat mode : {ASSUMED_MODE}')
    print('=' * 72 + '\n')
    curves = run_real_data_experiment(out_dir=args.out_dir, n_rounds=N_ROUNDS)
    df = process_curves(curves=curves, obs_depths=[30, 60, 90], window_len=60,
                        assumed_mode=ASSUMED_MODE, phase2_features_path=PHASE2_FEATS,
                        future_x=N_ROUNDS)
    p = os.path.join(args.out_dir, 'real_data_results.csv')
    df.to_csv(p, index=False)
    print(f'  Results saved to {p}')
    return 0


def main():
    args = parse_args()
    if args.retrain:
        return main_retrain(args)
    if args.roster_only:
        return main_roster_only(args)
    return main_reevaluate(args)


if __name__ == '__main__':
    sys.exit(main())
