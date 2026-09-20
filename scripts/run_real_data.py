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
    python scripts/run_real_data.py --retrain        # LEGACY: download + train
                                                     # (writes the 18-cell files)

Output directory: results/real_data/

New files
---------
    real_data_results_v2.csv    one row per (dataset, obs_depth, target_round, method);
                                perturb_iqr = perturbation IQR of the routed method
                                for the cell (repeated on every row of the cell)
    real_data_summary_v2.csv    one row per (dataset, obs_depth, target_round) with
                                perturb_iqr and the provenance of the regime
                                centroids (phase2_features_path,
                                phase2_features_rows, git_head)
    figure_rd_v2_01_skill.png   cascade skill heatmaps (dataset x depth, per target)

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
from src.trajectories import evaluate_recorded_curves, process_curves, FEATURE_COLS

OUT_DIR      = os.path.join('results', 'real_data')
PHASE2_FEATS = os.path.join('results', 'phase2', 'phase2_features.csv')
ASSUMED_MODE = CFG_MOD.ASSUMED_L_MODE   # "zero": real curves have no oracle
N_ROUNDS     = 500


def n_evaluations(curves_csv: str = None) -> dict:
    """Planned cells and method evaluations for the re-evaluation grid."""
    cfg = CFG_MOD.REAL_DATA
    csv = curves_csv or os.path.join(_ROOT, cfg['curves_csv'])
    n_datasets = 6
    if os.path.exists(csv):
        n_datasets = len([c for c in pd.read_csv(csv, nrows=1).columns if c != 'round'])
    pairs = [(d, t) for d in cfg['depths'] for t in cfg['targets'] if d < t]
    cells = n_datasets * len(pairs)
    return {'datasets': n_datasets, 'pairs': len(pairs), 'cells': cells,
            'methods': 7, 'evaluations': cells * 7}


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

    print(f'\n  Method median skill over all cells (skill < 1 beats the best trivial reference):')
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


def parse_args():
    p = argparse.ArgumentParser(description='Real-data experiment (redesign v2)')
    p.add_argument('--quick', action='store_true', help='accepted for pipeline uniformity')
    p.add_argument('--full', action='store_true', help='accepted for pipeline uniformity')
    p.add_argument('--retrain', action='store_true',
                   help='LEGACY path: download OpenML data, retrain XGBoost, write the '
                        '18-cell files (requires internet, xgboost, openml)')
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
    print(f'  Methods    : richardson_1, rational_fit, cascade + 4 trivial references')
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

    fig_skill_heatmap(df_sum, args.out_dir)
    print_summary_v2(df_long, df_sum)
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
    return main_reevaluate(args)


if __name__ == '__main__':
    sys.exit(main())
