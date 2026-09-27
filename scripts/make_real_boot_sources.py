"""
make_real_boot_sources.py
=========================
Build the generator inputs of the held-out real_boot_a / real_boot_b regimes
from two OpenML datasets that are NOT in the recorded-curve test set
(src.datasets.REAL_BOOT_SOURCE_IDS: electricity 151, nomao 1486; the test set
is src.datasets.DATASET_IDS).  Each dataset is trained exactly as the recorded
curves were (src.datasets.train_xgboost: N_ROUNDS rounds, the same parameters
and seed) and its validation log-loss curve is written to

    results/real_data/real_boot_sources.csv            (round 0..N_ROUNDS-1, one column per source)
    results/real_data/real_boot_sources_provenance.json

The provenance records the OpenML id and version, rows, features, classes,
the xgboost / scikit-learn / openml versions, the seed, the git head, the
date, and the three pre-specified inclusion criteria with each dataset's
measured value and pass / fail:

    rows        >= MIN_ROWS            (enough data for a stable curve)
    gap ratio   >= MIN_GAP_RATIO_90    gap(90) / gap(0) of the envelope profile
                                       that src.generators.real_boot_profile
                                       builds (the curve is still descending
                                       at the default observation depth)
    argmin      >= MIN_ARGMIN_ROUND    1-based round of the curve's minimum
                                       (still improving at that round)

A failing criterion is reported and written, never patched over: the script
does not substitute another dataset, and src/generators.py is only switched
to the new sources when both candidates pass.

    python scripts/make_real_boot_sources.py [--out-dir results/real_data]
"""

import argparse
import datetime as _dt
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402

import src.config as CFG_MOD             # noqa: E402
from src.datasets import (DATASET_IDS, N_ROUNDS, RANDOM_STATE, REAL_BOOT_SOURCE_IDS,  # noqa: E402
                          load_datasets, train_xgboost)
from src.generators import envelope_profile  # noqa: E402
from src.pipeline import git_head        # noqa: E402

MIN_ROWS = 20_000
MIN_GAP_RATIO_90 = 0.05
MIN_ARGMIN_ROUND = 150
OBS_DEPTH = CFG_MOD.OBS_IDX              # the default observation depth (90)


def curve_facts(curve: np.ndarray) -> dict:
    """Loss at rounds 1, OBS_DEPTH and N_ROUNDS (1-based), the argmin round
    (1-based) and the envelope gap ratio gap(OBS_DEPTH) / gap(0)."""
    n = np.arange(len(curve), dtype=float)
    prof = envelope_profile(n, curve)
    return dict(
        loss_round_1=float(curve[0]),
        loss_round_obs=float(curve[OBS_DEPTH - 1]),
        loss_round_last=float(curve[-1]),
        min_loss=float(np.min(curve)),
        argmin_round=int(np.argmin(curve)) + 1,
        gap_ratio_obs=float(prof[OBS_DEPTH] / prof[0]),
    )


def main() -> int:
    p = argparse.ArgumentParser(description="Build the real_boot generator sources from OpenML")
    p.add_argument("--out-dir", default=os.path.join(_ROOT, "results", "real_data"))
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    import sklearn, xgboost, openml   # noqa: E401  (versions for the provenance)

    print("=" * 72)
    print("  real_boot SOURCES  (generator inputs; never scored; outside the test set)")
    print("=" * 72)
    print(f"  sources   : {REAL_BOOT_SOURCE_IDS}")
    print(f"  test set  : {DATASET_IDS}  (disjoint by construction)")
    print(f"  training  : src.datasets.train_xgboost, {N_ROUNDS} rounds, seed {RANDOM_STATE}")
    print(f"  criteria  : rows >= {MIN_ROWS:,}; gap({OBS_DEPTH})/gap(0) >= {MIN_GAP_RATIO_90}; "
          f"argmin round >= {MIN_ARGMIN_ROUND}")
    print("=" * 72 + "\n")

    data = load_datasets(REAL_BOOT_SOURCE_IDS, with_meta=True)
    curves, prov = {}, {}
    all_pass = True
    for name, (X, y, meta) in data.items():
        print(f"\n  Training XGBoost on {name} ...")
        curve = train_xgboost(X, y, n_rounds=N_ROUNDS, seed=RANDOM_STATE)
        curves[name] = curve
        facts = curve_facts(curve)
        checks = {
            "rows": dict(value=int(X.shape[0]), threshold=MIN_ROWS,
                         passed=bool(X.shape[0] >= MIN_ROWS)),
            f"gap_ratio_{OBS_DEPTH}": dict(value=facts["gap_ratio_obs"], threshold=MIN_GAP_RATIO_90,
                                          passed=bool(facts["gap_ratio_obs"] >= MIN_GAP_RATIO_90)),
            "argmin_round": dict(value=facts["argmin_round"], threshold=MIN_ARGMIN_ROUND,
                                 passed=bool(facts["argmin_round"] >= MIN_ARGMIN_ROUND)),
        }
        passed = all(c["passed"] for c in checks.values())
        all_pass &= passed
        prov[name] = dict(**meta, n_rows=int(X.shape[0]), n_features=int(X.shape[1]),
                          n_classes=int(len(np.unique(y))), **facts, criteria=checks,
                          passed_all=passed)
        print(f"    loss at round 1 / {OBS_DEPTH} / {N_ROUNDS}: {facts['loss_round_1']:.6f} / "
              f"{facts['loss_round_obs']:.6f} / {facts['loss_round_last']:.6f}")
        print(f"    argmin round (1-based): {facts['argmin_round']}   min loss {facts['min_loss']:.6f}")
        print(f"    envelope gap ratio gap({OBS_DEPTH})/gap(0): {facts['gap_ratio_obs']:.4f}")
        for k, c in checks.items():
            print(f"    criterion {k:<14} value {c['value']:<12.6g} threshold {c['threshold']:<8g} "
                  f"{'PASS' if c['passed'] else 'FAIL'}")

    df = pd.DataFrame(curves)
    df.index.name = "round"
    csv_path = os.path.join(args.out_dir, "real_boot_sources.csv")
    df.to_csv(csv_path)
    payload = dict(
        purpose=("generator inputs of the held-out real_boot_a / real_boot_b regimes; never scored; "
                 "disjoint from the recorded-curve test set (src.datasets.DATASET_IDS)"),
        sources=REAL_BOOT_SOURCE_IDS, test_set=DATASET_IDS,
        training=dict(function="src.datasets.train_xgboost", n_rounds=N_ROUNDS, seed=RANDOM_STATE,
                      val_fraction=0.2, learning_rate=0.05, max_depth=6, subsample=0.8,
                      colsample_bytree=0.8),
        versions=dict(xgboost=xgboost.__version__, sklearn=sklearn.__version__,
                      openml=openml.__version__, numpy=np.__version__, pandas=pd.__version__),
        criteria=dict(min_rows=MIN_ROWS, min_gap_ratio_at_obs_depth=MIN_GAP_RATIO_90,
                      obs_depth=OBS_DEPTH, min_argmin_round=MIN_ARGMIN_ROUND,
                      gap_ratio_definition=("gap(obs_depth) / gap(0) of the smoothing-spline lower-envelope "
                                            "profile of src.generators.envelope_profile (the real_boot "
                                            "construction)")),
        git_head=git_head(), created=_dt.datetime.now().isoformat(timespec="seconds"),
        all_passed=all_pass, datasets=prov,
    )
    json_path = os.path.join(args.out_dir, "real_boot_sources_provenance.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n  Saved: {csv_path}  ({len(df)} rounds x {len(curves)} sources)")
    print(f"  Saved: {json_path}")
    print(f"\n  ALL CRITERIA PASSED: {'YES' if all_pass else 'NO'}"
          + ("" if all_pass else "  -> src/generators.py stays on its current sources; report the numbers"))
    print("=" * 72)
    return 0 if all_pass else 2


if __name__ == "__main__":
    sys.exit(main())
