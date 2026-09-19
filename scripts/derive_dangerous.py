"""
derive_dangerous.py
===================
Pipeline step between Phase 1 and Phases 2-5: re-derive the dangerous-method
set from the Phase-1 output and write the artifact that later phases read.

    python scripts/derive_dangerous.py                       # results/phase1 -> artifact
    python scripts/derive_dangerous.py --phase1-dir DIR --out FILE

Criterion (src/dangerous.py): pooled stability S < 0 on the core regimes,
pooled over the gap strata and noise levels, capped cells excluded, oracle
excluded.  Exit status 0 on success, 1 when the Phase-1 table is missing.
"""

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pandas as pd  # noqa: E402

import src.config as CFG_MOD  # noqa: E402
from src.dangerous import derive_dangerous, write_artifact  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Re-derive the dangerous-method set from Phase 1")
    p.add_argument("--phase1-dir", default=os.path.join("results", "phase1"))
    p.add_argument("--out", default=None,
                   help=f"artifact path (default: config.DANGEROUS_ARTIFACT = "
                        f"{CFG_MOD.DANGEROUS_ARTIFACT})")
    args = p.parse_args()

    src_csv = os.path.join(args.phase1_dir, "phase1_aggregated.csv")
    if not os.path.exists(src_csv):
        print(f"ERROR: {src_csv} not found. Run scripts/run_phase1.py first.")
        return 1

    print("=" * 72)
    print("  DANGEROUS-METHOD RE-DERIVATION  (Phase 1 -> artifact for Phases 2-5)")
    print("=" * 72)
    df = pd.read_csv(src_csv)
    dangerous, table = derive_dangerous(df)
    out = write_artifact(dangerous, table, args.out, source=os.path.abspath(src_csv))

    print(f"  source     : {src_csv}  ({len(df)} aggregated rows)")
    print(f"  criterion  : S < 0, core regimes, pooled over strata "
          f"{CFG_MOD.HORIZON_GAP_FRACTIONS}, capped cells excluded, oracle excluded")
    print(f"  methods    : {len(table)} scored")
    print(f"  dangerous  : {len(dangerous)}  -> {sorted(dangerous)}")
    print(f"  legacy set : {len(CFG_MOD.LEGACY_DANGEROUS_METHODS)} "
          f"(not consumed; for the record)")
    newly = sorted(dangerous - CFG_MOD.LEGACY_DANGEROUS_METHODS)
    cleared = sorted(CFG_MOD.LEGACY_DANGEROUS_METHODS - dangerous)
    print(f"  vs legacy  : +{newly}  -{cleared}")
    print(f"  artifact   : {out}")
    print()
    print(f"  {'method':<22} {'S':>8} {'valid':>7} {'cat':>7} {'beats':>7} {'cells':>6}  flag")
    print("  " + "-" * 70)
    for _, r in table.iterrows():
        flag = "DANGEROUS" if r["dangerous"] else ""
        print(f"  {r['method']:<22} {r['stability']:>8.4f} {r['valid_rate']:>7.3f} "
              f"{r['cat_rate']:>7.3f} {r['beats_rate']:>7.3f} {int(r['n_cells']):>6}  {flag}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
