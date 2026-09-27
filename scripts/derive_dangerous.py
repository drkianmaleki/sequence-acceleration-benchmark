"""
derive_dangerous.py
===================
Pipeline step between Phase 1 and Phases 2-5: re-derive the excluded-method
set ("dangerous" is its legacy implementation name) from the Phase-1 output
and write the artifact that later phases read.

    python scripts/derive_dangerous.py                       # results/phase1 -> artifact
    python scripts/derive_dangerous.py --phase1-dir DIR --out FILE

Criterion (src/dangerous.py): pooled valid_rate < config.RANK_MIN_VALID on
the core regimes at the Phase-1 depth, pooled over the gap strata and noise
levels with equal cell weights, capped cells excluded, oracle excluded.  Only
the accelerators are eligible; the trivial comparators are tabulated and
printed for the record but never written to the artifact.  Exit status 0 on
success, 1 when the Phase-1 table is missing.

The artifact's "source" field is the repo-relative path of the Phase-1 table
(results/phase1/phase1_aggregated.csv), so the committed JSON is machine-independent.
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
from src.dangerous import CRITERION, derive_dangerous, write_artifact  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Re-derive the excluded-method set from Phase 1")
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
    print("  EXCLUDED-METHOD RE-DERIVATION  (Phase 1 -> artifact for Phases 2-5)")
    print("  ('dangerous' is the legacy implementation name of the exclusion)")
    print("=" * 72)
    df = pd.read_csv(src_csv)
    dangerous, table = derive_dangerous(df)
    # Provenance: the artifact records its source as a repo-relative POSIX path
    # (the absolute path of the machine that ran the pipeline carries no
    # information for a fresh clone and differs between checkouts).
    source = os.path.relpath(os.path.abspath(src_csv), _ROOT).replace(os.sep, "/")
    out = write_artifact(dangerous, table, args.out, source=source)

    n_elig = int(table["eligible"].sum())
    n_triv = int((table["eligible"] == 0).sum())
    print(f"  source     : {src_csv}  ({len(df)} aggregated rows)")
    print(f"  criterion  : {CRITERION}")
    print(f"  floor      : RANK_MIN_VALID = {CFG_MOD.RANK_MIN_VALID}  "
          f"(strata {CFG_MOD.HORIZON_GAP_FRACTIONS} pooled with equal cell weights)")
    print(f"  methods    : {len(table)} tabulated = {n_elig} accelerators (eligible) "
          f"+ {n_triv} trivial comparators (tabulated for the record, never in the artifact)")
    print(f"  excluded   : {len(dangerous)}  -> {sorted(dangerous)}")
    print(f"  legacy set : {len(CFG_MOD.LEGACY_DANGEROUS_METHODS)} "
          f"(not consumed; for the record)")
    newly = sorted(dangerous - CFG_MOD.LEGACY_DANGEROUS_METHODS)
    cleared = sorted(CFG_MOD.LEGACY_DANGEROUS_METHODS - dangerous)
    print(f"  vs legacy  : +{newly}  -{cleared}")
    print(f"  artifact   : {out}")
    print()
    print(f"  {'method':<22} {'valid':>7} {'cat':>7} {'med_err':>10} {'cells':>6}  flag")
    print("  " + "-" * 72)
    for _, r in table.iterrows():
        if r["dangerous"]:
            flag = "EXCLUDED (valid_rate below the floor)"
        elif not r["eligible"]:
            flag = "trivial (tabulated only; not eligible, not in artifact)"
            if r["valid_rate"] < CFG_MOD.RANK_MIN_VALID:
                flag += "  [below the floor]"
        else:
            flag = ""
        print(f"  {r['method']:<22} {r['valid_rate']:>7.3f} {r['cat_rate']:>7.3f} "
              f"{r['med_error']:>10.5f} {int(r['n_cells']):>6}  {flag}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
