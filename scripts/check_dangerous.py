"""
check_dangerous.py
==================
Verify that the dangerous-method artifact matches what the committed Phase-1
output implies.

Redesign v2: the dangerous set lives in results/phase1/dangerous_methods.json
(written by scripts/derive_dangerous.py), not in a hard-coded constant.  This
guard re-derives the set from results/phase1/phase1_aggregated.csv and
compares it with the artifact, so a stale artifact after a Phase-1 re-run is
caught.

    python scripts/check_dangerous.py

Exit status is 0 when they match and 1 otherwise.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pandas as pd  # noqa: E402

from src.dangerous import artifact_path, derive_dangerous, load_dangerous  # noqa: E402

AGG_CSV = os.path.join("results", "phase1", "phase1_aggregated.csv")


def main() -> int:
    if not os.path.exists(AGG_CSV):
        print(f"ERROR: {AGG_CSV} not found. Run scripts/run_phase1.py first.")
        return 1
    try:
        declared = set(load_dangerous())
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 1

    observed, _ = derive_dangerous(pd.read_csv(AGG_CSV))
    observed = set(observed)

    print(f"Phase 1 output : {AGG_CSV}")
    print(f"artifact       : {artifact_path()}")
    print(f"declared in artifact          : {len(declared)} methods")
    print(f"re-derived from Phase 1 table : {len(observed)} methods")

    missing = observed - declared
    stale = declared - observed
    if not missing and not stale:
        print("\nMATCH: the dangerous-method artifact is up to date.")
        return 0

    print("\nMISMATCH: re-run  python scripts/derive_dangerous.py")
    if missing:
        print("  dangerous in Phase 1 but NOT in the artifact: " + ", ".join(sorted(missing)))
    if stale:
        print("  in the artifact but no longer dangerous:     " + ", ".join(sorted(stale)))
    return 1


if __name__ == "__main__":
    sys.exit(main())
