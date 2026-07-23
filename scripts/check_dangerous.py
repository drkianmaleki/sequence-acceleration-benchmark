"""
check_dangerous.py
==================
Verify that config.DANGEROUS_METHODS matches what Phase 1 actually produced.

DANGEROUS_METHODS is derived from Phase 1: a method is dangerous when its
pooled stability score is negative at one or more horizons, i.e. it fails
worse than making no prediction at all.  Because it is stored as a constant,
it can silently go stale after any change that alters Phase 1 results.

Run this after every Phase 1 re-run:

    python scripts/check_dangerous.py

Exit status is 0 when the constant matches Phase 1 output and 1 when it does
not, so it can also be used as a guard in a longer pipeline.
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.config as CFG_MOD  # noqa: E402

GLOBAL_CSV = os.path.join("results", "phase1", "phase1_global.csv")


def main() -> int:
    if not os.path.exists(GLOBAL_CSV):
        print(f"ERROR: {GLOBAL_CSV} not found. Run scripts/run_phase1.py first.")
        return 1

    df = pd.read_csv(GLOBAL_CSV)
    observed = set(df.loc[df["stability"] < 0.0, "method"].unique())
    declared = set(CFG_MOD.DANGEROUS_METHODS)

    print(f"Phase 1 output : {GLOBAL_CSV}")
    print(f"declared in config.DANGEROUS_METHODS : {len(declared)} methods")
    print(f"observed with stability < 0          : {len(observed)} methods")

    missing = observed - declared      # dangerous in fact, absent from constant
    stale = declared - observed        # listed as dangerous, no longer is

    if not missing and not stale:
        print("\nMATCH: config.DANGEROUS_METHODS is up to date.")
        return 0

    print("\nMISMATCH — update DANGEROUS_METHODS in src/config.py.")
    if missing:
        print("\n  dangerous in Phase 1 but NOT listed:")
        for m in sorted(missing):
            worst = df.loc[df["method"] == m, "stability"].min()
            print(f"    + {m:<20} worst stability = {worst:+.3f}")
    if stale:
        print("\n  listed but no longer dangerous:")
        for m in sorted(stale):
            worst = df.loc[df["method"] == m, "stability"].min()
            print(f"    - {m:<20} worst stability = {worst:+.3f}")

    print("\n  corrected set:\n")
    print("DANGEROUS_METHODS = frozenset({")
    for m in sorted(observed):
        print(f'    "{m}",')
    print("})")
    return 1


if __name__ == "__main__":
    sys.exit(main())
