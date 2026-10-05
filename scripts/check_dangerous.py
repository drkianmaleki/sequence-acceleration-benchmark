"""
check_dangerous.py
==================
Verify that the excluded-method artifact matches what the committed Phase-1
output implies ("dangerous" is the legacy implementation name of the set).

Redesign v2: the excluded set lives in results/phase1/dangerous_methods.json
(written by scripts/derive_dangerous.py), not in a hard-coded constant.  This
guard re-derives the set from phase1_aggregated.csv under the validity
criterion (pooled valid_rate < config.RANK_MIN_VALID on the core regimes,
capped cells excluded, oracle excluded, accelerators only) and compares it
with the artifact, so a stale artifact after a Phase-1 re-run is caught.  It
also verifies that the artifact declares that criterion and floor.

    python scripts/check_dangerous.py
    python scripts/check_dangerous.py --phase1-dir DIR      # a results snapshot elsewhere

Exit status is 0 when they match and 1 otherwise.
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
from src.dangerous import (CRITERION, SCHEMA, TABLE_DECIMALS, artifact_path,  # noqa: E402
                           derive_dangerous, load_artifact)


def main() -> int:
    p = argparse.ArgumentParser(description="Verify the excluded-method artifact against Phase 1")
    p.add_argument("--phase1-dir", default=os.path.join("results", "phase1"),
                   help="directory holding phase1_aggregated.csv and dangerous_methods.json")
    args = p.parse_args()
    agg_csv = os.path.join(args.phase1_dir, "phase1_aggregated.csv")
    art = os.path.join(args.phase1_dir, os.path.basename(CFG_MOD.DANGEROUS_ARTIFACT))

    if not os.path.exists(agg_csv):
        print(f"ERROR: {agg_csv} not found. Run scripts/run_phase1.py first.")
        return 1
    try:
        payload = load_artifact(art)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1
    declared = set(payload["dangerous_methods"])

    observed, table = derive_dangerous(pd.read_csv(agg_csv))
    observed = set(observed)

    print(f"Phase 1 output : {agg_csv}")
    print(f"artifact       : {artifact_path(art)}")
    print(f"schema         : {payload.get('schema')}")
    print(f"criterion      : {payload.get('criterion')}")
    print(f"floor          : rank_min_valid = {payload.get('rank_min_valid')}  "
          f"(config.RANK_MIN_VALID = {CFG_MOD.RANK_MIN_VALID})")
    print(f"declared in artifact          : {len(declared)} methods")
    print(f"re-derived from Phase 1 table : {len(observed)} methods")

    problems = []
    if payload.get("schema") != SCHEMA:
        problems.append(f"schema is {payload.get('schema')!r}, expected {SCHEMA!r}")
    if payload.get("criterion") != CRITERION:
        problems.append("the artifact's criterion text differs from src.dangerous.CRITERION")
    if payload.get("rank_min_valid") != float(CFG_MOD.RANK_MIN_VALID):
        problems.append(f"the artifact's floor {payload.get('rank_min_valid')} differs from "
                        f"config.RANK_MIN_VALID = {CFG_MOD.RANK_MIN_VALID}")
    flagged_rows = {r["method"] for r in payload.get("table", []) if r.get("dangerous") == 1}
    if flagged_rows != declared:
        problems.append("the artifact's table flags differ from its dangerous_methods list")
    # The stored valid_rate carries TABLE_DECIMALS decimals while the flag is decided on the
    # unrounded mean, so a rate within half a unit of the last stored decimal of the floor
    # cannot be judged from the table; every other row must agree with its flag.
    elig = table[table["eligible"] == 1]
    floor = float(CFG_MOD.RANK_MIN_VALID)
    tol = 0.5 * 10 ** (-TABLE_DECIMALS)
    surely_below = set(elig.loc[elig["valid_rate"] < floor - tol, "method"])
    surely_above = set(elig.loc[elig["valid_rate"] > floor + tol, "method"])
    if not surely_below <= observed or (surely_above & observed):
        problems.append("re-derived flags do not equal 'eligible and valid_rate below the floor' "
                        "outside the rounding band of the stored table")
    band = sorted(set(elig["method"]) - surely_below - surely_above)
    if band:
        print(f"note: {band} within {tol:g} of the floor in the stored table; the flag was decided on the unrounded mean")

    missing = observed - declared
    stale = declared - observed
    if not missing and not stale and not problems:
        print("\nMATCH: the excluded-method artifact is up to date and declares the validity criterion.")
        return 0

    print("\nMISMATCH: re-run  python scripts/derive_dangerous.py")
    if missing:
        print("  excluded per Phase 1 but NOT in the artifact: " + ", ".join(sorted(missing)))
    if stale:
        print("  in the artifact but no longer excluded:      " + ", ".join(sorted(stale)))
    for q in problems:
        print("  " + q)
    return 1


if __name__ == "__main__":
    sys.exit(main())
