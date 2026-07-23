"""
tests/test_accelerators.py
==========================
Unit tests for every accelerator in accelerators.py.

Four analytic test sequences with known exact limits are used.
Each has a specific difficulty type that matches the theoretical
strength of different method families.

Test cases
----------
GEOM      : s_n = 0.01 + 0.80 * 0.92^n          (geometric / single-exp)
HARMONIC  : s_n = 0.01 + 1.00 / (n+1)           (power-law, alpha=1)
LOGSLOW   : s_n = 0.01 + 0.80 / log(n + e)      (logarithmic convergence)
ALTGEOM   : s_n = 0.01 + 0.60 * (-0.85)^n       (alternating geometric)

All have true limit = 0.01.

Pass criterion
--------------
A method PASSES if:
    abs_error < EXACT_TOL  (exact result)
    OR improvement_ratio >= threshold

where improvement_ratio = |curr_err| / |estimate - true_limit|
and threshold differs per test case.

Output  (saved to ./results/)
------
phase0_unit_tests.csv   one row per (method, test_case), ~200 rows, ~30 KB
phase0_unit_tests.txt   human-readable PASS / fail / INVALID report
"""

import os
import sys
import math
import numpy as np
import pandas as pd

# Path setup: works when run as script or as module from project root
_THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_DIR = os.path.dirname(_THIS_DIR)
for _p in [_PACKAGE_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.accelerators import METHODS, METHOD_NAMES  # noqa: E402
import src.config as CFG_MOD                        # noqa: E402

# ── Configuration ──────────────────────────────────────────────────────────────

TRUE_LIMIT  = 0.01
OBS_IDX     = 90
FUTURE_X    = 5000
WINDOW_LEN  = 60
OUTPUT_DIR  = "results"
EXACT_TOL   = 1e-8    # abs_error below this = exact result = automatic PASS

PASS_THRESHOLDS = {
    "GEOM":     10.0,
    "HARMONIC":  5.0,
    "LOGSLOW":   2.0,
    "ALTGEOM":  20.0,
}

CFG = {
    "L_inf":          TRUE_LIMIT,
    "ridge":          CFG_MOD.RIDGE,
    "min_valid":      CFG_MOD.MIN_VALID,
    "max_valid":      CFG_MOD.MAX_VALID,
    "denom_tol":      CFG_MOD.DENOM_TOL,
    "win_shifts":     CFG_MOD.WIN_SHIFTS,
    "perturb_trials": CFG_MOD.PERTURB_TRIALS,
    "perturb_scale":  CFG_MOD.PERTURB_SCALE,
    "W_CAT":          CFG_MOD.W_CAT,
    "W_BEATS":        CFG_MOD.W_BEATS,
}

# ── Analytic sequences ─────────────────────────────────────────────────────────

def make_geom(n):
    return TRUE_LIMIT + 0.80 * (0.92 ** n)

def make_harmonic(n):
    return TRUE_LIMIT + 1.00 / (n + 1.0)

def make_logslow(n):
    return TRUE_LIMIT + 0.80 / np.log(n + np.e)

def make_altgeom(n):
    return TRUE_LIMIT + 0.60 * ((-0.85) ** n)

TEST_CASES = {
    "GEOM":     (make_geom,     "Geometric r=0.92 — algebraic methods should be exact"),
    "HARMONIC": (make_harmonic, "Harmonic (power-law alpha=1) — Richardson target"),
    "LOGSLOW":  (make_logslow,  "Log-slow — Brezinski-theta / Wynn-rho target"),
    "ALTGEOM":  (make_altgeom,  "Alternating geometric — Shanks/Aitken theoretically exact"),
}

# ── Runner ─────────────────────────────────────────────────────────────────────

def run_all_tests():
    n_arr   = np.arange(OBS_IDX + 1, dtype=float)
    records = []

    for test_name, (gen_fn, description) in TEST_CASES.items():
        seq_full  = gen_fn(n_arr)
        true_val  = float(TRUE_LIMIT)
        w_start   = max(0, OBS_IDX - WINDOW_LEN + 1)
        seq_win   = list(seq_full[w_start : OBS_IDX + 1])
        idx_win   = list(range(w_start, OBS_IDX + 1))
        curr_val  = float(seq_full[OBS_IDX])
        curr_err  = abs(curr_val - true_val)
        threshold = PASS_THRESHOLDS[test_name]

        for method_name in METHOD_NAMES:
            fn = METHODS[method_name]
            try:
                est = fn(seq_win, idx_win, float(FUTURE_X), CFG)
            except Exception:
                est = float("nan")

            valid = (math.isfinite(est)
                     and CFG["min_valid"] <= est <= CFG["max_valid"])
            err   = abs(est - true_val) if valid else float("nan")

            if valid and err > 1e-15:
                impv = curr_err / err
            elif valid and err <= 1e-15:
                impv = float("inf")
            else:
                impv = float("nan")

            # Pass: exact result OR above threshold
            if valid and err < EXACT_TOL:
                passed = True
            elif math.isfinite(impv):
                passed = bool(impv >= threshold)
            else:
                passed = False

            records.append({
                "test_case":     test_name,
                "description":   description,
                "method":        method_name,
                "true_limit":    true_val,
                "curr_val":      round(curr_val, 8),
                "curr_err":      round(curr_err, 8),
                "estimate":      round(est, 8) if valid else float("nan"),
                "abs_error":     round(err, 8) if valid else float("nan"),
                "improve_ratio": round(impv, 4) if math.isfinite(impv) else impv,
                "valid":         int(valid),
                "passed":        int(passed),
                "threshold":     threshold,
            })

    return pd.DataFrame(records)

# ── Report ─────────────────────────────────────────────────────────────────────

def _fmt_impv(v):
    if isinstance(v, float):
        if v == float("inf"):  return "exact"
        if math.isfinite(v):   return f"{v:.2f}x"
    return "—"

def write_report(df, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "phase0_unit_tests.csv")
    txt_path = os.path.join(out_dir, "phase0_unit_tests.txt")
    df.to_csv(csv_path, index=False)

    L = []
    L.append("=" * 72)
    L.append("  PHASE 0 UNIT TEST REPORT")
    L.append("  Sequence Acceleration — Validation on Known Analytic Sequences")
    L.append("=" * 72)

    for test_name, group in df.groupby("test_case", sort=False):
        desc      = group["description"].iloc[0]
        threshold = group["threshold"].iloc[0]
        n_pass    = int(group["passed"].sum())
        n_valid   = int(group["valid"].sum())
        n_total   = len(group)

        L.append(f"\n{'─'*72}")
        L.append(f"  {test_name}: {desc}")
        L.append(f"  Threshold: >= {threshold:.0f}x improvement  "
                 f"(or abs_error < {EXACT_TOL:.0e} = PASS)")
        L.append(f"  Pass: {n_pass}/{n_total}   Valid: {n_valid}/{n_total}")
        L.append(f"{'─'*72}")
        L.append(f"  {'Method':<24} {'Estimate':>12} {'AbsError':>12} "
                 f"{'Improve':>10} {'Result':>8}")
        L.append(f"  {'─'*24} {'─'*12} {'─'*12} {'─'*10} {'─'*8}")

        grp = group.sort_values(["passed", "improve_ratio"],
                                ascending=[False, False], na_position="last")
        for _, row in grp.iterrows():
            est_s  = f"{row['estimate']:.6f}" if row["valid"] else "INVALID"
            err_s  = f"{row['abs_error']:.2e}" if row["valid"] else "—"
            imp_s  = _fmt_impv(row["improve_ratio"])
            res_s  = ("PASS" if row["passed"]
                      else ("INVALID" if not row["valid"] else "fail"))
            L.append(f"  {row['method']:<24} {est_s:>12} {err_s:>12} "
                     f"{imp_s:>10} {res_s:>8}")

    # Global summary
    L.append(f"\n{'='*72}")
    L.append("  GLOBAL PASS SUMMARY  (✓ = PASS, · = fail or invalid)")
    L.append(f"  {'Method':<24} {'GEOM':>8} {'HARM':>8} "
             f"{'LOGS':>8} {'ALTG':>8} {'Total':>8}")
    L.append(f"  {'─'*24} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

    pivot = (df.pivot_table(index="method", columns="test_case",
                            values="passed", aggfunc="sum")
               .fillna(0).astype(int))
    for tc in ["GEOM", "HARMONIC", "LOGSLOW", "ALTGEOM"]:
        if tc not in pivot.columns:
            pivot[tc] = 0
    pivot["total"] = pivot[["GEOM", "HARMONIC", "LOGSLOW", "ALTGEOM"]].sum(axis=1)
    pivot = pivot.sort_values("total", ascending=False)

    for method, row in pivot.iterrows():
        g  = "✓" if row.get("GEOM", 0)     else "·"
        h  = "✓" if row.get("HARMONIC", 0) else "·"
        ls = "✓" if row.get("LOGSLOW", 0)  else "·"
        ag = "✓" if row.get("ALTGEOM", 0)  else "·"
        L.append(f"  {method:<24} {g:>8} {h:>8} {ls:>8} {ag:>8} "
                 f"{int(row['total']):>8}")

    # Implementation flags
    L.append(f"\n{'='*72}")
    L.append("  IMPLEMENTATION FLAGS")
    L.append(f"{'─'*72}")

    valid_max     = df.groupby("method")["valid"].max()
    always_inv    = sorted(valid_max[valid_max == 0].index.tolist())
    fail_all      = pivot[pivot["total"] == 0].index.tolist()
    valid_but_weak = sorted([m for m in fail_all if m not in always_inv])

    if always_inv:
        L.append("  ⚠  ALWAYS INVALID — likely implementation error, review these:")
        for m in always_inv:
            L.append(f"       {m}")
    else:
        L.append("  ✓  No method was always-invalid across all four tests.")

    L.append("")
    if valid_but_weak:
        L.append("  ℹ  VALID ESTIMATES but below threshold on all four tests.")
        L.append("     Check against theoretical expectations before flagging as broken.")
        for m in valid_but_weak:
            L.append(f"       {m}")
    else:
        L.append("  ✓  Every method that produced valid estimates passed >= 1 test.")

    L.append(f"{'='*72}\n")

    txt = "\n".join(L)
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write(txt)
    try:
        print(txt)
    except UnicodeEncodeError:
        # Legacy Windows consoles (cp1252) cannot render the report's
        # Unicode symbols; fall back to ASCII. Files on disk are UTF-8
        # and unaffected.
        print(txt.encode("ascii", errors="replace").decode("ascii"))
    print(f"Saved: {csv_path}")
    print(f"Saved: {txt_path}")


# ── pytest entry points ────────────────────────────────────────────────────────
# This module is primarily the Phase 0 runner, invoked by
# scripts/run_phase0_tests.py to produce the unit-test report.  The wrappers
# below also make it collectable by `pytest tests/`, which otherwise finds no
# test functions here and reports success without running anything.


def test_all_methods_are_exercised():
    """Every registered method must appear in the Phase 0 report."""
    df = run_all_tests()
    assert df["method"].nunique() == len(METHOD_NAMES)
    assert len(df) == len(METHOD_NAMES) * len(TEST_CASES)


def test_every_method_produces_at_least_one_valid_estimate():
    """A method that is never valid on any analytic case is broken, not merely weak."""
    df = run_all_tests()
    per_method = df.groupby("method")["valid"].max()
    dead = sorted(per_method[per_method == 0].index)
    assert not dead, f"methods never produced a valid estimate: {dead}"


def test_every_test_case_is_solved_by_someone():
    """Each analytic case must be passed by at least one method.

    If a case is passed by nobody, the case or its threshold is miscalibrated
    rather than every method being wrong.
    """
    df = run_all_tests()
    per_case = df.groupby("test_case")["passed"].sum()
    unsolved = sorted(per_case[per_case == 0].index)
    assert not unsolved, f"no method passed these cases: {unsolved}"


def test_baseline_is_not_reported_as_an_improvement():
    """current_value is the baseline; it cannot improve on itself."""
    df = run_all_tests()
    baseline = df[df["method"] == "current_value"]
    assert not baseline.empty
    assert baseline["passed"].sum() == 0


if __name__ == "__main__":
    print("=" * 72)
    print("  PHASE 0 — Unit Tests")
    print("=" * 72 + "\n")
    df = run_all_tests()
    write_report(df, OUTPUT_DIR)
