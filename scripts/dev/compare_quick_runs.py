"""
compare_quick_runs.py  (development tool)
=========================================
Before/after regression check for code changes that must not alter what any
existing method computes.

    python scripts/dev/compare_quick_runs.py BEFORE_DIR AFTER_DIR

BEFORE_DIR and AFTER_DIR are two results/ trees produced by
``python reproduce_all.py --quick`` at two commits (including the git-ignored
raw per-record files).  The script

  * loads phase1/phase1_records.csv from both, keys the records on
    (regime, noise, seed, target_g, method), restricts to the methods present
    in both, and asserts that estimate, error, valid, catastrophic, n_f,
    capped, L_true and L_hat are identical (exact equality, NaN == NaN) on
    every shared record; it reports the number of shared methods and of
    records compared and every mismatch;
  * asserts in the BEFORE file that beats_current == win_vs_last on every
    record (the retired column carried the same information as the kept one);
  * does the same key-wise comparison for phase4/phase4_raw.csv and
    phase5a/phase5a_raw.csv on the shared methods' estimate / error columns
    (keys include obs_idx);
  * compares phase1/phase1_aggregated.csv on valid_rate, cat_rate, med_error,
    med_skill and win_rate_vs_last for the shared methods, and checks that the
    new panel columns of the AFTER table satisfy q25 <= med <= q75 <= p90 and
    n_valid <= n_total on every row;
  * lists the methods only in AFTER and only in BEFORE.

Exit status 0 when everything matches, 1 otherwise.  Any mismatch means a
shared method's behaviour changed and must be explained before the change is
accepted.
"""

import os
import sys

import numpy as np
import pandas as pd

P1_KEYS = ["regime", "noise", "seed", "target_g", "method"]
P1_COLS = ["estimate", "error", "valid", "catastrophic", "n_f", "capped", "L_true", "L_hat"]
RAW_KEYS = ["regime", "obs_idx", "noise", "seed", "target_g", "method"]
RAW_COLS = ["estimate", "error"]
AGG_KEYS = ["method", "regime", "noise", "target_g"]
AGG_COLS = ["valid_rate", "cat_rate", "med_error", "med_skill", "win_rate_vs_last"]

PROBLEMS = []


def problem(msg: str):
    PROBLEMS.append(msg)
    print(f"  MISMATCH: {msg}")


def _equal(a: pd.Series, b: pd.Series) -> np.ndarray:
    """Element-wise exact equality with NaN == NaN (numeric or object)."""
    if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
        av, bv = a.to_numpy(dtype=float), b.to_numpy(dtype=float)
        return (av == bv) | (np.isnan(av) & np.isnan(bv))
    return (a.astype(str) == b.astype(str)).to_numpy()


def compare_keyed(name: str, before: pd.DataFrame, after: pd.DataFrame,
                  keys: list, cols: list) -> None:
    shared = sorted(set(before["method"]) & set(after["method"]))
    only_after = sorted(set(after["method"]) - set(before["method"]))
    only_before = sorted(set(before["method"]) - set(after["method"]))
    b = before[before["method"].isin(shared)]
    a = after[after["method"].isin(shared)]
    kb, ka = set(map(tuple, b[keys].to_numpy())), set(map(tuple, a[keys].to_numpy()))
    if kb != ka:
        problem(f"{name}: key sets differ for the shared methods "
                f"({len(kb - ka)} keys only in BEFORE, {len(ka - kb)} only in AFTER)")
    merged = b.merge(a, on=keys, how="inner", suffixes=("_b", "_a"))
    n_bad_total = 0
    for c in cols:
        if c + "_b" not in merged or c + "_a" not in merged:
            problem(f"{name}: column {c} missing in one of the files")
            continue
        ok = _equal(merged[c + "_b"], merged[c + "_a"])
        n_bad = int((~ok).sum())
        if n_bad:
            n_bad_total += n_bad
            bad = merged.loc[~ok, keys + [c + "_b", c + "_a"]].head(5)
            problem(f"{name}: {n_bad} of {len(merged)} records differ in {c}; first rows:\n{bad.to_string()}")
    print(f"  {name}: {len(shared)} shared methods, {len(merged):,} records compared on {cols}: "
          f"{'IDENTICAL' if n_bad_total == 0 and kb == ka else f'{n_bad_total} differing values'}")
    print(f"    methods only in AFTER : {only_after}")
    print(f"    methods only in BEFORE: {only_before}")


def main(before_dir: str, after_dir: str) -> int:
    print(f"BEFORE: {before_dir}\nAFTER : {after_dir}\n")

    # ── Phase 1 records ────────────────────────────────────────────────────────
    b1 = pd.read_csv(os.path.join(before_dir, "phase1", "phase1_records.csv"))
    a1 = pd.read_csv(os.path.join(after_dir, "phase1", "phase1_records.csv"))
    compare_keyed("phase1_records", b1, a1, P1_KEYS, P1_COLS)
    if "beats_current" in b1.columns and "win_vs_last" in b1.columns:
        same = (b1["beats_current"].to_numpy() == b1["win_vs_last"].to_numpy())
        if same.all():
            print(f"  BEFORE phase1_records: beats_current == win_vs_last on all {len(b1):,} records")
        else:
            problem(f"BEFORE phase1_records: beats_current != win_vs_last on {int((~same).sum())} records")
    else:
        problem("BEFORE phase1_records lacks beats_current or win_vs_last")
    if "E_last" in a1.columns:
        last = a1[a1["method"] == "last_value"].set_index([k for k in P1_KEYS if k != "method"])["error"]
        chk = a1.set_index([k for k in P1_KEYS if k != "method"])
        el = chk["E_last"].to_numpy(dtype=float)
        ref = last.reindex(chk.index).to_numpy(dtype=float)
        ok = (el == ref) | (np.isnan(el) & np.isnan(ref))
        print(f"  AFTER phase1_records: E_last equals last_value's error on {int(ok.sum()):,} of {len(ok):,} records"
              + ("" if ok.all() else "  <-- MISMATCH"))
        if not ok.all():
            problem("AFTER phase1_records: E_last differs from last_value's error")

    # ── Phase 4 and Phase 5a raw ───────────────────────────────────────────────
    for sub, fname in (("phase4", "phase4_raw.csv"), ("phase5a", "phase5a_raw.csv")):
        pb, pa = os.path.join(before_dir, sub, fname), os.path.join(after_dir, sub, fname)
        if not (os.path.exists(pb) and os.path.exists(pa)):
            problem(f"{fname} missing in one of the trees")
            continue
        compare_keyed(fname, pd.read_csv(pb), pd.read_csv(pa), RAW_KEYS, RAW_COLS)

    # ── Phase 1 aggregate ──────────────────────────────────────────────────────
    ba = pd.read_csv(os.path.join(before_dir, "phase1", "phase1_aggregated.csv"))
    aa = pd.read_csv(os.path.join(after_dir, "phase1", "phase1_aggregated.csv"))
    compare_keyed("phase1_aggregated", ba, aa, AGG_KEYS, AGG_COLS)
    panel = ["n_total", "n_valid", "mean_error", "sd_error", "med_error", "q25_error", "q75_error", "p90_error"]
    missing = [c for c in panel if c not in aa.columns]
    if missing:
        problem(f"AFTER phase1_aggregated lacks panel columns {missing}")
    else:
        ok_rows = aa[aa["n_valid"] > 0]
        eps = 1e-12
        order_ok = ((ok_rows["q25_error"] <= ok_rows["med_error"] + eps)
                    & (ok_rows["med_error"] <= ok_rows["q75_error"] + eps)
                    & (ok_rows["q75_error"] <= ok_rows["p90_error"] + eps))
        count_ok = (aa["n_valid"] <= aa["n_total"])
        nan_ok = aa.loc[aa["n_valid"] == 0, ["mean_error", "med_error", "q25_error", "q75_error", "p90_error"]].isna().all(axis=None)
        print(f"  AFTER phase1_aggregated: q25 <= med <= q75 <= p90 on {int(order_ok.sum()):,} of {len(ok_rows):,} rows with valid records; "
              f"n_valid <= n_total on {int(count_ok.sum()):,} of {len(aa):,} rows; "
              f"conditional fields NaN on every row without valid records: {bool(nan_ok)}")
        if not (order_ok.all() and count_ok.all() and nan_ok):
            problem("AFTER phase1_aggregated: panel-column consistency check failed")
        gone = [c for c in ("stability", "beats_rate") if c in aa.columns]
        if gone:
            problem(f"AFTER phase1_aggregated still carries {gone}")

    print()
    if PROBLEMS:
        print(f"RESULT: {len(PROBLEMS)} problem(s) -- a shared method's behaviour changed or a schema check failed")
        return 1
    print("RESULT: every shared method is byte-for-byte identical on every compared column; "
          "the roster differs only by the listed methods")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
