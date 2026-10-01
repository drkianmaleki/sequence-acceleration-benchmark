"""
compare_quick_runs.py  (development tool)
=========================================
Before/after regression check for code changes that must not alter what any
existing method computes.

    python scripts/dev/compare_quick_runs.py BEFORE_DIR AFTER_DIR
    python scripts/dev/compare_quick_runs.py --aggregate-only BEFORE_DIR AFTER_DIR

BEFORE_DIR and AFTER_DIR are two results/ trees produced by
``python reproduce_all.py --quick`` at two commits (including the git-ignored
raw per-record files).  With --aggregate-only the script compares nothing but
phase1/phase1_aggregated.csv med_error for the shared methods (exact,
round-trip floats) and reports the number of differing values and the largest
relative difference.  Otherwise the script

  * loads phase1/phase1_records.csv from both, keys the records on
    (regime, noise, seed, target_g, method), restricts to the methods present
    in both, and asserts that estimate, error, valid, catastrophic, n_f,
    capped, L_true and L_hat are identical (exact equality, NaN == NaN) on
    every shared record; it reports the number of shared methods and of
    records compared and every mismatch;
  * asserts, when the BEFORE file still has the retired beats_current column
    (a snapshot taken before Prompt R8a), that beats_current == win_vs_last on
    every record; a later snapshot skips this check;
  * does the same key-wise comparison for phase4/phase4_raw.csv and
    phase5a/phase5a_raw.csv on the shared methods' estimate, error and
    shift_iqr columns (keys include obs_idx), and reports separately how many
    rows differ in perturb_iqr: the perturbation diagnostic is paired per
    window since Prompt R8b Part B (one factor array per window shared by
    every method), so perturb_iqr is EXPECTED to differ from a snapshot taken
    before that change while estimate, error and shift_iqr are not;
  * compares phase5a/phase5a_ensemble.csv, phase5a_ensemble_holdout.csv and
    phase4/phase4_ensemble.csv selector by selector on mean_error, med_error
    and n_valid (a snapshot from before Prompt R8e Part C / R8f carries
    median_error and n, which are renamed first; rows of selectors without a
    valid record, which the older Phase 5a summary omitted, are reported, not
    counted as mismatches).  The selectors fall into three classes by what
    their formula (phases/phase5a.py::_compute_ensembles,
    phases/phase4.py::ensemble_comparison) reads (``selector_class``):
      - fixed: fixed name lists -- the trivial comparators, the fixed single
        methods, the Phase-2 cascade and the small-pool oracle / ensembles
        over src.pipeline.PHASE2_POOL; these must be identical;
      - roster/artifact-dependent: the whole accelerator roster or the
        roster minus the excluded set (oracle_<N>, equal_ensemble_<N>,
        *_safe); expected to differ across roster or artifact changes, which
        the script detects from the method sets of phase1_records.csv and
        the two dangerous_methods.json artifacts, and must be identical when
        neither changed;
      - diagnostic: reads perturb_iqr (diag_ensemble_*, capped_diag_*,
        threshold_ens_*); expected to differ from a snapshot taken before
        the pairing (R8b Part B) and listed with the number of values that
        changed; the roster-sized and *_safe members are also
        roster/artifact-dependent;
  * echoes phase0b/order_ladders_agreement.txt of the AFTER tree (the ladder
    script's own exact-agreement check against phase1_records.csv) and fails
    when it is missing or its verdict is not AGREE;
  * compares phase1/phase1_aggregated.csv on valid_rate, cat_rate, med_error,
    med_skill and win_rate_vs_last for the shared methods, and checks that the
    new panel columns of the AFTER table satisfy q25 <= med <= q75 <= p90 and
    n_valid <= n_total on every row;
  * lists the methods only in AFTER and only in BEFORE.

Exit status 0 when everything matches, 1 otherwise.  Any mismatch other than
the expected perturb_iqr / diagnostic-selector differences means a shared
method's behaviour changed and must be explained before the change is
accepted.
"""

import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.pipeline import PHASE2_POOL   # noqa: E402  (the small pool's size names the small-pool selectors)

P1_KEYS = ["regime", "noise", "seed", "target_g", "method"]
P1_COLS = ["estimate", "error", "valid", "catastrophic", "n_f", "capped", "L_true", "L_hat"]
RAW_KEYS = ["regime", "obs_idx", "noise", "seed", "target_g", "method"]
RAW_COLS = ["estimate", "error", "shift_iqr"]        # must be identical
RAW_EXPECTED_DIFF = ["perturb_iqr"]                   # paired since R8b Part B: expected to differ, reported only
ENS_KEYS = ["target_g", "regime_set", "selector"]
ENS_COLS = ["mean_error", "med_error", "n_valid"]
# Column names of the Phase 5a selector tables before Prompt R8e Part C (mean / median
# error and the count of valid records); a snapshot that still carries them is renamed
# to the descriptive-panel names before the comparison.
ENS_LEGACY = {"median_error": "med_error", "n": "n_valid"}
AGG_KEYS = ["method", "regime", "noise", "target_g"]
AGG_COLS = ["valid_rate", "cat_rate", "med_error", "med_skill", "win_rate_vs_last"]

PROBLEMS = []


def _read(path: str) -> pd.DataFrame:
    """CSV read that gives back exactly the floats the pipeline wrote."""
    return pd.read_csv(path, float_precision="round_trip")


def problem(msg: str):
    PROBLEMS.append(msg)
    print(f"  MISMATCH: {msg}")


def _equal(a: pd.Series, b: pd.Series) -> np.ndarray:
    """Element-wise exact equality with NaN == NaN (numeric or object)."""
    if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
        av, bv = a.to_numpy(dtype=float), b.to_numpy(dtype=float)
        return (av == bv) | (np.isnan(av) & np.isnan(bv))
    return (a.astype(str) == b.astype(str)).to_numpy()


FIXED_SELECTORS = {"fixed_rational", "fixed_richardson", "phase2_cascade",
                   "constant_assumed", "constant_oracle", "window_mean", "window_min", "last_value",
                   # Phase 4's table: the oracle over the fixed pool and the two-method proxy
                   "oracle", "phase2_proxy"}
P4_ENS_KEYS = ["target_g", "selector"]
SELECTOR_TABLES = (("phase5a", "phase5a_ensemble.csv", ENS_KEYS),
                   ("phase5a", "phase5a_ensemble_holdout.csv", ENS_KEYS),
                   ("phase4", "phase4_ensemble.csv", P4_ENS_KEYS))
_SIZED = re.compile(r"(oracle|equal_ensemble|diag_ensemble|capped_diag)_(\d+)")


def selector_class(name: str) -> str:
    """
    What a Phase 5a selector's formula reads (phases/phase5a.py::_compute_ensembles):
    'fixed' (fixed name lists: the trivials, the fixed single methods, the cascade,
    the small-pool oracle / ensembles over PHASE2_POOL), 'roster' (the whole roster
    or the roster minus the excluded set: oracle_<N>, equal_ensemble_<N>, *_safe),
    'diagnostic' (reads perturb_iqr: diag_ensemble_*, capped_diag_*, threshold_ens_*),
    or 'diagnostic, roster' for the diagnostic selectors over the roster / the
    safe set / the excluded-set-filtered pool (threshold_ens_*).
    """
    m = _SIZED.fullmatch(name)
    small = m is not None and int(m.group(2)) == len(PHASE2_POOL)
    diag = "diag" in name or name.startswith("threshold_ens")
    roster = name.endswith("_safe") or name.startswith("threshold_ens") or (m is not None and not small)
    if diag:
        return "diagnostic, roster" if roster else "diagnostic"
    if roster:
        return "roster"
    if name in FIXED_SELECTORS or small:
        return "fixed"
    return "roster"          # an unknown name is treated conservatively as roster-dependent


def compare_expected_diff(name: str, before: pd.DataFrame, after: pd.DataFrame,
                          keys: list, cols: list) -> None:
    """Report (never fail) how many shared records differ in the columns that the
    paired perturbation is expected to change."""
    shared = sorted(set(before["method"]) & set(after["method"]))
    b = before[before["method"].isin(shared)]
    a = after[after["method"].isin(shared)]
    merged = b.merge(a, on=keys, how="inner", suffixes=("_b", "_a"))
    for c in cols:
        if c + "_b" not in merged or c + "_a" not in merged:
            print(f"  {name}: column {c} absent in one of the files (nothing to report)")
            continue
        ok = _equal(merged[c + "_b"], merged[c + "_a"])
        both_finite = np.isfinite(merged[c + "_b"].to_numpy(dtype=float)) & np.isfinite(merged[c + "_a"].to_numpy(dtype=float))
        print(f"  {name}: {c} differs on {int((~ok).sum()):,} of {len(merged):,} shared records "
              f"({int((~ok & both_finite).sum()):,} with both values finite) -- EXPECTED: the perturbation "
              f"diagnostic is paired per window since R8b Part B; estimate / error / shift_iqr are checked above")


def _ens_frame(path: str, fname: str) -> pd.DataFrame:
    """A Phase 5a selector table with the descriptive-panel column names (ENS_LEGACY renamed)."""
    df = _read(path)
    ren = {old: new for old, new in ENS_LEGACY.items() if old in df.columns and new not in df.columns}
    if ren:
        print(f"  {fname}: {os.path.dirname(path)} carries the pre-R8e columns {sorted(ren)}; compared as {sorted(ren.values())}")
    return df.rename(columns=ren)


def _artifact_set(results_dir: str):
    """The excluded set of a results tree, or None when its artifact is absent / unreadable."""
    p = os.path.join(results_dir, "phase1", "dangerous_methods.json")
    try:
        with open(p, encoding="utf-8") as fh:
            return frozenset(json.load(fh)["dangerous_methods"])
    except Exception:
        return None


def compare_selectors(before_dir: str, after_dir: str, roster_same: bool, artifact_same) -> None:
    """
    Selector-level comparison of the Phase 5a ensemble tables.  ``roster_same``
    says whether the two trees evaluated the same method set, ``artifact_same``
    whether their excluded sets agree (None when an artifact is missing); the
    roster/artifact-dependent selectors must be identical only when both hold.
    """
    strict_roster = roster_same and artifact_same is True
    for sub_dir, fname, keys in SELECTOR_TABLES:
        pb, pa = os.path.join(before_dir, sub_dir, fname), os.path.join(after_dir, sub_dir, fname)
        if not (os.path.exists(pb) and os.path.exists(pa)):
            problem(f"{fname} missing in one of the trees")
            continue
        isel = keys.index("selector")
        b, a = _ens_frame(pb, fname), _ens_frame(pa, fname)
        shared = sorted(set(b["selector"]) & set(a["selector"]))
        classes = {s: selector_class(s) for s in shared}
        non_diag = [s for s in shared if classes[s] == "fixed"]
        roster_dep = [s for s in shared if classes[s] == "roster"]
        diag = [s for s in shared if classes[s].startswith("diagnostic")]
        only_after = sorted(set(a["selector"]) - set(b["selector"]))
        only_before = sorted(set(b["selector"]) - set(a["selector"]))
        merged = b.merge(a, on=keys, how="inner", suffixes=("_b", "_a"))
        kb = {k for k in map(tuple, b[keys].to_numpy()) if k[isel] in shared}
        ka = {k for k in map(tuple, a[keys].to_numpy()) if k[isel] in shared}
        # a selector without a valid record on a slice was omitted by the summary before R8e
        # Part C and keeps its row (n_valid == 0) since; such rows only in AFTER are expected
        omitted = set()
        if "n_valid" in a.columns:
            zero = a.set_index(keys)["n_valid"]
            omitted = {k for k in ka - kb if int(zero.loc[k]) == 0}
        if kb - ka or (ka - kb) - omitted:
            problem(f"{fname}: {tuple(keys)} key sets differ for the shared selectors "
                    f"({len(kb - ka)} only in BEFORE, {len((ka - kb) - omitted)} only in AFTER)")
        if omitted:
            print(f"  {fname}: {len(omitted)} row(s) only in AFTER with n_valid == 0 (selectors without a valid record on that "
                  f"slice, which the summary before R8e omitted): {sorted(omitted)}")
        def _n_diff(sel_list):
            counts = []
            for s in sel_list:
                ss = merged[merged["selector"] == s]
                n = sum(int((~_equal(ss[c + "_b"], ss[c + "_a"])).sum()) for c in ENS_COLS)
                counts.append((s, n, len(ss) * len(ENS_COLS)))
            return counts

        sub = merged[merged["selector"].isin(non_diag)]
        n_bad = 0
        for c in ENS_COLS:
            ok = _equal(sub[c + "_b"], sub[c + "_a"])
            if not ok.all():
                n_bad += int((~ok).sum())
                bad = sub.loc[~ok, keys + [c + "_b", c + "_a"]].head(5)
                problem(f"{fname}: {int((~ok).sum())} of {len(sub)} fixed-list selector rows differ in {c}; first rows:\n{bad.to_string()}")
        print(f"  {fname}: {len(non_diag)} fixed-list selectors, "
              f"{len(sub):,} rows compared on {ENS_COLS}: {'IDENTICAL' if n_bad == 0 else f'{n_bad} differing values'}")
        print(f"    fixed-list (must be identical): {non_diag}")
        rdep = _n_diff(roster_dep)
        tag = ("must be identical: same roster and excluded set" if strict_roster
               else "EXPECTED to differ across roster or artifact changes"
                    + ("" if artifact_same is not None else " (an artifact is missing)"))
        print(f"    roster/artifact-dependent ({tag}): {[f'{s} ({n} of {m} values differ)' for s, n, m in rdep]}")
        if strict_roster:
            for s, n, m in rdep:
                if n:
                    problem(f"{fname}: roster/artifact-dependent selector {s} differs in {n} of {m} values although the roster and the excluded set are unchanged")
        print(f"    diagnostic-dependent, EXPECTED to differ after the pairing (R8b Part B): "
              f"{[f'{s} ({n} of {m} values differ; {classes[s]})' for s, n, m in _n_diff(diag)]}")
        print(f"    selectors only in AFTER : {only_after}")
        print(f"    selectors only in BEFORE: {only_before}")


def aggregate_only(before_dir: str, after_dir: str) -> int:
    """
    phase1/phase1_aggregated.csv med_error for the shared methods, exact
    equality on round-trip floats: the number of differing values and the
    largest relative difference among them.  Exit 0 when nothing differs.
    """
    print(f"BEFORE: {before_dir}\nAFTER : {after_dir}\n")
    ba = _read(os.path.join(before_dir, "phase1", "phase1_aggregated.csv"))
    aa = _read(os.path.join(after_dir, "phase1", "phase1_aggregated.csv"))
    shared = sorted(set(ba["method"]) & set(aa["method"]))
    merged = (ba[ba["method"].isin(shared)][AGG_KEYS + ["med_error"]]
              .merge(aa[aa["method"].isin(shared)][AGG_KEYS + ["med_error"]], on=AGG_KEYS, how="inner",
                     suffixes=("_b", "_a")))
    x, y = merged["med_error_b"].to_numpy(dtype=float), merged["med_error_a"].to_numpy(dtype=float)
    ok = (x == y) | (np.isnan(x) & np.isnan(y))
    n_diff = int((~ok).sum())
    rel = 0.0
    if n_diff:
        d = ~ok & np.isfinite(x) & np.isfinite(y)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.abs(x[d] - y[d]) / np.maximum(np.abs(x[d]), np.abs(y[d]))
        rel = float(np.nanmax(r)) if r.size else float("nan")
        bad = merged.loc[~ok].head(5)
        print(bad.to_string(index=False))
    print(f"phase1_aggregated med_error: {len(shared)} shared methods, {len(merged):,} values compared "
          f"(exact, round-trip floats): {n_diff} differ; largest relative difference {rel:.3e}")
    print(f"RESULT: {'IDENTICAL' if n_diff == 0 else f'{n_diff} differing med_error values'}")
    return 0 if n_diff == 0 else 1


def echo_agreement(after_dir: str) -> None:
    """Print the Phase 0b agreement file of the AFTER tree; fail if missing or not AGREE."""
    p = os.path.join(after_dir, "phase0b", "order_ladders_agreement.txt")
    if not os.path.exists(p):
        problem(f"{p} missing (Phase 0b did not run or did not write its agreement check)")
        return
    text = open(p, encoding="utf-8").read()
    print("  AFTER phase0b/order_ladders_agreement.txt:")
    for line in text.rstrip().splitlines():
        print("    | " + line)
    if "VERDICT: AGREE" not in text:
        problem("Phase 0b agreement verdict is not AGREE")


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
    b1 = _read(os.path.join(before_dir, "phase1", "phase1_records.csv"))
    a1 = _read(os.path.join(after_dir, "phase1", "phase1_records.csv"))
    compare_keyed("phase1_records", b1, a1, P1_KEYS, P1_COLS)
    if "beats_current" in b1.columns:
        # BEFORE predates Prompt R8a: the retired column must carry the same information as the kept one.
        if "win_vs_last" not in b1.columns:
            problem("BEFORE phase1_records has beats_current but lacks win_vs_last")
        else:
            same = (b1["beats_current"].to_numpy() == b1["win_vs_last"].to_numpy())
            if same.all():
                print(f"  BEFORE phase1_records: beats_current == win_vs_last on all {len(b1):,} records")
            else:
                problem(f"BEFORE phase1_records: beats_current != win_vs_last on {int((~same).sum())} records")
    else:
        print("  BEFORE phase1_records: no beats_current column (snapshot taken after Prompt R8a); retired-column check not applicable")
    if "win_vs_last" not in a1.columns:
        problem("AFTER phase1_records lacks win_vs_last")
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
        rb, ra = _read(pb), _read(pa)
        cols = [c for c in RAW_COLS if c in rb.columns and c in ra.columns]
        if len(cols) < len(RAW_COLS):
            print(f"  {fname}: columns {sorted(set(RAW_COLS) - set(cols))} absent in one of the files; comparing {cols}")
        compare_keyed(fname, rb, ra, RAW_KEYS, cols)
        compare_expected_diff(fname, rb, ra, RAW_KEYS, RAW_EXPECTED_DIFF)

    # ── Phase 5a selector tables ───────────────────────────────────────────────
    roster_same = set(b1["method"]) == set(a1["method"])
    art_b, art_a = _artifact_set(before_dir), _artifact_set(after_dir)
    artifact_same = (art_b == art_a) if (art_b is not None and art_a is not None) else None
    print(f"  roster {'unchanged' if roster_same else 'CHANGED'} between the trees; excluded set "
          f"{'unchanged' if artifact_same else ('CHANGED' if artifact_same is False else 'unknown (artifact missing)')}")
    compare_selectors(before_dir, after_dir, roster_same, artifact_same)

    # ── Phase 0b agreement check (AFTER tree) ──────────────────────────────────
    echo_agreement(after_dir)

    # ── Phase 1 aggregate ──────────────────────────────────────────────────────
    ba = _read(os.path.join(before_dir, "phase1", "phase1_aggregated.csv"))
    aa = _read(os.path.join(after_dir, "phase1", "phase1_aggregated.csv"))
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
    print("RESULT: every shared method is byte-for-byte identical on every compared column "
          "(perturb_iqr and the diagnostic-dependent selectors excepted, as expected after the pairing); "
          "the roster differs only by the listed methods")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Before/after regression check of two --quick results trees")
    ap.add_argument("before_dir")
    ap.add_argument("after_dir")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="compare only phase1_aggregated.csv med_error for the shared methods (exact, round-trip floats)")
    args = ap.parse_args()
    if args.aggregate_only:
        sys.exit(aggregate_only(args.before_dir, args.after_dir))
    sys.exit(main(args.before_dir, args.after_dir))
