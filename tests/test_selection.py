"""
tests/test_selection.py
=======================
The selection-by-trial analysis (scripts/derive_selection.py) on
a hand-made records frame with known answers for both designs: a tie
(registry order), an invalid pilot record, an invalid final record (not
valid, never a win), a cell with a capped seed (excluded), the default
chosen (no strict win), the many_pilots validity floor, the pooled table
following from the cell table, and two runs giving identical files.  The
shared definitions (scripts/selection_defs.py) are checked for being
import-safe and self-consistent.  The catastrophe rate of the chosen
method and of the default (cat_rate_chosen, cat_rate_default) against
hand-computed values in both designs -- a chosen record that is valid and
catastrophic (the Cat marker), one that is invalid, a trial without a
choice, the default catastrophic on one final seed, the pooled means -- and
the two internal checks raising on a disagreement.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import src.config as C                                                           # noqa: E402
from src.pipeline import ACCEL_METHODS                                           # noqa: E402
from scripts import selection_defs as D                                          # noqa: E402
from scripts import derive_selection as S                                        # noqa: E402

RF, SE, LV = D.DEFAULT_METHOD, D.CONSERVATIVE_METHOD, S.LAST
# two more accelerators for the hand-made pools: one before and one after the named pair in registry order
BEFORE = ACCEL_METHODS[0]
AFTER = ACCEL_METHODS[-1]
assert D.registry_index(BEFORE) < D.registry_index(SE) < D.registry_index(RF) < D.registry_index(AFTER)
POOLS = {"lead": [RF, SE, AFTER], "named": [RF, SE], "classical": [BEFORE], "all": [BEFORE, SE, RF, AFTER]}


class Cat(float):
    """A hand-made error whose record is valid AND carries the catastrophic flag;
    a plain float is valid and not catastrophic, None is invalid (hence catastrophic)."""


def _is_cat(err, valid):
    return int((not valid) or isinstance(err, Cat))


def _records(cells):
    """cells: {(hold, regime, noise, g): {method: [(error, valid), ...] per seed}} -> a records frame
    with every method's record on every seed (error NaN when invalid); the catastrophic flag
    as Phase 1 records it: 1 when invalid, and 1 when the error is a Cat marker."""
    rows = []
    for (hold, regime, noise, g), methods in cells.items():
        for m, recs in methods.items():
            for seed, (err, valid) in enumerate(recs):
                rows.append(dict(regime=regime, is_holdout=hold, noise=noise, seed=seed, target_g=g, capped=0,
                                 method=m, valid=valid, error=(float(err) if valid else float("nan")),
                                 catastrophic=_is_cat(err, valid)))
    return pd.DataFrame(rows)


def _agg(cells, capped=()):
    """The cell rows of phase1_aggregated.csv the script reads: the capped flag and, as Phase 1
    stores it, cat_rate = the mean of the catastrophic flags over the seeds, four decimals."""
    rows = []
    for (hold, regime, noise, g), methods in cells.items():
        for m, recs in methods.items():
            rows.append(dict(is_holdout=hold, regime=regime, noise=noise, target_g=g, method=m,
                             capped=int((hold, regime, noise, g) in capped),
                             cat_rate=round(float(np.mean([_is_cat(e, v) for e, v in recs])), 4)))
    return pd.DataFrame(rows)


def _cell(rows):
    """{method: [(error, valid)]} from {method: [error or None]} (None = invalid)."""
    return {m: [((e if e is not None else float("nan")), int(e is not None)) for e in errs] for m, errs in rows.items()}


def test_defs_are_import_safe_and_consistent():
    assert D.DEFAULT_METHOD == "rational_fit" and D.CONSERVATIVE_METHOD == "single_exp_fit"
    assert D.noise_class("single_exp", 0.0) == "noise-free"
    assert all(D.noise_class(r, 0.0) == "intrinsic" for r in D.INTRINSIC_NOISE)
    assert D.noise_class("single_exp", 0.005) == "sigma=0.005"
    assert [D.noise_class_order(k) for k in ("noise-free", "intrinsic", "sigma=0.001", "sigma=0.005", D.NOISE_CLASS_ALL)] == \
        sorted(D.noise_class_order(k) for k in ("sigma=0.005", D.NOISE_CLASS_ALL, "intrinsic", "sigma=0.001", "noise-free"))
    assert D.noise_class_label("sigma=0.001") == r"$\sigma = 0.001$" and D.noise_class_label("sigma=0.001", tex=False) == "sigma=0.001"
    assert D.CLASSICAL and all(m in ACCEL_METHODS for m in D.CLASSICAL)
    pools = D.candidate_pools([RF, SE])
    assert tuple(pools) == D.POOL_NAMES and pools["named"] == [RF, SE] and pools["all"] == list(ACCEL_METHODS)
    assert pools["classical"] == D.CLASSICAL


def test_one_pilot_tie_invalid_pilot_invalid_final_and_default_chosen():
    # three seeds; the pool [BEFORE, SE, RF, AFTER] (registry order)
    #   seed 0: BEFORE and SE tie at 1.0 -> SE? no: the tie goes to the earlier registry entry, BEFORE
    #   seed 1: BEFORE invalid, SE 0.5 lowest -> SE chosen when seed 1 is the pilot
    #   seed 2: RF lowest (0.1) -> RF chosen when seed 2 is the pilot (never a strict win vs the default)
    # and SE is INVALID on seed 2 (the chosen record stays invalid there: not valid, never a win)
    cell = _cell({BEFORE: [1.0, None, 2.0], SE: [1.0, 0.5, None], RF: [3.0, 0.7, 0.1], AFTER: [5.0, 5.0, 5.0], LV: [2.0, 2.0, 2.0]})
    cells = {(0, "single_exp", 0.0, 0.1): cell}
    rec, agg = _records(cells), _agg(cells)
    out = S.derive_cells(rec, agg, POOLS, designs=("one_pilot",))
    r = out[out.pool == "all"].iloc[0]
    assert r.n_seeds == 3 and r.n_trials == 6 and r.n_no_choice == 0
    # pilots: 0 -> BEFORE (tie, registry order); 1 -> SE; 2 -> RF.  Trials (pilot, final):
    #   (0,1): BEFORE on seed 1 invalid        -> not valid, no win
    #   (0,2): BEFORE on seed 2 = 2.0 vs RF 0.1, LV 2.0 -> valid, no win vs default, no win vs last (not strictly below)
    #   (1,0): SE on seed 0 = 1.0 vs RF 3.0, LV 2.0  -> win vs default, win vs last
    #   (1,2): SE on seed 2 invalid            -> not valid, no win
    #   (2,0): RF on seed 0 = 3.0 (default chosen) -> valid, never a strict win vs default; vs last 2.0: no
    #   (2,1): RF on seed 1 = 0.7 (default chosen) -> valid; vs last 2.0: win
    assert r.chosen_valid_rate == pytest.approx(4 / 6)
    assert r.win_vs_default == pytest.approx(1 / 6)
    assert r.default_chosen == pytest.approx(2 / 6)
    assert r.win_vs_last == pytest.approx(2 / 6)
    assert r.med_error_chosen == pytest.approx(np.median([2.0, 1.0, 3.0, 0.7]))
    # the default's and the last value's medians over the trials' final seeds (every seed twice) = the seed medians
    assert r.med_error_default == pytest.approx(np.median([3.0, 0.7, 0.1])) and r.med_error_last == pytest.approx(2.0)
    assert r.med_ratio_vs_default == pytest.approx(np.median([2.0 / 0.1, 1.0 / 3.0, 1.0, 1.0]))
    assert r.top_chosen == BEFORE and r.top_chosen_share == pytest.approx(2 / 6) and r.n_distinct_chosen == 3
    # the named pool: pilot 0 -> SE (1.0 < 3.0); pilot 1 -> SE (0.5 < 0.7); pilot 2 -> RF (SE invalid there)
    n = out[out.pool == "named"].iloc[0]
    assert n.top_chosen == SE and n.n_distinct_chosen == 2 and n.default_chosen == pytest.approx(2 / 6)
    # a one-member pool is always chosen; BEFORE is invalid on seed 1, so pilot 1 has no valid member: no choice
    c = out[out.pool == "classical"].iloc[0]
    assert c.n_no_choice == 2 and c.top_chosen == BEFORE and c.top_chosen_share == 1.0
    assert c.chosen_valid_rate == pytest.approx(2 / 6)       # (0,2) and (2,0) valid; (0,1), (2,1) invalid; pilot 1: none


def test_many_pilots_validity_floor_and_median_choice():
    # four seeds -> three pilots per final; floor RANK_MIN_VALID = 0.9 means every pilot record must be valid
    # SE has the lowest median but is invalid on seed 3: eligible only when seed 3 is the final run
    cell = _cell({SE: [0.1, 0.2, 0.3, None], RF: [1.0, 1.0, 1.0, 1.0], AFTER: [0.5, 0.6, 0.7, 0.8], LV: [2.0, 2.0, 2.0, 2.0]})
    cells = {(0, "two_exp", 0.001, 0.5): cell}
    pools = {"lead": [RF, SE, AFTER], "named": [RF, SE], "classical": [AFTER], "all": [SE, RF, AFTER]}
    out = S.derive_cells(_records(cells), _agg(cells), pools, designs=("many_pilots",), min_valid=float(C.RANK_MIN_VALID))
    r = out[out.pool == "all"].iloc[0]
    # finals 0, 1, 2: SE has 2 of 3 valid pilots (< 0.9) -> AFTER (median 0.65 .. 0.7 < RF 1.0); final 3: SE eligible, chosen, invalid there
    assert r.n_trials == 4 and r.n_no_choice == 0
    assert r.top_chosen == AFTER and r.top_chosen_share == pytest.approx(3 / 4) and r.n_distinct_chosen == 2
    assert r.chosen_valid_rate == pytest.approx(3 / 4) and r.win_vs_default == pytest.approx(3 / 4) and r.win_vs_last == pytest.approx(3 / 4)
    assert r.default_chosen == 0.0
    assert r.med_error_chosen == pytest.approx(np.median([0.5, 0.6, 0.7]))
    assert r.med_error_default == pytest.approx(1.0) and r.med_error_last == pytest.approx(2.0)
    # with a floor of 0.5, SE (2 of 3 valid, median 0.2 or 0.25 or 0.15) is eligible and chosen on every final
    out2 = S.derive_cells(_records(cells), _agg(cells), pools, designs=("many_pilots",), min_valid=0.5)
    r2 = out2[out2.pool == "all"].iloc[0]
    assert r2.top_chosen == SE and r2.top_chosen_share == 1.0 and r2.chosen_valid_rate == pytest.approx(3 / 4)
    # the named pool under the 0.9 floor: SE ineligible on finals 0-2 -> RF chosen (default chosen, never a strict win)
    n = out[out.pool == "named"].iloc[0]
    assert n.default_chosen == pytest.approx(3 / 4) and n.win_vs_default == 0.0 and n.top_chosen == RF


def test_capped_cell_is_excluded_and_rows_are_ordered():
    cell = _cell({SE: [0.1, 0.2], RF: [1.0, 1.0], LV: [2.0, 2.0]})
    cells = {(0, "single_exp", 0.0, 0.1): cell, (0, "two_exp", 0.0, 0.1): cell, (1, "stretched_exp", 0.0, 0.1): cell}
    pools = {"lead": [RF, SE], "named": [RF, SE], "classical": [SE], "all": [SE, RF]}
    out = S.derive_cells(_records(cells), _agg(cells, capped={(0, "two_exp", 0.0, 0.1)}), pools)
    assert set(out.regime) == {"single_exp", "stretched_exp"}          # the capped cell is excluded
    assert list(out.columns) == S.CELL_COLUMNS
    assert out.regime_set.tolist()[:len(out) // 2] == ["core"] * (len(out) // 2)
    assert out.pool.tolist()[:8] == [p for p in D.POOL_NAMES for _ in D.DESIGNS]
    assert out.design.tolist()[:2] == list(D.DESIGNS)
    assert len(out) == 2 * len(D.POOL_NAMES) * len(D.DESIGNS)


def test_pooled_table_follows_from_the_cell_table():
    rng = np.random.RandomState(0)
    regimes = [("single_exp", 0), ("two_exp", 0), ("noisy_plateau", 0), ("stretched_exp", 1)]
    cells = {}
    for reg, hold in regimes:
        for noise in (0.0, 0.001):
            cells[(hold, reg, noise, 0.1)] = _cell({
                SE: [float(x) if x < 0.9 else None for x in rng.uniform(0.0, 1.0, 5)],
                RF: list(rng.uniform(0.1, 1.0, 5)), AFTER: list(rng.uniform(0.1, 1.0, 5)), LV: list(rng.uniform(0.5, 1.0, 5))})
    pools = {"lead": [RF, SE, AFTER], "named": [RF, SE], "classical": [AFTER], "all": [SE, RF, AFTER]}
    out = S.derive_cells(_records(cells), _agg(cells), pools)
    glob = S.pooled(out)
    assert list(glob.columns) == S.GLOBAL_COLUMNS
    core = out[(out.regime_set == "core") & (out.pool == "all") & (out.design == "one_pilot")]
    for cls in ("noise-free", "intrinsic", "sigma=0.001", D.NOISE_CLASS_ALL):
        sub = core if cls == D.NOISE_CLASS_ALL else core[core.noise_class == cls]
        g = glob[(glob.regime_set == "core") & (glob.pool == "all") & (glob.design == "one_pilot") & (glob.noise_class == cls)].iloc[0]
        assert g.families == sub.regime.nunique() and g.cells == len(sub)
        for col in ("chosen_valid_rate", "win_vs_default", "default_chosen", "win_vs_last"):
            assert g[col] == pytest.approx(sub[col].mean())
        for col in ("med_error_chosen", "med_error_default", "med_error_last", "med_ratio_vs_default"):
            assert g[col] == pytest.approx(sub[col].median())
        fam = sub.groupby("regime")[["win_vs_default", "win_vs_last"]].mean()
        assert g.families_win_default_k == int((fam.win_vs_default > 0.5).sum()) and g.families_win_default_n == len(fam)
        assert g.families_win_last_k == int((fam.win_vs_last > 0.5).sum()) and g.families_win_last_n == len(fam)
    for cls in ("noise-free", "intrinsic", "sigma=0.001", D.NOISE_CLASS_ALL):        # the two appended means
        sub = core if cls == D.NOISE_CLASS_ALL else core[core.noise_class == cls]
        g = glob[(glob.regime_set == "core") & (glob.pool == "all") & (glob.design == "one_pilot") & (glob.noise_class == cls)].iloc[0]
        for col in ("cat_rate_chosen", "cat_rate_default"):
            assert g[col] == pytest.approx(sub[col].mean())
    assert set(glob.noise_class) == {"noise-free", "intrinsic", "sigma=0.001", D.NOISE_CLASS_ALL}
    assert glob[glob.regime_set == "holdout"].noise_class.tolist().count("intrinsic") == 0
    # the pooled medians of the default and the last value at 'all' equal the Phase-1-style pooled medians
    for hold, rs in ((0, "core"), (1, "holdout")):
        for m, col in ((RF, "med_error_default"), (LV, "med_error_last")):
            seeds = [np.median([e for e, v in cells[k][m] if v]) for k in cells if k[0] == hold]
            g = glob[(glob.regime_set == rs) & (glob.noise_class == D.NOISE_CLASS_ALL)]
            assert np.allclose(g[col], np.median(seeds))


def _write_tree(root, cells, pools_lead):
    """A minimal results tree for main(): records, aggregated, the two pooled tables, a manifest."""
    os.makedirs(os.path.join(root, "phase1"), exist_ok=True)
    rec = _records(cells)
    rec.to_csv(os.path.join(root, "phase1", "phase1_records.csv"), index=False)
    agg = _agg(cells)
    agg.to_csv(os.path.join(root, "phase1", "phase1_aggregated.csv"), index=False)
    for hold, name in ((0, "phase1_global.csv"), (1, "phase1_global_holdout.csv")):
        rows = []
        sub = rec[rec.is_holdout == hold]
        for g in sorted(sub.target_g.unique()):
            sg = sub[sub.target_g == g]
            n_fam, n_cells = sg.regime.nunique(), sg.drop_duplicates(["regime", "noise"]).shape[0]
            for m in sorted(sg.method.unique()):
                sm = sg[sg.method == m]
                med = np.median([np.median(c.error[c.valid == 1]) for _, c in sm.groupby(["regime", "noise"])])
                # the pooled cat_rate as Phase 1 stores it: the mean over the cells of the cell cat_rate, four decimals
                cat = round(float(sm.groupby(["regime", "noise"]).catastrophic.mean().mean()), 4)
                rows.append(dict(method=m, is_trivial=int(m == LV), rank_eligible=int(m != LV), target_g=g, med_error=med, cat_rate=cat,
                                 win_rate_vs_last=0.5, n_regimes=n_fam, n_cells=n_cells))
        pd.DataFrame(rows).to_csv(os.path.join(root, "phase1", name), index=False)
    with open(os.path.join(root, "run_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump({"versions": S._versions()}, fh)


def test_main_writes_the_three_files_and_two_runs_are_identical(tmp_path, monkeypatch):
    rng = np.random.RandomState(1)
    cells = {}
    for reg, hold in (("single_exp", 0), ("two_exp", 0), ("stretched_exp", 1)):
        for noise in (0.0, 0.005):
            for g in (0.5, 0.1):
                cells[(hold, reg, noise, g)] = _cell({
                    SE: list(rng.uniform(0.0, 1.0, 3)), RF: list(rng.uniform(0.1, 1.0, 3)), AFTER: list(rng.uniform(0.1, 1.0, 3)),
                    BEFORE: list(rng.uniform(0.1, 1.0, 3)), LV: list(rng.uniform(0.5, 1.0, 3))})
    # the pools the script derives: LEAD from the hand-made pooled tables (every accelerator present is rank-eligible),
    # classical = the registry's classical variants -- they must be in the records, so patch the roster to the four methods
    members = [BEFORE, SE, RF, AFTER]
    monkeypatch.setattr(S, "ACCEL_METHODS", members)
    monkeypatch.setattr(D, "CLASSICAL", [BEFORE])
    monkeypatch.setattr(D, "ACCEL_METHODS", members)
    monkeypatch.setattr(D, "REAL_EVAL_METHODS", [RF])          # LEAD always adds the recorded-curve methods
    outs = []
    for run in ("a", "b"):
        root = str(tmp_path / run)
        _write_tree(root, cells, members)
        assert S.main(["--results", root, "--extract", os.path.join(root, "extract.csv.gz")]) == 0
        files = sorted(os.listdir(os.path.join(root, "phase1")))
        assert {"phase1_selection_cells.csv", "phase1_selection_global.csv", "phase1_selection_provenance.json"} <= set(files)
        outs.append(root)
    for name in ("phase1_selection_cells.csv", "phase1_selection_global.csv"):
        a = open(os.path.join(outs[0], "phase1", name), "rb").read()
        b = open(os.path.join(outs[1], "phase1", name), "rb").read()
        assert a == b and b"\r\n" not in a
    pa = json.load(open(os.path.join(outs[0], "phase1", "phase1_selection_provenance.json"), encoding="utf-8"))
    pb = json.load(open(os.path.join(outs[1], "phase1", "phase1_selection_provenance.json"), encoding="utf-8"))
    for k in ("started", "finished", "seconds"):
        pa.pop(k), pb.pop(k)
    assert pa == pb
    assert pa["records_rows"] == 3 * 2 * 2 * 5 * 3 and set(pa["pools"]) == set(D.POOL_NAMES)
    assert pa["pools"]["named"] == [RF, SE] and pa["pools"]["all"] == members and pa["pools"]["classical"] == [BEFORE]
    assert pa["cells"] == {"one_pilot": 12, "many_pilots": 12} and pa["trials"] == {"one_pilot": 12 * 6, "many_pilots": 12 * 3}
    assert all(e["agree"] for e in pa["phase1_agreement"]) and len(pa["phase1_agreement"]) == 4
    assert all(e["agree"] for e in pa["phase1_cat_agreement"]) and len(pa["phase1_cat_agreement"]) == 4
    assert all(e["max_abs_diff_cells"] <= S.CAT_CHECK_TOL and e["max_abs_diff_pooled"] <= S.CAT_CHECK_TOL for e in pa["phase1_cat_agreement"])
    assert "cat_rate_chosen" in pa["catastrophe_rate"] and "CAT_MULT" in pa["catastrophe_rate"]
    assert list(pa)[list(pa).index("phase1_agreement") + 1:list(pa).index("phase1_agreement") + 3] == ["catastrophe_rate", "phase1_cat_agreement"]
    cells_csv = pd.read_csv(os.path.join(outs[0], "phase1", "phase1_selection_cells.csv"))
    glob_csv = pd.read_csv(os.path.join(outs[0], "phase1", "phase1_selection_global.csv"))
    assert list(cells_csv.columns)[-2:] == ["cat_rate_chosen", "cat_rate_default"] and list(glob_csv.columns)[-2:] == ["cat_rate_chosen", "cat_rate_default"]
    assert pa["versions_equal_run_manifest"] is True and "evaluates no method" in pa["note"]
    ext = pd.read_csv(os.path.join(outs[0], "extract.csv.gz"))
    assert list(ext.columns) == ["regime", "is_holdout", "noise", "seed", "method", "valid", "error"]
    assert len(ext) == 3 * 2 * 3 * 5 and set(ext.method) == set(members) | {LV}        # g = 0.1 cells only


def test_internal_check_fails_on_a_disagreement():
    cell = _cell({SE: [0.1, 0.2], RF: [1.0, 3.0], LV: [2.0, 2.0]})
    cells = {(0, "single_exp", 0.0, 0.1): cell}
    pools = {"lead": [RF, SE], "named": [RF, SE], "classical": [SE], "all": [SE, RF]}
    glob = S.pooled(S.derive_cells(_records(cells), _agg(cells), pools))
    G = pd.DataFrame([dict(method=RF, target_g=0.1, med_error=2.0, n_regimes=1, n_cells=1),
                      dict(method=LV, target_g=0.1, med_error=2.0, n_regimes=1, n_cells=1)])
    H = G.iloc[0:0]
    ev = S.check_against_phase1(glob, G, H)
    assert len(ev) == 1 and ev[0]["agree"]
    with pytest.raises(AssertionError):
        S.check_against_phase1(glob, G.assign(med_error=[2.5, 2.0]), H)


# ── the catastrophe rate of the chosen method and of the default ───────
# cell A, three seeds, pool all = [BEFORE, SE, RF, AFTER] (registry order); the errors of the first test with
# the flags: BEFORE valid-and-catastrophic on seed 2, the default (RF) catastrophic on seed 0 only
CELL_A = {BEFORE: [1.0, None, Cat(2.0)], SE: [1.0, 0.5, None], RF: [Cat(3.0), 0.7, 0.1], AFTER: [5.0, 5.0, 5.0], LV: [2.0, 2.0, 2.0]}
# cell B, four seeds: SE invalid on seed 3, the default catastrophic on seed 1, AFTER valid-and-catastrophic on seed 2; BEFORE never valid
CELL_B = {BEFORE: [None, None, None, None], SE: [0.1, 0.2, 0.3, None], RF: [1.0, Cat(1.0), 1.0, 1.0], AFTER: [0.5, 0.6, Cat(0.7), 0.8],
          LV: [2.0, 2.0, 2.0, 2.0]}
CELLS_AB = {(0, "single_exp", 0.0, 0.1): _cell(CELL_A), (0, "two_exp", 0.0, 0.1): _cell(CELL_B)}


def test_cat_marker_is_a_valid_record_with_the_flag():
    rec = _records(CELLS_AB)
    r = rec[(rec.regime == "single_exp") & (rec.method == BEFORE)].sort_values("seed")
    assert r.valid.tolist() == [1, 0, 1] and r.catastrophic.tolist() == [0, 1, 1] and r.error.tolist()[2] == 2.0
    assert type(r.error.tolist()[2]) is float
    # every invalid record carries the flag (the script asserts the same on the real records)
    assert (rec[rec.valid == 0].catastrophic == 1).all()
    a = _agg(CELLS_AB)
    assert float(a[(a.regime == "single_exp") & (a.method == RF)].cat_rate.iloc[0]) == pytest.approx(round(1 / 3, 4))


def test_catastrophe_rate_one_pilot_by_hand():
    out = S.derive_cells(_records(CELLS_AB), _agg(CELLS_AB), POOLS, designs=("one_pilot",))
    A = out[(out.regime == "single_exp") & (out.pool == "all")].iloc[0]
    # cell A, pilots: 0 -> BEFORE (tie, registry order); 1 -> SE; 2 -> RF.  Chosen record on the final seed:
    #   (0,1) BEFORE invalid -> catastrophic;  (0,2) BEFORE valid AND flagged -> catastrophic
    #   (1,0) SE 1.0 -> no;                    (1,2) SE invalid -> catastrophic
    #   (2,0) RF flagged -> catastrophic;      (2,1) RF 0.7 -> no
    assert A.cat_rate_chosen == pytest.approx(4 / 6)
    # the default on the finals [1, 2, 0, 2, 0, 1]: flagged on seed 0 only -> 2 of 6
    assert A.cat_rate_default == pytest.approx(2 / 6)
    # the flags change no other statistic of the first test's cell (the errors are the same)
    assert A.chosen_valid_rate == pytest.approx(4 / 6) and A.win_vs_default == pytest.approx(1 / 6)
    assert A.default_chosen == pytest.approx(2 / 6) and A.win_vs_last == pytest.approx(2 / 6)
    assert A.med_error_chosen == pytest.approx(np.median([2.0, 1.0, 3.0, 0.7]))
    # the one-member pool [BEFORE]: pilot 1 has no valid member -> trials (1,0), (1,2) have no choice and count as catastrophic;
    #   (0,1) invalid, (0,2) flagged, (2,1) invalid -> catastrophic; (2,0) BEFORE 1.0 -> no: 5 of 6
    Ac = out[(out.regime == "single_exp") & (out.pool == "classical")].iloc[0]
    assert Ac.n_no_choice == 2 and Ac.cat_rate_chosen == pytest.approx(5 / 6) and Ac.cat_rate_default == pytest.approx(2 / 6)
    # the named pool [RF, SE]: pilots 0, 1 -> SE, pilot 2 -> RF: (0,1) SE 0.5 no; (0,2) SE invalid yes; (1,0) SE 1.0 no;
    #   (1,2) SE invalid yes; (2,0) RF flagged yes; (2,1) RF 0.7 no -> 3 of 6
    An = out[(out.regime == "single_exp") & (out.pool == "named")].iloc[0]
    assert An.cat_rate_chosen == pytest.approx(3 / 6)
    # cell B, pool all (BEFORE never valid): pilots 0, 1, 2 -> SE; pilot 3 -> AFTER (0.8; SE invalid there)
    #   SE on the finals of pilots 0-2: invalid on seed 3 (3 trials) -> catastrophic, the other 6 not;
    #   AFTER on finals 0, 1, 2: flagged on seed 2 -> 1 of 3: 4 of 12
    B = out[(out.regime == "two_exp") & (out.pool == "all")].iloc[0]
    assert B.n_trials == 12 and B.cat_rate_chosen == pytest.approx(4 / 12)
    # every seed is the final seed three times; the default flagged on seed 1 -> 3 of 12
    assert B.cat_rate_default == pytest.approx(3 / 12)
    # the pooled means over the two cells (class noise-free == all here)
    glob = S.pooled(out)
    for cls in ("noise-free", D.NOISE_CLASS_ALL):
        g = glob[(glob.regime_set == "core") & (glob.pool == "all") & (glob.design == "one_pilot") & (glob.noise_class == cls)].iloc[0]
        assert g.cells == 2 and g.cat_rate_chosen == pytest.approx((4 / 6 + 4 / 12) / 2) and g.cat_rate_default == pytest.approx((2 / 6 + 3 / 12) / 2)
    assert list(out.columns) == S.CELL_COLUMNS and list(glob.columns) == S.GLOBAL_COLUMNS


def test_catastrophe_rate_many_pilots_by_hand():
    out = S.derive_cells(_records(CELLS_AB), _agg(CELLS_AB), POOLS, designs=("many_pilots",), min_valid=float(C.RANK_MIN_VALID))
    # cell A (two pilots per final; the floor 0.9 needs both pilot records valid):
    #   final 0: pilots 1, 2: BEFORE and SE have one valid pilot -> ineligible; RF median 0.4 < AFTER 5 -> RF, flagged on seed 0 -> catastrophic
    #   final 1: pilots 0, 2: BEFORE median 1.5 < RF 1.55 -> BEFORE, invalid on seed 1 -> catastrophic
    #   final 2: pilots 0, 1: SE median 0.75 < RF 1.85 -> SE, invalid on seed 2 -> catastrophic
    A = out[(out.regime == "single_exp") & (out.pool == "all")].iloc[0]
    assert A.n_trials == 3 and A.n_no_choice == 0 and A.cat_rate_chosen == pytest.approx(1.0)
    assert A.cat_rate_default == pytest.approx(1 / 3) and A.chosen_valid_rate == pytest.approx(1 / 3) and A.default_chosen == pytest.approx(1 / 3)
    # cell B (three pilots per final): finals 0, 1, 2 -> AFTER (SE has 2 of 3 valid pilots, below the floor): 0.5, 0.6 not flagged,
    #   seed 2 flagged -> catastrophic; final 3: SE eligible (3 of 3) and chosen, invalid there -> catastrophic: 2 of 4
    B = out[(out.regime == "two_exp") & (out.pool == "all")].iloc[0]
    assert B.n_trials == 4 and B.top_chosen == AFTER and B.cat_rate_chosen == pytest.approx(2 / 4)
    assert B.cat_rate_default == pytest.approx(1 / 4)          # the default flagged on seed 1, every seed final once
    glob = S.pooled(out)
    g = glob[(glob.regime_set == "core") & (glob.pool == "all") & (glob.design == "many_pilots") & (glob.noise_class == D.NOISE_CLASS_ALL)].iloc[0]
    assert g.cat_rate_chosen == pytest.approx((1.0 + 2 / 4) / 2) and g.cat_rate_default == pytest.approx((1 / 3 + 1 / 4) / 2)
    # a cell without a trial (one seed): both rates are NaN
    one = {(0, "single_exp", 0.0, 0.1): _cell({SE: [0.1], RF: [1.0], LV: [2.0]})}
    o = S.derive_cells(_records(one), _agg(one), {"lead": [RF, SE], "named": [RF, SE], "classical": [SE], "all": [SE, RF]}, designs=("many_pilots",))
    assert o.n_trials.tolist() == [0] * 4 and o.cat_rate_chosen.isna().all() and o.cat_rate_default.isna().all()


def test_cat_internal_checks_agree_and_raise_on_a_disagreement():
    cells = S.derive_cells(_records(CELLS_AB), _agg(CELLS_AB), POOLS)
    glob = S.pooled(cells)
    agg = _agg(CELLS_AB)
    # the pooled Phase 1 cat_rate of the default as Phase 1 stores it: the mean of the cell rates (1/3 and 1/4), four decimals
    G = pd.DataFrame([dict(method=RF, target_g=0.1, cat_rate=round((1 / 3 + 1 / 4) / 2, 4), med_error=1.0, n_regimes=2, n_cells=2)])
    H = G.iloc[0:0]
    ev = S.check_cat_against_phase1(cells, glob, agg, G, H)
    assert len(ev) == 1 and ev[0]["agree"] and ev[0]["cells"] == len(cells) and ev[0]["rows"] == len(D.POOL_NAMES) * len(D.DESIGNS)
    assert ev[0]["max_abs_diff_cells"] <= S.CAT_CHECK_TOL and ev[0]["max_abs_diff_pooled"] <= S.CAT_CHECK_TOL
    assert ev[0]["max_abs_diff_cells"] == pytest.approx(abs(1 / 3 - round(1 / 3, 4)), abs=1e-12)        # the four-decimal rounding of 1/3
    # the per-cell check: the default's cat_rate of one cell off by one unit in the third decimal
    bad = agg.copy()
    bad.loc[(bad.regime == "two_exp") & (bad.method == RF), "cat_rate"] = 0.251
    with pytest.raises(AssertionError, match="disagrees"):
        S.check_cat_against_phase1(cells, glob, bad, G, H)
    # a cell without a Phase 1 row for the default
    with pytest.raises(AssertionError, match="without a Phase 1 row"):
        S.check_cat_against_phase1(cells, glob, agg[agg.regime != "two_exp"], G, H)
    # the pooled check: the Phase 1 pooled value off by one unit in the fourth decimal
    with pytest.raises(AssertionError, match="pooled"):
        S.check_cat_against_phase1(cells, glob, agg, G.assign(cat_rate=[G.cat_rate.iloc[0] + 0.0001]), H)
    # main() exits 2 and writes nothing when the committed aggregate disagrees (the pooled table doctored)


def test_main_exits_2_and_writes_nothing_on_a_cat_disagreement(tmp_path, monkeypatch):
    members = [BEFORE, SE, RF, AFTER]
    monkeypatch.setattr(S, "ACCEL_METHODS", members)
    monkeypatch.setattr(D, "CLASSICAL", [BEFORE])
    monkeypatch.setattr(D, "ACCEL_METHODS", members)
    monkeypatch.setattr(D, "REAL_EVAL_METHODS", [RF])
    cells = dict(CELLS_AB)
    cells[(1, "stretched_exp", 0.0, 0.1)] = _cell(CELL_B)     # a held-out cell, so that both pooled tables have rows
    good = str(tmp_path / "good")
    _write_tree(good, cells, members)
    assert S.main(["--results", good]) == 0                   # the hand-made tree agrees as it is
    prov = json.load(open(os.path.join(good, "phase1", "phase1_selection_provenance.json"), encoding="utf-8"))
    assert [e["agree"] for e in prov["phase1_cat_agreement"]] == [True, True]
    assert [(e["regime_set"], e["cells"], e["rows"]) for e in prov["phase1_cat_agreement"]] == [("core", 16, 8), ("holdout", 8, 8)]
    root = str(tmp_path / "bad")
    _write_tree(root, cells, members)
    p = os.path.join(root, "phase1", "phase1_global.csv")
    G = pd.read_csv(p)
    G.loc[G.method == RF, "cat_rate"] = G.loc[G.method == RF, "cat_rate"] + 0.001
    G.to_csv(p, index=False)
    assert S.main(["--results", root]) == 2
    assert not any(f.startswith("phase1_selection") for f in os.listdir(os.path.join(root, "phase1")))
