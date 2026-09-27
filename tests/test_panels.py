"""
tests/test_panels.py
====================
The descriptive error panel, the rule panel and the zero-denominator mask
(src/panels.py) on hand-built records.
"""

import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from src.panels import (CONDITIONAL_COLS, PANEL_COLS, RULE_COLS, error_panel,  # noqa: E402
                        rule_panel, threshold_rule, zero_denominator_flags)
from src.trivial import SKILL_EPS, skill_score  # noqa: E402

nan = float("nan")


def test_error_panel_on_hand_built_records():
    # six records: four valid (errors 1, 2, 3, 10), two invalid (NaN error);
    # the 10 is catastrophic (> 5 x E_last = 1); invalid records are catastrophic
    errors = [1.0, 2.0, 3.0, 10.0, nan, nan]
    valid = [1, 1, 1, 1, 0, 0]
    cat = [0, 0, 0, 1, 1, 1]
    e_last = [1.5, 1.5, 1.5, 1.0, 1.0, nan]
    p = error_panel(errors, valid, catastrophic=cat, last_errors=e_last)
    assert set(p) == set(PANEL_COLS)
    assert p["n_total"] == 6 and p["n_valid"] == 4
    assert p["valid_rate"] == pytest.approx(4 / 6)
    assert p["cat_rate"] == pytest.approx(3 / 6)
    # conditional on validity: the invalid records do not enter
    assert p["mean_error"] == pytest.approx(4.0)
    assert p["sd_error"] == pytest.approx(np.std([1, 2, 3, 10], ddof=1))
    assert p["med_error"] == pytest.approx(2.5)
    assert p["q25_error"] == pytest.approx(np.percentile([1, 2, 3, 10], 25))
    assert p["q75_error"] == pytest.approx(np.percentile([1, 2, 3, 10], 75))
    assert p["p90_error"] == pytest.approx(np.percentile([1, 2, 3, 10], 90))
    # win rate over ALL six records: only the record with error 1.0 < 1.5 wins
    assert p["win_rate_vs_last"] == pytest.approx(1 / 6)


def test_error_panel_empty_valid_set_and_single_record():
    p = error_panel([nan, nan], [0, 0], [1, 1], [1.0, 1.0])
    assert p["n_total"] == 2 and p["n_valid"] == 0 and p["valid_rate"] == 0.0
    assert all(math.isnan(p[c]) for c in CONDITIONAL_COLS)
    assert p["win_rate_vs_last"] == 0.0
    p1 = error_panel([0.5], [1], [0], [1.0])
    assert p1["n_valid"] == 1 and math.isnan(p1["sd_error"]) and p1["med_error"] == 0.5
    assert p1["win_rate_vs_last"] == 1.0
    p0 = error_panel([], [], [], [])
    assert p0["n_total"] == 0 and math.isnan(p0["valid_rate"]) and math.isnan(p0["win_rate_vs_last"])
    with pytest.raises(ValueError):
        error_panel([1.0], [1, 1], [0], [1.0])


def test_zero_denominator_mask_matches_the_skill_score_rule():
    e_m = np.array([0.0, 1e-13, 0.5, nan, 0.2, 0.3])
    e_l = np.array([0.0, 1e-13, 0.0, 0.0, nan, 0.4])
    mask = zero_denominator_flags(e_m, e_l)
    assert mask.tolist() == [True, True, True, False, False, False]
    # the branch gives 1.0 when the method is also exact, +inf otherwise
    assert skill_score(0.0, 0.0) == 1.0 and skill_score(1e-13, 1e-13) == 1.0
    assert skill_score(0.5, 0.0) == float("inf") and math.isnan(skill_score(nan, 0.0))
    assert SKILL_EPS == 1e-12
    with pytest.raises(ValueError):
        zero_denominator_flags([1.0], [1.0, 2.0])


def _records(method, cells, errs, e_last=1.0):
    rows = []
    for (cell, seeds) in zip(cells, errs):
        for seed, e in enumerate(seeds):
            valid = e is not None and math.isfinite(e)
            rows.append(dict(regime=cell[0], obs_idx=cell[1], noise=cell[2], seed=seed,
                             method=method, error=e if valid else nan, valid=int(valid),
                             catastrophic=int((not valid) or e > 5 * e_last), E_last=e_last))
    return pd.DataFrame(rows)


def test_rule_panel_on_a_hand_built_two_cell_example():
    # two eligible cells (A fires, B does not), two seeds each; a third cell
    # has a NaN feature and must not count
    cells = pd.DataFrame({"regime": ["A", "B", "C"], "obs_idx": [90, 90, 90], "noise": [0.0, 0.0, 0.0],
                          "richardson_r2": [0.2, 0.9, nan]})
    elig, fired = threshold_rule(cells, "richardson_r2", "<", 0.5)
    assert list(elig.regime) == ["A", "B"] and fired.tolist() == [True, False]
    keys = ["regime", "obs_idx", "noise"]
    cell_ids = [("A", 90, 0.0), ("B", 90, 0.0), ("C", 90, 0.0)]
    r1 = _records("richardson_1", cell_ids, [[1.0, 2.0], [0.5, 0.5], [9.0, 9.0]])
    alt = _records("rational_fit", cell_ids, [[0.5, nan], [1.0, 1.0], [0.1, 0.1]])
    out = rule_panel(elig, fired, r1, alt, keys)
    assert list(out) == list(RULE_COLS)
    assert out["n_cells_total"] == 2 and out["n_cells_fired"] == 1 and out["fire_rate"] == 0.5
    # fired cell A: r1 records (1, 2), alt records (0.5, invalid)
    assert out["r1_n_total"] == 2 and out["r1_n_valid"] == 2 and out["r1_med_error"] == 1.5
    assert out["alt_n_total"] == 2 and out["alt_n_valid"] == 1 and out["alt_valid_rate"] == 0.5
    assert out["alt_med_error"] == 0.5
    # records: seed 0 both valid and alt lower; seed 1 alt invalid -> not lower: 1 of 2
    assert out["lower_error_frac_records"] == 0.5
    # cells: alt median 0.5 < r1 median 1.5 on the one fired cell
    assert out["lower_error_frac_cells"] == 1.0
    assert out["median_rel_change"] == pytest.approx((0.5 - 1.5) / 1.5)
    # not-fired cell B: r1 (0.5, 0.5), alt (1, 1); cell C never enters
    assert out["nf_r1_n_total"] == 2 and out["nf_r1_med_error"] == 0.5
    assert out["nf_alt_n_total"] == 2 and out["nf_alt_med_error"] == 1.0
    # nothing fires: fired panels are empty, not-fired panels cover both cells
    none = rule_panel(elig, np.zeros(2, dtype=bool), r1, alt, keys)
    assert none["n_cells_fired"] == 0 and none["r1_n_total"] == 0 and math.isnan(none["r1_med_error"])
    assert math.isnan(none["lower_error_frac_records"]) and math.isnan(none["median_rel_change"])
    assert none["nf_r1_n_total"] == 4
    with pytest.raises(ValueError):
        threshold_rule(cells, "richardson_r2", "<=", 0.5)
    with pytest.raises(ValueError):
        rule_panel(elig, [True], r1, alt, keys)
