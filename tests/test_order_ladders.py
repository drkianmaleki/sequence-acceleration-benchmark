"""
tests/test_order_ladders.py
===========================
Phase 0b order ladders (scripts/order_ladders.py): every roster-marked variant
is the registered function itself and reproduces it bitwise on random
windows; every ladder has the specified size; ladder-only models are marked
is_roster = 0; the evaluation loop reproduces Phase 1's records exactly on a
tiny grid (the agreement check passes).
"""

import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from src.accelerators import METHODS  # noqa: E402
from src.evaluation import build_cfg, run_phase1  # noqa: E402
from src.pipeline import ACCEL_METHODS  # noqa: E402
from scripts.order_ladders import (COMPARATORS, LADDER_FAMILIES, agreement_check,  # noqa: E402
                                   all_variants, build_ladders, build_panels,
                                   comparator_variants, evaluate_grid, ladder_sizes)

EXPECTED_SIZES = {
    "shanks": 10, "wynn_eps": 8, "wynn_rho": 7, "levin_t": 8, "levin_u": 8, "levin_v": 8,
    "brezinski_theta": 6, "anderson": 6, "neville": 8, "pade": 28,
    "richardson_free": 4, "richardson_fixed": 7, "parametric": 6,
}
# the roster accelerators that carry an order parameter (all but linear,
# log_linear, geom_avg_diff and the three ensembles)
NO_LADDER = {"linear", "log_linear", "geom_avg_diff", "median_ensemble", "stability_weighted", "best_shanks_wynn"}


def _windows(n=6):
    rng = np.random.RandomState(7)
    out = []
    for _ in range(n):
        L = rng.uniform(0.005, 0.5)
        n_idx = np.arange(31, 91, dtype=float)
        gap = rng.uniform(0.2, 0.8) * (np.exp(-rng.uniform(0.01, 0.08) * n_idx) + rng.uniform(0, 0.3) / (n_idx + 1) ** 0.6)
        seq = L + gap + rng.uniform(0, 0.005) * rng.randn(n_idx.size)
        out.append((list(seq), list(range(31, 91))))
    return out


def test_ladder_sizes_and_families():
    sizes = ladder_sizes()
    assert sizes == EXPECTED_SIZES
    assert list(sizes) == LADDER_FAMILIES
    variants = build_ladders()
    assert len(variants) == sum(EXPECTED_SIZES.values())
    assert len({v.key for v in variants}) == len(variants)          # unique keys


def test_roster_variants_are_the_registered_functions():
    variants = build_ladders()
    roster = [v for v in variants if v.is_roster]
    names = sorted(v.roster_name for v in roster)
    assert len(names) == len(set(names))                            # each roster method in one ladder
    assert set(names) == set(ACCEL_METHODS) - NO_LADDER
    for v in roster:
        assert v.fn is METHODS[v.roster_name]                        # the registry entry itself
    for v in variants:
        if not v.is_roster:
            assert v.roster_name == "" and v.is_roster == 0
    comps = comparator_variants()
    assert [v.roster_name for v in comps] == COMPARATORS
    assert all(v.fn is METHODS[v.roster_name] and v.family == "trivial" for v in comps)


def test_roster_variants_reproduce_the_registry_bitwise_on_random_windows():
    cfg = build_cfg(500, 0.0, 0.1)
    for seq, idx in _windows():
        for v in all_variants():
            if not v.is_roster:
                continue
            a = v.fn(seq, idx, 500.0, cfg)
            b = METHODS[v.roster_name](seq, idx, 500.0, cfg)
            assert (math.isnan(a) and math.isnan(b)) or a == b, (v.key, a, b)


def test_ladder_only_models_run_and_are_marked():
    cfg = build_cfg(500, 0.0)
    seq = list(0.3 + 0.5 * np.exp(-0.05 * np.arange(31, 91)))
    idx = list(range(31, 91))
    only = [v for v in build_ladders() if not v.is_roster]
    assert {v.key for v in only} >= {"richardson_free:4 terms", "parametric:triple-exp",
                                     "parametric:rational two-term", "shanks:k=10", "pade:[9,10]",
                                     "richardson_fixed:alpha=0.25", "neville:d=1"}
    finite = 0
    for v in only:
        val = v.fn(seq, idx, 500.0, cfg)                             # must not raise
        assert isinstance(val, float)
        finite += int(math.isfinite(val))
    assert finite >= len(only) // 2                                  # most orders give an estimate here
    # the three ladder-only models fit the smooth single exponential closely
    for key in ("richardson_free:4 terms", "parametric:triple-exp", "parametric:rational two-term"):
        v = [x for x in only if x.key == key][0]
        val = v.fn(seq, idx, 500.0, cfg)
        assert math.isfinite(val) and abs(val - 0.3) < 0.05, (key, val)


def test_ladder_loop_reproduces_phase1_records_exactly(tmp_path):
    regimes = ["single_exp", "log_slow", "random_knots"]
    kw = dict(n_seeds=1, noise_levels=[0.0, 0.005], obs_idx=90, window_len=60)
    res = run_phase1(gap_fractions=[0.1], out_dir=str(tmp_path), regimes=regimes, verbose=False, **kw)
    assert (tmp_path / "phase1_records.csv").exists()
    df = evaluate_grid(regimes, g=0.1, jobs=1, verbose=False, **kw)
    n_var = len(all_variants())
    assert len(df) == n_var * len(regimes) * 1 * 2
    assert (df.target_g == 0.1).all() and set(df.regime) == set(regimes)
    text, ok = agreement_check(df, str(tmp_path / "phase1_records.csv"), 0.1)
    assert ok, text
    assert "VERDICT: AGREE" in text and "mismatches       : 0" in text
    # every roster method of Phase 1 that has a ladder was checked
    n_roster = sum(v.is_roster for v in all_variants())
    assert f"variants checked : {n_roster}" in text
    # a corrupted reference is caught
    bad = pd.read_csv(tmp_path / "phase1_records.csv")
    bad.loc[bad.method == "shanks_2", "estimate"] += 1e-9
    bad.to_csv(tmp_path / "bad.csv", index=False)
    text2, ok2 = agreement_check(df, str(tmp_path / "bad.csv"), 0.1)
    assert not ok2 and "shanks:k=2 -> shanks_2" in text2 and "MISMATCH" in text2
    # panels: one pooled row and one per noise level per (variant, regime set)
    panels = build_panels(df)
    core = panels[(panels.regime_set == "core") & (panels.variant == "shanks:k=2")]
    assert list(core.noise) == ["pooled", "0", "0.005"]
    assert (core.n_cells + core.n_cells_capped_excluded > 0).all()
    ls = df[(df.regime == "log_slow")]
    assert (ls.capped == 1).all()                                    # log_slow caps at g = 0.1
    pooled = panels[(panels.regime_set == "core") & (panels.noise == "pooled")]
    assert (pooled.n_cells_capped_excluded == 2).all()               # the two log_slow cells
    assert (pooled.n_total == pooled.n_cells).all()                  # one seed per cell
