"""
tests/test_pipeline_v2.py
=========================
Pipeline wiring under redesign v2 (Prompt 2): config v2 decisions, regime
resolution, the dangerous-set derivation and artifact, the capped-exclusion
rule in Phase 1, the Phase-2 gap-stratified sweep schema, the recorded-curve
real-data re-evaluation, and the Phase-0 exclusion of trivial comparators.
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

import src.config as CFG_MOD  # noqa: E402
from src.accelerators import METHOD_NAMES  # noqa: E402
from src.dangerous import (derive_dangerous, load_artifact, load_dangerous,  # noqa: E402
                           write_artifact)
from src.evaluation import run_phase1  # noqa: E402
from src.generators import HOLDOUT_REGIME_NAMES, REGIME_NAMES  # noqa: E402
from src.pipeline import (ACCEL_METHODS, TRIVIAL_NON_ORACLE, capped_block,  # noqa: E402
                          exclude_capped, resolve_regimes)
from src.trajectories import evaluate_recorded_curves  # noqa: E402
from src.trivial import TRIVIAL_METHOD_NAMES  # noqa: E402

CURVES_CSV = os.path.join(_ROOT, CFG_MOD.REAL_DATA["curves_csv"])


# ── config v2 ─────────────────────────────────────────────────────────────────

def test_config_v2_binding_decisions():
    assert CFG_MOD.HEADLINE_G == 0.1
    assert CFG_MOD.HORIZON_GAP_FRACTIONS == [0.5, 0.1, 0.02]
    assert CFG_MOD.PHASE5B_GAP_FRACTIONS == [0.5, 0.1]
    assert CFG_MOD.ASSUMED_L_MODE == "zero"
    assert CFG_MOD.EXCLUDE_CAPPED_FROM_POOLED is True
    assert CFG_MOD.RANK_METRIC == "med_error"
    assert not hasattr(CFG_MOD, "DANGEROUS_METHODS")          # no consumable constant
    assert isinstance(CFG_MOD.LEGACY_DANGEROUS_METHODS, frozenset)
    assert CFG_MOD.DANGEROUS_ARTIFACT.endswith("dangerous_methods.json")
    for name in ("PHASE1", "PHASE2", "PHASE4", "PHASE5A", "PHASE5B"):
        grid = getattr(CFG_MOD, name)
        assert set(grid) == {"full", "quick"}
        assert grid["quick"]["n_seeds"] == 2
        assert len(grid["quick"]["core_regimes"]) == 2
        if "holdout_regimes" in grid["quick"]:
            assert len(grid["quick"]["holdout_regimes"]) == 2
        assert grid["full"]["core_regimes"] is None
    assert CFG_MOD.PHASE1["full"]["n_seeds"] == 30
    assert len(CFG_MOD.PHASE2["full"]["obs_idx_list"]) == 13
    assert CFG_MOD.PHASE5B["full"]["assumed_modes"] == list(CFG_MOD.ASSUMED_L_MODES)
    assert CFG_MOD.REAL_DATA["depths"] == [30, 60, 90, 120, 150]
    assert CFG_MOD.REAL_DATA["targets"] == [300, 400, 500]


def test_resolve_regimes_and_pools():
    assert resolve_regimes() == REGIME_NAMES + HOLDOUT_REGIME_NAMES
    assert resolve_regimes(include_holdout=False) == REGIME_NAMES
    quick = resolve_regimes(CFG_MOD.QUICK_CORE_REGIMES, CFG_MOD.QUICK_HOLDOUT_REGIMES)
    assert quick == ["single_exp", "log_slow", "stretched_exp", "random_knots"]
    with pytest.raises(ValueError):
        resolve_regimes(["stretched_exp"])                    # not a core regime
    with pytest.raises(ValueError):
        resolve_regimes(None, ["single_exp"])                 # not a held-out regime
    assert len(ACCEL_METHODS) == 51
    assert not set(ACCEL_METHODS) & set(TRIVIAL_METHOD_NAMES)
    assert set(TRIVIAL_NON_ORACLE) == {"constant_assumed", "window_mean", "window_min", "last_value"}
    assert len(METHOD_NAMES) == 56


# ── dangerous derivation and artifact ─────────────────────────────────────────

def _agg_row(method, regime, g, vr, cr, br, capped=0, hold=0, oracle=0, trivial=0):
    return dict(method=method, regime=regime, target_g=g, noise=0.0,
                valid_rate=vr, cat_rate=cr, beats_rate=br, capped=capped,
                is_holdout=hold, is_oracle=oracle, is_trivial=trivial)


def test_dangerous_derivation_rules_and_artifact(tmp_path):
    rows = [
        # A: S < 0 on the non-capped core cells -> dangerous
        _agg_row("A", "r1", 0.5, 0.5, 0.9, 0.1),
        _agg_row("A", "r1", 0.1, 0.5, 0.9, 0.1),
        # A looks perfect on a CAPPED cell and on a held-out regime: both ignored
        _agg_row("A", "r2", 0.1, 1.0, 0.0, 1.0, capped=1),
        _agg_row("A", "h1", 0.1, 1.0, 0.0, 1.0, hold=1),
        # B: safe
        _agg_row("B", "r1", 0.5, 1.0, 0.0, 0.8),
        _agg_row("B", "r1", 0.1, 1.0, 0.1, 0.5),
        # C: awful only on held-out cells -> not dangerous (core-only rule)
        _agg_row("C", "r1", 0.1, 1.0, 0.0, 0.2),
        _agg_row("C", "h1", 0.1, 0.0, 1.0, 0.0, hold=1),
        # oracle: never scored
        _agg_row("constant_oracle", "r1", 0.1, 1.0, 0.0, 1.0, oracle=1, trivial=1),
    ]
    df = pd.DataFrame(rows)
    dangerous, table = derive_dangerous(df)
    assert dangerous == frozenset({"A"})
    assert set(table.method) == {"A", "B", "C"}
    a = table.set_index("method").loc["A"]
    assert a.n_cells == 2 and a.stability == pytest.approx(0.5 - 2 * 0.9 + 0.4 * 0.1, abs=1e-4)

    path = write_artifact(dangerous, table, str(tmp_path / "d.json"), source="unit test")
    payload = load_artifact(path)
    assert payload["schema"] == "dangerous_methods/v2"
    assert payload["dangerous_methods"] == ["A"]
    assert load_dangerous(path) == frozenset({"A"})

    missing = str(tmp_path / "missing.json")
    with pytest.raises(FileNotFoundError):
        load_dangerous(missing)                                 # pipeline phases fail loudly
    assert load_dangerous(missing, required=False) == frozenset()

    with pytest.raises(ValueError):
        derive_dangerous(df.drop(columns=["capped"]))           # pre-v2 table refused


# ── Phase 1: schema, capped exclusion, ranking by median error ────────────────

def test_phase1_capped_exclusion_schema_and_ranking(tmp_path):
    res = run_phase1(n_seeds=1, noise_levels=[0.0], gap_fractions=[0.1],
                     obs_idx=90, window_len=60, out_dir=str(tmp_path),
                     regimes=["single_exp", "log_slow"], verbose=False)
    rec = res["records"]
    for col in ("L_true", "L_hat", "target_g", "achieved_g", "n_f", "capped",
                "skill", "is_trivial", "is_oracle", "is_holdout"):
        assert col in rec.columns
    assert (rec.loc[rec.regime == "log_slow", "capped"] == 1).all()   # caps at g=0.1
    assert (rec.loc[rec.regime == "single_exp", "capped"] == 0).all()

    g = res["global"]
    assert (g.regime_set == "core").all()
    assert (g.n_regimes <= 1).all() and (g.n_capped_excluded == 1).all()
    ranked = g[(g.is_oracle == 0) & g.med_error.notna()]
    assert ranked.med_error.is_monotonic_increasing                    # sorted by med_error
    assert list(ranked["rank"]) == list(range(1, len(ranked) + 1))
    assert g.loc[g.is_oracle == 1, "rank"].isna().all()
    assert "constant_oracle" in set(g.method)                          # shown, unranked

    cap = res["capped"]
    assert set(cap.regime) == {"log_slow"} and (cap.achieved_g > 0.1).all()
    assert (cap.n_f == 50000).all()

    best = res["regime_best"]
    assert {"best_by_skill", "best_skill", "best_by_stability"} <= set(best.columns)
    assert list(best.columns).index("best_by_skill") < list(best.columns).index("best_by_stability")
    assert "constant_oracle" not in set(best.best_by_skill)

    agg = res["aggregated"]
    assert {"is_trivial", "is_oracle", "is_holdout", "L_true", "L_hat"} <= set(agg.columns)
    dangerous, table = derive_dangerous(agg)
    assert "constant_oracle" not in set(table.method)
    assert (tmp_path / "phase1_capped.csv").exists()
    assert (tmp_path / "phase1_global_holdout.csv").exists()


def test_capped_block_and_exclude_capped_helpers():
    df = pd.DataFrame({
        "regime": ["a", "a", "b"], "target_g": [0.1, 0.1, 0.1], "method": ["m", "n", "m"],
        "capped": [1, 1, 0], "n_f": [50000, 50000, 120], "achieved_g": [0.4, 0.4, 0.1],
        "error": [0.1, 0.3, 0.05],
    })
    assert len(exclude_capped(df)) == 1
    blk = capped_block(df, keys=["regime", "target_g", "method"], value_cols=["error"])
    assert len(blk) == 2 and set(blk.regime) == {"a"} and (blk.achieved_g == 0.4).all()
    assert list(blk.columns) == ["regime", "target_g", "method", "n_f", "achieved_g", "med_error", "n"]
    empty = capped_block(df[df.capped == 0], keys=["regime"], value_cols=["error"])
    assert empty.empty and list(empty.columns) == ["regime", "n_f", "achieved_g", "med_error", "n"]


# ── Phase 2: gap-stratified sweep schema ──────────────────────────────────────

def test_phase2_sweep_schema_and_rank_pool(tmp_path):
    from phases.phase2 import (PHASE2_METHODS, RANK_POOL, build_phase_diagrams,
                               run_sweep)
    assert len(PHASE2_METHODS) == 11 and "constant_oracle" in PHASE2_METHODS
    assert "constant_oracle" not in RANK_POOL and len(RANK_POOL) == 10

    df_agg, df_feat = run_sweep(obs_idx_list=[90], noise_list=[0.0],
                                gap_fractions=[0.5, 0.1], n_seeds=1, window_len=60,
                                out_dir=str(tmp_path), core_regimes=["single_exp", "log_slow"],
                                verbose=False)
    for col in ("target_g", "achieved_g", "n_f", "capped", "med_skill", "L_true", "L_hat",
                "is_trivial", "is_oracle"):
        assert col in df_agg.columns
    assert set(df_agg.method) == set(PHASE2_METHODS)
    assert (df_agg.L_hat == 0.0).all()                                 # mode zero
    ls = df_agg[(df_agg.regime == "log_slow") & (df_agg.target_g == 0.1)]
    assert (ls.capped == 1).all() and (ls.achieved_g > 0.1).all()
    assert {"L_true", "L_hat"} <= set(df_feat.columns)

    df_pd = build_phase_diagrams(df_agg, 0.1, str(tmp_path))
    assert {"capped", "n_f", "achieved_g", "richardson_rank", "richardson_err_rank"} <= set(df_pd.columns)
    assert df_pd.richardson_rank.max() <= len(RANK_POOL)
    assert (tmp_path / "phase2_capped.csv").exists()
    with pytest.raises(ValueError):
        run_sweep([90], [0.0], [0.5], 1, 60, str(tmp_path), core_regimes=["stretched_exp"])


# ── Real data: recorded-curve re-evaluation ───────────────────────────────────

@pytest.mark.skipif(not os.path.exists(CURVES_CSV), reason="recorded curves not present")
def test_real_data_reevaluation_grid_and_skill():
    df = pd.read_csv(CURVES_CSV)
    curves = {c: df[c].to_numpy(float) for c in df.columns if c != "round"}
    depths, targets = CFG_MOD.REAL_DATA["depths"], CFG_MOD.REAL_DATA["targets"]
    long, summ = evaluate_recorded_curves(curves, depths, targets, window_len=60,
                                          assumed_mode="zero")
    n_pairs = sum(1 for d in depths for t in targets if d < t)
    assert n_pairs == 15
    assert len(summ) == len(curves) * n_pairs
    assert (summ.obs_depth < summ.target_round).all()
    assert set(long.method) == {"cascade", "richardson_1", "rational_fit",
                                "constant_assumed", "last_value", "window_mean", "window_min"}
    assert len(long) == len(summ) * 7
    assert (long.L_hat == 0.0).all() and (long.assumed_mode == "zero").all()
    casc = long[long.method == "cascade"]
    assert casc.skill.notna().mean() > 0.9
    refs = long[long.is_trivial == 1]
    for _, cell in refs.groupby(["dataset", "obs_depth", "target_round"]):
        assert cell.skill.min() == pytest.approx(1.0)                  # best-of-four = 1
    # depth >= target yields no cell; oracle mode is refused on real curves
    long2, summ2 = evaluate_recorded_curves(curves, [500], [300, 500], 60, "zero")
    assert summ2.empty
    with pytest.raises(ValueError):
        evaluate_recorded_curves(curves, [30], [300], 60, "oracle")


# ── Phase 0 harness excludes trivial comparators ──────────────────────────────

def test_phase0_harness_excludes_trivials():
    from tests.test_accelerators import PHASE0_METHODS, CFG
    assert len(PHASE0_METHODS) == 51
    assert not set(PHASE0_METHODS) & set(TRIVIAL_METHOD_NAMES)
    assert "L_true" not in CFG
