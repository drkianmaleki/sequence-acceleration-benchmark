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
    assert len(ACCEL_METHODS) == len(METHOD_NAMES) - len(TRIVIAL_METHOD_NAMES)
    assert not set(ACCEL_METHODS) & set(TRIVIAL_METHOD_NAMES)
    assert set(TRIVIAL_NON_ORACLE) == {"constant_assumed", "window_mean", "window_min", "last_value"}
    assert len(METHOD_NAMES) == len(ACCEL_METHODS) + len(TRIVIAL_METHOD_NAMES)


# The accelerator roster, spelled out: every name that src.pipeline.ACCEL_METHODS
# must contain and nothing else.  A roster change (a method added or retired)
# is a deliberate act and must update this list in the same commit.
EXPECTED_ACCELERATORS = sorted([
    "linear", "log_linear", "geom_avg_diff",
    "richardson_1", "richardson_2", "richardson_3",
    "richardson_a05", "richardson_a10", "richardson_a20",
    "single_exp_fit", "double_exp_fit", "rational_fit", "log_fit",
    "shanks_1", "shanks_2", "shanks_3", "shanks_4",
    "wynn_eps_1", "wynn_eps_2", "wynn_eps_3",
    "wynn_rho_1", "wynn_rho_2", "wynn_rho_3",
    "pade_11", "pade_12", "pade_13", "pade_21", "pade_22", "pade_23",
    "pade_31", "pade_32", "pade_33", "pade_34", "pade_44", "pade_45",
    "levin_t1", "levin_t2", "levin_u1", "levin_u2", "levin_v1", "levin_v2",
    "brezinski_theta1", "brezinski_theta2",
    "neville_2", "neville_3", "neville_4",
    "anderson_1", "anderson_2", "anderson_3",
    "median_ensemble", "stability_weighted", "best_shanks_wynn",
])


def test_accelerator_roster_is_exactly_the_expected_list():
    assert sorted(ACCEL_METHODS) == EXPECTED_ACCELERATORS
    assert len(EXPECTED_ACCELERATORS) == len(set(EXPECTED_ACCELERATORS))
    assert "last_value" not in ACCEL_METHODS and "last_value" in METHOD_NAMES
    from src.evaluation import FAMILY, USES_FUTURE_X
    pade = {m for m in ACCEL_METHODS if FAMILY[m] == "pade"}
    assert {"pade_33", "pade_34", "pade_44", "pade_45"} <= pade
    assert all(USES_FUTURE_X[m] for m in pade)
    assert set(METHOD_NAMES) <= set(FAMILY)


# ── excluded-set derivation and artifact ("dangerous" is the legacy name) ────

def _agg_row(method, regime, g, vr, cr, me=0.01, capped=0, hold=0, oracle=0, trivial=0):
    return dict(method=method, regime=regime, target_g=g, noise=0.0,
                valid_rate=vr, cat_rate=cr, med_error=me, capped=capped,
                is_holdout=hold, is_oracle=oracle, is_trivial=trivial)


def test_dangerous_derivation_rules_and_artifact(tmp_path):
    A, B, C, D = "neville_2", "richardson_1", "shanks_1", "pade_22"   # accelerators (eligible)
    rows = [
        # A: pooled valid_rate 0.85 on the non-capped core cells -> excluded (below 0.9)
        _agg_row(A, "r1", 0.5, 0.80, 0.30, me=0.02),
        _agg_row(A, "r1", 0.1, 0.90, 0.30, me=0.03),
        # A looks perfect on a CAPPED cell and on a held-out regime: both ignored
        _agg_row(A, "r2", 0.1, 1.0, 0.0, capped=1),
        _agg_row(A, "h1", 0.1, 1.0, 0.0, hold=1),
        # B: pooled valid_rate 0.95 -> kept, whatever its catastrophe rate
        _agg_row(B, "r1", 0.5, 1.00, 0.90),
        _agg_row(B, "r1", 0.1, 0.90, 0.90),
        # C: awful only on held-out cells -> kept (core-only rule)
        _agg_row(C, "r1", 0.1, 1.0, 0.0),
        _agg_row(C, "h1", 0.1, 0.0, 1.0, hold=1),
        # D: exactly at the floor -> kept (the criterion is strictly below)
        _agg_row(D, "r1", 0.1, 0.9, 0.5),
        # trivial comparator with low validity: tabulated, never excluded, never in the artifact
        _agg_row("window_mean", "r1", 0.1, 0.5, 0.9, trivial=1),
        # trivial comparator that is fine: tabulated only
        _agg_row("last_value", "r1", 0.1, 1.0, 0.0, trivial=1),
        # oracle: never tabulated
        _agg_row("constant_oracle", "r1", 0.1, 1.0, 0.0, oracle=1, trivial=1),
    ]
    df = pd.DataFrame(rows)
    dangerous, table = derive_dangerous(df)
    assert dangerous == frozenset({A})                          # accelerators only
    assert set(table.method) == {A, B, C, D, "window_mean", "last_value"}   # trivials tabulated
    t = table.set_index("method")
    assert t.loc[A].n_cells == 2 and t.loc[A].valid_rate == pytest.approx(0.85, abs=1e-4)
    assert t.loc[A].med_error == pytest.approx(0.025)          # descriptive, not part of the criterion
    assert t.loc[B].valid_rate == pytest.approx(0.95) and t.loc[B].dangerous == 0 and t.loc[B].cat_rate == pytest.approx(0.9)
    assert t.loc[D].valid_rate == pytest.approx(0.9) and t.loc[D].dangerous == 0
    assert t.loc["window_mean"].valid_rate < CFG_MOD.RANK_MIN_VALID and t.loc["window_mean"].dangerous == 0
    assert t.loc["window_mean"].eligible == 0 and t.loc[A].eligible == 1
    assert set(table.loc[table.eligible == 1, "method"]) <= set(ACCEL_METHODS)
    assert list(table.valid_rate) == sorted(table.valid_rate)  # sorted by valid_rate ascending
    assert list(table.columns) == ["method", "is_trivial", "eligible", "valid_rate", "cat_rate",
                                   "med_error", "n_cells", "dangerous"]

    path = write_artifact(dangerous, table, str(tmp_path / "d.json"), source="unit test")
    payload = load_artifact(path)
    assert payload["schema"] == "dangerous_methods/v3"
    assert payload["dangerous_methods"] == [A]
    assert payload["rank_min_valid"] == CFG_MOD.RANK_MIN_VALID
    assert "RANK_MIN_VALID" in payload["criterion"]
    assert set(payload) == {"schema", "criterion", "legacy_name", "pool", "n_pool", "rank_min_valid",
                            "asymptote_mode", "assumed_mode", "gap_fractions", "source", "git_head",
                            "created", "dangerous_methods", "table"}
    assert payload["pool"] == "accelerators" and payload["n_pool"] == len(ACCEL_METHODS)
    written = {r["method"] for r in payload["table"]}
    assert written == {A, B, C, D}                              # trivials never written
    assert load_dangerous(path) == frozenset({A})
    with pytest.raises(ValueError):                             # a trivial can never be written
        write_artifact(frozenset({"window_mean"}), table, str(tmp_path / "bad.json"))

    missing = str(tmp_path / "missing.json")
    with pytest.raises(FileNotFoundError):
        load_dangerous(missing)                                 # pipeline phases fail loudly
    assert load_dangerous(missing, required=False) == frozenset()

    with pytest.raises(ValueError):
        derive_dangerous(df.drop(columns=["capped"]))           # pre-v2 table refused
    with open(path, encoding="utf-8") as fh:                    # a v2 artifact is refused
        old = fh.read().replace("dangerous_methods/v3", "dangerous_methods/v2")
    (tmp_path / "v2.json").write_text(old, encoding="utf-8")
    with pytest.raises(ValueError):
        load_artifact(str(tmp_path / "v2.json"))


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
    assert list(best.columns) == [
        "regime", "is_holdout", "target_g", "n_f", "achieved_g", "capped",
        "best_by_skill", "best_skill", "family", "method_type", "skill_best_med_error",
        "skill_best_valid_rate", "skill_best_cat_rate",
        "best_by_error", "err_best_med_error", "err_best_valid_rate", "err_best_cat_rate",
        "n_at_floor", "oracle_med_error"]
    assert "constant_oracle" not in set(best.best_by_skill)
    assert "constant_oracle" not in set(best.best_by_error)
    assert (best.n_at_floor >= 1).all()

    agg = res["aggregated"]
    from src.trivial import REFERENCE_TAGS
    # the descriptive panel of the seeds (src.panels.error_panel) next to the
    # existing columns; nothing else
    assert list(agg.columns) == [
        "method", "family", "method_type", "is_trivial", "is_oracle", "regime", "is_holdout",
        "noise", "target_g", "n_f", "achieved_g", "capped", "L_true", "L_hat",
        "n_total", "n_valid", "valid_rate", "cat_rate", "mean_error", "sd_error", "med_error",
        "q25_error", "q75_error", "p90_error", "med_improve", "med_skill", "n_seeds",
        *[c for _, t in REFERENCE_TAGS for c in (f"med_skill_vs_{t}", f"win_rate_vs_{t}")]]
    assert (agg.n_seeds == agg.n_total).all()
    ok = agg[agg.n_valid > 0]
    assert (ok.q25_error <= ok.med_error + 1e-15).all() and (ok.med_error <= ok.q75_error + 1e-15).all()
    assert (ok.q75_error <= ok.p90_error + 1e-15).all() and (agg.n_valid <= agg.n_total).all()
    assert "E_last" in rec.columns and rec.loc[rec.method == "last_value", "E_last"].equals(
        rec.loc[rec.method == "last_value", "error"])
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
    from phases.phase2 import (PHASE2_BASE_METHODS, PHASE2_METHODS, RANK_POOL,
                               UNRANKED_COMPARATORS, build_phase_diagrams, run_sweep)
    from phases.phase3 import CANDIDATES
    assert len(PHASE2_METHODS) == 11 and "constant_oracle" in PHASE2_METHODS
    # decision 4: the Richardson rank is over the original 9; constant_assumed
    # is reported alongside, unranked, like the oracle (and is no Phase-3 candidate)
    assert RANK_POOL == PHASE2_BASE_METHODS and len(RANK_POOL) == 9
    assert set(UNRANKED_COMPARATORS) == {"constant_assumed", "constant_oracle"}
    assert not set(UNRANKED_COMPARATORS) & set(RANK_POOL)
    assert CANDIDATES == RANK_POOL

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
    assert {"capped", "n_f", "achieved_g", "richardson_rank", "richardson_err_rank",
            "richardson_valid_rate", "richardson_below_floor", "n_rank_pool",
            "n_rank_eligible", "constant_assumed_stability", "constant_assumed_med_error",
            "constant_assumed_valid_rate", "constant_oracle_stability",
            "constant_oracle_med_error"} <= set(df_pd.columns)
    assert df_pd.richardson_rank.max() <= len(RANK_POOL)
    assert (df_pd.n_rank_pool == 9).all() and (df_pd.n_rank_eligible <= 9).all()
    assert not df_pd.best_method.isin(UNRANKED_COMPARATORS).any()
    assert df_pd.constant_assumed_stability.notna().all()      # reported alongside
    ranked = df_pd[df_pd.richardson_below_floor == 0]
    assert ranked.richardson_rank.notna().all()
    assert df_pd.loc[df_pd.richardson_below_floor == 1, "richardson_rank"].isna().all()
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

    # decision 5: perturb_iqr of the routed method on every cell, deterministic
    assert "perturb_iqr" in long.columns and "perturb_iqr" in summ.columns
    assert summ.perturb_iqr.notna().mean() > 0.9
    assert (summ.perturb_iqr.dropna() >= 0).all()
    cell_iqr = long.groupby(["dataset", "obs_depth", "target_round"]).perturb_iqr.nunique(dropna=False)
    assert (cell_iqr == 1).all()                                       # one value per cell
    from src.trajectories import perturb_iqr_real, perturb_seed, _DEFAULT_CFG
    import zlib
    r0 = summ.iloc[0]
    assert perturb_seed(r0.dataset, int(r0.obs_depth)) == zlib.crc32(
        f"{r0.dataset}:{int(r0.obs_depth)}".encode()) % 2 ** 31
    curve = curves[r0.dataset]
    d = int(r0.obs_depth); start = max(0, d - 60)
    cfg = _DEFAULT_CFG.copy(); cfg["L_inf"] = 0.0
    again = perturb_iqr_real(curve[start:d], np.arange(start + 1, d + 1, dtype=float),
                             r0.selected_method, float(r0.target_round), cfg,
                             perturb_seed(r0.dataset, d))
    assert again == pytest.approx(r0.perturb_iqr, rel=1e-12, nan_ok=True)
    # provenance metadata on the summary
    for col in ("phase2_features_path", "phase2_features_rows", "git_head"):
        assert col in summ.columns and summ[col].nunique() == 1
    assert summ.phase2_features_rows.iloc[0] == 0 and summ.nearest_regime.eq("unknown").all()
    feats = os.path.join(_ROOT, "results", "phase2", "phase2_features.csv")
    if os.path.exists(feats):
        _, summ3 = evaluate_recorded_curves(curves, [30], [300], 60, "zero",
                                            phase2_features_path=feats)
        assert summ3.phase2_features_rows.iloc[0] == len(pd.read_csv(feats))
        assert summ3.phase2_features_path.iloc[0] == feats
    # depth >= target yields no cell; oracle mode is refused on real curves
    long2, summ2 = evaluate_recorded_curves(curves, [500], [300, 500], 60, "zero")
    assert summ2.empty
    with pytest.raises(ValueError):
        evaluate_recorded_curves(curves, [30], [300], 60, "oracle")


# ── Rank validity floor (decision 3) ──────────────────────────────────────────

def test_assign_ranks_validity_floor_and_unranked_block():
    from src.pipeline import assign_ranks, unranked_block
    assert CFG_MOD.RANK_MIN_VALID == 0.9
    df = pd.DataFrame({
        "target_g": [0.1] * 5 + [0.5] * 2,
        "method": ["rare", "good", "tie_b", "tie_a", "constant_oracle", "x", "never"],
        "is_oracle": [0, 0, 0, 0, 1, 0, 0],
        "valid_rate": [0.5, 1.0, 0.95, 0.95, 1.0, 0.9, 0.0],
        "med_error": [0.001, 0.010, 0.020, 0.020, 0.0001, 0.3, np.nan],
    })
    out = assign_ranks(df, "med_error", group_cols=["target_g"])
    r = out.set_index("method")
    assert r.loc["good", "rank"] == 1                       # lowest eligible error
    assert r.loc["tie_a", "rank"] == 2 and r.loc["tie_b", "rank"] == 3   # name tie-break
    assert np.isnan(r.loc["rare", "rank"]) and r.loc["rare", "rank_eligible"] == 0   # 0.5 < 0.9
    assert np.isnan(r.loc["constant_oracle", "rank"])       # oracle never ranked
    assert r.loc["x", "rank"] == 1 and r.loc["x", "rank_eligible"] == 1   # 0.9 is eligible
    assert np.isnan(r.loc["never", "rank"])
    blk = unranked_block(out, ["target_g", "method", "valid_rate", "med_error"])
    assert list(blk.method) == ["rare", "never"]            # below floor, oracle excluded
    assert "valid_rate" in blk.columns
    desc = assign_ranks(df, "med_error", group_cols=["target_g"], ascending=False)
    assert desc.set_index("method").loc["tie_a", "rank"] == 1


def test_phase1_rank_floor_unranked_block_and_file(tmp_path):
    res = run_phase1(n_seeds=2, noise_levels=[0.005], gap_fractions=[0.1],
                     obs_idx=90, window_len=60, out_dir=str(tmp_path),
                     regimes=["single_exp"], verbose=False)
    g = res["global"]
    assert {"rank", "rank_eligible"} <= set(g.columns)
    below = g[(g.is_oracle == 0) & (g.valid_rate < CFG_MOD.RANK_MIN_VALID)]
    assert below["rank"].isna().all() and (below.rank_eligible == 0).all()
    ranked = g[g.rank_eligible == 1]
    assert (ranked.valid_rate >= CFG_MOD.RANK_MIN_VALID).all() and (ranked.is_oracle == 0).all()
    assert list(ranked["rank"]) == list(range(1, len(ranked) + 1))
    unr = res["unranked"]
    assert set(unr.method) == set(below.method)
    assert "valid_rate" in unr.columns
    assert (tmp_path / "phase1_unranked.csv").exists()


# ── Phase 5a: block-parallel evaluation is byte-identical to serial ───────────

def test_phase5a_block_parallel_byte_identical(tmp_path):
    from phases.phase5a import EVAL_METHODS, run_phase5a
    kw = dict(obs_idx_list=[60, 90], noise_list=[0.0, 0.005], gap_fractions=[0.1],
              n_seeds=1, window_len=60, perturb_trials=2, perturb_scale=0.02,
              dangerous={"neville_2"}, core_regimes=["single_exp"],
              holdout_regimes=["stretched_exp"], verbose=False)
    d1, d2 = tmp_path / "serial", tmp_path / "jobs2"
    df1 = run_phase5a(out_dir=str(d1), jobs=1, keep_shards=True, **kw)
    df2 = run_phase5a(out_dir=str(d2), jobs=2, **kw)
    assert len(df1) == 2 * 2 * 1 * 2 * 1 * len(EVAL_METHODS)
    b1 = (d1 / "phase5a_raw.csv").read_bytes()
    b2 = (d2 / "phase5a_raw.csv").read_bytes()
    assert b1 and b1 == b2                                  # byte-identical raw table
    pd.testing.assert_frame_equal(df1, df2)
    # shards: one per (obs_idx, noise) block; their textual concatenation is the raw file
    shards = sorted((d1 / "shards").glob("phase5a_raw_part*.csv"))
    assert len(shards) == 4 and shards[0].name.endswith("obs60_sigma0.csv")
    body = b"".join(s.read_bytes().split(b"\n", 1)[1] for s in shards[1:])
    assert shards[0].read_bytes() + body == b1
    assert not (d2 / "shards").exists()                     # removed after concatenation


# ── Phase 0 harness excludes trivial comparators ──────────────────────────────

def test_phase0_harness_excludes_trivials():
    from tests.test_accelerators import PHASE0_METHODS, CFG
    assert len(PHASE0_METHODS) == len(ACCEL_METHODS)
    assert not set(PHASE0_METHODS) & set(TRIVIAL_METHOD_NAMES)
    assert "L_true" not in CFG
