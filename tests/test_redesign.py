"""
tests/test_redesign.py
======================
Invariants of redesign v2, plus the referee's finding as a regression test.

Sections
--------
1. Determinism of the hidden asymptote L_true(regime, seed)
2. Gap preservation: legacy mode reproduces the rejected design's means
3. Horizon inversion: gap-stratified n_f hits its target; cap cases flagged
4. THE FLAW: under legacy mode at n_f = 5000 the median |L_true - target|
   over the 18 regimes is ~0.00103 (the referee's number), and under the
   redesign the trivial constant predictor no longer has privileged access
5. Trivial comparators: exact outputs, registration, skill score
6. Held-out generators: finite, decreasing envelope, correct L_true
7. Isolation: nothing but the oracle reads L_true; no hidden 0.01 default
8. The evaluation loop carries (target_g, achieved_g, n_f) and skill
"""

import math
import os
import sys
import zlib

import numpy as np
import pandas as pd
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import src.config as CFG_MOD  # noqa: E402
from src.accelerators import METHODS, METHOD_NAMES, accel_current_value  # noqa: E402
from src.asymptote import assumed_asymptote  # noqa: E402
from src.evaluation import build_cfg, run_phase1  # noqa: E402
from src.generators import (  # noqa: E402
    ALL_REGIME_NAMES, GAP, GENERATORS, HOLDOUT, HOLDOUT_REGIME_NAMES,
    REGIME_NAMES, TRUTH, mean, random_knots_params, real_boot_profile,
    regime_functions, true_asymptote,
)
from src.horizons import horizon_for_gap, horizon_table  # noqa: E402
from src.trivial import (  # noqa: E402
    ORACLE_METHODS, SKILL_REFERENCE_METHODS, TRIVIAL_METHOD_NAMES,
    best_reference_error, skill_score,
)

FIXTURE = os.path.join(_HERE, "fixtures", "legacy_means_d5dc041.csv")
REFEREE_MEDIAN = 0.00103          # median |L* - target| at n_f = 5000, old design
N_OBS = 90
GS = tuple(CFG_MOD.HORIZON_GAP_FRACTIONS)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Determinism of L_true
# ═══════════════════════════════════════════════════════════════════════════════

def test_L_true_is_deterministic_and_matches_the_spec():
    lo, hi = math.log10(0.005), math.log10(0.5)
    for regime in ("single_exp", "log_slow", "staircase", "random_knots", "real_boot_a"):
        for seed in (0, 7, 29):
            a = true_asymptote(regime, seed)
            b = true_asymptote(regime, seed)
            assert a == b
            rs = np.random.RandomState(
                zlib.crc32(f"{regime}:{seed}:Lstar".encode()) % 2**31)
            assert a == 10.0 ** rs.uniform(lo, hi)     # bit-exact re-derivation


def test_L_true_in_range_and_never_shared():
    vals = {(r, s): true_asymptote(r, s) for r in ALL_REGIME_NAMES for s in range(30)}
    assert all(0.005 <= v <= 0.5 for v in vals.values())
    for r in ALL_REGIME_NAMES:                       # distinct across seeds
        assert len({vals[r, s] for s in range(30)}) == 30
    for s in range(30):                              # distinct across regimes
        assert len({vals[r, s] for r in ALL_REGIME_NAMES}) == len(ALL_REGIME_NAMES)
    med = float(np.median(list(vals.values())))      # log-uniform: median ~ 0.05
    assert 0.03 < med < 0.08


def test_legacy_mode_is_the_old_shared_constant_and_not_the_default():
    assert CFG_MOD.ASYMPTOTE_MODE == "hetero"
    assert CFG_MOD.LEGACY_L_INF == 0.01
    for r in ("single_exp", "log_slow", "stretched_exp"):
        for s in (0, 5):
            assert true_asymptote(r, s, mode="legacy") == 0.01
    with pytest.raises(ValueError):
        true_asymptote("single_exp", 0, mode="nonsense")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Gap preservation
# ═══════════════════════════════════════════════════════════════════════════════

def test_legacy_mode_reproduces_old_means():
    """L_star + gap(n) at L_star = 0.01 equals the means recorded from the
    unmodified code (commit d5dc041) on a 308-point grid, to within
    floating-point re-association (max deviation reported in the message)."""
    fx = pd.read_csv(FIXTURE, float_precision="round_trip")
    n = fx["n"].to_numpy(dtype=float)
    assert set(REGIME_NAMES) <= set(fx.columns)
    worst = 0.0
    for regime in REGIME_NAMES:
        old = fx[regime].to_numpy(dtype=float)
        new = np.asarray(mean(regime, n, CFG_MOD.LEGACY_L_INF), dtype=float)
        via_registry = np.asarray(TRUTH[regime](n, true_asymptote(regime, 0, "legacy")), dtype=float)
        d = float(np.abs(old - new).max())
        worst = max(worst, d)
        assert d <= 1e-15, f"{regime}: legacy mean deviates by {d:.3e}"
        assert np.array_equal(new, via_registry)
    print(f"\nLEGACY GAP CHECK: max |old mean - (0.01 + gap)| = {worst:.3e}")


@pytest.mark.parametrize("regime", ALL_REGIME_NAMES)
def test_mean_is_asymptote_plus_gap(regime):
    n = np.arange(0, 2001, dtype=float)
    g = np.asarray(GAP[regime](n, seed=0), dtype=float)
    for L in (0.005, 0.0731, 0.5):
        m = np.asarray(mean(regime, n, L, seed=0), dtype=float)
        assert np.allclose(m - L, g, rtol=0.0, atol=1e-15)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Horizon inversion
# ═══════════════════════════════════════════════════════════════════════════════

# Monotone (non-increasing) gap shapes: first crossing must be exact.
MONOTONE = [
    "single_exp", "two_exp", "three_exp", "four_exp", "power_law",
    "rational_decay", "mixed_pow_rat", "multiphase", "log_slow",
    "delayed_plateau", "noisy_plateau", "heavy_noise_plat", "broken_power_law",
    "stretched_exp", "logistic_tail", "inv_sqrt_log", "random_knots",
    "real_boot_a", "real_boot_b",
]
# Monotone shapes whose per-step relative decay beyond n = 90 is below 1%, so
# the integer crossing lands within 1% of the target fraction.  (Exponential
# regimes decay several percent per step; for them integer quantisation, not
# the search, sets the residual, and the first-crossing test above applies.)
SLOW = [
    "power_law", "rational_decay", "mixed_pow_rat", "multiphase", "log_slow",
    "broken_power_law", "stretched_exp", "inv_sqrt_log", "random_knots",
]


@pytest.mark.parametrize("g", GS)
@pytest.mark.parametrize("regime", MONOTONE)
def test_horizon_is_the_first_crossing(regime, g):
    hz = horizon_for_gap(regime, N_OBS, g, seed=0)
    gap = GAP[regime]
    g_obs = float(gap(N_OBS, 0))
    assert hz.target_g == g and hz.n_obs == N_OBS and hz.n_f > N_OBS
    if hz.capped:
        assert hz.n_f == CFG_MOD.HORIZON_N_CAP
        assert hz.achieved_g > g                      # honestly short of target
        assert math.isfinite(hz.achieved_g)
        return
    thr = g * g_obs
    assert float(gap(hz.n_f, 0)) <= thr
    assert float(gap(hz.n_f - 1, 0)) > thr
    assert hz.achieved_g <= g
    assert hz.achieved_g == pytest.approx(float(gap(hz.n_f, 0)) / g_obs)


@pytest.mark.parametrize("g", GS)
@pytest.mark.parametrize("regime", SLOW)
def test_slow_regimes_hit_target_within_one_percent(regime, g):
    hz = horizon_for_gap(regime, N_OBS, g, seed=0)
    if hz.capped:
        return                                        # covered by the cap test
    assert 0.99 * g <= hz.achieved_g <= g, (regime, g, hz.achieved_g)


def test_cap_cases_are_flagged_not_hidden():
    hz = horizon_for_gap("log_slow", N_OBS, 0.02)
    assert hz.capped and hz.n_f == 50_000
    assert 0.02 < hz.achieved_g < 1.0 and math.isfinite(hz.achieved_g)
    assert not horizon_for_gap("log_slow", N_OBS, 0.5).capped
    df = horizon_table(ALL_REGIME_NAMES, N_OBS, GS, seed=0)
    assert {"regime", "target_g", "achieved_g", "n_f", "capped"} <= set(df.columns)
    assert len(df) == len(ALL_REGIME_NAMES) * len(GS)
    capped = df[df["capped"]]
    assert {"log_slow", "log_oscillatory"} <= set(capped["regime"])
    assert (capped["n_f"] == 50_000).all()
    assert (capped["achieved_g"] > capped["target_g"]).all()
    assert (df.loc[~df["capped"], "achieved_g"] <= df.loc[~df["capped"], "target_g"]).all()


def test_seed_dependent_shapes_need_a_seed():
    with pytest.raises(ValueError):
        horizon_for_gap("random_knots", N_OBS, 0.5)
    a = horizon_for_gap("random_knots", N_OBS, 0.1, seed=0)
    b = horizon_for_gap("random_knots", N_OBS, 0.1, seed=1)
    assert a.n_f != b.n_f or a.gap_obs != b.gap_obs


# ═══════════════════════════════════════════════════════════════════════════════
# 4. THE FLAW as a regression test
# ═══════════════════════════════════════════════════════════════════════════════

def test_referee_flaw_legacy_median_gap_at_5000():
    """Under the rejected design (shared L* = 0.01) at n_f = 5000, the median
    |L* - target| over the 18 regimes is ~0.00103: half the regimes had
    converged to the constant the methods were handed."""
    n_f = 5000
    devs = []
    for regime in REGIME_NAMES:
        L_true = true_asymptote(regime, 0, mode="legacy")
        target = float(TRUTH[regime](n_f, L_true))
        devs.append(abs(L_true - target))
    med = float(np.median(devs))
    print(f"\nFLAW REGRESSION: median |L_true - target(n_f=5000)| over 18 legacy "
          f"regimes = {med:.6f}  (referee: {REFEREE_MEDIAN})")
    assert med == pytest.approx(REFEREE_MEDIAN, rel=0.01), f"measured {med:.6f}"
    assert sum(d < 1.1e-3 for d in devs) >= 9      # at least half converged


def test_redesign_removes_the_privileged_constant():
    """Under hetero L_true and the default 'zero' mode, constant_assumed
    returns 0, whose error at n_f = 5000 is the whole target level, not the
    ~1e-3 residual of the rejected design."""
    assert CFG_MOD.ASSUMED_L_MODE == "zero"
    errs = []
    for regime in REGIME_NAMES:
        for seed in range(5):
            _, truth, L_true = regime_functions(regime, seed)
            L_hat = assumed_asymptote(L_true, [1.0], "zero")
            cfg = build_cfg(5000, L_hat, L_true)
            est = METHODS["constant_assumed"]([1.0], [0], 5000.0, cfg)
            errs.append(abs(est - float(truth(5000))))
    assert float(np.median(errs)) > 0.03            # ~ median L_true ~ 0.05
    for L in (0.01, 0.2):
        for m in ("zero", "half", "double"):
            assert assumed_asymptote(L, [0.3], m) != L
    assert assumed_asymptote(0.2, [0.3], "oracle") == 0.2


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Trivial comparators
# ═══════════════════════════════════════════════════════════════════════════════

def test_trivial_methods_exact_outputs_on_toy_window():
    seq, idx = [1.0, 2.0, 3.0, 4.0], [10, 11, 12, 13]
    cfg = {"L_inf": 0.25, "L_true": 0.125}
    assert METHODS["constant_assumed"](seq, idx, 99.0, cfg) == 0.25
    assert METHODS["constant_oracle"](seq, idx, 99.0, cfg) == 0.125
    assert METHODS["window_mean"](seq, idx, 99.0, cfg) == 2.5
    assert METHODS["window_min"](seq, idx, 99.0, cfg) == 1.0
    assert METHODS["last_value"](seq, idx, 99.0, cfg) == 4.0
    assert METHODS["current_value"](seq, idx, 99.0, cfg) == 4.0
    assert math.isnan(METHODS["constant_oracle"](seq, idx, 99.0, {"L_inf": 0.25}))
    assert math.isnan(METHODS["constant_assumed"](seq, idx, 99.0, {}))


def test_trivial_methods_are_registered_and_flagged():
    for m in TRIVIAL_METHOD_NAMES:
        assert m in METHODS and m in METHOD_NAMES
    assert METHODS["last_value"] is accel_current_value        # literal alias
    assert ORACLE_METHODS == frozenset({"constant_oracle"})
    assert "constant_oracle" not in SKILL_REFERENCE_METHODS
    assert set(SKILL_REFERENCE_METHODS) == {
        "constant_assumed", "last_value", "window_mean", "window_min"}
    assert len(METHOD_NAMES) == 56 and len(set(METHOD_NAMES)) == 56


def test_skill_score_definition():
    errs = {"constant_assumed": 0.4, "last_value": 0.2, "window_mean": 0.3,
            "window_min": 0.25, "constant_oracle": 0.0, "richardson_1": 0.1}
    assert best_reference_error(errs) == 0.2               # oracle excluded
    assert skill_score(0.1, 0.2) == 0.5
    assert skill_score(0.2, 0.2) == 1.0
    assert skill_score(0.4, 0.2) == 2.0
    assert math.isnan(skill_score(float("nan"), 0.2))
    assert math.isnan(skill_score(0.1, float("nan")))
    assert skill_score(0.0, 0.0) == 1.0
    assert skill_score(0.1, 0.0) == float("inf")
    assert math.isnan(best_reference_error({"constant_oracle": 0.0}))


def test_assumed_asymptote_modes():
    win = [0.30, 0.20, 0.25]
    assert assumed_asymptote(0.1, win, "zero") == 0.0
    assert assumed_asymptote(0.1, win, "half") == 0.05
    assert assumed_asymptote(0.1, win, "oracle") == 0.1
    assert assumed_asymptote(0.1, win, "double") == 0.2
    assert assumed_asymptote(0.1, win, "winmin") == pytest.approx(0.18)
    assert assumed_asymptote(None, [-0.1, 0.2], "winmin") == 0.0
    assert assumed_asymptote(None, win, "zero") == 0.0
    with pytest.raises(ValueError):
        assumed_asymptote(None, win, "half")            # needs L_true
    with pytest.raises(ValueError):
        assumed_asymptote(0.1, win, "guess")
    assert assumed_asymptote(0.1, win) == 0.0            # config default: zero


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Held-out generators
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("seed", [0, 7])
@pytest.mark.parametrize("regime", HOLDOUT_REGIME_NAMES)
def test_holdout_regime_contract(regime, seed):
    n = np.arange(0, 50_001, dtype=float)
    g = np.asarray(GAP[regime](n, seed), dtype=float)
    assert np.all(np.isfinite(g)) and np.all(g >= 0.0)
    block_max = g[:50_000].reshape(500, 100).max(axis=1)
    assert np.all(np.diff(block_max) <= 1e-12), "envelope is not decreasing"
    assert g[-1] < 0.05 * g[0]
    L_true = true_asymptote(regime, seed)
    gen, truth, L = regime_functions(regime, seed)
    assert L == L_true and 0.005 <= L <= 0.5
    obs = gen(n[:300], np.random.RandomState(seed), 0.0)
    assert np.allclose(obs, L_true + g[:300], rtol=0.0, atol=1e-12)
    assert abs(float(truth(50_000)) - L_true) < 0.05 * g[0]
    assert GAP[regime].holdout and regime in HOLDOUT and regime not in REGIME_NAMES


def test_holdout_gap_at_zero_matches_the_specification():
    assert float(GAP["stretched_exp"](0)) == pytest.approx(0.7)
    assert float(GAP["logistic_tail"](0)) == pytest.approx(0.6 / (1 + math.exp(-60 / 18)))
    assert float(GAP["inv_sqrt_log"](0)) == pytest.approx(0.9)
    for s in range(5):
        assert float(GAP["random_knots"](0, s)) == pytest.approx(0.7)
    assert float(GAP["real_boot_a"](0)) == pytest.approx(0.7)
    assert float(GAP["real_boot_b"](0)) == pytest.approx(0.7)


def test_random_knots_is_seeded_continuous_and_in_spec():
    seen = set()
    for seed in range(6):
        knots, exps, consts = random_knots_params(seed)
        assert len(knots) == 3 and len(set(knots)) == 3
        assert knots.min() >= 20 and knots.max() <= 200
        assert np.all(np.diff(knots) > 0)
        assert len(exps) == 4 and np.all((0.3 <= exps) & (exps <= 0.9))
        for k in knots:
            left = float(GAP["random_knots"](k - 1e-6, seed))
            right = float(GAP["random_knots"](k, seed))
            assert abs(left - right) <= 1e-5 * max(right, 1e-12)
        assert random_knots_params(seed)[0].tolist() == knots.tolist()
        seen.add(tuple(knots))
    assert len(seen) >= 3                              # genuinely seeded


def test_real_boot_profiles_come_from_the_recorded_curves():
    for ds, regime in (("adult", "real_boot_a"), ("higgs", "real_boot_b")):
        n, prof = real_boot_profile(ds)
        assert len(n) == 500 and prof[0] == pytest.approx(0.7) and prof[-1] == 0.0
        assert np.all(np.diff(prof) <= 1e-12)
        assert float(GAP[regime](600)) == 0.0
        assert float(GAP[regime](90)) > 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Isolation: nobody but the oracle reads L_true; no hidden 0.01 default
# ═══════════════════════════════════════════════════════════════════════════════

def test_no_method_except_the_oracle_reads_L_true():
    gen, _, L_true = regime_functions("power_law", 3)
    seq = gen(np.arange(91, dtype=float), np.random.RandomState(3), 0.005)
    win, idx = list(seq[31:91]), list(range(31, 91))
    for L_hat in (0.0, 0.05):
        cfg_with = build_cfg(2000, L_hat, L_true)
        cfg_without = build_cfg(2000, L_hat, None)
        for m in METHOD_NAMES:
            if m in ORACLE_METHODS:
                continue
            a = METHODS[m](win, idx, 2000.0, cfg_with)
            b = METHODS[m](win, idx, 2000.0, cfg_without)
            assert (math.isnan(a) and math.isnan(b)) or a == b, m
    assert "L_true" not in build_cfg(2000, 0.0)


def test_accelerators_have_no_hidden_default_asymptote():
    win = list(0.3 + 0.5 * np.exp(-0.05 * np.arange(31, 91)))
    idx = list(range(31, 91))
    bare = {"ridge": 1e-8, "min_valid": -0.5, "max_valid": 500.0, "denom_tol": 1e-14}
    for m in ("log_linear", "richardson_1", "richardson_a10",
              "single_exp_fit", "rational_fit", "log_fit"):
        with pytest.raises(KeyError):
            METHODS[m](win, idx, 1000.0, bare)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Evaluation loop end-to-end (small grid)
# ═══════════════════════════════════════════════════════════════════════════════

def test_run_phase1_records_carry_horizons_and_skill(tmp_path):
    res = run_phase1(n_seeds=1, noise_levels=[0.0], gap_fractions=[0.5, 0.02],
                     obs_idx=N_OBS, window_len=60, out_dir=str(tmp_path),
                     regimes=["single_exp", "log_slow", "random_knots"],
                     verbose=False)
    rec = res["records"]
    for col in ("target_g", "achieved_g", "n_f", "capped", "L_true", "L_hat",
                "assumed_mode", "skill", "ref_error", "is_oracle", "is_holdout",
                "is_trivial"):
        assert col in rec.columns
    assert len(rec) == 3 * 2 * len(METHOD_NAMES)

    ls = rec[(rec.regime == "log_slow") & (rec.target_g == 0.02)]
    assert (ls.capped == 1).all() and (ls.n_f == 50_000).all() and (ls.achieved_g > 0.02).all()
    se = rec[(rec.regime == "single_exp") & (rec.target_g == 0.5)]
    assert (se.capped == 0).all() and (se.achieved_g <= 0.5).all()

    assert (rec.assumed_mode == "zero").all() and (rec.L_hat == 0.0).all()
    assert (rec.loc[rec.method == "constant_assumed", "estimate"] == 0.0).all()
    orc = rec[rec.method == "constant_oracle"]
    assert (orc.is_oracle == 1).all()
    assert np.allclose(orc.estimate, orc.L_true)

    for _, cell in rec.groupby(["regime", "seed", "target_g"]):
        ref = cell[cell.method.isin(SKILL_REFERENCE_METHODS)]
        assert ref.skill.min() == pytest.approx(1.0)
        assert (cell.loc[cell.method == "random_knots", :].empty)
    assert (rec.loc[rec.regime == "random_knots", "is_holdout"] == 1).all()
    assert (rec.loc[rec.regime != "random_knots", "is_holdout"] == 0).all()
    assert (rec.loc[rec.method == "window_mean", "is_trivial"] == 1).all()
    assert (rec.loc[rec.method == "richardson_1", "is_trivial"] == 0).all()

    g = res["global"]
    assert set(g.regime_set) == {"core"} and "random_knots" not in set(res["aggregated"][res["aggregated"].is_holdout == 0].regime)
    assert g.loc[g.is_oracle == 1, "rank"].isna().all()
    # ranks by med_error over the rank-eligible methods: non-oracle, finite
    # med_error, valid_rate >= RANK_MIN_VALID (Report-2 review, decision 3)
    eligible = (g.is_oracle == 0) & g.med_error.notna() & (g.valid_rate >= CFG_MOD.RANK_MIN_VALID)
    assert (g.rank_eligible == eligible.astype(int)).all()
    ranked = g[eligible]
    assert ranked["rank"].notna().all()
    assert g.loc[~eligible, "rank"].isna().all()            # below the floor: unranked
    for _, grp in ranked.groupby("target_g"):
        assert grp.med_error.is_monotonic_increasing        # sorted by med_error
        assert list(grp["rank"]) == list(range(1, len(grp) + 1))
    assert "constant_oracle" in set(g.method)               # shown, unranked
    assert set(res["global_holdout"].regime_set) == {"holdout"}
    assert "constant_oracle" not in set(res["regime_best"].best_by_skill)
    assert "constant_oracle" not in set(res["regime_best"].best_by_stability)
    assert "phase1_capped.csv" in {p.name for p in tmp_path.iterdir()}

    for f in ("phase1_records.csv", "phase1_aggregated.csv", "phase1_global.csv",
              "phase1_global_holdout.csv", "phase1_regime_best.csv",
              "phase1_horizons.csv", "phase1_heatmap_g0.5.csv"):
        assert (tmp_path / f).exists(), f
