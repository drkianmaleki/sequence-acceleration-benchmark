"""
tests/test_review_edge_cases.py  (adopted from review R8d)
=======================================================
Adversarial probes of the edges the R8a / R8b definitions imply:

  * error_panel on all-invalid input, on a single valid record, and on a
    record whose error equals E_last exactly (a tie is not a win);
  * rule_panel on a rule that fires nowhere and on one that fires everywhere
    (the nf_ panel is then empty);
  * the R_R denominator rule (E_last exactly 0 with Richardson exact / not
    exact / invalid) through skill_score, zero_denominator_flags and
    richardson_targets; Spearman with a +inf target ranks it largest;
  * classifier folds with one depth removed (no leakage, every row tested
    once) and the documented degradation with a single noise level;
  * every ladder constructor called with a roster order reproduces the
    registry function bitwise on 50 random windows, including windows with
    NaN and windows too short for the higher orders;
  * perturbation factors: trial count changes the shape but not the leading
    rows (prefix property of one RandomState stream); reversing the regime
    list leaves every perturb_iqr unchanged, and the pairing test fails when
    the key is deliberately made order-dependent;
  * the exclusion boundary: valid rate 0.899 is excluded, 0.900 is not.
"""

import itertools
import math
import os
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

import src.config as CFG_MOD  # noqa: E402
from src.accelerators import METHODS  # noqa: E402
from src.dangerous import derive_dangerous  # noqa: E402
from src.diagnostics import perturbation_factors, perturbation_key  # noqa: E402
from src.evaluation import build_cfg  # noqa: E402
from src.panels import (CONDITIONAL_COLS, RULE_COLS, error_panel, rule_panel,  # noqa: E402
                        threshold_rule, zero_denominator_flags)
from src.trivial import SKILL_EPS, skill_score  # noqa: E402
from phases.phase2 import FEATURE_COLS, WINDOW_KEYS, richardson_targets, run_correlation_analysis  # noqa: E402
from phases.phase3 import classifier_folds, classifier_table, regime_classifier  # noqa: E402
from tests.test_phase3_classifier import DEPTHS, NOISES, REGIMES, _feature_table  # noqa: E402

nan = float("nan")
inf = float("inf")


# ── error_panel edges ────────────────────────────────────────────────────────

def test_error_panel_all_invalid_input():
    p = error_panel([nan, nan, nan], [0, 0, 0], [1, 1, 1], [0.5, 0.5, nan])
    assert p["n_total"] == 3 and p["n_valid"] == 0
    assert p["valid_rate"] == 0.0 and p["cat_rate"] == 1.0
    assert all(math.isnan(p[c]) for c in CONDITIONAL_COLS)
    assert p["win_rate_vs_last"] == 0.0
    # a record flagged invalid never wins even if its error column carried a number
    q = error_panel([0.1, 0.2], [0, 0], [1, 1], [1.0, 1.0])
    assert q["n_valid"] == 0 and q["win_rate_vs_last"] == 0.0 and math.isnan(q["med_error"])


def test_error_panel_single_valid_record():
    p = error_panel([0.25, nan], [1, 0], [0, 1], [1.0, 1.0])
    assert p["n_total"] == 2 and p["n_valid"] == 1 and p["valid_rate"] == 0.5
    assert p["mean_error"] == 0.25 and p["med_error"] == 0.25
    assert p["q25_error"] == 0.25 and p["q75_error"] == 0.25 and p["p90_error"] == 0.25
    assert math.isnan(p["sd_error"])                   # sample sd undefined below two records
    assert p["win_rate_vs_last"] == 0.5                # 1 win of 2 records


def test_error_panel_tie_with_E_last_is_not_a_win():
    e_last = [0.3, 0.3, 0.3, 0.3]
    errors = [0.3, np.nextafter(0.3, 0.0), np.nextafter(0.3, 1.0), nan]
    p = error_panel(errors, [1, 1, 1, 0], [0, 0, 0, 1], e_last)
    assert p["win_rate_vs_last"] == pytest.approx(1 / 4)   # only the strictly-below record wins
    # an invalid E_last never yields a win either
    q = error_panel([0.1], [1], [0], [nan])
    assert q["n_valid"] == 1 and q["win_rate_vs_last"] == 0.0


# ── rule_panel edges ─────────────────────────────────────────────────────────

def _records(method, cell_ids, per_cell_errors, e_last=1.0):
    rows = []
    for cell, errs in zip(cell_ids, per_cell_errors):
        for seed, e in enumerate(errs):
            valid = e is not None and math.isfinite(e)
            rows.append(dict(regime=cell[0], obs_idx=cell[1], noise=cell[2], seed=seed, method=method,
                             error=e if valid else nan, valid=int(valid),
                             catastrophic=int((not valid) or e > 5 * e_last), E_last=e_last))
    return pd.DataFrame(rows)


def test_rule_panel_fires_nowhere_and_everywhere():
    cells = pd.DataFrame({"regime": ["A", "B", "C"], "obs_idx": [90] * 3, "noise": [0.0] * 3,
                          "feat": [0.1, 0.2, 0.3]})
    ids = [("A", 90, 0.0), ("B", 90, 0.0), ("C", 90, 0.0)]
    r1 = _records("richardson_1", ids, [[1.0, 2.0], [1.0, 1.0], [3.0, nan]])
    alt = _records("rational_fit", ids, [[0.5, 0.5], [2.0, 2.0], [1.0, 1.0]])
    keys = list(WINDOW_KEYS)
    # fires nowhere: feat < 0 never true
    elig, fired = threshold_rule(cells, "feat", "<", 0.0)
    assert len(elig) == 3 and not fired.any()
    none = rule_panel(elig, fired, r1, alt, keys)
    assert list(none) == list(RULE_COLS)
    assert none["n_cells_total"] == 3 and none["n_cells_fired"] == 0 and none["fire_rate"] == 0.0
    assert none["r1_n_total"] == 0 and none["alt_n_total"] == 0
    assert math.isnan(none["r1_med_error"]) and math.isnan(none["alt_valid_rate"])
    assert math.isnan(none["lower_error_frac_records"]) and math.isnan(none["lower_error_frac_cells"])
    assert math.isnan(none["median_rel_change"])
    assert none["nf_r1_n_total"] == 6 and none["nf_alt_n_total"] == 6
    assert none["nf_r1_n_valid"] == 5 and none["nf_r1_valid_rate"] == pytest.approx(5 / 6)
    # fires everywhere: feat > 0 always true -> the not-fired panels are empty
    elig, fired = threshold_rule(cells, "feat", ">", 0.0)
    assert fired.all()
    every = rule_panel(elig, fired, r1, alt, keys)
    assert every["n_cells_fired"] == 3 and every["fire_rate"] == 1.0
    assert every["nf_r1_n_total"] == 0 and every["nf_alt_n_total"] == 0
    assert math.isnan(every["nf_r1_med_error"]) and math.isnan(every["nf_alt_win_rate_vs_last"])
    assert every["r1_n_total"] == 6 and every["r1_n_valid"] == 5
    # record-wise: A (0.5<1, 0.5<2) yes yes; B (2<1) no no; C (1<3) yes, (nan) no -> 3 of 6
    assert every["lower_error_frac_records"] == pytest.approx(3 / 6)
    # cell-wise medians: A 0.5<1.5 yes; B 2<1 no; C 1<3 yes -> 2 of 3
    assert every["lower_error_frac_cells"] == pytest.approx(2 / 3)
    assert every["median_rel_change"] == pytest.approx(np.median([(0.5 - 1.5) / 1.5, (2 - 1) / 1, (1 - 3) / 3]))


# ── R_R denominator rule and Spearman with +inf ──────────────────────────────

def _rec(regime, obs_idx, seed, err, e_last, g=0.1, capped=0):
    valid = math.isfinite(err)
    return dict(method="richardson_1", regime=regime, obs_idx=obs_idx, noise=0.0, seed=seed,
                target_g=g, estimate=0.0, error=err if valid else nan, valid=int(valid),
                catastrophic=int(not valid), E_last=e_last, capped=capped, L_true=0.1, L_hat=0.0,
                n_f=300, achieved_g=g)


def test_R_R_denominator_rule_exact_ratio_inf_and_nan(tmp_path):
    assert skill_score(0.0, 0.0) == 1.0                       # E_last exactly 0, Richardson exact
    assert skill_score(SKILL_EPS, 0.0) == 1.0                 # at the threshold: still exact
    assert skill_score(2 * SKILL_EPS, 0.0) == inf             # E_last 0, Richardson not exact
    assert math.isnan(skill_score(nan, 0.0))                  # Richardson invalid -> NaN, not the branch
    assert math.isnan(skill_score(0.5, nan))
    mask = zero_denominator_flags([0.0, 2 * SKILL_EPS, nan, 0.5], [0.0, 0.0, 0.0, 0.5])
    assert mask.tolist() == [True, True, False, False]
    # through richardson_targets: cell X (E_last = 0, Richardson exact on seed 0, not exact on
    # seed 1, invalid on seed 2) -> ratios [1, inf] -> median inf; n_zero_denominator counts the
    # two finite records, the invalid one is excluded from the branch
    rows = [_rec("X", 90, 0, 0.0, 0.0), _rec("X", 90, 1, 1e-6, 0.0), _rec("X", 90, 2, nan, 0.0),
            _rec("Y", 90, 0, 0.2, 0.4), _rec("Y", 90, 1, 0.1, 0.4)]
    df_t = richardson_targets(pd.DataFrame(rows), 0.1, str(tmp_path))
    x = df_t[df_t.regime == "X"].iloc[0]
    y = df_t[df_t.regime == "Y"].iloc[0]
    assert x.R_R_med == inf and x.n_zero_denominator == 2 and x.zero_denominator_cell == 1
    assert x.n_valid_R == 2 and x.n_total == 3
    assert y.R_R_med == pytest.approx(np.median([0.5, 0.25])) and y.n_zero_denominator == 0
    assert y.log_med_error_R == pytest.approx(math.log(0.15))
    # a cell with exact Richardson and E_last = 0 gives log(0) = -inf, which the correlation
    # keeps as a value (notna) rather than dropping it
    assert x.med_error_R == pytest.approx(5e-7) and math.isfinite(x.log_med_error_R)


def test_spearman_with_plus_inf_in_the_target_ranks_it_largest(tmp_path):
    # 20 cells of one regime; the feature rises with the cell index; R_R_med rises too and is
    # +inf on the last cell (the E_last <= SKILL_EPS branch): the rank correlation must be 1
    obs = list(range(20, 40))
    feat_rows, tgt_rows = [], []
    for i, o in enumerate(obs):
        for seed in range(2):
            feat_rows.append(dict(regime="A", obs_idx=o, noise=0.0, seed=seed, L_true=0.1, L_hat=0.0,
                                  **{f: float(i) + 0.01 * seed for f in FEATURE_COLS}))
        rr = inf if i == len(obs) - 1 else 1.0 + 0.1 * i
        tgt_rows.append(dict(regime="A", obs_idx=o, noise=0.0, target_g=0.1, n_f=300.0, achieved_g=0.1,
                             capped=0, R_R_med=rr, log_med_error_R=math.log(0.01 + 0.001 * i),
                             med_error_R=0.01 + 0.001 * i, n_valid_R=2, n_total=2,
                             n_zero_denominator=int(rr == inf), zero_denominator_cell=int(rr == inf)))
    df_corr, base = run_correlation_analysis(pd.DataFrame(feat_rows), pd.DataFrame(tgt_rows), 0.1,
                                             str(tmp_path), "_review")
    row = df_corr[(df_corr.regime == "ALL") & (df_corr.feature == FEATURE_COLS[0])].iloc[0]
    assert row.n_cells == 20 and row.n_dropped_nan == 0          # the +inf cell is not dropped
    assert row.spearman_vs_RR == pytest.approx(1.0)
    assert row.spearman_vs_log_err == pytest.approx(1.0)
    # the same statement directly: +inf ranks largest, exactly like a huge finite value
    x = np.arange(20, dtype=float)
    y = x.copy(); y[-1] = inf
    y2 = x.copy(); y2[-1] = 1e300
    assert spearmanr(x, y).statistic == spearmanr(x, y2).statistic == pytest.approx(1.0)


# ── classifier protocols ─────────────────────────────────────────────────────

def test_classifier_folds_with_one_depth_removed_have_no_leakage():
    df = _feature_table(True, np.random.RandomState(11))
    df = df[df.obs_idx != DEPTHS[2]]                              # one depth removed from the grid
    table = classifier_table(df)
    assert table.obs_idx.nunique() == len(DEPTHS) - 1
    folds, note = classifier_folds(table, "grouped_by_depth")
    assert len(folds) == len(DEPTHS) - 1 and "one per obs_idx" in note
    depth = table.obs_idx.values
    tested = np.zeros(len(table), dtype=int)
    for tr, te in folds:
        assert not set(depth[tr]) & set(depth[te])               # no depth on both sides
        assert len(set(depth[te])) == 1
        assert len(tr) + len(te) == len(table)
        tested[te] += 1
    assert (tested == 1).all()
    # rows of the same regime at the neighbouring depths are in training (transfer across depth)
    tr, te = folds[0]
    held = depth[te][0]
    for regime in REGIMES:
        assert ((table.regime.values[tr] == regime) & (depth[tr] != held)).any()


def test_classifier_single_noise_level_degrades_as_documented(tmp_path):
    df = _feature_table(True, np.random.RandomState(12))
    one = df[df.noise == NOISES[0]]
    folds, note = classifier_folds(classifier_table(one), "grouped_by_noise")
    assert folds == [] and note == "only 1 distinct noise value(s): the protocol cannot hold out a group"
    out = regime_classifier(one, str(tmp_path))
    gn = out[out.protocol == "grouped_by_noise"]
    assert (gn.n_samples == 0).all() and gn.accuracy.isna().all()
    assert (gn.note == note).all()
    ov = out[(out.protocol == "grouped_by_depth") & (out.regime == "__OVERALL__")].iloc[0]
    assert ov.n_samples == len(REGIMES) * len(DEPTHS) and ov.accuracy == 1.0
    leg = out[(out.protocol == "stratified_5fold_legacy") & (out.regime == "__OVERALL__")].iloc[0]
    assert leg.n_samples == len(REGIMES) * len(DEPTHS)
    assert "5 stratified folds" in leg.note


# ── ladders: constructors with a roster order reproduce the registry bitwise ─

def _call(fn, seq, idx, fx, cfg):
    try:
        return ("v", float(fn(seq, idx, fx, cfg)))
    except Exception as exc:                                   # both sides must raise alike
        return ("x", type(exc).__name__)


def _same_result(a, b):
    if a[0] != b[0]:
        return False
    if a[0] == "x":
        return a[1] == b[1]
    return (math.isnan(a[1]) and math.isnan(b[1])) or a[1] == b[1]


def test_ladder_constructors_with_roster_orders_equal_the_registry_on_50_windows():
    from scripts.order_ladders import LADDERS, all_variants
    rng = np.random.RandomState(2026)
    windows = []
    for k in range(50):
        n = int(rng.choice([3, 4, 5, 6, 8, 12, 20, 40, 60]))       # short windows starve high orders
        start = int(rng.randint(1, 60))
        idx = list(range(start, start + n))
        L = rng.uniform(0.005, 0.5)
        gap = rng.uniform(0.2, 0.8) * np.exp(-rng.uniform(0.01, 0.1) * np.arange(start, start + n))
        seq = L + gap + rng.uniform(0, 0.01) * rng.randn(n)
        if k % 5 == 0:                                             # windows with a NaN inside
            seq[int(rng.randint(0, n))] = np.nan
        if k % 7 == 0:                                             # a flat window (zero differences)
            seq[:] = L
        windows.append((list(seq), idx))
    cfg = build_cfg(500, 0.0, 0.1)
    n_checked = 0
    for family, orders, roster, make, label in LADDERS:
        for order, name in roster.items():
            assert order in list(orders)
            fn_ctor = make(order)                                    # the constructor path
            fn_reg = METHODS[name]                                   # the registry entry
            for seq, idx in windows:
                a = _call(fn_ctor, seq, idx, 500.0, cfg)
                b = _call(fn_reg, seq, idx, 500.0, cfg)
                assert _same_result(a, b), (family, name, len(seq), a, b)
                n_checked += 1
    assert n_checked == 50 * sum(len(r) for _, _, r, _, _ in LADDERS)
    # and the variant objects themselves, as the ladder script evaluates them
    for v in all_variants():
        if v.is_roster:
            for seq, idx in windows:
                assert _same_result(_call(v.fn, seq, idx, 500.0, cfg), _call(METHODS[v.roster_name], seq, idx, 500.0, cfg))


# ── perturbations ────────────────────────────────────────────────────────────

def test_perturbation_factors_trial_count_changes_shape_but_not_the_leading_rows():
    f3 = perturbation_factors("single_exp", 0, 90, 0.005, 60, 3, 0.02)
    f5 = perturbation_factors("single_exp", 0, 90, 0.005, 60, 5, 0.02)
    assert f3.shape == (3, 60) and f5.shape == (5, 60)
    assert np.array_equal(f5[:3], f3)                              # one stream, row-major: a prefix
    assert not np.array_equal(f5[3], f3[0])
    # the window length is part of the shape, not of the key: the first draws coincide
    f30 = perturbation_factors("single_exp", 0, 90, 0.005, 30, 3, 0.02)
    assert np.array_equal(f30.ravel(), f3.ravel()[:90])
    assert perturbation_key("single_exp", 0, 90, 0.005) == perturbation_key("single_exp", 0, 90, 0.005)
    # sigma enters through int(sigma * 1e6) % 9973: two sigmas that collide give the same key
    assert perturbation_key("single_exp", 0, 90, 0.0) == perturbation_key("single_exp", 0, 90, 0.009973)


def test_reversing_regimes_leaves_perturb_iqr_unchanged_and_a_broken_key_is_caught(monkeypatch):
    import phases.phase5a as P5
    regimes = ["single_exp", "power_law", "rational_decay"]
    keys = ["regime", "seed", "target_g", "method"]
    methods = ["richardson_1", "rational_fit", "pade_22", "last_value", "constant_assumed",
               "window_mean", "window_min", "constant_oracle"]
    pert = ["richardson_1", "rational_fit", "pade_22"]

    def block(regs):
        return P5.evaluate_block(90, 0.0, n_max=90, n_seeds=1, regimes=regs, gap_fractions=[0.1],
                                 window_len=60, perturb_trials=3, perturb_scale=0.02, dangerous=set(),
                                 methods=methods, perturb_methods=pert)

    def same(a, b):
        av, bv = a.to_numpy(dtype=float), b.to_numpy(dtype=float)
        return bool(np.all((av == bv) | (np.isnan(av) & np.isnan(bv))))

    fwd, rev = block(regimes), block(list(reversed(regimes)))
    m = fwd.merge(rev, on=keys, suffixes=("_f", "_r"))
    assert len(m) == len(fwd) == len(rev)
    assert same(m["perturb_iqr_f"], m["perturb_iqr_r"])
    assert m[m.method.isin(pert)].perturb_iqr_f.notna().all()

    # break the pairing on purpose: a key that also counts the calls made so far, so the
    # factors of a window depend on how many windows were evaluated before it
    calls = {"n": 0}
    real = perturbation_factors

    def order_dependent(regime, seed, obs_idx, sigma, n_points, n_trials, scale):
        calls["n"] += 1
        rng = np.random.RandomState((perturbation_key(regime, seed, obs_idx, sigma) + calls["n"]) % 2 ** 32)
        return 1.0 + float(scale) * rng.randn(int(n_trials), int(n_points))

    monkeypatch.setattr(P5, "perturbation_factors", order_dependent)
    fwd2, rev2 = block(regimes), block(list(reversed(regimes)))
    m2 = fwd2.merge(rev2, on=keys, suffixes=("_f", "_r"))
    sub = m2[m2.method.isin(pert)]
    assert not same(sub["perturb_iqr_f"], sub["perturb_iqr_r"])   # the pairing test would fail
    assert same(m2["estimate_f"], m2["estimate_r"])                 # central estimates untouched
    monkeypatch.setattr(P5, "perturbation_factors", real)


# ── exclusion boundary ───────────────────────────────────────────────────────

def _agg_row(method, regime, g, vr, cr=0.0, me=0.01, capped=0, hold=0, oracle=0, trivial=0):
    return dict(method=method, regime=regime, target_g=g, noise=0.0, valid_rate=vr, cat_rate=cr,
                med_error=me, capped=capped, is_holdout=hold, is_oracle=oracle, is_trivial=trivial)


def test_exclusion_boundary_0899_excluded_0900_kept():
    floor = CFG_MOD.RANK_MIN_VALID
    assert floor == 0.9
    rows = [
        _agg_row("shanks_1", "r1", 0.1, 0.899),                         # below the floor -> excluded
        _agg_row("shanks_2", "r1", 0.1, 0.900),                         # at the floor -> kept
        _agg_row("shanks_3", "r1", 0.5, 0.898), _agg_row("shanks_3", "r1", 0.1, 0.900),   # mean 0.899
        _agg_row("shanks_4", "r1", 0.5, 0.880), _agg_row("shanks_4", "r1", 0.1, 0.920),   # mean 0.900
        _agg_row("wynn_eps_1", "r1", 0.1, 0.8999996),                   # stored as 0.900000 (six decimals), still excluded
        _agg_row("wynn_eps_2", "r1", 0.1, 0.5), _agg_row("wynn_eps_2", "r2", 0.1, 1.0, capped=1),  # capped cell ignored
        _agg_row("last_value", "r1", 0.1, 0.1, trivial=1),              # a trivial is never excluded
        _agg_row("constant_oracle", "r1", 0.1, 1.0, oracle=1, trivial=1),
    ]
    dangerous, table = derive_dangerous(pd.DataFrame(rows))
    t = table.set_index("method")
    assert dangerous == frozenset({"shanks_1", "shanks_3", "wynn_eps_1", "wynn_eps_2"})
    assert t.loc["shanks_2"].dangerous == 0 and t.loc["shanks_4"].dangerous == 0
    assert t.loc["shanks_3"].valid_rate == pytest.approx(0.899)
    assert t.loc["shanks_4"].valid_rate == pytest.approx(0.900)
    # the flag is decided on the unrounded rate while the table stores six decimals (the
    # criterion text says so)
    assert t.loc["wynn_eps_1"].valid_rate == 0.9 and t.loc["wynn_eps_1"].dangerous == 1
    assert t.loc["wynn_eps_2"].n_cells == 1 and t.loc["wynn_eps_2"].valid_rate == 0.5
    assert t.loc["last_value"].dangerous == 0 and "constant_oracle" not in t.index
