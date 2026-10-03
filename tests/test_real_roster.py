"""
tests/test_real_roster.py
=========================
The roster evaluation of the recorded curves
(src.trajectories.evaluate_recorded_curves_roster, R9d Part B): the
per-record scoring on hand-made cases (through the one function, with the
registry entries of a few methods replaced by constants), the schema and
row count, the roster membership (every accelerator, no oracle), the
valid_strict rule, and the agreement of richardson_1, rational_fit and the
four trivial predictors with the legacy path (evaluate_recorded_curves) on
a subset of the committed curves.  The last test pins the legacy path to
the committed real_data_results_v2.csv, which is what the fingerprint
entries of src/trajectories.py and scripts/run_real_data.py rely on.
"""

import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import src.config as C                                                   # noqa: E402
from src.accelerators import METHODS                                      # noqa: E402
from src.pipeline import ACCEL_METHODS                                    # noqa: E402
from src.trivial import ORACLE_METHODS, SKILL_REFERENCE_METHODS           # noqa: E402
from src.trajectories import (ROSTER_COLUMNS, ROSTER_REAL_METHODS,        # noqa: E402
                              evaluate_recorded_curves, evaluate_recorded_curves_roster)

CURVES_CSV = os.path.join(_ROOT, "results", "real_data", "real_data_curves.csv")
LEGACY_CSV = os.path.join(_ROOT, "results", "real_data", "real_data_results_v2.csv")
SUBSET = dict(datasets=("adult", "higgs"), depths=[30, 60], targets=[300])


def _load_curves(names=None):
    df = pd.read_csv(CURVES_CSV)
    cols = [c for c in df.columns if c != "round" and (names is None or c in names)]
    return {c: df[c].to_numpy(dtype=float) for c in cols}


def _same(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return bool(np.all((a == b) | (np.isnan(a) & np.isnan(b))))


# ── scoring on hand-made cases ────────────────────────────────────────────────
def test_scoring_on_hand_made_cases(monkeypatch):
    # a linear descent: window = rounds 1..30, target round 60; the last value is 1.0 - 29 * 0.01
    curve = 1.0 - 0.01 * np.arange(60, dtype=float)
    last, true = curve[29], curve[59]
    E_last = abs(last - true)
    assert E_last > 1e-12
    cat = C.CAT_MULT * E_last
    const = lambda v: (lambda seq, idx, fx, cfg: v)                      # noqa: E731
    cases = {
        "richardson_1":   true + cat,                 # error exactly CAT_MULT * E_last: not catastrophic
        "richardson_2":   float("nan"),               # NaN: invalid, catastrophic, improve NaN
        "richardson_3":   C.MAX_VALID + 1.0,          # out of range: invalid
        "rational_fit":   true,                       # error 0 <= 1e-12: improve 1.0, skill vs last 0, wins
        "single_exp_fit": true + cat * (1 + 1e-6),    # just above the threshold: catastrophic
        "log_linear":     true - 0.5 * E_last,        # a valid, better estimate: improve 2
        "pade_11":        2.0 * np.max(curve[:30]) + 1e-9,   # valid for the benchmark, not strictly valid
        "pade_12":        -1e-9,                      # below zero: valid for the benchmark, not strictly valid
    }
    for m, v in cases.items():
        monkeypatch.setitem(METHODS, m, const(v))
    df = evaluate_recorded_curves_roster({"c": curve}, depths=[30], targets=[60], window_len=30, assumed_mode="zero")
    assert list(df.columns) == ROSTER_COLUMNS and len(df) == len(ROSTER_REAL_METHODS)
    r = df.set_index("method")
    assert r.loc["last_value", "error"] == pytest.approx(E_last) and (r.E_last == r.loc["last_value", "error"]).all()
    assert r.loc["last_value", "improve_ratio"] == 1.0 and r.loc["last_value", "win_vs_last"] == 0

    x = r.loc["richardson_1"]
    assert x.valid == 1 and x.catastrophic == 0 and x.error == pytest.approx(cat) and x.improve_ratio == pytest.approx(1 / C.CAT_MULT)
    assert x.win_vs_last == 0 and x.skill_vs_last == pytest.approx(C.CAT_MULT)
    x = r.loc["single_exp_fit"]
    assert x.valid == 1 and x.catastrophic == 1
    x = r.loc["richardson_2"]
    assert math.isnan(x.prediction_raw) and x.valid == 0 and x.valid_strict == 0 and x.catastrophic == 1
    assert math.isnan(x.prediction) and math.isnan(x.error) and math.isnan(x.improve_ratio) and math.isnan(x.skill)
    assert math.isnan(x.skill_vs_last) and x.win_vs_last == 0 and x.win_vs_assumed == 0
    x = r.loc["richardson_3"]
    assert x.prediction_raw == C.MAX_VALID + 1.0 and x.valid == 0 and math.isnan(x.prediction) and x.catastrophic == 1
    x = r.loc["rational_fit"]
    assert x.valid == 1 and x.error == 0.0 and x.improve_ratio == 1.0 and x.catastrophic == 0
    assert x.win_vs_last == 1 and x.skill_vs_last == 0.0 and x.skill == 0.0
    x = r.loc["log_linear"]
    assert x.improve_ratio == pytest.approx(2.0) and x.skill_vs_last == pytest.approx(0.5) and x.win_vs_last == 1
    assert r.loc["pade_11", "valid"] == 1 and r.loc["pade_11", "valid_strict"] == 0
    assert r.loc["pade_12", "valid"] == 1 and r.loc["pade_12", "valid_strict"] == 0
    assert (r.window_max == float(np.max(curve[:30]))).all() and (r.L_hat == 0.0).all() and (r.assumed_mode == "zero").all()
    assert (r.true_val == true).all() and (r.obs_depth == 30).all() and (r.target_round == 60).all()
    # ref_error = the best of the four deployable trivials; the trivials' skill >= 1 with a minimum of exactly 1
    triv = r.loc[list(SKILL_REFERENCE_METHODS)]
    assert (r.ref_error == triv.error.min()).all() and triv.skill.min() == pytest.approx(1.0)


def test_exact_zero_last_value_error(monkeypatch):
    # a flat curve: the last value equals the truth, E_last = 0
    curve = np.full(60, 0.3)
    monkeypatch.setitem(METHODS, "richardson_1", lambda s, i, f, c: 0.31)       # error 0.01 > 1e-12
    monkeypatch.setitem(METHODS, "rational_fit", lambda s, i, f, c: 0.3)        # exact
    df = evaluate_recorded_curves_roster({"flat": curve}, depths=[30], targets=[60], window_len=30, assumed_mode="zero")
    r = df.set_index("method")
    assert r.loc["last_value", "error"] == 0.0 and (r.E_last == 0.0).all()
    x = r.loc["richardson_1"]
    assert x.valid == 1 and x.catastrophic == 0                  # E_last <= 1e-12: the catastrophe rule cannot fire
    assert x.improve_ratio == 0.0                                 # E_last / error with E_last = 0 (Phase-1 rule)
    assert math.isinf(x.skill_vs_last) and x.win_vs_last == 0    # skill_score: exact reference, inexact method
    x = r.loc["rational_fit"]
    assert x.error == 0.0 and x.improve_ratio == 1.0 and x.skill_vs_last == 1.0 and x.win_vs_last == 0


# ── schema, roster, valid_strict and agreement with the legacy path ───────────
@pytest.mark.skipif(not os.path.exists(CURVES_CSV), reason="recorded curves not present")
def test_schema_roster_and_valid_strict():
    curves = _load_curves(SUBSET["datasets"])
    df = evaluate_recorded_curves_roster(curves, SUBSET["depths"], SUBSET["targets"], window_len=60, assumed_mode="zero")
    cells = len(curves) * len(SUBSET["depths"]) * len(SUBSET["targets"])
    assert list(df.columns) == ROSTER_COLUMNS
    assert len(df) == cells * len(ROSTER_REAL_METHODS)
    assert ROSTER_REAL_METHODS == list(ACCEL_METHODS) + list(SKILL_REFERENCE_METHODS)
    assert set(ACCEL_METHODS) <= set(df.method) and not (set(df.method) & set(ORACLE_METHODS))
    assert (df[df.is_trivial == 1].method.isin(SKILL_REFERENCE_METHODS)).all()
    per_cell = df.groupby(["dataset", "obs_depth", "target_round"]).method.apply(list)
    assert all(ms == ROSTER_REAL_METHODS for ms in per_cell)
    raw = df.prediction_raw.to_numpy(dtype=float)
    strict = np.isfinite(raw) & (raw >= 0.0) & (raw <= 2.0 * df.window_max.to_numpy(dtype=float))
    assert (df.valid_strict.to_numpy() == strict.astype(int)).all()
    valid = np.isfinite(raw) & (raw >= C.MIN_VALID) & (raw <= C.MAX_VALID)
    assert (df.valid.to_numpy() == valid.astype(int)).all()
    assert _same(df.prediction, np.where(valid, raw, np.nan))
    assert (df.catastrophic[df.valid == 0] == 1).all()
    for (_, _, _), cell in df.groupby(["dataset", "obs_depth", "target_round"]):
        last = cell[cell.method == "last_value"].iloc[0]
        assert (cell.E_last == last.error).all()


@pytest.mark.skipif(not os.path.exists(CURVES_CSV), reason="recorded curves not present")
def test_roster_agrees_with_the_legacy_path_on_the_shared_methods():
    curves = _load_curves(SUBSET["datasets"])
    roster = evaluate_recorded_curves_roster(curves, SUBSET["depths"], SUBSET["targets"], window_len=60, assumed_mode="zero")
    legacy, _ = evaluate_recorded_curves(curves, SUBSET["depths"], SUBSET["targets"], window_len=60, assumed_mode="zero")
    keys = ["dataset", "obs_depth", "target_round", "method"]
    shared = ["richardson_1", "rational_fit", *SKILL_REFERENCE_METHODS]
    m = (legacy[legacy.method.isin(shared)][keys + ["prediction", "error"]]
         .merge(roster[keys + ["prediction", "error"]], on=keys, suffixes=("_legacy", "_roster")))
    assert len(m) == len(curves) * len(SUBSET["depths"]) * len(SUBSET["targets"]) * len(shared)
    assert _same(m.prediction_legacy, m.prediction_roster)
    assert _same(m.error_legacy, m.error_roster)


@pytest.mark.skipif(not (os.path.exists(CURVES_CSV) and os.path.exists(LEGACY_CSV)), reason="committed real-data files not present")
def test_legacy_path_reproduces_the_committed_results_on_a_subset():
    """The legacy recorded-curve outputs are unchanged (the guarantee behind the
    fingerprint records of src/trajectories.py and scripts/run_real_data.py)."""
    curves = _load_curves(SUBSET["datasets"])
    legacy, _ = evaluate_recorded_curves(curves, SUBSET["depths"], SUBSET["targets"], window_len=60, assumed_mode="zero")
    committed = pd.read_csv(LEGACY_CSV, float_precision="round_trip")
    keys = ["dataset", "obs_depth", "target_round", "method"]
    m = legacy[keys + ["prediction", "error", "perturb_iqr", "skill"]].merge(
        committed[keys + ["prediction", "error", "perturb_iqr", "skill"]], on=keys, suffixes=("_now", "_committed"))
    assert len(m) == len(legacy) > 0
    for col in ("prediction", "error", "perturb_iqr", "skill"):
        assert _same(m[f"{col}_now"], m[f"{col}_committed"]), col
