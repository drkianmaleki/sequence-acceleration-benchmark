"""
tests/test_perturbations.py
===========================
Paired perturbation diagnostics (src/diagnostics.py, Phases 4 and 5a):

  * the perturbation factors of a window are a deterministic function of
    (regime, seed, obs_idx, sigma) and do not depend on the call order;
  * a method's perturb_iqr does not depend on which other methods are
    evaluated (Phase 5a block with one method removed) or on the order of the
    regimes (Phase 5a block and Phase 4 run with the regime list reversed);
  * every method evaluated on a cell receives identical perturbed windows
    (two spy methods record what they are handed);
  * shift_iqr is unchanged: pinned values on a fixed window.
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

import src.accelerators as ACC  # noqa: E402
from src.diagnostics import (perturb_iqr_with_factors, perturbation_factors,  # noqa: E402
                             perturbation_key)
from tests.tested_environment import SKIP_REASON, is_tested_environment  # noqa: E402
from phases.phase5a import EVAL_METHODS, PERTURB_METHODS, evaluate_block  # noqa: E402

# deterministic regimes (no intrinsic noise): at sigma = 0 the windows do not
# depend on the order in which the regimes consume the noise stream, so the
# perturbation factors are the only randomness of the diagnostic
REGIMES = ["single_exp", "power_law"]
OBS, SIGMA, N_TRIALS, SCALE = 90, 0.0, 3, 0.02
KEYS = ["regime", "seed", "target_g", "method"]


def _block(regimes, methods=None, perturb_methods=None):
    return evaluate_block(OBS, SIGMA, n_max=OBS, n_seeds=1, regimes=regimes, gap_fractions=[0.1],
                          window_len=60, perturb_trials=N_TRIALS, perturb_scale=SCALE,
                          dangerous=set(), methods=methods, perturb_methods=perturb_methods)


def _same(a: pd.Series, b: pd.Series) -> bool:
    av, bv = a.to_numpy(dtype=float), b.to_numpy(dtype=float)
    return bool(np.all((av == bv) | (np.isnan(av) & np.isnan(bv))))


def test_factors_are_deterministic_and_independent_of_call_order():
    a1 = perturbation_factors("single_exp", 0, OBS, SIGMA, 60, N_TRIALS, SCALE)
    b1 = perturbation_factors("power_law", 0, OBS, SIGMA, 60, N_TRIALS, SCALE)
    b2 = perturbation_factors("power_law", 0, OBS, SIGMA, 60, N_TRIALS, SCALE)
    a2 = perturbation_factors("single_exp", 0, OBS, SIGMA, 60, N_TRIALS, SCALE)
    assert a1.shape == (N_TRIALS, 60)
    assert np.array_equal(a1, a2) and np.array_equal(b1, b2)
    assert not np.array_equal(a1, b1)
    key = (0 * 999 + OBS * 7 + int(SIGMA * 1e6) % 9973 + zlib.crc32(b"single_exp") % 100003) % 2 ** 32
    assert perturbation_key("single_exp", 0, OBS, SIGMA) == key
    assert np.array_equal(a1, 1.0 + SCALE * np.random.RandomState(key).randn(N_TRIALS, 60))
    # every component of the key matters
    base = perturbation_key("single_exp", 3, 60, 0.005)
    assert base != perturbation_key("single_exp", 4, 60, 0.005)
    assert base != perturbation_key("single_exp", 3, 90, 0.005)
    assert base != perturbation_key("single_exp", 3, 60, 0.02)
    assert base != perturbation_key("power_law", 3, 60, 0.005)
    assert 0 <= base < 2 ** 32


def test_perturb_iqr_with_factors_rules():
    seq = list(0.3 + 0.5 * np.exp(-0.05 * np.arange(31, 91)))
    idx = list(range(31, 91))
    cfg = {"L_inf": 0.0, "ridge": 1e-8, "min_valid": -0.5, "max_valid": 500.0, "denom_tol": 1e-14}
    F = perturbation_factors("single_exp", 0, 90, 0.0, len(seq), 5, 0.02)
    v1 = perturb_iqr_with_factors(ACC.METHODS["rational_fit"], seq, idx, 500.0, cfg, F)
    v2 = perturb_iqr_with_factors(ACC.METHODS["rational_fit"], seq, idx, 500.0, cfg, F)
    assert math.isfinite(v1) and v1 >= 0 and v1 == v2
    # fewer than two valid perturbed estimates -> NaN; a raising method is invalid
    assert math.isnan(perturb_iqr_with_factors(lambda s, i, x, c: float("nan"), seq, idx, 500.0, cfg, F))
    assert math.isnan(perturb_iqr_with_factors(ACC.METHODS["rational_fit"], seq, idx, 500.0, cfg, F[:1]))

    def boom(s, i, x, c):
        raise ValueError("no")
    assert math.isnan(perturb_iqr_with_factors(boom, seq, idx, 500.0, cfg, F))


def test_perturb_iqr_does_not_depend_on_the_other_methods():
    full = _block(REGIMES)
    dropped = "pade_22"
    reduced = _block(REGIMES, methods=[m for m in EVAL_METHODS if m != dropped],
                     perturb_methods=[m for m in PERTURB_METHODS if m != dropped])
    assert dropped in set(full.method) and dropped not in set(reduced.method)
    m = full.merge(reduced, on=KEYS, suffixes=("_full", "_red"))
    assert len(m) == len(reduced)
    assert _same(m["perturb_iqr_full"], m["perturb_iqr_red"])
    assert _same(m["estimate_full"], m["estimate_red"])
    # the diagnostic was actually computed for the pool members
    assert full[full.method.isin(PERTURB_METHODS)].perturb_iqr.notna().mean() > 0.5


def test_perturb_iqr_does_not_depend_on_regime_order_phase5a():
    fwd = _block(REGIMES)
    rev = _block(list(reversed(REGIMES)))
    m = fwd.merge(rev, on=KEYS, suffixes=("_f", "_r"))
    assert len(m) == len(fwd) == len(rev)
    assert _same(m["perturb_iqr_f"], m["perturb_iqr_r"])
    assert _same(m["estimate_f"], m["estimate_r"])


def test_perturb_iqr_does_not_depend_on_regime_order_phase4(tmp_path):
    from phases.phase4 import run_phase4
    kw = dict(obs_idx_list=[OBS], noise_list=[SIGMA], gap_fractions=[0.1], n_seeds=1, window_len=60,
              shifts=[-1, 0, 1], perturb_trials=N_TRIALS, perturb_scale=SCALE, holdout_regimes=[], verbose=False)
    fwd = run_phase4(out_dir=str(tmp_path / "f"), core_regimes=REGIMES, **kw)
    rev = run_phase4(out_dir=str(tmp_path / "r"), core_regimes=list(reversed(REGIMES)), **kw)
    m = fwd.merge(rev, on=KEYS, suffixes=("_f", "_r"))
    assert len(m) == len(fwd) == len(rev)
    assert _same(m["perturb_iqr_f"], m["perturb_iqr_r"])
    assert _same(m["shift_iqr_f"], m["shift_iqr_r"])
    assert _same(m["estimate_f"], m["estimate_r"])
    pool = fwd[fwd.method.isin(["richardson_1", "rational_fit", "pade_22"])]
    assert pool.perturb_iqr.notna().all()


def test_every_method_receives_identical_perturbed_windows(monkeypatch):
    seen = {"spy_a": [], "spy_b": []}

    def make_spy(name):
        def spy(seq, indices, future_x, cfg):
            seen[name].append(np.asarray(seq, dtype=float).copy())
            return float(seq[-1])
        return spy

    monkeypatch.setitem(ACC.METHODS, "spy_a", make_spy("spy_a"))
    monkeypatch.setitem(ACC.METHODS, "spy_b", make_spy("spy_b"))
    methods = ["richardson_1", "spy_a", "spy_b"]
    df = _block(["single_exp"], methods=methods, perturb_methods=methods)
    assert set(df.method) == set(methods)
    # one central window plus N_TRIALS perturbed windows per spy (one horizon)
    assert len(seen["spy_a"]) == len(seen["spy_b"]) == 1 + N_TRIALS
    for wa, wb in zip(seen["spy_a"], seen["spy_b"]):
        assert np.array_equal(wa, wb)
    central = seen["spy_a"][0]
    F = perturbation_factors("single_exp", 0, OBS, SIGMA, len(central), N_TRIALS, SCALE)
    for t in range(N_TRIALS):
        assert np.array_equal(seen["spy_a"][1 + t], central * F[t])
        assert not np.array_equal(seen["spy_a"][1 + t], central)


@pytest.mark.skipif(not is_tested_environment(), reason=SKIP_REASON)
def test_shift_iqr_is_unchanged_pinned_values():
    from phases.phase4 import _cfg, _shift_iqr
    from tests.test_input_dependence import windows
    regime, sigma, seed, seq, idx = [w for w in windows() if w[0] == "single_exp" and w[1] == 0.005 and w[2] == 0][0]
    assert (len(seq), idx[0], idx[-1]) == (60, 31, 90)
    assert float(seq[0]) == pytest.approx(0.27891313775674553, rel=1e-12)
    cfg = _cfg(500, 0.0)
    pinned = {
        "richardson_1":   0.00013653365772742232,
        "rational_fit":   6.498680734914886e-05,
        "pade_22":        0.006668126024174086,
        "shanks_2":       0.0,
        "log_linear":     0.00012241382694934064,
        "richardson_a10": 0.000871109639299511,
    }
    for method, value in pinned.items():
        got = _shift_iqr(list(seq), list(idx), 500.0, method, cfg, [-2, -1, 0, 1, 2])
        assert got == pytest.approx(value, rel=1e-12, abs=1e-18), method
