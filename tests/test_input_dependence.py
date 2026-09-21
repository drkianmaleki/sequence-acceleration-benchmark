"""
test_input_dependence.py
========================
Permanent guard against degenerate accelerators (redesign v2, Prompt 5A).

Motivation: weniger_d1 / weniger_d2 returned 0 for every input (the remainder
estimate w_n = s_n cancelled the sequence out of the numerator) and went
unnoticed through every phase because 0 is a legal estimate: with the
deployment-honest assumed asymptote L_hat = 0 they were an exact copy of
the constant_assumed comparator.  Two properties every accelerator must have:

    (a) REACTION   its output changes under a 1 % multiplicative perturbation
                   of the window on at least REACT_MIN of the windows where it
                   is finite (a finite value that turns NaN counts as a change);
    (b) NON-TRIVIALITY   its output coincides (within EQ_TOL) with one of the
                   deployable trivial predictors -- last value, window mean,
                   window min, L_hat -- on at most EQ_MAX of the windows.

Exempt: current_value (it IS the last value by design).

Windows: 8 regimes x 6 seeds x sigma in {0, 0.005} = 96 windows, n_obs = 90,
window length 60, L_hat = 0, generated exactly as Phase 1 generates them.

The module also prints the L_hat-consumer list: the accelerators whose output
changes when L_hat moves from 0 to 0.5 * min(window).  Phase 5b sweep 1 uses
this list (phases.phase5b.LHAT_CONSUMERS must equal it).

Run with  pytest tests/test_input_dependence.py -s  to see the per-method table.
"""

import os
import sys
from functools import lru_cache

import numpy as np
import pandas as pd
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import src.config as CFG_MOD                                      # noqa: E402
from src.accelerators import METHODS                              # noqa: E402
from src.evaluation import build_cfg                              # noqa: E402
from src.generators import regime_functions                       # noqa: E402
from src.pipeline import ACCEL_METHODS                            # noqa: E402

REGIMES = ["single_exp", "two_exp", "power_law", "rational_decay",
           "mixed_pow_rat", "multiphase", "osc_exp", "log_slow"]
SEEDS = range(6)
NOISE = [0.0, 0.005]
OBS_IDX = CFG_MOD.OBS_IDX          # 90
WINDOW_LEN = CFG_MOD.WINDOW_LEN    # 60
FUTURE_X = 500.0
L_HAT = 0.0
PERTURB_SCALE = 0.01
REACT_MIN = 0.90
EQ_MAX = 0.10
EQ_TOL = 1e-12
EXEMPT = {"current_value"}

# L_hat consumers.  The Report-4 review expected the nine direct consumers
# (the curve fits and Richardson fits use L_hat as a starting value or offset;
# stability_weighted is an ensemble over a pool that contains them).  The
# measured set also contains median_ensemble, whose pool likewise contains the
# consumers, so its median moves on the windows where a consumer's value
# crosses it.  The test asserts the measured set so that a change is noticed.
EXPECTED_LHAT_CONSUMERS = {
    "log_linear", "double_exp_fit", "richardson_3", "richardson_2",
    "single_exp_fit", "rational_fit", "richardson_1", "log_fit",
    "stability_weighted", "median_ensemble",
}


@lru_cache(maxsize=None)
def windows():
    """The 96 evaluation windows, built as Phase 1 builds them."""
    n_arr = np.arange(OBS_IDX + 1, dtype=float)
    w_start = max(0, OBS_IDX - WINDOW_LEN + 1)
    idx_win = list(range(w_start, OBS_IDX + 1))
    out = []
    for regime in REGIMES:
        for sigma in NOISE:
            for seed in SEEDS:
                gen, _, _ = regime_functions(regime, seed)
                rng = np.random.RandomState(seed * 137 + int(sigma * 1e6) % 9973)
                seq_full = gen(n_arr, rng, sigma)
                seq = np.asarray(seq_full[w_start:OBS_IDX + 1], dtype=float)
                out.append((regime, sigma, seed, seq, idx_win))
    assert len(out) == len(REGIMES) * len(NOISE) * len(SEEDS) == 96
    return tuple(out)


def _call(method, seq, idx, L_hat):
    cfg = build_cfg(int(FUTURE_X), L_hat)
    try:
        return float(METHODS[method](list(seq), list(idx), FUTURE_X, cfg))
    except Exception:
        return float("nan")


@lru_cache(maxsize=None)
def analysis():
    """One row per accelerator: reaction rate, trivial-equality rate, L_hat use."""
    rows = []
    for k, method in enumerate(ACCEL_METHODS):
        n_fin = n_react = n_eq = n_lhat = 0
        for w, (regime, sigma, seed, seq, idx) in enumerate(windows()):
            base = _call(method, seq, idx, L_HAT)
            rng = np.random.RandomState(12345 + w)
            pert = seq * (1.0 + PERTURB_SCALE * rng.uniform(-1.0, 1.0, size=seq.size))
            alt = _call(method, pert, idx, L_HAT)
            if np.isfinite(base):
                n_fin += 1
                # a finite estimate that becomes NaN (out of range) under the
                # perturbation has changed; only an unchanged value is a non-reaction
                if (not np.isfinite(alt)) or abs(alt - base) > EQ_TOL:
                    n_react += 1
                trivials = (seq[-1], float(np.mean(seq)), float(np.min(seq)), L_HAT)
                if any(abs(base - t) <= EQ_TOL for t in trivials):
                    n_eq += 1
            lh = _call(method, seq, idx, 0.5 * float(np.min(seq)))
            if (np.isfinite(base) != np.isfinite(lh)) or (np.isfinite(base) and abs(lh - base) > EQ_TOL):
                n_lhat += 1
        n_w = len(windows())
        rows.append(dict(method=method, n_windows=n_w, n_finite=n_fin,
                         reaction_rate=(n_react / n_fin) if n_fin else float("nan"),
                         equality_rate=n_eq / n_w,
                         lhat_windows=n_lhat, uses_lhat=n_lhat > 0))
    return pd.DataFrame(rows)


def _flag(df):
    react_fail = df[(~df.method.isin(EXEMPT)) & (df.reaction_rate < REACT_MIN)]
    eq_fail = df[(~df.method.isin(EXEMPT)) & (df.equality_rate > EQ_MAX)]
    return react_fail, eq_fail


def report(df):
    react_fail, eq_fail = _flag(df)
    lines = ["", f"input-dependence over {int(df.n_windows.iloc[0])} windows "
                 f"({len(REGIMES)} regimes x {len(SEEDS)} seeds x sigma {NOISE}; n_obs {OBS_IDX}, window {WINDOW_LEN}, L_hat {L_HAT})",
             f"  {'method':<20} {'finite':>6} {'reaction':>9} {'equality':>9}  flag"]
    for _, r in df.sort_values(["reaction_rate", "equality_rate"], ascending=[True, False]).iterrows():
        flags = []
        if r.method in EXEMPT:
            flags.append("exempt")
        else:
            if r.reaction_rate < REACT_MIN:
                flags.append(f"REACTION < {REACT_MIN:.0%}")
            if r.equality_rate > EQ_MAX:
                flags.append(f"EQUALS TRIVIAL > {EQ_MAX:.0%}")
        if flags or r.reaction_rate < 0.999 or r.equality_rate > 0:
            lines.append(f"  {r.method:<20} {int(r.n_finite):>6} {r.reaction_rate:>9.3f} {r.equality_rate:>9.3f}  {', '.join(flags)}")
    lines.append(f"  ... {int((df.reaction_rate >= 0.999).sum())} of {len(df)} accelerators react on >= 99.9 % of finite windows; "
                 f"{int((df.equality_rate == 0).sum())} never equal a trivial")
    lines.append(f"  flagged: reaction {sorted(react_fail.method)}; equality {sorted(eq_fail.method)}")
    consumers = sorted(df[df.uses_lhat].method, key=lambda m: -int(df.set_index('method').loc[m, 'lhat_windows']))
    lines.append(f"  L_hat consumers ({len(consumers)}; output differs between L_hat = 0 and 0.5 * min(window)): "
                 + ", ".join(f"{m} ({int(df.set_index('method').loc[m, 'lhat_windows'])})" for m in consumers))
    return "\n".join(lines)


def test_every_accelerator_reacts_to_its_input():
    df = analysis()
    print(report(df))
    react_fail, _ = _flag(df)
    assert react_fail.empty, ("output does not react to a 1 % window perturbation: "
                              + ", ".join(f"{r.method} ({r.reaction_rate:.3f})" for _, r in react_fail.iterrows()))


def test_no_accelerator_is_a_trivial_predictor():
    df = analysis()
    _, eq_fail = _flag(df)
    assert eq_fail.empty, ("output equals a deployable trivial on too many windows: "
                           + ", ".join(f"{r.method} ({r.equality_rate:.3f})" for _, r in eq_fail.iterrows()))


def test_lhat_consumer_list():
    df = analysis()
    consumers = set(df[df.uses_lhat].method)
    print("\n  L_hat consumers:", sorted(consumers))
    assert consumers == EXPECTED_LHAT_CONSUMERS, (
        f"+{sorted(consumers - EXPECTED_LHAT_CONSUMERS)} -{sorted(EXPECTED_LHAT_CONSUMERS - consumers)}")


def test_phase5b_consumer_list_matches():
    """Phase 5b sweep 1b evaluates exactly the measured L_hat consumers."""
    from phases.phase5b import LHAT_CONSUMERS, SWEEP1_METHODS
    df = analysis()
    assert set(LHAT_CONSUMERS) == set(df[df.uses_lhat].method)
    assert set(SWEEP1_METHODS) == set(LHAT_CONSUMERS) | {"constant_assumed"}
