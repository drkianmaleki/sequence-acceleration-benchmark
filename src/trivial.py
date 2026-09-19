"""
trivial.py
==========
Trivial comparators, registered as first-class methods.

The rejected paper was beaten 7.8x at n_f = 5000 by a predictor that simply
returned the configured asymptote.  That predictor, and its equally trivial
relatives, are now benchmark methods so that no accelerator can be ranked
without being compared against them.

All five share the accelerator signature  fn(seq, indices, future_x, cfg).

    constant_assumed   returns L_hat  (cfg['L_inf']); what a deployed system
                       could actually do with its asymptote guess
    constant_oracle    returns L_true (cfg['L_true']); evaluation-only
                       reference.  Flagged in ORACLE_METHODS so rankings can
                       exclude it while every table can still show it.
    window_mean        mean of the observation window
    window_min         min of the observation window
    last_value         alias of current_value (registered in accelerators.py
                       as the very same function object)

Skill score
-----------
    skill(method) = err(method) / err(best of SKILL_REFERENCE_METHODS)

where the reference set is {constant_assumed, last_value, window_mean,
window_min}: the oracle is never in the denominator.  skill < 1 means the
method beat every non-oracle trivial predictor on that cell.
"""

import math
from typing import Dict, Optional

import numpy as np

TRIVIAL_METHOD_NAMES = [
    "constant_assumed",
    "constant_oracle",
    "window_mean",
    "window_min",
    "last_value",
]

ORACLE_METHODS = frozenset({"constant_oracle"})

SKILL_REFERENCE_METHODS = (
    "constant_assumed", "last_value", "window_mean", "window_min",
)

_SKILL_EPS = 1e-12


def trivial_constant_assumed(seq, indices, future_x: float, cfg: dict) -> float:
    """Return the assumed asymptote L_hat supplied under cfg['L_inf']."""
    L_hat = cfg.get("L_inf", None)
    return float(L_hat) if L_hat is not None else float("nan")


def trivial_constant_oracle(seq, indices, future_x: float, cfg: dict) -> float:
    """Return the true asymptote L_true (evaluation-only reference).

    The evaluation harness places L_true under cfg['L_true'] for this method
    alone; no other method reads that key (tests/test_redesign.py enforces
    it).  Without the key the oracle returns NaN rather than guessing.
    """
    L_true = cfg.get("L_true", None)
    return float(L_true) if L_true is not None else float("nan")


def trivial_window_mean(seq, indices, future_x: float, cfg: dict) -> float:
    arr = np.asarray(seq, dtype=float)
    return float(np.mean(arr)) if arr.size else float("nan")


def trivial_window_min(seq, indices, future_x: float, cfg: dict) -> float:
    arr = np.asarray(seq, dtype=float)
    return float(np.min(arr)) if arr.size else float("nan")


def trivial_last_value(seq, indices, future_x: float, cfg: dict) -> float:
    """Last observed value.  accelerators.py overrides this entry with the
    current_value function object itself so the alias is literal."""
    return float(seq[-1]) if len(seq) else float("nan")


TRIVIAL_METHODS = {
    "constant_assumed": trivial_constant_assumed,
    "constant_oracle":  trivial_constant_oracle,
    "window_mean":      trivial_window_mean,
    "window_min":       trivial_window_min,
    "last_value":       trivial_last_value,
}


def is_oracle(method: str) -> bool:
    return method in ORACLE_METHODS


def best_reference_error(errors: Dict[str, float]) -> float:
    """Smallest finite error among SKILL_REFERENCE_METHODS, or NaN if none."""
    vals = [errors.get(m, float("nan")) for m in SKILL_REFERENCE_METHODS]
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    return min(vals) if vals else float("nan")


def skill_score(err_method: Optional[float], err_reference: Optional[float]) -> float:
    """
    err(method) / err(best trivial reference).

    NaN when the method is invalid or no reference is valid.  When the best
    reference is exact (error below 1e-12, e.g. staircase after its last step
    under oracle assumptions) the ratio is 1.0 if the method is also exact and
    +inf otherwise, so medians remain well defined.
    """
    if err_method is None or err_reference is None:
        return float("nan")
    if not (math.isfinite(err_method) and math.isfinite(err_reference)):
        return float("nan")
    if err_reference <= _SKILL_EPS:
        return 1.0 if err_method <= _SKILL_EPS else float("inf")
    return float(err_method / err_reference)
