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

Skill scores
------------
Two kinds, both per record (one method on one cell):

  hindsight best-of-four (strict)
    skill(method) = err(method) / err(best of SKILL_REFERENCE_METHODS on that cell)
    The denominator needs the truth to pick the trivial, so this is a
    HINDSIGHT bar: skill < 1 means the method beat every deployable trivial
    predictor there, including the one only hindsight could have chosen.
    Column ``skill``; aggregated as ``med_skill``.

  fixed-reference (deployable)
    skill_vs_<ref> = err(method) / err(<ref>)   and
    win_vs_<ref>   = 1 if err(method) < err(<ref>) else 0
    for each deployable trivial separately (<ref> in assumed, last, wmean,
    wmin = constant_assumed, last_value, window_mean, window_min).  Each
    denominator is a predictor a deployment could actually run.  Aggregated
    as ``med_skill_vs_<ref>`` (median) and ``win_rate_vs_<ref>`` (mean).

The oracle is never in any denominator.  ``skill_vs_table`` builds the
eight per-record columns; ``aggregate_skill_vs`` the eight aggregates.
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

# Fixed-reference skill: one (method, tag) per deployable trivial.
REFERENCE_TAGS = (
    ("constant_assumed", "assumed"),
    ("last_value",       "last"),
    ("window_mean",      "wmean"),
    ("window_min",       "wmin"),
)
SKILL_VS_COLS     = tuple(f"skill_vs_{t}"     for _, t in REFERENCE_TAGS)
WIN_VS_COLS       = tuple(f"win_vs_{t}"       for _, t in REFERENCE_TAGS)
MED_SKILL_VS_COLS = tuple(f"med_skill_vs_{t}" for _, t in REFERENCE_TAGS)
WIN_RATE_VS_COLS  = tuple(f"win_rate_vs_{t}"  for _, t in REFERENCE_TAGS)
SKILL_VS_RECORD_COLS = SKILL_VS_COLS + WIN_VS_COLS
SKILL_VS_AGG_COLS    = MED_SKILL_VS_COLS + WIN_RATE_VS_COLS

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


def skill_vs_table(err_method: Optional[float], errors: Dict[str, float]) -> Dict[str, float]:
    """
    The eight fixed-reference columns for one record:
        skill_vs_<tag> = err_method / err(<ref>)   (NaN when either is invalid)
        win_vs_<tag>   = 1 if err_method < err(<ref>) else 0  (0 when invalid)
    ``errors`` maps method names (including the four deployable trivials) to
    absolute errors; NaN marks an invalid estimate.
    """
    out: Dict[str, float] = {}
    valid = (err_method is not None and math.isfinite(err_method))
    for ref, tag in REFERENCE_TAGS:
        e_ref = errors.get(ref, float("nan"))
        e_ref = float("nan") if e_ref is None else float(e_ref)
        if valid and math.isfinite(e_ref):
            out[f"skill_vs_{tag}"] = skill_score(err_method, e_ref)
            out[f"win_vs_{tag}"] = int(err_method < e_ref)
        else:
            out[f"skill_vs_{tag}"] = float("nan")
            out[f"win_vs_{tag}"] = 0
    return out


def skill_vs_from_arrays(err_method, ref_errors: Dict[str, "np.ndarray"]) -> Dict[str, float]:
    """
    Aggregates for a selector evaluated on many records at once:
    err_method and each ref_errors[tag] are equal-length arrays.  Returns
    med_skill_vs_<tag> (median over records where both are finite) and
    win_rate_vs_<tag> (mean of the win indicator over records where the
    method is finite; an invalid method never wins).
    """
    err_method = np.asarray(err_method, dtype=float)
    out: Dict[str, float] = {}
    ok_m = np.isfinite(err_method)
    for _, tag in REFERENCE_TAGS:
        e_ref = np.asarray(ref_errors[tag], dtype=float)
        both = ok_m & np.isfinite(e_ref)
        if both.any():
            sk = np.array([skill_score(a, b) for a, b in zip(err_method[both], e_ref[both])], dtype=float)
            out[f"med_skill_vs_{tag}"] = round(float(np.median(sk)), 4)
        else:
            out[f"med_skill_vs_{tag}"] = float("nan")
        wins = np.zeros(err_method.shape, dtype=float)
        wins[both] = (err_method[both] < e_ref[both]).astype(float)
        out[f"win_rate_vs_{tag}"] = round(float(wins[ok_m | np.isfinite(e_ref)].mean()), 4) if (ok_m | np.isfinite(e_ref)).any() else float("nan")
    return out


def aggregate_skill_vs(df, prefix_med: str = "med_skill_vs_", prefix_win: str = "win_rate_vs_") -> Dict[str, float]:
    """
    Aggregate the eight per-record columns of a DataFrame slice:
    median of skill_vs_<tag> (NaN skipped) and mean of win_vs_<tag>.
    """
    out: Dict[str, float] = {}
    for _, tag in REFERENCE_TAGS:
        col_s, col_w = f"skill_vs_{tag}", f"win_vs_{tag}"
        s = df[col_s].dropna() if col_s in df else None
        out[f"{prefix_med}{tag}"] = (round(float(np.median(s)), 4) if s is not None and len(s) else float("nan"))
        out[f"{prefix_win}{tag}"] = (round(float(df[col_w].mean()), 4) if col_w in df and len(df) else float("nan"))
    return out


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
