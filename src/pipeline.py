"""
pipeline.py
===========
Shared helpers for the redesign-v2 phase loops.

Every phase evaluates cells (regime, obs_idx, noise, seed, g).  This module
holds the pieces they all need so that the rules from the Report-1 review
are applied in exactly one way:

    resolve_regimes      core / held-out regime lists (holdout = evaluation only)
    horizon_meta         (target_g, achieved_g, n_f, capped) for one cell
    exclude_capped       the capped-exclusion rule for pooled statistics
    capped_block         the separate "capped" report block with achieved_g
    assign_ranks         rank eligibility (oracle excluded, finite metric,
                         valid_rate >= config.RANK_MIN_VALID) and the rank
    unranked_block       the separate block of below-floor methods
    skill_table          best-of-four trivial reference and per-method skill
    method_flags         is_trivial / is_oracle for output schemas
    git_head             short commit hash for provenance fields
    ACCEL_METHODS        the 51 accelerators (ensemble / pool candidates)
"""

import math
import os
import subprocess
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import src.config as CFG_MOD
from src.accelerators import METHOD_NAMES
from src.generators import HOLDOUT, HOLDOUT_REGIME_NAMES, REGIME_NAMES
from src.horizons import horizon_for_gap
from src.trivial import (ORACLE_METHODS, SKILL_REFERENCE_METHODS,
                         TRIVIAL_METHOD_NAMES, best_reference_error,
                         skill_score)

# The 51 accelerators: everything that is not a trivial comparator.  Ensemble
# pools, selector candidates and the dangerous derivation draw from these
# plus the non-oracle trivial comparators where a phase says so; the oracle
# never enters a pool.
ACCEL_METHODS: List[str] = [m for m in METHOD_NAMES if m not in TRIVIAL_METHOD_NAMES]
TRIVIAL_NON_ORACLE: List[str] = [m for m in TRIVIAL_METHOD_NAMES if m not in ORACLE_METHODS]
REFERENCE_METHODS: List[str] = list(SKILL_REFERENCE_METHODS)

HORIZON_COLS = ["target_g", "achieved_g", "n_f", "capped"]


def is_trivial(method: str) -> bool:
    return method in TRIVIAL_METHOD_NAMES


def is_oracle(method: str) -> bool:
    return method in ORACLE_METHODS


def method_flags(method: str) -> Dict[str, int]:
    return {"is_trivial": int(is_trivial(method)), "is_oracle": int(is_oracle(method))}


def resolve_regimes(core_regimes: Optional[Iterable[str]] = None,
                    holdout_regimes: Optional[Iterable[str]] = None,
                    include_holdout: bool = True) -> List[str]:
    """
    Regime list for a phase: core regimes first, then held-out regimes.

    None means "all" for either group.  Held-out regimes are evaluation-only
    everywhere; callers that train anything must pass include_holdout=False.
    """
    core = list(REGIME_NAMES) if core_regimes is None else list(core_regimes)
    bad = [r for r in core if r not in REGIME_NAMES]
    if bad:
        raise ValueError(f"not core regimes: {bad}")
    if not include_holdout:
        return core
    hold = (list(HOLDOUT_REGIME_NAMES) if holdout_regimes is None
            else list(holdout_regimes))
    bad = [r for r in hold if r not in HOLDOUT]
    if bad:
        raise ValueError(f"not held-out regimes: {bad}")
    return core + hold


def is_holdout(regime: str) -> int:
    return int(regime in HOLDOUT)


def horizon_meta(regime: str, obs_idx: int, g: float, seed: int) -> Dict[str, float]:
    """(target_g, achieved_g, n_f, capped) for one evaluation cell."""
    hz = horizon_for_gap(regime, obs_idx, g, seed=seed)
    return {"target_g": float(g), "achieved_g": hz.achieved_g,
            "n_f": int(hz.n_f), "capped": int(hz.capped)}


def exclude_capped(df: pd.DataFrame) -> pd.DataFrame:
    """The capped-exclusion rule: pooled cross-regime statistics never include
    cells whose horizon search hit the cap."""
    if not CFG_MOD.EXCLUDE_CAPPED_FROM_POOLED or "capped" not in df.columns:
        return df
    return df[df["capped"] == 0]


def capped_block(df: pd.DataFrame, keys: Sequence[str],
                 value_cols: Sequence[str] = ("error",)) -> pd.DataFrame:
    """
    The separate report block for capped cells: one row per key combination
    with n_f, achieved_g and the median of each value column.  Empty frame
    (with the right columns) when nothing was capped.
    """
    cols = list(keys) + ["n_f", "achieved_g"] + [f"med_{c}" for c in value_cols] + ["n"]
    if "capped" not in df.columns:
        return pd.DataFrame(columns=cols)
    sub = df[df["capped"] == 1]
    if sub.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for key_vals, grp in sub.groupby(list(keys), sort=False):
        if not isinstance(key_vals, tuple):
            key_vals = (key_vals,)
        row = dict(zip(keys, key_vals))
        row["n_f"] = float(grp["n_f"].median())
        row["achieved_g"] = float(grp["achieved_g"].median())
        for c in value_cols:
            vals = grp[c].dropna()
            row[f"med_{c}"] = float(vals.median()) if len(vals) else float("nan")
        row["n"] = int(len(grp))
        rows.append(row)
    return pd.DataFrame(rows)[cols]


def assign_ranks(df: pd.DataFrame, metric: Optional[str] = None,
                 group_cols: Sequence[str] = ("target_g",),
                 ascending: bool = True,
                 min_valid: Optional[float] = None,
                 valid_col: str = "valid_rate") -> pd.DataFrame:
    """
    The rank rule for every ranked table (Report-2 review, decision 3).

    A row is rank-eligible when it is not the oracle comparator, its metric
    is finite and its valid_rate is at least config.RANK_MIN_VALID.  Ranks
    (1 = best) are assigned within each group over the eligible rows, sorted
    by the metric with the method name as a deterministic tie-break.  Returns
    a copy with two columns: ``rank`` (float, NaN when ineligible) and
    ``rank_eligible`` (int).  Ineligible rows stay in the table; the ones
    below the validity floor are what unranked_block() reports.
    """
    metric = CFG_MOD.RANK_METRIC if metric is None else metric
    floor = CFG_MOD.RANK_MIN_VALID if min_valid is None else float(min_valid)
    out = df.copy()
    out["rank"] = np.nan
    if out.empty:
        out["rank_eligible"] = pd.Series(dtype=int)
        return out
    eligible = out[metric].notna() & np.isfinite(out[metric].astype(float))
    if "is_oracle" in out.columns:
        eligible &= out["is_oracle"] == 0
    if valid_col in out.columns:
        eligible &= out[valid_col].notna() & (out[valid_col] >= floor)
    out["rank_eligible"] = eligible.astype(int)
    keys = list(group_cols) if group_cols else []
    groups = out[eligible].groupby(keys, sort=False) if keys else [(None, out[eligible])]
    for _, grp in groups:
        sub = grp.sort_values([metric, "method"], ascending=[ascending, True], kind="mergesort")
        out.loc[sub.index, "rank"] = np.arange(1, len(sub) + 1, dtype=float)
    return out


def unranked_block(df: pd.DataFrame, keep_cols: Optional[Sequence[str]] = None,
                   valid_col: str = "valid_rate") -> pd.DataFrame:
    """
    The separate block of methods that a ranked table shows unranked because
    they sit below the validity floor (oracle rows are excluded: they are
    unranked by design, not by validity).  Sorted by valid_rate descending
    within the table's groups; ``valid_rate`` is always among the columns.
    """
    if df.empty or "rank_eligible" not in df.columns:
        return pd.DataFrame(columns=list(keep_cols) if keep_cols else [])
    sub = df[df["rank_eligible"] == 0]
    if "is_oracle" in sub.columns:
        sub = sub[sub["is_oracle"] == 0]
    if valid_col in sub.columns:
        sub = sub[sub[valid_col].notna() & (sub[valid_col] < CFG_MOD.RANK_MIN_VALID)]
    if keep_cols:
        cols = [c for c in keep_cols if c in sub.columns]
        sub = sub[cols]
    sort_by = [c for c in ("target_g", "cat_mult") if c in sub.columns]
    if valid_col in sub.columns:
        return sub.sort_values(sort_by + [valid_col, "method"],
                               ascending=[True] * len(sort_by) + [False, True]
                               ).reset_index(drop=True)
    return sub.reset_index(drop=True)


def git_head() -> str:
    """Short hash of the checked-out commit (\"unknown\" outside a git checkout)."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=root, stderr=subprocess.DEVNULL
                                       ).decode().strip()
    except Exception:
        return "unknown"


def skill_table(errors: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    """Best-of-four trivial reference error and skill for every method."""
    ref = best_reference_error(errors)
    return ref, {m: (skill_score(e, ref) if (e is not None and math.isfinite(e))
                     else float("nan"))
                 for m, e in errors.items()}


def median_skill(series: pd.Series) -> float:
    vals = series.dropna()
    return float(vals.median()) if len(vals) else float("nan")
