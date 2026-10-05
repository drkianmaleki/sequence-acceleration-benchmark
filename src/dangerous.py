"""
dangerous.py
============
The excluded-method set under redesign v2.

``dangerous`` is the legacy implementation name for the exclusion condition
(pooled validity below config.RANK_MIN_VALID); the paper calls it the
excluded set.  The name is kept in the code, the artifact path and the
column names so that every phase reads one artifact under one name.

An accelerator is excluded when its pooled valid rate on the core regimes at
the Phase-1 depth is below config.RANK_MIN_VALID: uncapped cells only, the
oracle excluded, pooled over the gap strata and noise levels with equal cell
weights (the mean of the per-cell valid_rate of phase1_aggregated.csv).  No
composite score enters the criterion; the pooled catastrophe rate and median
error are recorded alongside for the reader, never used to decide.

The set is derived from results/phase1/phase1_aggregated.csv by
scripts/derive_dangerous.py and stored in the artifact
config.DANGEROUS_ARTIFACT (JSON, schema "dangerous_methods/v3").  Phases 2-5
obtain it through load_dangerous(); if the artifact is missing they stop with
instructions, which is how the pipeline ordering (Phase 1 -> derivation ->
Phases 2-5) is enforced.  reproduce_all.py runs the derivation step
explicitly.

Report-2 review (decision 2): only the accelerators (src.pipeline
.ACCEL_METHODS) are eligible for the flag.  The non-oracle trivial
comparators are still tabulated (and printed by the derivation script, for
the record) but they are never written to the artifact: neither into
``dangerous_methods`` nor into the artifact's ``table``.
"""

import datetime as _dt
import json
import os
import subprocess
from typing import Dict, FrozenSet, Optional, Tuple

import pandas as pd

import src.config as CFG_MOD
from src.pipeline import ACCEL_METHODS

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ELIGIBLE = frozenset(ACCEL_METHODS)
SCHEMA = "dangerous_methods/v3"
CRITERION = ("pooled valid_rate < RANK_MIN_VALID on the core regimes at the Phase-1 depth, "
             "pooled over gap strata and noise, capped cells excluded, oracle excluded; "
             "accelerators only; the flag is computed on the unrounded pooled mean")
TABLE_DECIMALS = 6      # valid_rate, cat_rate and med_error are stored with this many decimals


def artifact_path(path: Optional[str] = None) -> str:
    """Absolute path of the artifact (relative paths resolve from the repo root)."""
    p = CFG_MOD.DANGEROUS_ARTIFACT if path is None else path
    return p if os.path.isabs(p) else os.path.join(_ROOT, p)


def derive_dangerous(df_agg: pd.DataFrame) -> Tuple[FrozenSet[str], pd.DataFrame]:
    """
    Derive the excluded set from a Phase-1 aggregated table.

    Rules: core regimes only (is_holdout == 0), capped cells excluded,
    oracle excluded, pooled over strata, noise levels and regimes with equal
    cell weights.  A method is flagged when it is eligible (an accelerator)
    and its pooled valid_rate is below config.RANK_MIN_VALID.  The non-oracle
    trivial comparators are tabulated for the record but can never be flagged.
    Returns (dangerous, table); the table has one row per tabulated method
    with the pooled valid_rate, cat_rate and med_error (stored with
    TABLE_DECIMALS decimals; the flag itself is computed on the unrounded
    pooled mean), the cell count, the eligibility and the flag, sorted by
    valid_rate ascending.
    """
    required = {"method", "valid_rate", "cat_rate", "med_error", "capped",
                "is_holdout", "is_oracle"}
    missing = required - set(df_agg.columns)
    if missing:
        raise ValueError(f"phase1_aggregated.csv lacks columns {sorted(missing)}; "
                         "re-run Phase 1 under redesign v2")
    pool = df_agg[(df_agg["is_holdout"] == 0) & (df_agg["is_oracle"] == 0)]
    if CFG_MOD.EXCLUDE_CAPPED_FROM_POOLED:
        pool = pool[pool["capped"] == 0]
    floor = float(CFG_MOD.RANK_MIN_VALID)
    rows = []
    for method, grp in pool.groupby("method", sort=False):
        vr = float(grp["valid_rate"].mean())
        cr = float(grp["cat_rate"].mean())
        me = grp["med_error"].dropna()
        eligible = method in _ELIGIBLE
        rows.append({
            "method": method,
            "is_trivial": int(grp["is_trivial"].iloc[0]) if "is_trivial" in grp else 0,
            "eligible": int(eligible),
            "valid_rate": round(vr, TABLE_DECIMALS), "cat_rate": round(cr, TABLE_DECIMALS),
            "med_error": round(float(me.median()), TABLE_DECIMALS) if len(me) else float("nan"),
            "n_cells": int(len(grp)),
            "dangerous": int(eligible and vr < floor),      # the unrounded mean decides
        })
    table = (pd.DataFrame(rows).sort_values(["valid_rate", "method"])
                                .reset_index(drop=True))
    dangerous = frozenset(table.loc[table["dangerous"] == 1, "method"])
    assert dangerous <= _ELIGIBLE
    return dangerous, table


def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=_ROOT, stderr=subprocess.DEVNULL
                                       ).decode().strip()
    except Exception:
        return "unknown"


def write_artifact(dangerous: FrozenSet[str], table: pd.DataFrame,
                   path: Optional[str] = None, source: str = "",
                   extra: Optional[Dict] = None) -> str:
    """Write the JSON artifact; returns its path.

    Only accelerator rows go into the artifact: trivial comparators are
    dropped from ``table`` and refused in ``dangerous``.
    """
    bad = sorted(set(dangerous) - _ELIGIBLE)
    if bad:
        raise ValueError(f"only the {len(_ELIGIBLE)} accelerators can be dangerous; got {bad}")
    if "eligible" in table.columns:
        table = table[table["eligible"] == 1]
    else:
        table = table[table["method"].isin(_ELIGIBLE)]
    p = artifact_path(path)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    payload = {
        "schema": SCHEMA,
        "criterion": CRITERION,
        "legacy_name": ("'dangerous' is the legacy implementation name for the exclusion "
                        "condition (pooled validity below RANK_MIN_VALID); the paper calls "
                        "it the excluded set"),
        "pool": "accelerators",
        "n_pool": len(_ELIGIBLE),
        "rank_min_valid": float(CFG_MOD.RANK_MIN_VALID),
        "asymptote_mode": CFG_MOD.ASYMPTOTE_MODE,
        "assumed_mode": CFG_MOD.ASSUMED_L_MODE,
        "gap_fractions": list(CFG_MOD.HORIZON_GAP_FRACTIONS),
        "source": source,
        "git_head": _git_head(),
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "dangerous_methods": sorted(dangerous),
        "table": table.to_dict(orient="records"),
    }
    if extra:
        payload.update(extra)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return p


def load_artifact(path: Optional[str] = None) -> Dict:
    p = artifact_path(path)
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"dangerous-method artifact not found at {p}.\n"
            "Run Phase 1 and then  python scripts/derive_dangerous.py  "
            "(reproduce_all.py does this in order) before phases 2-5.")
    with open(p, encoding="utf-8") as fh:
        payload = json.load(fh)
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"{p}: unexpected schema {payload.get('schema')!r}; "
                         f"expected {SCHEMA!r} (re-run scripts/derive_dangerous.py)")
    return payload


def load_dangerous(path: Optional[str] = None, required: bool = True) -> FrozenSet[str]:
    """
    The excluded set phases 2-5 consume.  With required=False a missing
    artifact yields an empty set (for optional figure scripts); pipeline
    phases keep the default and fail loudly.
    """
    try:
        payload = load_artifact(path)
    except FileNotFoundError:
        if required:
            raise
        return frozenset()
    return frozenset(payload["dangerous_methods"])
