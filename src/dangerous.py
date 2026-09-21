"""
dangerous.py
============
The dangerous-method set under redesign v2.

A method is dangerous when its pooled stability score

    S = valid_rate - W_CAT * cat_rate + W_BEATS * beats_rate

is negative on the core regimes, pooled over the three gap strata and the
noise levels, with capped cells excluded and the oracle comparator excluded.
The set is derived from results/phase1/phase1_aggregated.csv by
scripts/derive_dangerous.py and stored in the artifact
config.DANGEROUS_ARTIFACT (JSON).  Phases 2-5 obtain it through
load_dangerous(); if the artifact is missing they stop with instructions,
which is how the pipeline ordering (Phase 1 -> derivation -> Phases 2-5)
is enforced.  reproduce_all.py runs the derivation step explicitly.

Report-2 review (decision 2): only the accelerators (src.pipeline
.ACCEL_METHODS) are eligible for the dangerous flag.  The non-oracle trivial
comparators are still scored (and printed by the derivation script, for the
record) but they are never written to the artifact: neither into
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


def artifact_path(path: Optional[str] = None) -> str:
    """Absolute path of the artifact (relative paths resolve from the repo root)."""
    p = CFG_MOD.DANGEROUS_ARTIFACT if path is None else path
    return p if os.path.isabs(p) else os.path.join(_ROOT, p)


def derive_dangerous(df_agg: pd.DataFrame) -> Tuple[FrozenSet[str], pd.DataFrame]:
    """
    Derive the dangerous set from a Phase-1 aggregated table.

    Rules: core regimes only (is_holdout == 0), capped cells excluded,
    oracle excluded, pooled over strata, noise levels and regimes.  Only the
    accelerators are eligible for the flag (``eligible`` column); the
    non-oracle trivial comparators are scored for the record but can never
    be dangerous.
    Returns (dangerous, table) where table has one row per scored method
    with the pooled rates, S, eligibility and the flag.
    """
    required = {"method", "valid_rate", "cat_rate", "beats_rate", "capped",
                "is_holdout", "is_oracle"}
    missing = required - set(df_agg.columns)
    if missing:
        raise ValueError(f"phase1_aggregated.csv lacks columns {sorted(missing)}; "
                         "re-run Phase 1 under redesign v2")
    pool = df_agg[(df_agg["is_holdout"] == 0) & (df_agg["is_oracle"] == 0)]
    if CFG_MOD.EXCLUDE_CAPPED_FROM_POOLED:
        pool = pool[pool["capped"] == 0]
    rows = []
    for method, grp in pool.groupby("method", sort=False):
        vr = float(grp["valid_rate"].mean())
        cr = float(grp["cat_rate"].mean())
        br = float(grp["beats_rate"].mean())
        s = vr - CFG_MOD.W_CAT * cr + CFG_MOD.W_BEATS * br
        eligible = method in _ELIGIBLE
        rows.append({
            "method": method,
            "is_trivial": int(grp["is_trivial"].iloc[0]) if "is_trivial" in grp else 0,
            "eligible": int(eligible),
            "valid_rate": round(vr, 4), "cat_rate": round(cr, 4),
            "beats_rate": round(br, 4), "stability": round(s, 4),
            "n_cells": int(len(grp)),
            "dangerous": int(eligible and s < 0.0),
        })
    table = (pd.DataFrame(rows).sort_values("stability").reset_index(drop=True))
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
        "schema": "dangerous_methods/v2",
        "criterion": ("pooled stability S = valid_rate - W_CAT*cat_rate + "
                      "W_BEATS*beats_rate < 0 on core regimes, pooled over "
                      "gap strata and noise, capped cells excluded, oracle excluded; "
                      f"the {len(_ELIGIBLE)} accelerators only (trivial comparators never eligible)"),
        "pool": "accelerators",
        "n_pool": len(_ELIGIBLE),
        "W_CAT": CFG_MOD.W_CAT, "W_BEATS": CFG_MOD.W_BEATS,
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
    if payload.get("schema") != "dangerous_methods/v2":
        raise ValueError(f"{p}: unexpected schema {payload.get('schema')!r}")
    return payload


def load_dangerous(path: Optional[str] = None, required: bool = True) -> FrozenSet[str]:
    """
    The dangerous set phases 2-5 consume.  With required=False a missing
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
