"""
horizons.py
===========
Gap-stratified prediction horizons.

The rejected design evaluated every regime at fixed indices (150, 1000,
5000).  At n = 5000 most targets had converged to the asymptote, so a
predictor returning the asymptote constant beat every method.  Horizons are
now defined by how much of the observed gap remains:

    n_f(regime, g) = first integer n > n_obs with gap(n) <= g * gap(n_obs)

found on the noiseless mean by forward search and capped at
config.HORIZON_N_CAP (50,000).  When the cap binds (e.g. log_slow at
g = 0.02) n_f = cap and the *achieved* fraction gap(cap)/gap(n_obs) is
recorded, so no evaluation record can pretend a stratum was reached.

For oscillatory regimes "first n" means the first crossing, which is what a
forward search returns; the achieved fraction is recorded in every case.
"""

from dataclasses import dataclass, asdict
from functools import lru_cache
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

import src.config as CFG_MOD
from src.generators import GAP, SEED_DEPENDENT_SHAPE


@dataclass(frozen=True)
class Horizon:
    regime: str
    n_obs: int
    target_g: float
    n_f: int
    achieved_g: float      # gap(n_f) / gap(n_obs); NaN when gap(n_obs) == 0
    capped: bool           # True when the search hit HORIZON_N_CAP first
    gap_obs: float
    gap_f: float
    degenerate: bool = False   # gap(n_obs) == 0: already converged at n_obs
    seed: Optional[int] = None  # only set for seed-dependent shapes

    def as_dict(self) -> dict:
        return asdict(self)


def _shape_seed(regime: str, seed: Optional[int]) -> Optional[int]:
    """Only seed-dependent shapes key their horizon on the seed."""
    if regime in SEED_DEPENDENT_SHAPE:
        if seed is None:
            raise ValueError(f"{regime} needs a seed to define its horizon")
        return int(seed)
    return None


@lru_cache(maxsize=None)
def _horizon_cached(regime: str, n_obs: int, g: float,
                    seed: Optional[int], cap: int) -> Horizon:
    gap = GAP[regime]
    if not (0.0 < g < 1.0):
        raise ValueError(f"gap fraction must lie in (0, 1); got {g}")
    if n_obs >= cap:
        raise ValueError(f"n_obs={n_obs} must be below the cap {cap}")

    gap_obs = float(gap(float(n_obs), seed))
    if not np.isfinite(gap_obs):
        raise ValueError(f"{regime}: gap({n_obs}) is not finite")
    if gap_obs <= 0.0:
        return Horizon(regime, n_obs, g, n_obs + 1, float("nan"), False,
                       gap_obs, float(gap(float(n_obs + 1), seed)),
                       degenerate=True, seed=seed)

    threshold = g * gap_obs
    ns = np.arange(n_obs + 1, cap + 1, dtype=float)
    gaps = np.asarray(gap(ns, seed), dtype=float)
    hit = np.flatnonzero(gaps <= threshold)
    if hit.size:
        k = int(hit[0])
        return Horizon(regime, n_obs, g, int(ns[k]), float(gaps[k] / gap_obs),
                       False, gap_obs, float(gaps[k]), seed=seed)
    return Horizon(regime, n_obs, g, int(cap), float(gaps[-1] / gap_obs),
                   True, gap_obs, float(gaps[-1]), seed=seed)


def horizon_for_gap(regime: str, n_obs: int, g: float,
                    seed: Optional[int] = None,
                    cap: Optional[int] = None) -> Horizon:
    """
    n_f for observation depth n_obs and target remaining-gap fraction g.

    Parameters
    ----------
    regime : key of src.generators.GAP
    n_obs  : observation depth (the last observed index)
    g      : target remaining-gap fraction in (0, 1)
    seed   : sequence seed; required only for seed-dependent shapes
    cap    : search cap (defaults to config.HORIZON_N_CAP)
    """
    cap = CFG_MOD.HORIZON_N_CAP if cap is None else int(cap)
    return _horizon_cached(regime, int(n_obs), float(g),
                           _shape_seed(regime, seed), cap)


def horizon_table(regimes: Iterable[str], n_obs: int,
                  gap_fractions: Optional[Iterable[float]] = None,
                  seed: int = 0,
                  cap: Optional[int] = None) -> pd.DataFrame:
    """One row per (regime, g): n_f, achieved_g and the capped flag."""
    gs: List[float] = list(CFG_MOD.HORIZON_GAP_FRACTIONS
                           if gap_fractions is None else gap_fractions)
    rows = []
    for regime in regimes:
        for g in gs:
            rows.append(horizon_for_gap(regime, n_obs, g, seed=seed, cap=cap).as_dict())
    return pd.DataFrame(rows)


def format_horizon_table(df: pd.DataFrame) -> str:
    """Markdown table: regimes as rows, one 'n_f (achieved g)' cell per g."""
    gs = sorted(df["target_g"].unique(), reverse=True)
    header = "| regime | " + " | ".join(f"g = {g:g}" for g in gs) + " |"
    sep = "|---|" + "|".join("---" for _ in gs) + "|"
    lines = [header, sep]
    for regime, grp in df.groupby("regime", sort=False):
        cells = []
        for g in gs:
            r = grp[grp["target_g"] == g].iloc[0]
            if r["degenerate"]:
                cells.append(f"{int(r['n_f'])} (already converged)")
            elif r["capped"]:
                cells.append(f"**{int(r['n_f'])} CAP** (achieved g = {r['achieved_g']:.3f})")
            else:
                cells.append(f"{int(r['n_f'])} (g = {r['achieved_g']:.4f})")
        lines.append(f"| {regime} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
