"""
selection_defs.py  (redesign v2, R9f Part A)
============================================
The definitions shared by the table generator (scripts/make_paper_tables.py)
and the selection analysis (scripts/derive_selection.py), in one place so
that neither re-implements them:

    noise_class(), noise_class_order(), noise_class_label(), INTRINSIC_NOISE
        the noise class of a (regime, noise) cell: 'noise-free' when sigma = 0
        and the family has no intrinsic noise; 'intrinsic' when sigma = 0 and
        the family is one of the intrinsic-noise generators (two core families
        carry their own noise at sigma = 0, so "sigma = 0" is not
        "noise-free"); 'sigma=<level>' otherwise; NOISE_CLASS_ALL = 'all' is
        the pooled class of every cell
    CLASSICAL_FAMILIES, CLASSICAL
        the classical (limit-estimating) variants, derived from the registry's
        family and type maps; the count is derived, never typed
    LEAD_TOP_CORE, LEAD_TOP_HOLD, LEAD_TOP_WIN, rank_accelerators(),
    leading_methods(), lead_rule()
        the leading-method rule LEAD: the union of the LEAD_TOP_CORE lowest
        core median errors, the LEAD_TOP_HOLD lowest held-out median errors,
        the LEAD_TOP_WIN highest core win rates against the last value
        (rank-eligible accelerators at the headline stratum) and the two
        methods of the recorded-curve path; ordered by core rank, then
        held-out rank
    DEFAULT_METHOD, CONSERVATIVE_METHOD
        the paper's default (rational_fit) and the conservative alternative it
        names (single_exp_fit)
    POOL_NAMES, candidate_pools(), DESIGNS, registry_index()
        the four candidate pools of the selection analysis -- lead, named,
        classical, all -- and the two designs (one_pilot, many_pilots)

This module does nothing when imported: it reads no file and touches
results/ nowhere.  It lives under scripts/ (not src/ or phases/) and is
imported by the table generators only, so the code fingerprint's path set
(scripts/run_code_fingerprint.py) neither changes nor grows; no
result-producing step script may import it.
"""

import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.evaluation import FAMILY, METHOD_TYPE            # noqa: E402  (the registry's family and type maps)
from src.generators import _INTRINSIC_NOISE_GENERATORS    # noqa: E402  (the families with their own noise at sigma = 0)
from src.pipeline import ACCEL_METHODS                    # noqa: E402  (the accelerator roster, registry order)
from src.trajectories import REAL_EVAL_METHODS            # noqa: E402  (the two methods of the recorded-curve path)

# ── the two named methods ────────────────────────────────────────────────────
DEFAULT_METHOD = "rational_fit"            # the paper's default
CONSERVATIVE_METHOD = "single_exp_fit"     # the conservative alternative the paper names (K-40)
assert DEFAULT_METHOD in ACCEL_METHODS and CONSERVATIVE_METHOD in ACCEL_METHODS, (DEFAULT_METHOD, CONSERVATIVE_METHOD)

# ── noise classes ────────────────────────────────────────────────────────────
INTRINSIC_NOISE = tuple(_INTRINSIC_NOISE_GENERATORS)
NOISE_CLASS_ALL = "all"                    # the pooled class: every cell of the set and stratum


def noise_class(regime, noise):
    """'noise-free', 'intrinsic' or 'sigma=<level>' for a (regime, noise) cell."""
    if float(noise) == 0.0:
        return "intrinsic" if regime in INTRINSIC_NOISE else "noise-free"
    return f"sigma={float(noise):g}"


def noise_class_order(key):
    """noise-free, then intrinsic noise only, then sigma ascending, then the pooled class."""
    if key == "noise-free":
        return (0, 0.0)
    if key == "intrinsic":
        return (1, 0.0)
    if key == NOISE_CLASS_ALL:
        return (3, 0.0)
    return (2, float(key.split("=", 1)[1]))


def noise_class_label(key, tex=True):
    """The printed label of a noise class (TeX by default; plain text for FACTS rows)."""
    if key == "noise-free":
        return "noise-free"
    if key == "intrinsic":
        return "intrinsic noise only"
    if key == NOISE_CLASS_ALL:
        return "all cells"
    level = key.split("=", 1)[1]
    return rf"$\sigma = {level}$" if tex else f"sigma={level}"


# ── the classical variants ───────────────────────────────────────────────────
CLASSICAL_FAMILIES = ("shanks", "wynn_eps", "wynn_rho", "levin", "brezinski", "anderson")   # the classical (limit-estimating) families
# The limit-estimating members of the classical families, from the registry's family and
# type maps; the type filter must keep every member of those families (they are all limit
# estimators).  Ordered by family, then name (the order of f04).
CLASSICAL = sorted([m for m in ACCEL_METHODS if FAMILY[m] in CLASSICAL_FAMILIES and METHOD_TYPE[m] == "limit"],
                   key=lambda m: (CLASSICAL_FAMILIES.index(FAMILY[m]), m))
assert len(CLASSICAL) == sum(1 for m in ACCEL_METHODS if FAMILY[m] in CLASSICAL_FAMILIES), \
    "a member of a classical family is not a limit estimator"


# ── the leading methods (LEAD) ───────────────────────────────────────────────
LEAD_TOP_CORE, LEAD_TOP_HOLD, LEAD_TOP_WIN = 10, 5, 3


def rank_accelerators(G, strata):
    """Rank among the accelerators only, per stratum, med_error ascending, over the
    rank-eligible rows (rank_eligible == 1: valid_rate >= RANK_MIN_VALID) of a
    pooled Phase 1 table; {g: {method: rank}}."""
    out = {}
    for g in strata:
        s = G[(G.target_g == g) & (G.is_trivial == 0) & (G.rank_eligible == 1)]
        s = s.sort_values(["med_error", "method"])
        out[g] = {m: i + 1 for i, m in enumerate(s.method)}
    return out


def leading_methods(G_core, rank_core, rank_hold, headline_g):
    """LEAD: see the module docstring.  G_core = phase1_global.csv; rank_core / rank_hold =
    rank_accelerators() of the core and held-out tables."""
    rc = sorted(rank_core[headline_g], key=rank_core[headline_g].get)
    rh = sorted(rank_hold[headline_g], key=rank_hold[headline_g].get)
    sg = G_core[(G_core.target_g == headline_g) & (G_core.is_trivial == 0) & (G_core.rank_eligible == 1)]
    top_win = list(sg.sort_values(["win_rate_vs_last", "method"], ascending=[False, True]).method[:LEAD_TOP_WIN])
    members = list(dict.fromkeys(rc[:LEAD_TOP_CORE] + rh[:LEAD_TOP_HOLD] + top_win + list(REAL_EVAL_METHODS)))
    members.sort(key=lambda m: (rank_core[headline_g].get(m, math.inf), rank_hold[headline_g].get(m, math.inf), m))
    return members


def lead_rule(headline_g):
    """The sentence that defines LEAD (printed in fragment filters and FACTS rows)."""
    return (f"the union of the {LEAD_TOP_CORE} lowest core median errors, the {LEAD_TOP_HOLD} lowest held-out median errors, "
            f"the {LEAD_TOP_WIN} highest core win rates against the last value (rank-eligible accelerators at g = {headline_g:g}) "
            f"and the recorded-curve methods {', '.join(REAL_EVAL_METHODS)}; ordered by core rank, then held-out rank")


# ── the candidate pools and the designs of the selection analysis ────────────
POOL_NAMES = ("lead", "named", "classical", "all")
DESIGNS = ("one_pilot", "many_pilots")


def candidate_pools(lead):
    """The four candidate pools, given LEAD: lead = the leading methods; named = the
    default and the conservative alternative; classical = the classical variants;
    all = every accelerator (registry order).  Members are listed in the order of
    their definition; ties in the analysis are broken by registry_index()."""
    pools = {"lead": list(lead), "named": [DEFAULT_METHOD, CONSERVATIVE_METHOD],
             "classical": list(CLASSICAL), "all": list(ACCEL_METHODS)}
    assert tuple(pools) == POOL_NAMES
    for name, members in pools.items():
        assert members and len(set(members)) == len(members), (name, members)
        assert all(m in ACCEL_METHODS for m in members), (name, [m for m in members if m not in ACCEL_METHODS])
    return pools


def registry_index(method):
    """The position of an accelerator in src.pipeline.ACCEL_METHODS (the tie-break of every choice)."""
    return ACCEL_METHODS.index(method)
