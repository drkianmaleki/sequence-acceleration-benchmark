#!/usr/bin/env python3
"""
make_paper_tables.py  (v3, redesign v2)
=======================================
Turn the committed results/ of the redesign-v2 pipeline into the LaTeX table
fragments of the paper and into FACTS.md, the list of every headline number
with its provenance (file, filter, formula).  Read-only over results/.

    python scripts/make_paper_tables.py                      # defaults below
    python scripts/make_paper_tables.py --results results --out paper_fragments --facts FACTS.md
    python scripts/make_paper_tables.py --no-raw             # accepted; no effect (see below)

Inputs are the per-stratum files (phase1_global.csv keyed by target_g,
phase2_*_g{g}.csv, phase5b_sweep*_global.csv keyed by target_g, ...).  The
legacy-named headline copies phase2_correlations.csv and phase2_rules.csv
are never read.  This script reads no git-ignored raw per-record file: the
ten facts that only the raw files can supply (richardson_3 validity by
depth and the invalid rates from phase5a_raw.csv and phase1_records.csv;
the record-level skill split of the fixed default) are computed once by
scripts/derive_raw_facts.py into results/raw_facts.csv and emitted from
there; without that file the "not recomputed" row is written instead.
--no-raw is accepted for compatibility and has no effect.  Every fragment
and every FACTS row is therefore regenerable from the committed files alone
(scripts/check_tables.py verifies it).

Fragments (paper_fragments/, one tabular per file, booktabs, no \\begin{table})
------------------------------------------------------------------------------
  f01_trivial_baseline.tex          trivial baseline: win rate and skill vs predict-L_hat and vs last_value (primary), strict hindsight skill, oracle reference
  f02_ranking_g{0.5,0.1,0.02}.tex   main ranking (accelerators above the validity floor) + unranked block
  f03_skill_summary.tex             fraction of cells with median skill < 1 per family, per stratum, core | held-out
  f04_classical_noop.tex            the 21 classical variants vs the +/-10 % band, in MI and in win-rate-vs-last terms, plus strict skill
  f05_sweep1_modes.tex              assumed-asymptote mode sweep, cascade rows (Phase 5b sweep 1a; clamped features): what happened when the cascade fired
  f05b_sweep1_consumers_{core,holdout}.tex   sweep 1b: per L_hat consumer, median error under each mode and win rate vs last_value (g = 0.5, 0.1); constant_assumed = the value of knowing the floor
  f06_generalisation.tex            held-out vs core rank shift per method
  f07_real_data_v2.tex              real-data re-evaluation over the (depth x target) grid, with skill
  f07b_real_data_legacy18.tex       the preserved 18-cell legacy real-data run
  f08_real_perturb_diagnostic.tex   perturb_iqr of the routed method vs cell failure: all cells, pre-/post-minimum targets, by routed method (AUC)
  f09a_selectors_phase3.tex         Phase 3 selectors: validity rate, n_valid/n_total, median / q25-q75 / p90 error of the chosen records per stratum (conditional on validity)
  f09b_ensemble_phase5a_core.tex    Phase 5a selectors / ensembles, core, error and skill per stratum
  f09c_ensemble_phase5a_holdout.tex Phase 5a, held-out regimes
  f09d_ablation_phase5a.tex         Phase 5a ablation comparisons per stratum
  f10a_diagnostics_correlations.tex Phase 4 diagnostic-error correlations, core pooled | held-out
  f10b_diagnostics_depth.tex        Phase 4 perturb_iqr correlation by observation depth (headline stratum)
  f10c_diagnostics_ensemble.tex     Phase 4 selectors with trivial references, error and skill
  f10d_diagnostics_filter.tex       Phase 4 cascade + perturb_iqr screen
  f11_capped_block.tex              every capped (regime x stratum) cell with achieved_g; depth-grid counts
  f12_validity_by_depth.tex         validity by observation depth: every method below the 0.9 floor at some depth
  f14_real_fixed_methods.tex        real data (Table tab:realmethods): rational_fit, richardson_1 and the cascade per dataset and pooled,
                                    pre- / post-minimum targets: n, win vs last, fail, valid, median error and skill
  f15_bootstrap.tex                 family bootstrap over core regimes (Table tab:boot): rational_fit's win rate vs last and its
                                    cell-median error against each comparator, point and 2.5 / 97.5 percentiles
  f16_generalisation_summary.tex    generalisation summary (Table tab:gen): eligible accelerators core / held-out / both, Spearman rho,
                                    ranks of rational_fit and log_linear
  f18_ladders_classical.tex         Phase 0b order ladders of the classical families (every order next to the roster orders)
  f19_ladders_fits.tex              Phase 0b order ladders of the fits (Richardson terms, fixed exponents, parametric models)
  f20_roster.tex                    the method roster by family (Table tab:roster), derived from the registry
  f04b_classical_noop_holdout.tex   f04 on the held-out families
  f21_noop_by_noise.tex             the classical no-op by noise class (noise-free / intrinsic noise only / sigma), per set and stratum
  f21b_noop_by_noise_variants_{core,holdout}.tex   every classical variant's MI and win rate in each noise class, headline stratum
  f22_per_family_g{0.5,0.1,0.02}.tex   rational_fit and single_exp_fit family by family: last value, rational_fit, single_exp_fit, the best rank-eligible accelerator
  f23_family_head_to_head.tex       rational_fit against every other leading method and the last value, families and cells won
  f24_leading_fits.tex              the leading methods side by side: core, held-out, L_hat sensitivity, recorded curves
  f25_real_roster.tex               every method on the recorded curves (real_data_roster_v2.csv), pre- / post-minimum
  f26_real_classical.tex            the classical variants on the recorded curves: MI, wins, validity per split
  f27_selection_one_pilot.tex       choosing a method by trial on the synthetic families, design one_pilot (phase1_selection_global.csv,
                                    scripts/derive_selection.py): per set, noise class and candidate pool, the chosen method against the
                                    default and the last value
  f27b_selection_many_pilots.tex    the same for the design many_pilots
  f28_selection_recorded.tex        choosing a method by trial on the recorded curves (pre-minimum cells): leave one depth out, leave one
                                    dataset out, per candidate pool
  f29_real_named_by_dataset.tex     rational_fit and single_exp_fit on the recorded curves per dataset and pooled, before / after the minimum
  f30_real_final_loss_by_depth.tex  the datasets whose minimum is at the final recorded round: error as a share of the final loss by depth,
                                    last value / rational_fit / single_exp_fit

The definitions shared with scripts/derive_selection.py -- the noise class of
a cell, the classical variants, the leading-method rule LEAD, DEFAULT_METHOD
and CONSERVATIVE_METHOD, the candidate pools -- are imported from
scripts/selection_defs.py and not re-implemented here.

Printed text follows the paper's terminology: method families through
FAMILY_LABEL, ladder orders in math mode, "curve family" (never "regime"),
"the excluded set", "last value"; FACTS rows, CSV files and fragment names
keep the code identifiers.

Macros used (defined in the paper preamble): \\meth{}, \\diag{},
\\rhoC, \\rhoV, \\TE, \\LE, \\nobs, \\nf.
"""

import argparse
import json
import math
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import src.config as C                                    # noqa: E402
from src.evaluation import FAMILY, METHOD_TYPE            # noqa: E402  (the registry's family and type maps)
from src.pipeline import ACCEL_METHODS, PHASE2_POOL, TRIVIAL_NON_ORACLE, resolve_regimes   # noqa: E402
from src.trivial import ORACLE_METHODS, SKILL_REFERENCE_METHODS  # noqa: E402
from phases.phase5a import (ORACLE_SMALL, EQUAL_SMALL, DIAG_SMALL,  # noqa: E402
                            N_SMALL as N_POOL_SMALL,
                            EQUAL_POOL, DIAG_POOL, CAPPED_POOL,
                            ABL_THRESHOLD_VS_EQUAL, ABL_DIAG_VS_EQUAL, ABL_WEIGHTING_SMALL)
from phases.phase5b import LHAT_CONSUMERS                 # noqa: E402
from scripts.analyze_by_ltrue import load_manifest, provenance, write_fragment_index   # noqa: E402  (the header of every generated file; the index)
from scripts.selection_defs import (CLASSICAL, CLASSICAL_FAMILIES, CONSERVATIVE_METHOD, DEFAULT_METHOD,   # noqa: E402
                                    INTRINSIC_NOISE, NOISE_CLASS_ALL, POOL_NAMES, candidate_pools, lead_rule,
                                    leading_methods, noise_class, noise_class_label, noise_class_order,
                                    rank_accelerators, registry_index)

STRATA = list(C.HORIZON_GAP_FRACTIONS)          # [0.5, 0.1, 0.02]
HEADLINE_G = float(C.HEADLINE_G)                # 0.1
ORACLE = "constant_oracle"
DEPLOYABLE = list(SKILL_REFERENCE_METHODS)      # constant_assumed, last_value, window_mean, window_min
NOOP_BAND = (0.9, 1.1)                          # median improvement factor within +/-10 % of 1
SKILL_NOOP = 0.9                                # median skill >= 0.9: no better than 10 % over the best trivial
WIN_BAND = (0.4, 0.6)                           # win rate vs last_value within a coin flip +/- 0.1
# Family bootstrap (paper §5.4, fragment f15): the resampling unit is the core regime.
BOOT_SEED = 20260927                            # RandomState seed of the bootstrap draws; fixed once, never tuned
BOOT_N = 5000                                   # resamples of the core regimes (with replacement)
BOOT_COMPARATORS = ["log_linear", "richardson_1", "richardson_2", "richardson_3", "double_exp_fit"]   # rational_fit against each
FAMILY_ORDER = ["richardson", "parametric", "pade", "neville", "baseline", "ensemble",
                "shanks", "wynn_eps", "wynn_rho", "levin", "brezinski", "anderson", "trivial"]
TYPE_MACRO = {"trajectory": r"\TE", "limit": r"\LE"}
REAL_DATASETS = ["adult", "bank_marketing", "covertype", "higgs", "jannis", "miniboone"]

# ── Shared definitions (R9d Part C, section 5.1; R9f: scripts/selection_defs.py) ──
# The noise class of a (regime, noise) cell (noise_class, noise_class_order,
# noise_class_label, INTRINSIC_NOISE), the classical variants (CLASSICAL), the
# leading-method rule (LEAD_TOP_*, rank_accelerators, leading_methods, lead_rule),
# DEFAULT_METHOD, CONSERVATIVE_METHOD and the candidate pools are imported above
# from scripts.selection_defs, which scripts/derive_selection.py shares.


# Printed method-family names: one map used by every fragment that prints a
# method family (the registry's families and the ladder families of Phase 0b).
# FACTS rows, CSV files and fragment file names keep the code identifiers.
FAMILY_LABEL = {
    "richardson": "Richardson", "parametric": "parametric fit", "pade": r"Pad\'{e}", "neville": "Neville",
    "baseline": "baseline", "ensemble": "ensemble", "shanks": "Shanks", "wynn_eps": r"Wynn $\varepsilon$",
    "wynn_rho": r"Wynn $\rho$", "levin": "Levin", "brezinski": r"Brezinski $\theta$", "anderson": "Anderson",
    "trivial": "trivial",
    # ladder families (results/phase0b/order_ladders_panels.csv)
    "levin_t": "Levin $t$", "levin_u": "Levin $u$", "levin_v": "Levin $v$", "brezinski_theta": r"Brezinski $\theta$",
    "richardson_free": "Richardson, free exponents", "richardson_fixed": "Richardson, fixed exponent",
}


def fam_label(fam):
    try:
        return FAMILY_LABEL[fam]
    except KeyError:
        raise KeyError(f"no printed label for the method family {fam!r}; add it to FAMILY_LABEL") from None


_ORDER_PARAM = re.compile(r"^(k|m|d|alpha)=(.+)$")


def order_label(s):
    """Ladder order labels of the form k=n, m=n, d=n, alpha=x print in math mode; the others as they are."""
    m = _ORDER_PARAM.match(str(s))
    if m:
        sym = r"\alpha" if m.group(1) == "alpha" else m.group(1)
        return f"${sym} = {m.group(2)}$"
    return esc(s)


# The leading methods (LEAD, scripts.selection_defs.leading_methods), derived at
# the headline stratum from the rank-eligible accelerators: the union of the
# LEAD_TOP_CORE lowest median errors on the core set, the LEAD_TOP_HOLD lowest
# on the held-out set, the LEAD_TOP_WIN highest win rates against the last
# value on the core set, and the two methods of the recorded-curve path
# (REAL_EVAL_METHODS); ordered by core rank, then held-out rank.  Built once
# the ranks are known (below).


# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Paper table fragments + FACTS.md (redesign v2)")
    p.add_argument("--results", default=os.path.join(_ROOT, "results"))
    p.add_argument("--out", default=os.path.join(_ROOT, "paper_fragments"))
    p.add_argument("--facts", default=os.path.join(_ROOT, "FACTS.md"))
    p.add_argument("--no-raw", action="store_true",
                   help="accepted for compatibility; no effect (the raw-derived facts come from results/raw_facts.csv, "
                        "written by scripts/derive_raw_facts.py)")
    return p.parse_args()


ARGS = parse_args()
RES = ARGS.results
OUT = ARGS.out
os.makedirs(OUT, exist_ok=True)


# The run the results come from (results/run_manifest.json), not the git state when the generator runs.
PROV = provenance("scripts/make_paper_tables.py", RES)

# The raw-derived facts (scripts/derive_raw_facts.py -> results/raw_facts.csv); None when absent.
_RAW_FACTS_PATH = os.path.join(RES, "raw_facts.csv")
RAW_FACTS = (pd.read_csv(_RAW_FACTS_PATH, dtype=str, keep_default_na=False, encoding="utf-8")
             if os.path.exists(_RAW_FACTS_PATH) else None)


# ── helpers ──────────────────────────────────────────────────────────────────
def rel(path):
    return os.path.relpath(path, _ROOT).replace(os.sep, "/")


def read(*parts, **kw):
    return pd.read_csv(os.path.join(RES, *parts), **kw)


def esc(s):
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def mth(s):
    return r"\meth{" + esc(s) + "}"


def tt(s):
    return r"\texttt{" + esc(s) + "}"


def f4(v):
    return "--" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.4f}"


def f3(v):
    return "--" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.3f}"


def f2(v):
    return "--" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.2f}"


def fint(v):
    return "--" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{int(v)}"


def signed(v, nd=4):
    return "--" if not math.isfinite(v) else f"${v:+.{nd}f}$"


def pct(v, nd=0):
    return "--" if not math.isfinite(v) else f"{100 * v:.{nd}f}\\%"


def gname(g):
    return f"{g:g}"


FRAGMENTS = []   # (file, description, sources)


def frag(name, colspec, rows, sources, filters, description, notes=(), extra=()):
    """Write one booktabs tabular with a provenance comment header (``extra``:
    further comment lines between the sources and the filter, e.g. the
    provenance of a separately produced input)."""
    path = os.path.join(OUT, name)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("% AUTO-GENERATED -- do not hand-edit.  " + PROV + "\n")
        f.write("% " + description + "\n")
        for s in sources:
            f.write(f"% source : {s}\n")
        for s in extra:
            f.write(f"% {s}\n")
        f.write(f"% filter : {filters}\n")
        for n in notes:
            f.write(f"% note   : {n}\n")
        f.write("\\begin{tabular}{" + colspec + "}\n\\toprule\n")
        for r in rows:
            f.write(r + "\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    FRAGMENTS.append((name, description, list(sources)))
    print(f"  wrote {name:<36} {len(rows):>4} lines")


FACTS = []   # dicts: section, fact, value, file, filter, formula


def fact(section, name, value, file, filt, formula):
    FACTS.append(dict(section=section, fact=name, value=value, file=file,
                      filter=filt, formula=formula))


def mid(text, ncol):
    return r"\multicolumn{" + str(ncol) + "}{l}{" + text + r"} \\"


# ── load the Phase-1 tables ──────────────────────────────────────────────────
G_CORE = read("phase1", "phase1_global.csv")
G_HOLD = read("phase1", "phase1_global_holdout.csv")
AGG = read("phase1", "phase1_aggregated.csv")
UNRANKED = read("phase1", "phase1_unranked.csv")
HORIZONS = read("phase1", "phase1_horizons.csv")
CAPPED1 = read("phase1", "phase1_capped.csv")
with open(os.path.join(RES, "phase1", "dangerous_methods.json"), encoding="utf-8") as fh:
    ARTIFACT = json.load(fh)
DANGEROUS = set(ARTIFACT["dangerous_methods"])
LEGACY = set(C.LEGACY_DANGEROUS_METHODS)
# One comparison of the excluded set with the legacy hard-coded set; every sentence about it derives from these.
SAME_AS_LEGACY = DANGEROUS == LEGACY
ADDED_VS_LEGACY = sorted(DANGEROUS - LEGACY)
REMOVED_VS_LEGACY = sorted(LEGACY - DANGEROUS)
FAM = G_CORE.drop_duplicates("method").set_index("method")["family"].to_dict()
TYP = G_CORE.drop_duplicates("method").set_index("method")["method_type"].to_dict()
assert set(ACCEL_METHODS) == set(G_CORE[G_CORE.is_trivial == 0].method), "accelerator roster mismatch"
N_ACC = len(ACCEL_METHODS)          # the roster size; never hard-coded below
assert len(DEPLOYABLE) == len(SKILL_REFERENCE_METHODS)
# The classical variants (CLASSICAL, scripts.selection_defs): the limit-estimating members of the
# classical families, from the registry's family and type maps; the count is derived.
assert all(FAM[m] == FAMILY[m] for m in CLASSICAL), "family labels of phase1_global.csv differ from the registry"

SRC_G = ["results/phase1/phase1_global.csv (core regimes, all three strata, capped cells excluded)",
         "results/phase1/phase1_global_holdout.csv (held-out regimes)"]


def dag(m):
    return r"$\dagger$" if m in DANGEROUS else ""


# Rank among the accelerators only (N_ACC), per stratum, med_error ascending, over the
# rank-eligible rows (scripts.selection_defs.rank_accelerators).
RANK_CORE = rank_accelerators(G_CORE, STRATA)
RANK_HOLD = rank_accelerators(G_HOLD, STRATA)


def row_of(G, g, m):
    s = G[(G.target_g == g) & (G.method == m)]
    return s.iloc[0] if len(s) else None


LEAD = leading_methods(G_CORE, RANK_CORE, RANK_HOLD, HEADLINE_G)
LEAD_RULE = lead_rule(HEADLINE_G)
POOLS = candidate_pools(LEAD)                   # the four candidate pools of the selection analysis (f27, f27b, f28)


# Family-level statistics (per regime and method, uncapped cells only): median
# error = median over the family's cells of med_error; win rate = mean over
# cells of win_rate_vs_last; validity and catastrophe rate = means over cells.
# This is the aggregation of the ranking tables (median of cell medians), so
# the per-family tables decompose the pooled ones.
def family_table(g, hold):
    sub = AGG[(AGG.target_g == g) & (AGG.is_holdout == hold) & (AGG.capped == 0)]
    return (sub.groupby(["regime", "method"])
               .agg(med_error=("med_error", "median"), win=("win_rate_vs_last", "mean"),
                    valid=("valid_rate", "mean"), cat_rate=("cat_rate", "mean"), n=("noise", "size")))


def cell_table(g, hold):
    """Cell-level med_error, (regime, noise) x method, uncapped cells."""
    sub = AGG[(AGG.target_g == g) & (AGG.is_holdout == hold) & (AGG.capped == 0)]
    return sub.pivot_table(index=["regime", "noise"], columns="method", values="med_error")


FAMILY_TABLES = {(g, h): family_table(g, h) for g in STRATA for h in (0, 1)}
CELL_TABLES = {(g, h): cell_table(g, h) for g in STRATA for h in (0, 1)}
SET_REGIMES = {h: [r for r in resolve_regimes() if r in set(AGG[AGG.is_holdout == h].regime)] for h in (0, 1)}   # canonical order
FAMILY_STAT_NOTE = ("family-level statistics over the family's uncapped (noise) cells: median error = median of the cell med_error, "
                    "win rate = mean of win_rate_vs_last, validity and catastrophe rate = means over cells (the aggregation of the "
                    "pooled tables, median of cell medians)")

# Selector labels of the Phase 5a tables (f09b / f09c), reused by f09d and f10c.
SELECTOR_ORDER5 = [(f"oracle_{N_ACC}", f"oracle over the {N_ACC}-method pool"), ("constant_oracle", r"\meth{constant\_oracle} \textit{(reference)}"),
                   (ORACLE_SMALL, f"oracle over the {N_POOL_SMALL}-method pool"),
                   ("fixed_rational", r"fixed \meth{rational\_fit}"), ("fixed_richardson", r"fixed \meth{richardson\_1}"),
                   ("phase2_cascade", "Phase-2 cascade"),
                   (EQUAL_SMALL, f"equal ensemble ({N_POOL_SMALL})"), (DIAG_SMALL, f"diagnostic-weighted ensemble ({N_POOL_SMALL})"),
                   (f"equal_ensemble_{N_ACC}", f"equal ensemble ({N_ACC})"), (f"diag_ensemble_{N_ACC}", f"diagnostic-weighted ensemble ({N_ACC})"),
                   (f"capped_diag_{N_ACC}", f"capped diagnostic-weighted ({N_ACC})"),
                   ("equal_ensemble_safe", "equal ensemble (safe)"), ("diag_ensemble_safe", "diagnostic-weighted (safe)"),
                   ("capped_diag_safe", "capped diagnostic-weighted (safe)"),
                   ("threshold_ens_010", "threshold ensemble, IQR $\\leq 0.10$"), ("threshold_ens_050", "threshold ensemble, IQR $\\leq 0.50$"),
                   ("threshold_ens_safe", "threshold ensemble, no threshold"),
                   ("constant_assumed", r"\meth{constant\_assumed}"), ("last_value", r"\meth{last\_value}"),
                   ("window_min", r"\meth{window\_min}"), ("window_mean", r"\meth{window\_mean}")]
SELECTOR_LABEL = dict(SELECTOR_ORDER5)


def selector_label(sel):
    try:
        return SELECTOR_LABEL[sel]
    except KeyError:
        raise KeyError(f"no printed label for the selector {sel!r}; add it to SELECTOR_ORDER5") from None


# The Phase 5a ablation comparisons (first, second) by name -- the `comparisons`
# list of phases/phase5a.py; each prints "<first> vs <second>" through the f09 labels.
ABLATION_PAIRS = {
    ABL_THRESHOLD_VS_EQUAL:          ("threshold_ens_010", EQUAL_POOL),
    "threshold_vs_rational":         ("threshold_ens_010", "fixed_rational"),
    "capped_diag_vs_equal":          (CAPPED_POOL, EQUAL_POOL),
    ABL_DIAG_VS_EQUAL:               (DIAG_POOL, EQUAL_POOL),
    ABL_WEIGHTING_SMALL:             (DIAG_SMALL, EQUAL_SMALL),
    "filter_benefit":                ("threshold_ens_010", "threshold_ens_safe"),
    "pool_expansion_gain":           (EQUAL_POOL, EQUAL_SMALL),
    "threshold_vs_constant_assumed": ("threshold_ens_010", "constant_assumed"),
    "rational_vs_window_min":        ("fixed_rational", "window_min"),
}


def ablation_label(cmp):
    try:
        a, b = ABLATION_PAIRS[cmp]
    except KeyError:
        raise KeyError(f"no (first, second) pair for the ablation comparison {cmp!r}; add it to ABLATION_PAIRS") from None
    return f"{selector_label(a)} vs {selector_label(b)}"


# Phase 4 selectors (f10c): the f09 wording for the same selector; the oracle and
# the diagnostic ensemble are those of the len(PHASE2_POOL)-method pool.
PHASE4_SELECTOR_LABEL = {
    "oracle":           selector_label(ORACLE_SMALL),
    "diag_ensemble":    selector_label(DIAG_SMALL),
    "fixed_rational":   selector_label("fixed_rational"),
    "fixed_richardson": selector_label("fixed_richardson"),
    "phase2_proxy":     r"lower error of \meth{richardson\_1} and \meth{rational\_fit} (hindsight)",
    **{m: selector_label(m) for m in ("constant_assumed", "last_value", "window_mean", "window_min")},
}


def phase4_selector_label(sel):
    try:
        return PHASE4_SELECTOR_LABEL[sel]
    except KeyError:
        raise KeyError(f"no printed label for the Phase 4 selector {sel!r}; add it to PHASE4_SELECTOR_LABEL") from None


def screen_label(s):
    """Phase 4 cascade screens (f10d): no_filter -> 'no screen'; perturb_iqr>t -> the screen at the diagnostic."""
    if s == "no_filter":
        return "no screen"
    m = re.fullmatch(r"perturb_iqr>(.+)", str(s))
    if m:
        return rf"screen at \diag{{perturb\_IQR}} $> {m.group(1)}$"
    raise KeyError(f"no printed label for the cascade screen {s!r}")


# ═════════════════════════════════════════════════════════════════════════════
# F01  trivial baseline -- THE PAPER'S FIRST RESULT
# ═════════════════════════════════════════════════════════════════════════════
def f01():
    """Trivial baseline, Prompt-5B layout: fixed-reference (deployable) columns
    vs predict-L_hat (constant_assumed) and vs last_value are primary, the
    hindsight best-of-four (strict) skill is one column, the oracle constant is
    a labelled reference row; no pooled-median-vs-best-single-trivial count."""
    if "win_rate_vs_assumed" not in G_CORE.columns:
        print("  (phase1_global.csv has no win_rate_vs_* columns: f01 needs a run at or after Prompt 5A; skipped)")
        return
    COLS = ["med_error", "win_rate_vs_assumed", "med_skill_vs_assumed", "win_rate_vs_last", "med_skill_vs_last", "med_skill"]
    FMT = [f4, f3, f3, f3, f3, f3]

    SELF = {"constant_assumed": ("win_rate_vs_assumed", "med_skill_vs_assumed"),
            "last_value": ("win_rate_vs_last", "med_skill_vs_last")}

    def cells(r):
        if r is None:
            return ["--"] * len(COLS)
        own = SELF.get(str(r["method"]), ())
        return ["--" if c in own else f(float(r[c])) for f, c in zip(FMT, COLS)]

    rows = [r"Stratum / row & \multicolumn{6}{c}{core} & \multicolumn{6}{c}{held-out} \\",
            r"\cmidrule(lr){2-7}\cmidrule(lr){8-13}",
            r" & med.\ err & win vs $\hat L$ & skill vs $\hat L$ & win vs last & skill vs last & strict skill "
            r"& med.\ err & win vs $\hat L$ & skill vs $\hat L$ & win vs last & skill vs last & strict skill \\",
            r"\midrule"]
    for g in STRATA:
        # the families and cells that enter the pooled statistics of this stratum (n_regimes, n_cells of the pooled table)
        n_fam = {k: int(G[G.target_g == g].n_regimes.max()) for k, G in (("core", G_CORE), ("holdout", G_HOLD))}
        n_cel = {k: int(G[G.target_g == g].n_cells.max()) for k, G in (("core", G_CORE), ("holdout", G_HOLD))}
        rows.append(mid(rf"\textit{{$g = {gname(g)}$}}" + (" (headline; " if g == HEADLINE_G else " (")
                        + f"core {n_fam['core']} families, {n_cel['core']} cells; held-out {n_fam['holdout']} families, {n_cel['holdout']} cells)", 13))
        labels = {"constant_assumed": r"predict $\hat L$ (\meth{constant\_assumed})", "last_value": r"\meth{last\_value}",
                  "window_mean": r"\meth{window\_mean}", "window_min": r"\meth{window\_min}"}
        for m in ("constant_assumed", "last_value", "window_mean", "window_min"):
            rows.append(f"{labels[m]} & " + " & ".join(cells(row_of(G_CORE, g, m)) + cells(row_of(G_HOLD, g, m))) + r" \\")
        rows.append(f"{mth(ORACLE)} \\textit{{(reference, not deployable)}} & " +
                    " & ".join(cells(row_of(G_CORE, g, ORACLE)) + cells(row_of(G_HOLD, g, ORACLE))) + r" \\")
        rows.append(r"\addlinespace[2pt]")
        stats = {}
        for key, G, RK in (("core", G_CORE, RANK_CORE), ("holdout", G_HOLD, RANK_HOLD)):
            sg = G[G.target_g == g]
            acc = sg[sg.is_trivial == 0]
            best_m = min(RK[g], key=RK[g].get)
            stats[key] = dict(
                best_m=best_m, n_eligible=int(acc.rank_eligible.sum()),
                n_win_lhat=int((acc.win_rate_vs_assumed > 0.5).sum()), n_win_last=int((acc.win_rate_vs_last > 0.5).sum()),
                n_strict=int((acc.med_skill < 1.0).sum()),
                med={c: float(acc[c].median()) for c in COLS},
                oracle_e=float(sg[sg.method == ORACLE].med_error.iloc[0]),
                lhat_e=float(sg[sg.method == "constant_assumed"].med_error.iloc[0]),
                last_e=float(sg[sg.method == "last_value"].med_error.iloc[0]))
        for key, label in (("core", "core"), ("holdout", "held-out")):
            m = stats[key]["best_m"]
            rows.append(f"best accelerator, {label} rank 1: {mth(m)} & " +
                        " & ".join(cells(row_of(G_CORE, g, m)) + cells(row_of(G_HOLD, g, m))) + r" \\")
        sc, sh = stats["core"], stats["holdout"]
        rows.append(f"median accelerator ({N_ACC}) & " +
                    " & ".join([f(sc["med"][c]) for f, c in zip(FMT, COLS)] + [f(sh["med"][c]) for f, c in zip(FMT, COLS)]) + r" \\")
        rows.append(rf"accelerators with win rate $> 0.5$ / strict skill $< 1$ & & {sc['n_win_lhat']}/{N_ACC} & & {sc['n_win_last']}/{N_ACC} & & {sc['n_strict']}/{N_ACC}"
                    rf" & & {sh['n_win_lhat']}/{N_ACC} & & {sh['n_win_last']}/{N_ACC} & & {sh['n_strict']}/{N_ACC} \\")
        if g != STRATA[-1]:
            rows.append(r"\midrule")
        for key, label in (("core", "core"), ("holdout", "held-out")):
            st = stats[key]
            sec = "trivial baseline"
            fil = SRC_G[0] if key == "core" else SRC_G[1]
            flt = f"target_g == {g}, regime_set == {key}, capped cells excluded (pooled table)"
            fact(sec, f"g={gname(g)} {label}: predict-L_hat / last_value / oracle median error",
                 f"{st['lhat_e']:.4f} / {st['last_e']:.4f} / {st['oracle_e']:.4f}", fil, flt, "med_error rows of constant_assumed, last_value, constant_oracle")
            bm = st["best_m"]; bro = row_of(G_CORE if key == "core" else G_HOLD, g, bm)
            fact(sec, f"g={gname(g)} {label}: best accelerator (rank 1 by median error)",
                 f"{bm}: med. err {bro.med_error:.4f}; win rate vs L_hat {bro.win_rate_vs_assumed:.3f}, vs last {bro.win_rate_vs_last:.3f}; "
                 f"median skill vs L_hat {bro.med_skill_vs_assumed:.3f}, vs last {bro.med_skill_vs_last:.3f}; strict {bro.med_skill:.3f}",
                 fil, flt + ", is_trivial == 0, rank_eligible == 1", "argmin med_error; win_rate_vs_* = mean over cells of the per-cell win rate")
            fact(sec, f"g={gname(g)} {label}: accelerators with win rate > 0.5 vs L_hat / vs last_value; with strict pooled skill < 1",
                 f"{st['n_win_lhat']} / {st['n_win_last']} / {st['n_strict']} of {N_ACC} ({st['n_eligible']} rank-eligible)",
                 fil, flt + ", is_trivial == 0", "count(win_rate_vs_assumed > 0.5); count(win_rate_vs_last > 0.5); count(med_skill < 1)")
            fact(sec, f"g={gname(g)} {label}: median accelerator",
                 f"med. err {st['med']['med_error']:.4f}; win vs L_hat {st['med']['win_rate_vs_assumed']:.3f}, vs last {st['med']['win_rate_vs_last']:.3f}; strict skill {st['med']['med_skill']:.3f}",
                 fil, flt + ", is_trivial == 0", f"median over the {N_ACC} accelerators")
    frag("f01_trivial_baseline.tex", "l" + "r" * 12, rows, SRC_G,
         "per stratum, core | held-out; win vs X = per-cell win rate (fraction of seeds with error below X's, mean over cells); "
         "skill vs X = median over cells of the per-cell median error ratio vs X; strict skill = hindsight best-of-four "
         "(err / best of the four deployable trivials on the same cell); the oracle is excluded from every count",
         "Trivial baseline (the paper's first result): the deployable trivials, the oracle constant as a labelled reference, the rank-1 "
         "and median accelerator, and how many accelerators beat predict-L_hat / last_value on more than half of their cells, "
         "with the hindsight (strict) skill alongside",
         notes=["the Prompt-4 'pooled median error below the best single trivial' count was dropped (Report-5A review): it "
                "compared pooled medians of different cells; the win rates are per-cell statements"])


# ═════════════════════════════════════════════════════════════════════════════
# F02  main ranking per stratum, core | held-out
# ═════════════════════════════════════════════════════════════════════════════
def f02():
    for g in STRATA:
        head = [r"$r$ & Method & Family & T & \multicolumn{4}{c}{core} & \multicolumn{5}{c}{held-out} \\",
                r"\cmidrule(lr){5-8}\cmidrule(lr){9-13}",
                r" & & & & med.\ err & med.\ skill & $\rhoC$ & $\rhoV$ & $r$ & med.\ err & med.\ skill & $\rhoC$ & $\rhoV$ \\",
                r"\midrule"]
        rows = list(head)
        ranked = sorted(RANK_CORE[g], key=RANK_CORE[g].get)
        for m in ranked:
            rc, rh = row_of(G_CORE, g, m), row_of(G_HOLD, g, m)
            rh_rank = RANK_HOLD[g].get(m, float("nan"))
            rows.append(f"{RANK_CORE[g][m]} & {dag(m)}{mth(m)} & {fam_label(FAM[m])} & {TYPE_MACRO[TYP[m]]} & "
                        f"{f4(rc.med_error)} & {f3(rc.med_skill)} & {f3(rc.cat_rate)} & {f3(rc.valid_rate)} & "
                        f"{fint(rh_rank)} & {f4(rh.med_error)} & {f3(rh.med_skill)} & {f3(rh.cat_rate)} & {f3(rh.valid_rate)} \\\\")
        below = G_CORE[(G_CORE.target_g == g) & (G_CORE.is_trivial == 0) & (G_CORE.rank_eligible == 0)]
        below = below.sort_values(["valid_rate", "method"], ascending=[False, True])
        rows.append(r"\midrule")
        rows.append(mid(rf"\textit{{Unranked: below the validity floor $\rhoV < {C.RANK_MIN_VALID:g}$ on the core families "
                        rf"({len(below)} methods; shown, never ranked)}}", 13))
        for _, rc in below.iterrows():
            m = rc.method
            rh = row_of(G_HOLD, g, m)
            rh_rank = RANK_HOLD[g].get(m, float("nan"))
            rows.append(f"-- & {dag(m)}{mth(m)} & {fam_label(FAM[m])} & {TYPE_MACRO[TYP[m]]} & "
                        f"{f4(rc.med_error)} & {f3(rc.med_skill)} & {f3(rc.cat_rate)} & {f3(rc.valid_rate)} & "
                        f"{fint(rh_rank)} & {f4(rh.med_error)} & {f3(rh.med_skill)} & {f3(rh.cat_rate)} & {f3(rh.valid_rate)} \\\\")
        frag(f"f02_ranking_g{gname(g)}.tex", "rlll" + "rrrr" + "rrrrr", rows, SRC_G,
             f"target_g == {g}; rank r = position by med_error among the {N_ACC} accelerators with rank_eligible == 1 "
             f"(valid_rate >= {C.RANK_MIN_VALID}), core and held-out ranked separately; trivial comparators not ranked "
             f"(see f01); dagger = in the dangerous artifact",
             f"Main ranking at g = {gname(g)}: rank pool = accelerators above the validity floor, core and held-out side by side, "
             f"unranked block appended")
        fact("ranking", f"g={gname(g)}: rank-eligible accelerators (core / held-out)",
             f"{len(RANK_CORE[g])} / {len(RANK_HOLD[g])} of {N_ACC}", SRC_G[0], f"target_g == {g}, is_trivial == 0",
             f"count(rank_eligible == 1); floor valid_rate >= {C.RANK_MIN_VALID}")
        fact("ranking", f"g={gname(g)}: below-floor accelerators (core)", ", ".join(below.method), SRC_G[0],
             f"target_g == {g}, is_trivial == 0, rank_eligible == 0", "the unranked block")


# ═════════════════════════════════════════════════════════════════════════════
# F03  skill summary per family
# ═════════════════════════════════════════════════════════════════════════════
def f03():
    A = AGG[(AGG.capped == 0) & (AGG.is_oracle == 0)].copy()
    A["beats"] = A.med_skill < 1.0
    fams = [f for f in FAMILY_ORDER if f in set(A.family)]
    head = [r"Family & $K$ & T & \multicolumn{2}{c}{$g = 0.5$} & \multicolumn{2}{c}{$g = 0.1$} & \multicolumn{2}{c}{$g = 0.02$} \\",
            r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
            r" & & & core & held-out & core & held-out & core & held-out \\", r"\midrule"]
    rows = list(head)

    def cell(sub):
        return pct(float(sub.beats.mean()), 0) if len(sub) else "--"

    for fam in fams:
        s = A[A.family == fam]
        K = s.method.nunique()
        types = sorted(set(TYP[m] for m in s.method.unique()))
        t = TYPE_MACRO[types[0]] if len(types) == 1 else "mixed"
        vals = []
        for g in STRATA:
            for h in (0, 1):
                vals.append(cell(s[(s.target_g == g) & (s.is_holdout == h)]))
        label = fam_label(fam) + (r" \textit{(4 deployable)}" if fam == "trivial" else "")
        if fam == "trivial":
            rows.append(r"\midrule")
        rows.append(f"{label} & {K} & {t} & " + " & ".join(vals) + r" \\")
        for g in STRATA:
            for h, lab in ((0, "core"), (1, "held-out")):
                sub = s[(s.target_g == g) & (s.is_holdout == h)]
                if g == HEADLINE_G:
                    fact("skill summary", f"{fam}: cells with median skill < 1, g={gname(g)}, {lab}",
                         f"{100 * sub.beats.mean():.1f}% ({int(sub.beats.sum())}/{len(sub)} cells)",
                         "results/phase1/phase1_aggregated.csv",
                         f"family == {fam}, target_g == {g}, is_holdout == {h}, capped == 0, is_oracle == 0",
                         "mean over (method, regime, noise) cells of [med_skill < 1]")
    acc = A[A.is_trivial == 0]
    vals = [cell(acc[(acc.target_g == g) & (acc.is_holdout == h)]) for g in STRATA for h in (0, 1)]
    rows.append(r"\midrule")
    rows.append(rf"\textbf{{all accelerators}} & {N_ACC} & & " + " & ".join(vals) + r" \\")
    for g in STRATA:
        for h, lab in ((0, "core"), (1, "held-out")):
            sub = acc[(acc.target_g == g) & (acc.is_holdout == h)]
            fact("skill summary", f"all accelerators: cells with median skill < 1, g={gname(g)}, {lab}",
                 f"{100 * sub.beats.mean():.1f}% ({int(sub.beats.sum())}/{len(sub)} cells)",
                 "results/phase1/phase1_aggregated.csv",
                 f"is_trivial == 0, target_g == {g}, is_holdout == {h}, capped == 0",
                 "mean over (method, regime, noise) cells of [med_skill < 1]")
    frag("f03_skill_summary.tex", "lrl" + "r" * 6, rows,
         [f"results/phase1/phase1_aggregated.csv (per (method, regime, noise, g) cell: median skill over {int(AGG.n_total.max())} seeds)"],
         "capped == 0, is_oracle == 0; a cell counts when its median-seed skill is < 1 (NaN median skill counts as not < 1)",
         "Skill summary: fraction of (method x regime x noise) cells whose median skill is below 1 (the method beat the "
         "hindsight best-of-four deployable trivial (strict) on the median seed), per family, per stratum, core vs held-out",
         notes=["K = methods in the family; T = method type (\\TE trajectory extrapolator / \\LE limit estimator)"])


# ═════════════════════════════════════════════════════════════════════════════
# F04  classical no-op table
# ═════════════════════════════════════════════════════════════════════════════
NOOP_BAND_COUNTS = {}     # {'core' | 'holdout': {g: variants in the MI band}}; f21's internal check reads it


def _classical_noop(key):
    """Classical no-op table on the classical variants: MI vs the +/-10 % band and,
    in win-rate terms, the per-cell win rate vs last_value within 0.5 +/- 0.1;
    key = 'core' (f04, phase1_global.csv) or 'holdout' (f04b, phase1_global_holdout.csv)."""
    G, src, label = (G_CORE, SRC_G[0], "core") if key == "core" else (G_HOLD, SRC_G[1], "held-out")
    has_win = "win_rate_vs_last" in G.columns
    per = 3 if has_win else 2
    head = [r"Family & Method & " + " & ".join(rf"\multicolumn{{{per}}}{{c}}{{$g = {gname(g)}$}}" for g in STRATA) + r" \\",
            "".join(rf"\cmidrule(lr){{{3 + per * i}-{2 + per * (i + 1)}}}" for i in range(len(STRATA))),
            r" & & " + " & ".join((r"MI & win vs last & med.\ skill" if has_win else r"MI & med.\ skill") for _ in STRATA) + r" \\", r"\midrule"]
    rows = list(head)
    inband = {g: 0 for g in STRATA}
    wband = {g: 0 for g in STRATA}
    snoop = {g: 0 for g in STRATA}
    outside = {g: [] for g in STRATA}      # (method, MI) of the variants outside the band
    wins = {g: [] for g in STRATA}
    last_fam = None
    for m in CLASSICAL:
        fam = FAM[m]
        cells = []
        for g in STRATA:
            r = row_of(G, g, m)
            mi, sk = float(r.med_improve), float(r.med_skill)
            ib = math.isfinite(mi) and NOOP_BAND[0] <= mi <= NOOP_BAND[1]
            sn = math.isfinite(sk) and sk >= SKILL_NOOP
            inband[g] += int(ib)
            snoop[g] += int(sn)
            if not ib:
                outside[g].append((m, mi))
            cells.append((r"\textbf{" + f3(mi) + "}") if ib else f3(mi))
            if has_win:
                w = float(r.win_rate_vs_last)
                wb = math.isfinite(w) and WIN_BAND[0] <= w <= WIN_BAND[1]
                wband[g] += int(wb)
                wins[g].append(w)
                cells.append((r"\textbf{" + f3(w) + "}") if wb else f3(w))
            cells.append((r"\textbf{" + f3(sk) + "}") if sn else f3(sk))
        if last_fam is not None and fam != last_fam:
            rows.append(r"\addlinespace[2pt]")
        last_fam = fam
        rows.append(f"{fam_label(fam)} & {dag(m)}{mth(m)} & " + " & ".join(cells) + r" \\")
    n = len(CLASSICAL)
    rows.append(r"\midrule")
    rows.append(r"\multicolumn{2}{l}{in the $\pm 10\%$ band (MI in $[0.9, 1.1]$)} & " +
                " & ".join(f"{inband[g]}/{n}" + " & " * (per - 1) for g in STRATA) + r" \\")
    if has_win:
        rows.append(r"\multicolumn{2}{l}{win rate vs last in $[0.4, 0.6]$ (coin flip)} & " +
                    " & ".join(f" & {wband[g]}/{n} & " for g in STRATA) + r" \\")
    rows.append(r"\multicolumn{2}{l}{median skill $\geq 0.9$} & " +
                " & ".join(" & " * (per - 1) + f"{snoop[g]}/{n}" for g in STRATA) + r" \\")
    for g in STRATA:
        fact("classical no-op", f"g={gname(g)} {label}: classical variants with MI in [0.9, 1.1]", f"{inband[g]} of {n}", src,
             f"target_g == {g}, family in {CLASSICAL_FAMILIES}", "count(0.9 <= med_improve <= 1.1); MI = median of curr_err / err")
        if has_win:
            fact("classical no-op", f"g={gname(g)} {label}: classical variants with win rate vs last_value in [0.4, 0.6]", f"{wband[g]} of {n}", src,
                 f"target_g == {g}, family in {CLASSICAL_FAMILIES}", "count(0.4 <= win_rate_vs_last <= 0.6); a no-op wins against the last value about half the time")
        fact("classical no-op", f"g={gname(g)} {label}: classical variants with median skill >= 0.9", f"{snoop[g]} of {n}", src,
             f"target_g == {g}, family in {CLASSICAL_FAMILIES}", "count(med_skill >= 0.9)")
    for g in STRATA:
        fact("classical no-op", f"g={gname(g)} {label}: classical variants outside the MI band [0.9, 1.1], with their MI",
             "; ".join(f"{m} {mi:.3f}" for m, mi in outside[g]) or "none", src,
             f"target_g == {g}, family in {CLASSICAL_FAMILIES}", "med_improve of the variants with med_improve outside [0.9, 1.1]")
        if has_win:
            fact("classical no-op", f"g={gname(g)} {label}: mean win rate vs last_value over the classical variants",
                 f"{float(np.mean(wins[g])):.3f} (min {min(wins[g]):.3f}, max {max(wins[g]):.3f})", src,
                 f"target_g == {g}, family in {CLASSICAL_FAMILIES}", "mean / min / max over the variants of win_rate_vs_last")
    NOOP_BAND_COUNTS[key] = dict(inband)
    name = "f04_classical_noop.tex" if key == "core" else "f04b_classical_noop_holdout.tex"
    frag(name, "ll" + "r" * (per * len(STRATA)), rows, [src],
         f"{'core regimes' if key == 'core' else 'held-out families'}, capped excluded; the {n} classical variants = families shanks, wynn_eps, wynn_rho, levin, brezinski, anderson "
         "(the Weniger pair was retired: numerically identical to levin_t1/t2); MI = med_improve (median over seeds and cells of "
         "current-value error / method error), bold = inside the +/-10 % band; win vs last = per-cell win rate vs last_value, bold = "
         "inside [0.4, 0.6]; med. skill = hindsight best-of-four (strict), bold = >= 0.9",
         "Classical no-op table by stratum: median improvement factor vs the +/-10 % band, win rate vs the last value, and strict skill"
         + ("" if key == "core" else " -- the held-out families (the layout and statistics of f04 on phase1_global_holdout.csv)"))


def f04():
    _classical_noop("core")


def f04b():
    _classical_noop("holdout")


# ═════════════════════════════════════════════════════════════════════════════
# F05  assumed-asymptote mode sweep (Phase 5b sweep 1)
# ═════════════════════════════════════════════════════════════════════════════
def f05():
    S = read("phase5b", "phase5b_sweep1_global.csv")
    gs = [g for g in STRATA if g in set(S.target_g)]
    head = [r"Curve families & $\hat{L}$ mode & $\sigma$ & " +
            " & ".join(rf"\multicolumn{{5}}{{c}}{{$g = {gname(g)}$}}" for g in gs) + r" \\",
            "".join(rf"\cmidrule(lr){{{4 + 5 * i}-{8 + 5 * i}}}" for i in range(len(gs))),
            r" & & & " + " & ".join(r"fire & lower rec. & lower cell & $\Delta$med & V$_{\mathrm{alt}}$" for _ in gs) + r" \\", r"\midrule"]
    rows = list(head)
    modes = [m for m in C.ASSUMED_L_MODES if m in set(S.assumed_mode)]
    for rs in ("core", "holdout"):
        for i, mode in enumerate(modes):
            for j, nz in enumerate(sorted(S.noise.unique())):
                cells = []
                for g in gs:
                    r = S[(S.regime_set == rs) & (S.assumed_mode == mode) & (S.noise == nz) & (S.target_g == g)]
                    if len(r):
                        r = r.iloc[0]
                        cells += [f3(r.fire_rate), f3(float(r.lower_error_frac_records)), f3(float(r.lower_error_frac_cells)),
                                  signed(float(r.median_rel_change), 3), f3(float(r.alt_valid_rate))]
                    else:
                        cells += ["--"] * 5
                lab_rs = ("core" if rs == "core" else "held-out") if (i == 0 and j == 0) else ""
                lab_mode = (esc(mode) + (r" \textit{(oracle)}" if mode == "oracle" else "")) if j == 0 else ""
                rows.append(f"{lab_rs} & {lab_mode} & {nz:g} & " + " & ".join(cells) + r" \\")
            if mode != modes[-1]:
                rows.append(r"\addlinespace[1pt]")
        if rs == "core":
            rows.append(r"\midrule")
    core_h = S[(S.regime_set == "core") & (S.target_g == HEADLINE_G)]
    nonzero = [m for m in modes if m != "zero"]
    for col, label in (("fire_rate", "cascade fire rate"), ("lower_error_frac_records", "fraction of fired windows with lower routed error")):
        piv = core_h.pivot_table(index="noise", columns="assumed_mode", values=col)
        spread = float((piv.max(axis=1) - piv.min(axis=1)).max())
        piv_nz = piv[[m for m in nonzero if m in piv.columns]]
        spread_nz = float((piv_nz.max(axis=1) - piv_nz.min(axis=1)).max()) if len(piv_nz.columns) else float("nan")
        fact("sweep 1 (assumed asymptote)", f"g={gname(HEADLINE_G)} core: max spread of the {label} across the L_hat modes",
             f"{spread:.4f} (across the non-zero modes {spread_nz:.4f})", "results/phase5b/phase5b_sweep1_global.csv",
             f"regime_set == core, target_g == {HEADLINE_G}", f"max over noise of (max - min over assumed_mode of {col})")
    frag("f05_sweep1_modes.tex", "lll" + "rrrrr" * len(gs), rows,
         ["results/phase5b/phase5b_sweep1_global.csv (Phase 5b sweep 1a, pooled over regimes, capped excluded; src.panels.rule_panel)"],
         "all rows; one window (regime, noise, seed) is one cell of the cascade evaluation; fire = fraction of windows where the "
         "cascade routed to rational_fit; lower rec. / lower cell = fraction of fired windows where rational_fit's error was below "
         "richardson_1's (both valid; an invalid record never counts as lower; on one-record cells the two coincide); Delta med = "
         "median over fired windows of (rational_fit error - richardson_1 error) / richardson_1 error; V_alt = validity rate of "
         "rational_fit on the fired windows (errors conditional on validity)",
         "Assumed-asymptote mode sweep: what happened when the Phase-2 cascade fired, under each L_hat mode, core and held-out, per stratum",
         notes=["the cascade features clamp L0 = max(0, min(L_hat, 0.5 * min(window))), so the four non-zero modes coincide "
                "whenever L_hat >= 0.5 * min(window); see FACTS.md"])


# ═════════════════════════════════════════════════════════════════════════════
# F06  held-out vs core generalisation
# ═════════════════════════════════════════════════════════════════════════════
def f05b():
    """Sweep 1b: per L_hat-consuming method, median error under each mode and the
    win rate vs last_value (the mode-invariant reference), at g = 0.1 and 0.5,
    core and held-out; constant_assumed's own error per mode = the value of
    knowing the floor."""
    path = os.path.join(RES, "phase5b", "phase5b_sweep1_consumers.csv")
    if not os.path.exists(path):
        print("  (phase5b_sweep1_consumers.csv absent: fragment f05b skipped; produced by a run at or after Prompt 5A)")
        return
    Cn = pd.read_csv(path)
    from phases.phase5b import SWEEP1_METHODS
    modes = [m for m in C.ASSUMED_L_MODES if m in set(Cn.assumed_mode)]
    gs = [g for g in (0.5, 0.1) if g in set(Cn.target_g)]
    head = [r"Method & row & " + " & ".join(rf"\multicolumn{{{len(modes)}}}{{c}}{{$g = {gname(g)}$}}" for g in gs) + r" \\",
            "".join(rf"\cmidrule(lr){{{3 + len(modes) * i}-{2 + len(modes) * (i + 1)}}}" for i in range(len(gs))),
            r" & & " + " & ".join(" & ".join(esc(m) for m in modes) for _ in gs) + r" \\", r"\midrule"]
    for rs in ("core", "holdout"):
        rows = list(head)
        for m in SWEEP1_METHODS:
            e_cells, w_cells = [], []
            for g in gs:
                for mode in modes:
                    r = Cn[(Cn.method == m) & (Cn.assumed_mode == mode) & (Cn.regime_set == rs) & (Cn.target_g == g)]
                    if len(r):
                        r = r.iloc[0]
                        e_cells.append(f4(float(r.med_error)))
                        w_cells.append("--" if m == "last_value" else f3(float(r.win_rate_vs_last)))
                    else:
                        e_cells.append("--"); w_cells.append("--")
            label = (r"predict $\hat L$ (\meth{constant\_assumed}) -- the value of knowing the floor" if m == "constant_assumed" else mth(m))
            rows.append(f"{label} & med.\\ err & " + " & ".join(e_cells) + r" \\")
            rows.append(r" & win vs last & " + " & ".join(w_cells) + r" \\")
            rows.append(r"\addlinespace[1pt]")
        frag(f"f05b_sweep1_consumers_{rs}.tex", "ll" + "r" * (len(modes) * len(gs)), rows,
             ["results/phase5b/phase5b_sweep1_consumers.csv (Phase 5b sweep 1b; capped excluded; pooled over noise)"],
             f"regime_set == {rs}; med. err = median error of the method under that L_hat mode; win vs last = per-record win rate "
             "against last_value, whose prediction does not depend on the mode (the mode-invariant reference); the vs-L_hat columns "
             "stay in the CSV but are not the headline comparison because their reference changes with the mode",
             f"Assumed-asymptote sweep 1b, {rs} regimes: the L_hat-consuming accelerators under every mode, and the value of knowing the floor")
    # L_hat-invariance facts: max change in median error across modes per consumer
    for rs in ("core", "holdout"):
        for g in gs:
            parts = []
            for m in SWEEP1_METHODS:
                ch = lhat_change(Cn, m, rs, g)
                if ch is not None:
                    d, rel = ch
                    parts.append(f"{m} {d:.5f} ({rel:.1f} %)")
            fact("sweep 1b (L_hat invariance)", f"max change of median error across the five L_hat modes, g={gname(g)}, {rs}",
                 "; ".join(parts), "results/phase5b/phase5b_sweep1_consumers.csv", f"regime_set == {rs}, target_g == {g}",
                 "max over modes - min over modes of med_error per method (relative to the min)")
            ca = Cn[(Cn.method == "constant_assumed") & (Cn.regime_set == rs) & (Cn.target_g == g)]
            fact("sweep 1b (L_hat invariance)", f"the value of knowing the floor: constant_assumed median error per mode, g={gname(g)}, {rs}",
                 ", ".join(f"{r.assumed_mode} {r.med_error:.4f}" for _, r in ca.iterrows()),
                 "results/phase5b/phase5b_sweep1_consumers.csv", f"method == constant_assumed, regime_set == {rs}, target_g == {g}", "med_error per assumed_mode")
            best = Cn[(Cn.regime_set == rs) & (Cn.target_g == g) & (Cn.method != "constant_assumed")]
            fact("sweep 1b (L_hat invariance)", f"win rate vs last_value per consumer under the zero mode, g={gname(g)}, {rs}",
                 ", ".join(f"{r.method} {r.win_rate_vs_last:.3f}" for _, r in best[best.assumed_mode == "zero"].sort_values("win_rate_vs_last", ascending=False).iterrows()),
                 "results/phase5b/phase5b_sweep1_consumers.csv", f"assumed_mode == zero, regime_set == {rs}, target_g == {g}", "win_rate_vs_last")


def sweep1_consumers():
    """phase5b_sweep1_consumers.csv (None when absent)."""
    path = os.path.join(RES, "phase5b", "phase5b_sweep1_consumers.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


def lhat_change(Cn, m, rs, g):
    """(max - min, 100 * (max - min) / min) of med_error across the assumed-asymptote
    modes for one consumer, regime set and stratum; None with fewer than two modes.
    The one computation behind the sweep-1b invariance rows and the f24 column."""
    sub_ = Cn[(Cn.method == m) & (Cn.regime_set == rs) & (Cn.target_g == g)]
    if len(sub_) < 2:
        return None
    lo, hi = float(sub_.med_error.min()), float(sub_.med_error.max())
    return hi - lo, (100 * (hi - lo) / lo if lo > 0 else float("nan"))


def facts_lhat_invariance_full_precision():
    """Assumed-asymptote invariance at full precision, per consumer: the largest
    absolute and relative difference of med_error across the modes (scientific
    notation) and the spread of the win rate vs last_value, per set and stratum."""
    Cn = sweep1_consumers()
    if Cn is None:
        return
    gs = [g for g in (0.5, 0.1) if g in set(Cn.target_g)]
    for m in LHAT_CONSUMERS:
        parts = []
        for rs in ("core", "holdout"):
            for g in gs:
                sub_ = Cn[(Cn.method == m) & (Cn.regime_set == rs) & (Cn.target_g == g)]
                if len(sub_) < 2:
                    continue
                lo, hi = float(sub_.med_error.min()), float(sub_.med_error.max())
                wlo, whi = float(sub_.win_rate_vs_last.min()), float(sub_.win_rate_vs_last.max())
                rel = (hi - lo) / lo if lo > 0 else float("nan")
                parts.append(f"{rs} g={gname(g)}: |d med_error| max {hi - lo:.3e} (rel {rel:.3e}), win rate spread {whi - wlo:.4f}")
        if parts:
            fact("sweep 1b (L_hat invariance)", f"{m}: assumed-asymptote invariance at full precision",
                 "; ".join(parts), "results/phase5b/phase5b_sweep1_consumers.csv", f"method == {m}, per regime_set and target_g",
                 "max - min over the assumed modes of med_error (absolute, and relative to the min); max - min of win_rate_vs_last")


def generalisation_rhos():
    """Per stratum: Spearman rho between the core and held-out med_error ranks over
    the accelerators rank-eligible in both regime sets, and that count.  The one
    computation behind the generalisation facts (f06) and the summary table (f16)."""
    out = []
    for g in STRATA:
        both = [m for m in ACCEL_METHODS if m in RANK_CORE[g] and m in RANK_HOLD[g]]
        rho = spearmanr([RANK_CORE[g][m] for m in both], [RANK_HOLD[g][m] for m in both]).correlation if len(both) >= 2 else float("nan")
        out.append((g, float(rho), len(both)))
    return out


def f06():
    head = [r"Method & \multicolumn{3}{c}{$g = 0.5$} & \multicolumn{5}{c}{$g = 0.1$} & \multicolumn{3}{c}{$g = 0.02$} \\",
            r"\cmidrule(lr){2-4}\cmidrule(lr){5-9}\cmidrule(lr){10-12}",
            r" & $r_c$ & $r_h$ & $\Delta$ & $r_c$ & $r_h$ & $\Delta$ & err$_c$ & err$_h$ & $r_c$ & $r_h$ & $\Delta$ \\",
            r"\midrule"]
    rows = list(head)

    def key(m):
        rc = RANK_CORE[HEADLINE_G].get(m)
        return (0, rc) if rc is not None else (1, RANK_HOLD[HEADLINE_G].get(m, 999), m)

    for m in sorted(ACCEL_METHODS, key=key):
        cells = []
        for g in STRATA:
            rc, rh = RANK_CORE[g].get(m, float("nan")), RANK_HOLD[g].get(m, float("nan"))
            d = rh - rc if (isinstance(rc, int) and isinstance(rh, int)) else float("nan")
            cells += [fint(rc), fint(rh), ("--" if not math.isfinite(d) else f"${d:+d}$")]
            if g == HEADLINE_G:
                cells += [f4(row_of(G_CORE, g, m).med_error), f4(row_of(G_HOLD, g, m).med_error)]
        label = dag(m) + mth(m)
        if m in ("rational_fit", "log_linear"):
            label = r"\textbf{" + label + "}"
        rows.append(f"{label} & " + " & ".join(cells) + r" \\")
    rows.append(r"\midrule")
    rhos = generalisation_rhos()
    rows.append(r"\multicolumn{12}{l}{Spearman $\rho$ of core vs held-out rank over methods ranked in both: " +
                ", ".join(rf"$g = {gname(g)}$: {rho:.3f} ($n = {n}$)" for g, rho, n in rhos) + r"} \\")
    for g, rho, n in rhos:
        fact("generalisation", f"g={gname(g)}: Spearman rho of core vs held-out accelerator rank", f"{rho:.3f} (n = {n})",
             "; ".join(SRC_G), f"target_g == {g}, rank-eligible in both regime sets", "spearmanr(rank_core, rank_holdout)")
    for m in ("rational_fit", "log_linear"):
        for g in STRATA:
            fact("generalisation", f"{m}: core / held-out rank at g={gname(g)}",
                 f"{RANK_CORE[g].get(m, 'unranked')} / {RANK_HOLD[g].get(m, 'unranked')}", "; ".join(SRC_G),
                 f"target_g == {g}", "rank among rank-eligible accelerators by med_error")
    frag("f06_generalisation.tex", "l" + "rrr" + "rrrrr" + "rrr", rows, SRC_G,
         f"ranks among the {N_ACC} accelerators (rank-eligible only; -- = below the validity floor in that regime set); "
         "Delta = r_h - r_c (positive = worse on the held-out regimes); rows sorted by core rank at the headline stratum",
         "Held-out vs core generalisation: per-method rank shift per stratum, with median errors at the headline stratum; "
         "rational_fit and log_linear in bold")


# ═════════════════════════════════════════════════════════════════════════════
# F07  real data v2 grid, and the legacy 18-cell table
# ═════════════════════════════════════════════════════════════════════════════
def real_summary_with_minima():
    """real_data_summary_v2.csv plus argmin_round / post_min_target, computed from
    the recorded curves when the summary predates those columns."""
    S = read("real_data", "real_data_summary_v2.csv")
    curves = read("real_data", "real_data_curves.csv")
    minima = {}
    for d in [c for c in curves.columns if c != "round"]:
        v = curves[d].to_numpy(dtype=float)
        i = int(np.nanargmin(v))
        minima[d] = dict(argmin_round=i + 1, min_value=float(v[i]), value_at_500=float(v[min(500, len(v)) - 1]),
                         rise_from_min=(float(v[min(500, len(v)) - 1]) - float(v[i])) / float(v[i]))
    if "post_min_target" not in S.columns:
        S["argmin_round"] = S.dataset.map(lambda d: minima[d]["argmin_round"])
        S["rise_from_min"] = S.dataset.map(lambda d: minima[d]["rise_from_min"])
        S["post_min_target"] = (S.target_round > S.argmin_round).astype(int)
    return S, minima


REAL_STRATA = (("all", None), ("pre-minimum", 0), ("post-minimum", 1))


def f07():
    S, minima = real_summary_with_minima()
    targets = sorted(S.target_round.unique())
    ABBR = {"richardson_1": "R1", "rational_fit": "RF"}
    head = [r"Dataset & $\nobs$ & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{target round {int(t)}}}" for t in targets) + r" \\",
            "".join(rf"\cmidrule(lr){{{3 + 3 * i}-{5 + 3 * i}}}" for i in range(len(targets))),
            r" & & " + " & ".join(r"routed & err & skill" for _ in targets) + r" \\", r"\midrule"]
    rows = list(head)
    for d in REAL_DATASETS:
        sd = S[S.dataset == d]
        for i, ob in enumerate(sorted(sd.obs_depth.unique())):
            cells = []
            for t in targets:
                r = sd[(sd.obs_depth == ob) & (sd.target_round == t)]
                if len(r):
                    r = r.iloc[0]
                    sk = float(r.cascade_skill)
                    sks = (r"\textbf{" + f2(sk) + "}") if sk >= 1 else f2(sk)
                    if int(r.post_min_target) == 1:
                        sks += r"$^{+}$"
                    cells += [ABBR.get(r.selected_method, esc(r.selected_method)), f4(float(r.cascade_err)), sks]
                else:
                    cells += ["--"] * 3
            rows.append(f"{tt(d) if i == 0 else ''} & {int(ob)} & " + " & ".join(cells) + r" \\")
        if d != REAL_DATASETS[-1]:
            rows.append(r"\addlinespace[2pt]")
    rows.append(r"\midrule")
    med = [f"{S[S.target_round == t].cascade_skill.median():.2f}" for t in targets]
    nfail = [int((S[S.target_round == t].cascade_skill >= 1).sum()) for t in targets]
    rows.append(r"\multicolumn{2}{l}{median skill / cells with skill $\geq 1$ (of 30)} & " +
                " & ".join(f" & & {m} / {n}" for m, n in zip(med, nfail)) + r" \\")
    for label, flag in REAL_STRATA[1:]:
        sub = S if flag is None else S[S.post_min_target == flag]
        rows.append(rf"\multicolumn{{2}}{{l}}{{{label} targets: cells / skill $\geq 1$ / median skill}} & " +
                    " & ".join(f" & & {len(sub[sub.target_round == t])} / {int((sub[sub.target_round == t].cascade_skill >= 1).sum())} / "
                               f"{sub[sub.target_round == t].cascade_skill.median():.2f}" if len(sub[sub.target_round == t]) else " & & --"
                               for t in targets) + r" \\")
    for d, m in minima.items():
        fact("real data (v2)", f"{d}: recorded-curve minimum", f"argmin round {m['argmin_round']}, min {m['min_value']:.5f}, "
             f"value at round 500 {m['value_at_500']:.5f}, rise from minimum {100 * m['rise_from_min']:+.2f}%",
             "results/real_data/real_data_curves.csv", f"column {d}", "argmin over rounds (1-based); (v[500] - min) / min")
    for label, flag in REAL_STRATA:
        sub = S if flag is None else S[S.post_min_target == flag]
        fact("real data (v2)", f"{label} targets: cells / cascade skill >= 1 / median skill / median improvement",
             f"{len(sub)} / {int((sub.cascade_skill >= 1).sum())} / {sub.cascade_skill.median():.3f} / {sub.improvement.median():+.3f}",
             "results/real_data/real_data_summary_v2.csv", "all rows" if flag is None else f"post_min_target == {flag} (target_round > argmin_round)",
             "count(cascade_skill >= 1); median(cascade_skill); median(improvement)")
        for col, tag in (("cascade_win_vs_assumed", "constant_assumed"), ("cascade_win_vs_last", "last_value")):
            if col in sub.columns:
                fact("real data (v2)", f"{label} targets: cascade win rate vs {tag}", f"{sub[col].mean():.3f}",
                     "results/real_data/real_data_summary_v2.csv", "as above", f"mean({col})")
    n_fail = int((S.cascade_skill >= 1).sum())
    n_neg = int((S.improvement < 0).sum())
    n_cells = len(S)
    fact("real data (v2)", "cells on the grid",
         f"{n_cells} = {S.dataset.nunique()} datasets x {S.obs_depth.nunique()} depths x {S.target_round.nunique()} targets",
         "results/real_data/real_data_summary_v2.csv", "all rows", "row count; nunique of dataset, obs_depth, target_round")
    fact("real data (v2)", f"median cascade skill over the {n_cells} cells", f"{S.cascade_skill.median():.3f}",
         "results/real_data/real_data_summary_v2.csv", "all rows", "median(cascade_skill); skill = hindsight best-of-four (strict): cascade_err / best-of-four trivial error")
    fact("real data (v2)", "cells with cascade skill >= 1 (cascade no better than the best trivial)", f"{n_fail} of {n_cells}",
         "results/real_data/real_data_summary_v2.csv", "all rows", "count(cascade_skill >= 1)")
    fact("real data (v2)", "cells with negative improvement over the last observed value", f"{n_neg} of {n_cells}",
         "results/real_data/real_data_summary_v2.csv", "all rows", "count(improvement < 0); improvement = (current_err - cascade_err) / current_err, current_err = error of the last observed value")
    fail = S[S.cascade_skill >= 1]
    fact("real data (v2)", "failing cells by dataset", ", ".join(f"{k} {v}" for k, v in fail.dataset.value_counts().items()),
         "results/real_data/real_data_summary_v2.csv", "cascade_skill >= 1", "value counts of dataset")
    fact("real data (v2)", "failing cells by depth", ", ".join(f"{int(k)}: {v}" for k, v in fail.obs_depth.value_counts().sort_index().items()),
         "results/real_data/real_data_summary_v2.csv", "cascade_skill >= 1", "value counts of obs_depth")
    w = S.loc[S.cascade_skill.idxmax()]
    fact("real data (v2)", "worst cell", f"{w.dataset} depth {int(w.obs_depth)} -> round {int(w.target_round)} ({w.selected_method}): skill {w.cascade_skill:.1f}, improvement {w.improvement:+.1f}",
         "results/real_data/real_data_summary_v2.csv", "argmax cascade_skill", "-")
    fact("real data (v2)", "best trivial reference per cell", ", ".join(f"{k} {v}" for k, v in S.ref_best_method.value_counts().items()),
         "results/real_data/real_data_summary_v2.csv", "all rows", "value counts of ref_best_method (why skill >= 1 coincides with improvement <= 0)")
    fact("real data (v2)", "routing", f"richardson_1 {int((S.selected_method == 'richardson_1').sum())}, rational_fit {int((S.selected_method == 'rational_fit').sum())} cells",
         "results/real_data/real_data_summary_v2.csv", "all rows", "value counts of selected_method")
    fact("real data (v2)", "provenance recorded on every row", f"git_head {S.git_head.iloc[0]}, phase2_features_rows {int(S.phase2_features_rows.iloc[0])}",
         "results/real_data/real_data_summary_v2.csv", "all rows", "columns git_head, phase2_features_rows")
    frag("f07_real_data_v2.tex", "lr" + "lrr" * len(targets), rows,
         ["results/real_data/real_data_summary_v2.csv (recorded XGBoost validation curves re-evaluated under the v2 design; "
          "L_hat mode zero; cascade = Phase-2 rules on the window features)"],
         f"all {len(S)} cells; routed: R1 = richardson_1, RF = rational_fit; err = |cascade prediction - recorded value at the "
         "target round|; skill = hindsight best-of-four (strict): err / best-of-four trivial error on the same cell, bold when >= 1; a trailing + marks a post-minimum target (target round beyond the recorded curve's argmin)",
         "Real-data re-evaluation over the (depth x target) grid: routed method, cascade error and skill per cell")

    # legacy 18-cell run, preserved as its own fragment
    L = read("real_data", "real_data_results.csv")
    L["improvement"] = (L.current_err - L.cascade_err) / L.current_err
    rows = [r"Dataset & $\nobs$ & routed & predicted & cascade err & last-value err & improv. \\", r"\midrule"]
    for d in REAL_DATASETS:
        for _, r in L[L.dataset == d].sort_values("obs_depth").iterrows():
            ce = (r"\textbf{" + f4(float(r.cascade_err)) + "}") if r.improvement < 0 else f4(float(r.cascade_err))
            rows.append(f"{tt(d)} & {int(r.obs_depth)} & {mth(r.selected_method)} & {f4(float(r.predicted_val))} & {ce} & "
                        f"{f4(float(r.current_err))} & ${100 * r.improvement:+.1f}$\\% \\\\")
        if d != REAL_DATASETS[-1]:
            rows.append(r"\addlinespace[2pt]")
    rows.append(r"\midrule")
    parts = []
    for ob in sorted(L.obs_depth.unique()):
        s = L[L.obs_depth == ob]
        red = 100 * (s.current_err.mean() - s.cascade_err.mean()) / s.current_err.mean()
        parts.append(f"$\\nobs = {int(ob)}$: ${red:+.1f}$\\%")
        fact("real data (legacy 18-cell run)", f"reduction in mean error at depth {int(ob)}", f"{red:+.1f}%",
             "results/real_data/real_data_results.csv", f"obs_depth == {int(ob)}",
             "100 * (mean current_err - mean cascade_err) / mean current_err")
    red_all = 100 * (L.current_err.mean() - L.cascade_err.mean()) / L.current_err.mean()
    fact("real data (legacy 18-cell run)", f"reduction in mean error, all {len(L)} cells", f"{red_all:+.1f}%",
         "results/real_data/real_data_results.csv", "all rows", "100 * (mean current_err - mean cascade_err) / mean current_err")
    fact("real data (legacy 18-cell run)", "cells worse than the last observed value", f"{int((L.improvement < 0).sum())} of {len(L)}",
         "results/real_data/real_data_results.csv", "all rows", "count(cascade_err > current_err); current_err = error of the last observed value")
    rows.append(r"\multicolumn{7}{l}{reduction in mean error: " + ", ".join(parts) + f"; all cells ${red_all:+.1f}$\\%" + r"} \\")
    frag("f07b_real_data_legacy18.tex", "lrlrrrr", rows,
         ["results/real_data/real_data_results.csv (the pre-redesign 18-cell run: fixed target = final recorded round, "
          "legacy assumed asymptote 0.01; kept for the record, regenerable with scripts/analyze_real_diagnostics_legacy.py)"],
         f"all {len(L)} rows; bold = cascade worse than the last observed value",
         "Legacy 18-cell real-data table (pre-redesign design), preserved unchanged")


# ═════════════════════════════════════════════════════════════════════════════
# F08  real-data perturbation diagnostic
# ═════════════════════════════════════════════════════════════════════════════
def f08():
    S, _ = real_summary_with_minima()
    S = S[S.perturb_iqr.notna()].copy()
    defs = [("skill $\\geq 1$", S.cascade_skill >= 1.0, "cascade_skill >= 1"),
            ("improvement $< 0$", S.improvement < 0, "improvement < 0"),
            ("either", (S.cascade_skill >= 1.0) | (S.improvement < 0), "cascade_skill >= 1 or improvement < 0")]
    rows = [r"Cells & failure definition & $n_{\mathrm{fail}}$ & $n_{\mathrm{succ}}$ & med.\ IQR$_{\mathrm{fail}}$ & "
            r"med.\ IQR$_{\mathrm{succ}}$ & AUC & $p$ & ordering \\", r"\midrule"]
    answer_lines = []

    def auc_row(label, sub, fail, dlabel, dexpr):
        f, s = sub.perturb_iqr[fail], sub.perturb_iqr[~fail]
        if len(f) == 0 or len(s) == 0:
            rows.append(f"{label} & {dlabel} & {len(f)} & {len(s)} & -- & -- & -- & -- & -- \\\\")
            return None
        u = mannwhitneyu(f, s, alternative="two-sided")
        auc = float(u.statistic) / (len(f) * len(s))          # P(IQR_fail > IQR_succ) (+ half ties)
        order = "failing $>$ succeeding" if auc > 0.5 else ("failing $<$ succeeding" if auc < 0.5 else "no ordering")
        rows.append(f"{label} & {dlabel} & {len(f)} & {len(s)} & {f4(float(f.median()))} & {f4(float(s.median()))} & "
                    f"{auc:.3f} & {u.pvalue:.3f} & {order} \\\\")
        return dict(n_f=len(f), n_s=len(s), med_f=float(f.median()), med_s=float(s.median()), auc=auc, p=float(u.pvalue),
                    order=order.replace("$", ""), dexpr=dexpr)

    res = {}
    for dlabel, fail, dexpr in defs:
        res[dexpr] = auc_row(f"all {len(S)}", S, fail, dlabel, dexpr)
    rows.append(r"\addlinespace[2pt]")
    for label, flag in REAL_STRATA[1:]:
        sub = S[S.post_min_target == flag]
        fail = (sub.cascade_skill >= 1.0) | (sub.improvement < 0)
        res[f"either|{label}"] = auc_row(f"{label} targets ({len(sub)})", sub, fail, "either", f"either, post_min_target == {flag}")
    rows.append(r"\addlinespace[2pt]")
    for m in ("richardson_1", "rational_fit"):
        sub = S[S.selected_method == m]
        fail = (sub.cascade_skill >= 1.0) | (sub.improvement < 0)
        res[f"either|{m}"] = auc_row(f"routed {mth(m)} ({len(sub)})", sub, fail, "either", "either, routed == " + m)
    r = res["cascade_skill >= 1 or improvement < 0"]
    if r is not None:
        strength = ("does not separate" if (abs(r["auc"] - 0.5) < 0.1 or r["p"] > 0.05) else
                    ("weakly separates" if abs(r["auc"] - 0.5) < 0.25 else "separates"))
        direction = ("" if strength == "does not separate" else
                     (" -- in the expected direction (failing cells have the higher IQR)" if r["auc"] > 0.5 else
                      " -- but in the INVERSE direction (failing cells have the LOWER IQR)"))
        verdict = strength
        answer = (f"On the {len(S)}-cell grid the perturb_iqr of the routed method {verdict} failing cells "
                  f"(skill >= 1 or negative improvement; n = {r['n_f']}) from succeeding ones (n = {r['n_s']}){direction}: "
                  f"AUC = {r['auc']:.3f} with a higher IQR read as 'failure', two-sided Mann-Whitney p = {r['p']:.3f}; "
                  f"ordering: {r['order']} (median IQR {r['med_f']:.4f} vs {r['med_s']:.4f}).")
        for k, lab in (("cascade_skill >= 1", "skill >= 1 alone"), ("improvement < 0", "negative improvement alone")):
            rr = res[k]
            if rr is not None:
                answer += f" With {lab}: AUC = {rr['auc']:.3f}, p = {rr['p']:.3f}, {rr['order']} (n_fail = {rr['n_f']})."
        for label, _ in REAL_STRATA[1:]:
            rr = res.get(f"either|{label}")
            if rr is not None:
                answer += (f" Within {label} targets only: AUC = {rr['auc']:.3f}, p = {rr['p']:.3f}, {rr['order']} "
                           f"(n_fail = {rr['n_f']} of {rr['n_f'] + rr['n_s']}).")
            else:
                answer += f" Within {label} targets: one group empty, no AUC."
        for m in ("richardson_1", "rational_fit"):
            rr = res.get(f"either|{m}")
            if rr is not None:
                answer += f" Routed {m} only: AUC = {rr['auc']:.3f}, p = {rr['p']:.3f}, {rr['order']} (n_fail = {rr['n_f']} of {rr['n_f'] + rr['n_s']})."
        same = bool(((S.cascade_skill >= 1.0) == (S.improvement < 0)).all())
        answer += (" The two failure definitions select the same cells." if same else " The two failure definitions differ.")
        answer_lines.append(answer)
        fact("real data (v2) perturbation diagnostic", "does perturb_iqr of the routed method separate failing from succeeding cells?",
             answer, "results/real_data/real_data_summary_v2.csv", f"all {len(S)} cells with a finite perturb_iqr",
             "AUC = Mann-Whitney U(fail, succ) / (n_fail * n_succ); failure = cascade_skill >= 1 or improvement < 0")
    frag("f08_real_perturb_diagnostic.tex", "llrrrrrrl", rows,
         ["results/real_data/real_data_summary_v2.csv (perturb_iqr = IQR of the routed method's prediction over 5 "
          "evaluations on 2 %-perturbed windows, crc32 seed per (dataset, depth))"],
         f"all {len(S)} cells with a finite perturb_iqr, then pre-minimum (target round <= argmin round of the recorded curve) and post-minimum targets, then by routed method; "
         "AUC = P(IQR_fail > IQR_succ) from the Mann-Whitney U statistic (ties count one half); p two-sided",
         "Real-data perturbation diagnostic: does the routed method's perturb_iqr separate failing cells from succeeding ones?",
         notes=answer_lines)
    return answer_lines


# ═════════════════════════════════════════════════════════════════════════════
# F09  selectors / ensembles
# ═════════════════════════════════════════════════════════════════════════════
def f09():
    # a) Phase 3
    SC = read("phase3", "phase3_selector_comparison.csv")
    NAME3 = {"oracle": r"oracle \textit{(hindsight: lowest cell-median error)}", "phase2_cascade": "Phase-2 cascade",
             "enhanced_cascade": "enhanced cascade",
             "fixed_rational": r"fixed \meth{rational\_fit}", "fixed_richardson": r"fixed \meth{richardson\_1}",
             "fixed_single_exp": r"fixed \meth{single\_exp\_fit}", "fixed_last": r"fixed \meth{last\_value} (the trivial floor)"}
    order3 = ["oracle", "phase2_cascade", "enhanced_cascade", "fixed_rational", "fixed_richardson", "fixed_single_exp", "fixed_last"]
    order3 = [s for s in order3 if s in set(SC.selector)]
    rows = [r"Selector & valid rate & $n_{\mathrm{valid}}/n_{\mathrm{total}}$ & med.\ err & q25--q75 & p90 & win vs last \\",
            r"\midrule"]
    for g in STRATA:
        sg = SC[SC.target_g == g]
        if sg.empty:
            continue
        n_excl = int(sg.n_capped_excluded.max())
        rows.append(mid(rf"\textit{{$g = {gname(g)}$}}" + (r" (headline)" if g == HEADLINE_G else "")
                        + rf"; {int(sg.n_cells.max())} cells, {n_excl} capped cells excluded", 7))
        for s in order3:
            r = sg[sg.selector == s]
            if r.empty:
                continue
            r = r.iloc[0]
            rows.append(f"{NAME3[s]} & {f3(float(r.valid_rate))} & {int(r.n_valid)}/{int(r.n_total)} & {f4(float(r.med_error))} & "
                        f"{f4(float(r.q25_error))}--{f4(float(r.q75_error))} & {f4(float(r.p90_error))} & {f3(float(r.win_rate_vs_last))} \\\\")
        if g != STRATA[-1]:
            rows.append(r"\addlinespace[2pt]")
    for s in order3:
        r = SC[(SC.selector == s) & (SC.target_g == HEADLINE_G)]
        if len(r):
            r = r.iloc[0]
            fact("selectors (Phase 3)", f"{s} at g={gname(HEADLINE_G)}: validity rate; n_valid/n_total; median error (conditional on validity); win rate vs last",
                 f"{r.valid_rate:.4f}; {int(r.n_valid)}/{int(r.n_total)}; {r.med_error:.5f} (q25 {r.q25_error:.5f}, q75 {r.q75_error:.5f}, p90 {r.p90_error:.5f}); {r.win_rate_vs_last:.3f}",
                 "results/phase3/phase3_selector_comparison.csv", f"selector == {s}, target_g == {HEADLINE_G}",
                 "src.panels.error_panel over the chosen records of every uncapped cell x seed; no fallback for an invalid choice")
    spread = SC.groupby("target_g").valid_rate.agg(lambda v: float(v.max() - v.min()))
    fact("selectors (Phase 3)", "validity-rate spread across selectors per stratum (the Phase-3 warning fires above 0.001)",
         ", ".join(f"g={gname(g)}: {v:.4f}" for g, v in spread.items()),
         "results/phase3/phase3_selector_comparison.csv", "per target_g", "max - min of valid_rate over selectors")
    frag("f09a_selectors_phase3.tex", "lrrrrrr", rows,
         ["results/phase3/phase3_selector_comparison.csv (core regimes, capped cells excluded; src.panels.error_panel over the chosen records)"],
         "all selectors, per stratum; a selector picks one method per cell and is scored on every record of that cell (the chosen method's "
         "record; an invalid choice stays invalid, no fallback); valid rate and n_valid/n_total over all records; med. err, q25--q75 and p90 "
         "over the valid records only; win vs last = fraction of all records where the chosen record is valid and below the last-value error; "
         "the cascades' thresholds are fixed constants, so the per-regime panel (phase3_regime_results.csv) is the per-family evidence and "
         "the held-out families of Phase 5a the out-of-sample test",
         "Phase 3 selectors: conditional on validity; validity rate alongside")

    # b/c) Phase 5a ensembles
    ORDER5 = SELECTOR_ORDER5
    for fname, path, label in (("f09b_ensemble_phase5a_core.tex", "phase5a_ensemble.csv", "core"),
                               ("f09c_ensemble_phase5a_holdout.tex", "phase5a_ensemble_holdout.csv", "held-out")):
        E = read("phase5a", path)
        # per stratum: validity rate, n_valid/n_total, (mean error at the headline stratum), median error, median skill
        per = {g: (5 if g == HEADLINE_G else 4) for g in STRATA}
        head = [r"Selector & " + " & ".join(rf"\multicolumn{{{per[g]}}}{{c}}{{$g = {gname(g)}$}}" for g in STRATA) + r" \\"]
        cmid, start = "", 2
        for g in STRATA:
            cmid += rf"\cmidrule(lr){{{start}-{start + per[g] - 1}}}"
            start += per[g]
        head.append(cmid)
        head.append(r" & " + " & ".join((r"$\rhoV$ & $n_v/n$ & mean err & med.\ err & med.\ skill" if g == HEADLINE_G
                                          else r"$\rhoV$ & $n_v/n$ & med.\ err & med.\ skill") for g in STRATA) + r" \\")
        head.append(r"\midrule")
        rows = list(head)
        for i, (sel, name) in enumerate(ORDER5):
            cells = []
            for g in STRATA:
                r = E[(E.selector == sel) & (E.target_g == g)]
                if len(r):
                    r = r.iloc[0]
                    cells += [f3(float(r.valid_rate)), f"{int(r.n_valid)}/{int(r.n_total)}"]
                    if g == HEADLINE_G:
                        cells.append(f4(float(r.mean_error)))
                    cells += [f4(float(r.med_error)), f3(float(r.med_skill))]
                else:
                    cells += ["--"] * per[g]
            if sel in ("fixed_rational", EQUAL_SMALL, "constant_assumed"):
                rows.append(r"\addlinespace[2pt]")
            rows.append(f"{name} & " + " & ".join(cells) + r" \\")
        for sel in (f"oracle_{N_ACC}", "constant_oracle", ORACLE_SMALL, "fixed_rational", "fixed_richardson", "phase2_cascade", EQUAL_SMALL, "constant_assumed"):
            r = E[(E.selector == sel) & (E.target_g == HEADLINE_G)]
            if len(r):
                r = r.iloc[0]
                fact("ensembles (Phase 5a)", f"{sel} ({label}) at g={gname(HEADLINE_G)}: validity rate; n_valid/n_total; mean error; median error (conditional on validity); med. skill",
                     f"{r.valid_rate:.4f}; {int(r.n_valid)}/{int(r.n_total)}; {r.mean_error:.6f}; {r.med_error:.6f}; {r.med_skill:.4f}",
                     f"results/phase5a/{path}", f"selector == {sel}, target_g == {HEADLINE_G}",
                     "src.panels.error_panel over the selector's records (an invalid choice stays invalid); med_skill = hindsight best-of-four (strict), median over the valid records of err / best-of-four trivial")
        frag(fname, "l" + "r" * sum(per.values()), rows,
             [f"results/phase5a/{path} ({label} regimes; the Phase-5a depth grid; capped cells excluded; src.panels.error_panel over each selector's records)"],
             "all selectors; a selector's record on a cell is the record of the method it chose or the ensemble value, an invalid choice stays "
             "invalid; rho_V = validity rate over all records, n_v/n = n_valid/n_total; mean err and med. err over the valid records only "
             "(conditional on validity); skill = hindsight best-of-four (strict): err / best-of-four trivial error per record, median over the valid records",
             f"Phase 5a selectors and ensembles, {label} regimes: validity rate and n_valid/n_total next to the median error and median skill per stratum "
             f"(mean error at the headline stratum), conditional on validity; oracle_{N_ACC} vs constant_oracle vs the fixed defaults")

    # d) ablation
    AB = read("phase5a", "phase5a_ablation.csv")
    comps = list(dict.fromkeys(AB.comparison))
    rows = [r"Comparison (positive = first is better) & " + " & ".join(rf"$g = {gname(g)}$" for g in STRATA) + r" & $n_{\mathrm{both}}/n$ \\", r"\midrule"]
    for cmp in comps:
        vals = []
        for g in STRATA:
            r = AB[(AB.comparison == cmp) & (AB.target_g == g)]
            vals.append(signed(float(r.mean_improvement.iloc[0]), 4) if len(r) else "--")
        rh = AB[(AB.comparison == cmp) & (AB.target_g == HEADLINE_G)]
        n_txt = f"{int(rh.n_both_valid.iloc[0])}/{int(rh.n_total.iloc[0])}" if len(rh) else "--"
        rows.append(f"{ablation_label(cmp)} & " + " & ".join(vals) + f" & {n_txt} \\\\")
    frag("f09d_ablation_phase5a.tex", "l" + "r" * (len(STRATA) + 1), rows,
         ["results/phase5a/phase5a_ablation.csv (core regimes, capped excluded)"],
         "all comparisons, printed as '<first> vs <second>' with the selector labels of f09b (the (first, second) pairs are the "
         "comparisons list of phases/phase5a.py; the comparison names of the CSV are " + ", ".join(comps) + "); "
         "mean_improvement = mean over the records where both selectors are valid of (error of the second selector - error of "
         "the first); n_both/n = those records over all records at the headline stratum",
         "Phase 5a ablation: paired mean error differences between selectors per stratum, with the count of records where both are valid")


# ═════════════════════════════════════════════════════════════════════════════
# F10  Phase-4 diagnostics
# ═════════════════════════════════════════════════════════════════════════════
def f10():
    D = read("phase4", "phase4_diagnostic_correlations.csv")
    H = read("phase4", "phase4_diagnostic_correlations_holdout.csv")
    methods = list(dict.fromkeys(D.method))

    def get(T, m, d, col):
        r = T[(T.method == m) & (T.diagnostic == d)]
        return float(r[col].iloc[0]) if len(r) else float("nan")

    head = [r"Method & \multicolumn{3}{c}{core (pooled)} & \multicolumn{3}{c}{held-out} \\",
            r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
            r" & $r$(\diag{perturb\_IQR}) & $r$(\diag{shift\_IQR}) & $n$ & $r$(\diag{perturb\_IQR}) & $r$(\diag{shift\_IQR}) & $n$ \\",
            r"\midrule"]
    rows = list(head)
    for m in methods:
        lab = r"\textit{all methods pooled}" if m == "ALL" else mth(m)
        rows.append(f"{lab} & {f3(get(D, m, 'perturb_iqr', 'spearman_r'))} & {f3(get(D, m, 'shift_iqr', 'spearman_r'))} & "
                    f"{fint(get(D, m, 'perturb_iqr', 'n'))} & {f3(get(H, m, 'perturb_iqr', 'spearman_r'))} & "
                    f"{f3(get(H, m, 'shift_iqr', 'spearman_r'))} & {fint(get(H, m, 'perturb_iqr', 'n'))} \\\\")
        if m == "ALL":
            rows.append(r"\addlinespace[2pt]")
    for m in ("ALL", "richardson_1", "rational_fit", "log_linear"):
        fact("diagnostics (Phase 4)", f"{m}: Spearman r(perturb_iqr, error), core / held-out",
             f"{get(D, m, 'perturb_iqr', 'spearman_r'):.4f} / {get(H, m, 'perturb_iqr', 'spearman_r'):.4f}",
             "results/phase4/phase4_diagnostic_correlations.csv; results/phase4/phase4_diagnostic_correlations_holdout.csv",
             f"method == {m}, diagnostic == perturb_iqr", "spearman_r column (valid records, capped excluded)")
    frag("f10a_diagnostics_correlations.tex", "l" + "rrr" * 2, rows,
         ["results/phase4/phase4_diagnostic_correlations.csv (core regimes pooled over obs 30/60/90/120, noise, seeds, strata; capped excluded)",
          "results/phase4/phase4_diagnostic_correlations_holdout.csv (held-out regimes)"],
         "all rows; Spearman correlation between the diagnostic and the absolute error over valid records; -- = diagnostic undefined "
         "(a NaN Spearman r in the stored table: the diagnostic is constant or undefined for that method)",
         "Phase 4 diagnostics: correlation of perturb_IQR and shift_IQR with error, core pooled and held-out")

    OB = read("phase4", "phase4_obs_reliability.csv")
    ob = OB[(OB.diagnostic == "perturb_iqr") & (OB.target_g == HEADLINE_G)]
    depths = sorted(ob.obs_idx.unique())
    if not depths:
        print("  (phase4_obs_reliability.csv has no perturb_iqr rows at the headline stratum: fragment f10b skipped)")
        return _f10_rest()
    rows = [r"Method & " + " & ".join(rf"$\nobs = {int(d)}$" for d in depths) + r" \\", r"\midrule"]
    for m in list(dict.fromkeys(ob.method)):
        vals = []
        for d in depths:
            r = ob[(ob.method == m) & (ob.obs_idx == d)]
            vals.append(f3(float(r.spearman_r.iloc[0])) if len(r) else "--")
        rows.append(f"{mth(m)} & " + " & ".join(vals) + r" \\")
    r1 = ob[ob.method == "richardson_1"].sort_values("obs_idx")
    fact("diagnostics (Phase 4)", f"richardson_1: r(perturb_iqr, error) by depth at g={gname(HEADLINE_G)}",
         ", ".join(f"{int(d)}: {v:.3f}" for d, v in zip(r1.obs_idx, r1.spearman_r)),
         "results/phase4/phase4_obs_reliability.csv", f"method == richardson_1, diagnostic == perturb_iqr, target_g == {HEADLINE_G}",
         "spearman_r per obs_idx")
    frag("f10b_diagnostics_depth.tex", "l" + "r" * len(depths), rows,
         ["results/phase4/phase4_obs_reliability.csv (core + held-out, per depth)"],
         f"diagnostic == perturb_iqr, target_g == {HEADLINE_G}; Spearman r between perturb_IQR and error at each observation depth",
         "Phase 4: perturb_IQR reliability by observation depth at the headline stratum")

    _f10_rest()


def _f10_rest():
    E4 = read("phase4", "phase4_ensemble.csv")
    rows = [r"Selector & $\rhoV$ & $n_v/n$ & mean err & med.\ err & med.\ skill & win vs last \\", r"\midrule"]
    for _, r in E4.iterrows():
        lab = phase4_selector_label(r.selector)
        rows.append(f"{lab} & {f3(float(r.valid_rate))} & {int(r.n_valid)}/{int(r.n_total)} & "
                    f"{f4(float(r.mean_error))} & {f4(float(r.med_error))} & {f3(float(r.med_skill))} & {f3(float(r.win_rate_vs_last))} \\\\")
        fact("diagnostics (Phase 4)", f"{r.selector} at g={gname(float(r.target_g))}: validity rate; n_valid/n_total; mean error; median error (conditional on validity); med. skill; win rate vs last",
             f"{r.valid_rate:.4f}; {int(r.n_valid)}/{int(r.n_total)}; {r.mean_error:.6f}; {r.med_error:.6f}; {r.med_skill:.4f}; {r.win_rate_vs_last:.4f}",
             "results/phase4/phase4_ensemble.csv", f"selector == {r.selector}, target_g == {gname(float(r.target_g))}",
             "src.panels.error_panel over the selector's records (an invalid choice stays invalid); med_skill = hindsight best-of-four (strict), median over the valid records")
    frag("f10c_diagnostics_ensemble.tex", "lrrrrrr", rows,
         ["results/phase4/phase4_ensemble.csv (core regimes, headline stratum, capped excluded; src.panels.error_panel over each selector's records)"],
         f"all rows; a selector's record on a window is the record of the method it chose or the combined value, an invalid choice stays invalid; "
         f"rho_V = validity rate over all windows, n_v/n = n_valid/n_total; mean err and med. err over the valid records only (conditional on "
         f"validity); selectors printed with the labels of f09b: oracle = oracle over the {len(PHASE2_POOL)}-method pool; phase2_proxy = the lower "
         f"error of richardson_1 and rational_fit (a hindsight proxy of the Phase-2 cascade); diag_ensemble = "
         f"perturb_IQR-weighted ensemble of the {len(PHASE2_POOL)}-method pool (eight accelerators plus the last observed value), weights "
         f"1/(perturb_IQR + eps) over the paired perturbations",
         "Phase 4 selectors with the trivial references at the headline stratum: validity rate and n_valid/n_total next to the error and skill, "
         "conditional on validity; validity rate alongside")

    CF = read("phase4", "phase4_cascade_filter.csv")
    rows = [r"Screen & $\rhoV$ & $n_v/n$ & med.\ err & mean err & win vs last \\", r"\midrule"]
    for _, r in CF.iterrows():
        rows.append(f"{screen_label(r['filter'])} & {f3(float(r.valid_rate))} & {int(r.n_valid)}/{int(r.n_total)} & {f4(float(r.med_error))} & "
                    f"{f4(float(r.mean_error))} & {f3(float(r.win_rate_vs_last))} \\\\")
    base = CF[CF['filter'] == 'no_filter'].iloc[0]
    best = CF.loc[CF.mean_error.idxmin()]
    fact("diagnostics (Phase 4)", "perturb_IQR screen on the cascade: lowest mean error (conditional on validity) vs no filter, validity rate alongside",
         f"{best['filter']}: mean err {best.mean_error:.6f} (valid {best.valid_rate:.4f}, {int(best.n_valid)}/{int(best.n_total)}) vs no_filter "
         f"{base.mean_error:.6f} (valid {base.valid_rate:.4f}, {int(base.n_valid)}/{int(base.n_total)}); {100 * (best.mean_error - base.mean_error) / base.mean_error:+.2f}%",
         "results/phase4/phase4_cascade_filter.csv", "all rows", "argmin mean_error; src.panels.error_panel over the chosen records, no evaluation fallback")
    fact("diagnostics (Phase 4)", "perturb_IQR screen on the cascade: validity rate and win rate vs last per threshold",
         "; ".join(f"{r['filter']}: valid {r.valid_rate:.4f}, win vs last {r.win_rate_vs_last:.4f}" for _, r in CF.iterrows()),
         "results/phase4/phase4_cascade_filter.csv", "all rows", "valid_rate, win_rate_vs_last")
    frag("f10d_diagnostics_filter.tex", "lrrrrr", rows,
         ["results/phase4/phase4_cascade_filter.csv (core regimes, headline stratum; src.panels.error_panel over the chosen records)"],
         "all rows (no_filter printed as 'no screen', perturb_iqr>t as the screen at that threshold); a window whose routed prediction "
         "has perturb_IQR above the threshold is routed to the last observed value (last_value), "
         "which is part of the screened method; the chosen record is then taken as it is (an invalid choice stays invalid, no evaluation "
         "fallback); rho_V = validity rate, n_v/n = n_valid/n_total; med. err and mean err over the valid records (conditional on validity); "
         "win vs last = fraction of all windows where the chosen record is valid and below the last-value error",
         "Phase 4: the perturb_IQR screen applied to the Phase-2 cascade, conditional on validity; validity rate alongside")


# ═════════════════════════════════════════════════════════════════════════════
# F11  capped block
# ═════════════════════════════════════════════════════════════════════════════
def f11():
    cells = (CAPPED1.groupby(["regime", "is_holdout", "target_g"], as_index=False)
             .agg(n_f=("n_f", "first"), achieved_g=("achieved_g", "first"), n_methods=("method", "nunique"), n_cells=("n", "first")))
    hz = HORIZONS.set_index(["regime", "target_g"])
    rows = [r"Curve family & set & $g$ & $\nf$ & achieved $g$ & gap$(\nobs)$ & gap$(\nf)$ & note \\", r"\midrule"]
    inv = []
    for _, r in cells.sort_values(["is_holdout", "regime", "target_g"], ascending=[True, True, False]).iterrows():
        h = hz.loc[(r.regime, r.target_g)] if (r.regime, r.target_g) in hz.index else None
        seed_dep = h is not None and not pd.isna(h.seed)
        note = ("seed-dependent shape: capped on some seeds (median $\\nf$ shown)" if seed_dep and int(r.n_f) < C.HORIZON_N_CAP
                else ("seed-dependent shape" if seed_dep else ""))
        gap_o = f'{h.gap_obs:.4f}' if h is not None else '--'
        gap_f = f'{h.gap_f:.4f}' if h is not None else '--'
        rows.append(f"{tt(r.regime)} & {'held-out' if r.is_holdout else 'core'} & {r.target_g:g} & {int(r.n_f)} & "
                    f"{r.achieved_g:.4f} & {gap_o} & {gap_f} & {note} \\\\")
        inv.append(f"{r.regime} g={r.target_g:g} (n_f {int(r.n_f)}, achieved {r.achieved_g:.3f})")
    rows.append(r"\midrule")
    counts = []
    for ph, path, depths in (("Phase 2", "phase2/phase2_capped.csv", 13), ("Phase 4", "phase4/phase4_capped.csv", 4), ("Phase 5a", "phase5a/phase5a_capped.csv", 4)):
        cp = read(*path.split("/"))
        d = cp.drop_duplicates(["regime", "obs_idx", "target_g"])
        counts.append((ph, len(d), depths, sorted(d.regime.unique())))
    c3 = read("phase3", "phase3_capped_cells.csv").drop_duplicates(["regime", "obs_idx", "noise", "target_g"])
    rows.append(r"\multicolumn{8}{l}{Capped cells in the depth grids (curve family $\times$ depth $\times$ $g$): " +
                "; ".join(f"{ph}: {n} over {dep} depths" for ph, n, dep, _ in counts) +
                f"; Phase 3: {len(c3)} (curve family, depth, noise, $g$) cells" + r"} \\")
    rows.append(r"\multicolumn{8}{l}{Curve families ever capped in the depth grids: " +
                ", ".join(tt(x) for x in sorted(set().union(*[set(c[3]) for c in counts]))) + r"} \\")
    fact("capped cells", "Phase 1 (obs 90) capped (regime, g) cells", f"{len(cells)}: " + "; ".join(inv),
         "results/phase1/phase1_capped.csv; results/phase1/phase1_horizons.csv", "all rows",
         f"distinct (regime, target_g); a cell is capped when the horizon search hit HORIZON_N_CAP = {C.HORIZON_N_CAP} on any seed")
    for ph, n, dep, regs in counts:
        fact("capped cells", f"{ph} capped (regime, depth, g) cells", f"{n} over {dep} depths; regimes {', '.join(regs)}",
             f"results/{ph.lower().replace(' ', '')}/{ph.lower().replace(' ', '')}_capped.csv", "all rows", "distinct (regime, obs_idx, target_g)")
    fact("capped cells", "capped cells never enter a pooled statistic", "EXCLUDE_CAPPED_FROM_POOLED = True", "src/config.py", "-", "src.pipeline.exclude_capped")
    frag("f11_capped_block.tex", "lllrrrrl", rows,
         ["results/phase1/phase1_capped.csv (the capped block of the main run: every method x capped cell)",
          "results/phase1/phase1_horizons.csv (n_f search per regime and stratum at n_obs = 90; seed 0 for seed-dependent shapes)",
          "results/phase2/phase2_capped.csv; results/phase4/phase4_capped.csv; results/phase5a/phase5a_capped.csv; results/phase3/phase3_capped_cells.csv"],
         f"capped == 1; n_f = HORIZON_N_CAP = {C.HORIZON_N_CAP} unless the shape is seed-dependent (median over seeds shown); "
         f"achieved g = gap(n_f) / gap(n_obs) actually reached; these cells are excluded from every pooled table and figure",
         "Capped block: every capped (regime x stratum) cell of the main run with the achieved gap fraction, and the capped-cell "
         "counts of the depth grids")


# ═════════════════════════════════════════════════════════════════════════════
# F12  validity by depth (every method with any depth below the floor)
# ═════════════════════════════════════════════════════════════════════════════
def f12():
    path = os.path.join(RES, "phase5a", "phase5a_validity_by_depth.csv")
    if not os.path.exists(path):
        print("  (phase5a_validity_by_depth.csv absent: fragment f12 skipped; produced by a run at or after Prompt 5A)")
        return
    V = pd.read_csv(path)
    V = V[V.is_oracle == 0]
    depths = sorted(V.obs_idx.unique())
    pooled = (V.groupby(["method", "obs_idx"]).apply(lambda g: (g.valid_rate * g.n).sum() / g.n.sum())
               .unstack("obs_idx"))
    worst = V.groupby("method").valid_rate.min()
    flagged = pooled[(pooled < C.RANK_MIN_VALID).any(axis=1)].copy()
    flagged["min"] = pooled.min(axis=1)
    flagged = flagged.sort_values("min")
    rows = [r"Method & " + " & ".join(rf"$\nobs = {int(d)}$" for d in depths) + r" & min (depth, $\sigma$) \\", r"\midrule"]
    for m, r in flagged.iterrows():
        cells = [(r"\textbf{" + f3(float(r[d])) + "}") if float(r[d]) < C.RANK_MIN_VALID else f3(float(r[d])) for d in depths]
        rows.append(f"{dag(m)}{mth(m)} & " + " & ".join(cells) + f" & {f3(float(worst[m]))} \\\\")
    rows.append(r"\midrule")
    p1_depth = int(C.PHASE1["full"]["obs_idx"])      # the excluded set is derived at the Phase-1 depth only
    rows.append(rf"\multicolumn{{{len(depths) + 2}}}{{l}}{{{len(flagged)} of {int((V.is_trivial == 0).sum() and V[V.is_trivial == 0].method.nunique())} accelerators fall below "
                rf"$\rhoV = {C.RANK_MIN_VALID:g}$ at some depth; the excluded set (validity below the floor at depth {p1_depth}) contains {int(flagged.index.isin(DANGEROUS).sum())} of them}} \\")
    for m, r in flagged.iterrows():
        if m not in DANGEROUS:
            fact("validity by depth", f"{m}: valid rate by depth (pooled over noise)", ", ".join(f"obs {int(d)}: {float(r[d]):.3f}" for d in depths),
                 "results/phase5a/phase5a_validity_by_depth.csv", f"method == {m}", "n-weighted mean of valid_rate over noise per obs_idx")
    fact("validity by depth", "accelerators below the 0.9 floor at some depth but not in the dangerous artifact",
         ", ".join(m for m in flagged.index if m not in DANGEROUS) or "none",
         "results/phase5a/phase5a_validity_by_depth.csv; results/phase1/dangerous_methods.json", "min over depth of the noise-pooled valid rate < 0.9",
         "the artifact is derived at obs 90 only")
    frag("f12_validity_by_depth.tex", "l" + "r" * (len(depths) + 1), rows,
         ["results/phase5a/phase5a_validity_by_depth.csv (method x obs_idx x noise valid rate over every regime, seed and stratum)"],
         f"methods whose noise-pooled valid rate is below {C.RANK_MIN_VALID} at any depth (bold); last column = the minimum over (depth, noise) cells; dagger = the excluded set",
         "Validity by observation depth: every method that falls below the rank floor at some depth")


# ═════════════════════════════════════════════════════════════════════════════
# F14  real data, fixed methods and the cascade (Table tab:realmethods, §8)
# ═════════════════════════════════════════════════════════════════════════════
def f14():
    """Per dataset and pooled, split by pre-/post-minimum target and total: for
    rational_fit, richardson_1 and the cascade row, n_cells, n_win_vs_last,
    n_fail (skill >= 1 or non-finite error), valid rows, median error and
    median skill over the valid rows (conditional on validity)."""
    L = read("real_data", "real_data_results_v2.csv")
    if "post_min_target" not in L.columns:
        print("  (real_data_results_v2.csv has no post_min_target column: fragment f14 skipped)")
        return
    SEL = [("rational_fit", L.method == "rational_fit"), ("richardson_1", L.method == "richardson_1"),
           ("cascade", L.is_cascade == 1)]
    SPLITS = [("pre-minimum", 0), ("post-minimum", 1), ("total", None)]
    datasets = [d for d in REAL_DATASETS if d in set(L.dataset)] + sorted(set(L.dataset) - set(REAL_DATASETS))

    def block(sub):
        n = int(len(sub))
        err = sub.error.to_numpy(dtype=float)
        valid = np.isfinite(err)
        fail = (sub.skill.to_numpy(dtype=float) >= 1.0) | ~valid
        ok = sub[valid]
        return dict(n=n, win=int((sub.win_vs_last == 1).sum()), fail=int(fail.sum()), valid=int(valid.sum()),
                    med_err=float(ok.error.median()) if len(ok) else float("nan"),
                    med_skill=float(ok.skill.median()) if len(ok) else float("nan"))

    rows = [r"Dataset & split & method & $n$ & win vs last & fail & valid & med.\ err & med.\ skill \\", r"\midrule"]
    sec = "real data fixed methods (§8)"
    for d in ["pooled"] + datasets:
        sd = L if d == "pooled" else L[L.dataset == d]
        first_split = True
        for label, flag in SPLITS:
            ss = sd if flag is None else sd[sd.post_min_target == flag]
            for i, (mname, mask) in enumerate(SEL):
                b = block(ss[mask.loc[ss.index]])
                lab_d = (r"\textbf{pooled}" if d == "pooled" else tt(d)) if (first_split and i == 0) else ""
                lab_s = label if i == 0 else ""
                rows.append(f"{lab_d} & {lab_s} & {mth(mname) if mname != 'cascade' else 'cascade'} & {b['n']} & "
                            f"{b['win']}/{b['n']} & {b['fail']}/{b['n']} & {b['valid']}/{b['n']} & {f4(b['med_err'])} & {f3(b['med_skill'])} \\\\")
                fact(sec, f"{d}, {label} targets, {mname}: win vs last / fail / valid (of n); median error; median skill",
                     f"{b['win']}/{b['n']} / {b['fail']}/{b['n']} / {b['valid']}/{b['n']}; {b['med_err']:.5f}; {b['med_skill']:.3f}",
                     "results/real_data/real_data_results_v2.csv",
                     ("all datasets" if d == "pooled" else f"dataset == {d}") + ("" if flag is None else f", post_min_target == {flag}")
                     + (f", is_cascade == 1" if mname == "cascade" else f", method == {mname}"),
                     "win = count(win_vs_last == 1); fail = count(skill >= 1 or error not finite); medians over rows with finite error")
            first_split = False
            if flag is not None:
                rows.append(r"\addlinespace[1pt]")
        rows.append(r"\midrule" if d == "pooled" else r"\addlinespace[3pt]")
    frag("f14_real_fixed_methods.tex", "lllrrrrrr", rows,
         ["results/real_data/real_data_results_v2.csv (recorded curves re-evaluated on the (depth x target) grid; L_hat mode zero)"],
         "method in {rational_fit, richardson_1} and the cascade row (is_cascade == 1); split = pre-minimum (target_round <= argmin_round) / "
         "post-minimum / total; win vs last = rows with win_vs_last == 1 (error below the last observed value's); fail = skill >= 1 or "
         "error not finite (no better than the hindsight best trivial; the definition of real_data_strata_v2.csv); valid = rows with a "
         "finite error; med. err and med. skill over the valid rows only (conditional on validity)",
         "Real data (Table tab:realmethods): fixed rational_fit and richardson_1 next to the cascade, per dataset and pooled, pre- and "
         "post-minimum targets: n, win vs last, fail, valid, median error and skill")


# ═════════════════════════════════════════════════════════════════════════════
# F15  family bootstrap (Table tab:boot, §5.4)
# ═════════════════════════════════════════════════════════════════════════════
def f15():
    """Resampling unit = core regime.  Cells = uncapped (regime, noise) cells at
    the headline stratum; each draw samples the core regimes with replacement
    and takes their cells.  Point estimates on the unresampled cells; 2.5th /
    97.5th percentiles over BOOT_N draws from RandomState(BOOT_SEED)."""
    A = AGG[(AGG.is_holdout == 0) & (AGG.capped == 0) & (AGG.target_g == HEADLINE_G)]
    cells = A.pivot_table(index=["regime", "noise"], columns="method", values="med_error")
    win = A[A.method == "rational_fit"].set_index(["regime", "noise"]).win_rate_vs_last.reindex(cells.index)
    regimes = sorted(cells.index.get_level_values("regime").unique())
    codes = np.array([regimes.index(r) for r in cells.index.get_level_values("regime")])
    by_regime = [np.flatnonzero(codes == k) for k in range(len(regimes))]
    rat = cells["rational_fit"].to_numpy(dtype=float)
    winv = win.to_numpy(dtype=float)
    # The comparator list: the typed list in its order, then every accelerator ranked above
    # rational_fit on the core set at the headline stratum that is not yet in it, then the
    # rank-eligible accelerator with the highest core win rate against the last value.
    comp_list = list(BOOT_COMPARATORS)
    rk = RANK_CORE[HEADLINE_G]
    above = [m for m in sorted(rk, key=rk.get) if "rational_fit" in rk and rk[m] < rk["rational_fit"]]
    sg = G_CORE[(G_CORE.target_g == HEADLINE_G) & (G_CORE.is_trivial == 0) & (G_CORE.rank_eligible == 1)]
    top_win = list(sg.sort_values(["win_rate_vs_last", "method"], ascending=[False, True]).method[:1])
    for m in above + top_win:
        if m not in comp_list:
            comp_list.append(m)
    assert all(m in comp_list for m in above), (above, comp_list)
    comps = {m: cells[m].to_numpy(dtype=float) for m in comp_list if m in cells.columns}
    missing = [m for m in comp_list if m not in cells.columns]
    if missing:
        print(f"  (bootstrap comparators absent from phase1_aggregated.csv: {missing})")
    winc = {m: A[A.method == m].set_index(["regime", "noise"]).win_rate_vs_last.reindex(cells.index).to_numpy(dtype=float)
            for m in comps}

    def stats(idx):
        out = {"win_rate_vs_last": float(np.nanmean(winv[idx]))}
        for m, arr in comps.items():
            both = np.isfinite(rat[idx]) & np.isfinite(arr[idx])
            r, c = rat[idx][both], arr[idx][both]
            out[f"lower_error_frac:{m}"] = float((r < c).mean()) if both.any() else float("nan")
            out[f"delta_median:{m}"] = float(np.median(r) - np.median(c)) if both.any() else float("nan")
            out[f"n_cells_excluded:{m}"] = int((~both).sum())     # every cell dropped: either median missing
            out[f"n_cells:{m}"] = int(both.sum())
        for m in comps:                                           # the comparators' own win rates, same draws
            out[f"win_rate:{m}"] = float(np.nanmean(winc[m][idx]))
        return out

    all_idx = np.arange(len(cells))
    point = stats(all_idx)
    rng = np.random.RandomState(BOOT_SEED)
    draws = rng.randint(0, len(regimes), size=(BOOT_N, len(regimes)))
    boot = {k: np.empty(BOOT_N) for k in point}
    for b, draw in enumerate(draws):
        idx = np.concatenate([by_regime[k] for k in draw])
        st = stats(idx)
        for k in point:
            boot[k][b] = st[k]

    def ci(k):
        v = boot[k][np.isfinite(boot[k])]
        return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if v.size else (float("nan"), float("nan"))

    sec = "family bootstrap (§5.4)"
    core = AGG[(AGG.is_holdout == 0) & (AGG.target_g == HEADLINE_G)]
    all_capped = core.groupby("regime").capped.min() == 1
    fully_capped = sorted(all_capped[all_capped].index)
    assert len(regimes) == core.regime.nunique() - len(fully_capped), (regimes, fully_capped)
    fact(sec, f"resampling population at g={gname(HEADLINE_G)}: core families with at least one uncapped cell; core families in total; fully capped families",
         f"{len(regimes)}; {core.regime.nunique()}; {', '.join(fully_capped) or 'none'}",
         "results/phase1/phase1_aggregated.csv", f"is_holdout == 0, target_g == {HEADLINE_G}",
         "regimes with capped == 0 in at least one cell; all regimes; regimes with capped == 1 in every cell (they contribute no cell and are never drawn)")
    rows = [r"Statistic & comparator & point & 2.5\% & 97.5\% & cells & excl. \\", r"\midrule"]
    lo, hi = ci("win_rate_vs_last")
    rows.append(rf"\meth{{rational\_fit}} win rate vs last & -- & {f3(point['win_rate_vs_last'])} & {f3(lo)} & {f3(hi)} & {len(cells)} & -- \\")
    fact(sec, f"rational_fit win rate vs last_value at g={gname(HEADLINE_G)} core: point [2.5%, 97.5%] over {BOOT_N} regime resamples",
         f"{point['win_rate_vs_last']:.4f} [{lo:.4f}, {hi:.4f}] ({len(cells)} cells, {len(regimes)} regimes)",
         "results/phase1/phase1_aggregated.csv", f"is_holdout == 0, capped == 0, target_g == {HEADLINE_G}, method == rational_fit",
         f"mean over drawn cells of win_rate_vs_last; regimes resampled with replacement, RandomState({BOOT_SEED}), {BOOT_N} draws")
    rows.append(r"\addlinespace[2pt]")
    for m in comps:
        lo1, hi1 = ci(f"lower_error_frac:{m}")
        lo2, hi2 = ci(f"delta_median:{m}")
        n_c, n_x = point[f"n_cells:{m}"], point[f"n_cells_excluded:{m}"]
        rows.append(rf"lower-error fraction & {mth(m)} & {f3(point[f'lower_error_frac:{m}'])} & {f3(lo1)} & {f3(hi1)} & {n_c} & {n_x} \\")
        rows.append(rf"$\Delta$ median error & {mth(m)} & {signed(point[f'delta_median:{m}'], 5)} & {signed(lo2, 5)} & {signed(hi2, 5)} & {n_c} & {n_x} \\")
        fact(sec, f"rational_fit vs {m} at g={gname(HEADLINE_G)} core: lower-error fraction of cells, point [2.5%, 97.5%]",
             f"{point[f'lower_error_frac:{m}']:.4f} [{lo1:.4f}, {hi1:.4f}] ({n_c} cells with both medians finite; {n_x} excluded)",
             "results/phase1/phase1_aggregated.csv", f"is_holdout == 0, capped == 0, target_g == {HEADLINE_G}, methods rational_fit and {m}",
             "fraction of drawn cells (both medians finite) with med_error(rational_fit) < med_error(M)")
        fact(sec, f"rational_fit vs {m} at g={gname(HEADLINE_G)} core: delta of median cell errors (positive = rational worse), point [2.5%, 97.5%]",
             f"{point[f'delta_median:{m}']:+.6f} [{lo2:+.6f}, {hi2:+.6f}]",
             "results/phase1/phase1_aggregated.csv", f"is_holdout == 0, capped == 0, target_g == {HEADLINE_G}, methods rational_fit and {m}",
             "median over drawn cells of med_error(rational_fit) - median over drawn cells of med_error(M), cells with both finite")
    # the win rate against the last value of every comparator, from the same draws
    rows.append(r"\addlinespace[2pt]")
    for m in comps:
        lo3, hi3 = ci(f"win_rate:{m}")
        rows.append(rf"win rate vs last & {mth(m)} & {f3(point[f'win_rate:{m}'])} & {f3(lo3)} & {f3(hi3)} & {len(cells)} & -- \\")
        fact(sec, f"{m} win rate vs last_value at g={gname(HEADLINE_G)} core: point [2.5%, 97.5%] over {BOOT_N} regime resamples (same draws)",
             f"{point[f'win_rate:{m}']:.4f} [{lo3:.4f}, {hi3:.4f}] ({len(cells)} cells, {len(regimes)} regimes)",
             "results/phase1/phase1_aggregated.csv", f"is_holdout == 0, capped == 0, target_g == {HEADLINE_G}, method == {m}",
             f"mean over drawn cells of win_rate_vs_last; the draws of the rational_fit rows (RandomState({BOOT_SEED}), {BOOT_N})")
    fact(sec, f"comparators at g={gname(HEADLINE_G)}", ", ".join(comps),
         "results/phase1/phase1_global.csv", f"target_g == {HEADLINE_G}, is_trivial == 0, rank_eligible == 1",
         f"the typed list {BOOT_COMPARATORS} in its order, then every accelerator ranked above rational_fit on the core set "
         f"({', '.join(above) or 'none'}), then the rank-eligible accelerator with the highest core win rate vs last_value ({', '.join(top_win) or 'none'})")
    frag("f15_bootstrap.tex", "llrrrrr", rows,
         ["results/phase1/phase1_aggregated.csv (core regimes, uncapped cells at the headline stratum)"],
         f"is_holdout == 0, capped == 0, target_g == {gname(HEADLINE_G)}; resampling unit = core regime ({len(regimes)} regimes, "
         f"{len(cells)} cells), {BOOT_N} draws with replacement from RandomState({BOOT_SEED}); point = unresampled cells; interval = "
         "2.5th / 97.5th percentile of the draws; lower-error fraction and delta median over the cells where both methods have a finite "
         "cell-median error (excl. = cells dropped because either method has none; cells + excl. = the cell total); positive delta = rational_fit worse; "
         "comparators = the typed list, then every accelerator ranked above rational_fit on the core families at this stratum, then the "
         "rank-eligible accelerator with the highest core win rate vs the last value; the last block = every comparator's own win rate vs the "
         "last value from the same draws",
         "Family bootstrap over core regimes (Table tab:boot): rational_fit's win rate vs the last value and its cell-median error against "
         "each comparator, point and 2.5 / 97.5 percentiles")


# ═════════════════════════════════════════════════════════════════════════════
# F16  generalisation summary (Table tab:gen)
# ═════════════════════════════════════════════════════════════════════════════
def _rank_or_unranked(G, RK, g, m):
    if m in RK[g]:
        return str(RK[g][m])
    r = row_of(G, g, m)
    return "unranked (valid rate " + (f3(float(r.valid_rate)) if r is not None else "--") + ")"


def f16():
    rows = [r"$g$ & eligible core & eligible held-out & eligible both & Spearman $\rho$ & "
            r"\meth{rational\_fit} core / held-out & \meth{log\_linear} core / held-out \\", r"\midrule"]
    sec = "generalisation"
    rhos = {g: (rho, n) for g, rho, n in generalisation_rhos()}
    for g in STRATA:
        n_core, n_hold = len(RANK_CORE[g]), len(RANK_HOLD[g])
        rho, n_both = rhos[g]
        rf_c, rf_h = _rank_or_unranked(G_CORE, RANK_CORE, g, "rational_fit"), _rank_or_unranked(G_HOLD, RANK_HOLD, g, "rational_fit")
        ll_c, ll_h = _rank_or_unranked(G_CORE, RANK_CORE, g, "log_linear"), _rank_or_unranked(G_HOLD, RANK_HOLD, g, "log_linear")
        rows.append(f"{gname(g)} & {n_core}/{N_ACC} & {n_hold}/{N_ACC} & {n_both} & {f3(float(rho))} & {rf_c} / {rf_h} & {ll_c} / {ll_h} \\\\")
        fact(sec, f"g={gname(g)}: accelerators eligible (valid_rate >= {C.RANK_MIN_VALID}) on core / held-out / both",
             f"{n_core} / {n_hold} / {n_both} of {N_ACC}", "; ".join(SRC_G), f"target_g == {g}, is_trivial == 0", "count(rank_eligible == 1)")
        fact(sec, f"g={gname(g)}: rational_fit and log_linear ranks, core / held-out",
             f"rational_fit {rf_c} / {rf_h}; log_linear {ll_c} / {ll_h}", "; ".join(SRC_G), f"target_g == {g}",
             "rank by med_error among the rank-eligible accelerators; below the floor: unranked with the valid rate")
    frag("f16_generalisation_summary.tex", "lrrrrll", rows, SRC_G,
         f"per stratum; eligible = accelerators with valid_rate >= {C.RANK_MIN_VALID} (rank_eligible == 1) in that regime set; rho = Spearman "
         "correlation of the core and held-out med_error ranks over the accelerators eligible on both (the generalisation FACTS rows); "
         "ranks are among the eligible accelerators of that regime set",
         "Generalisation summary (Table tab:gen): eligible accelerators per regime set, rank correlation core vs held-out, and the two headline fits")


# ═════════════════════════════════════════════════════════════════════════════
# F18 / F19  order ladders (Phase 0b)
# ═════════════════════════════════════════════════════════════════════════════
LADDERS_CLASSICAL = ["shanks", "wynn_eps", "wynn_rho", "levin_t", "levin_u", "levin_v", "brezinski_theta", "anderson", "neville", "pade"]
LADDERS_FITS = ["richardson_free", "richardson_fixed", "parametric"]


def _ladder_panels():
    path = os.path.join(RES, "phase0b", "order_ladders_panels.csv")
    if not os.path.exists(path):
        print("  (results/phase0b/order_ladders_panels.csv absent: fragments f18 / f19 skipped; produced by scripts/order_ladders.py)")
        return None
    P = pd.read_csv(path)
    return P[P.noise.astype(str) == "pooled"]


def _ladder_fragment(name, families, description):
    P = _ladder_panels()
    if P is None:
        return
    P = P[P.family.isin(families)]
    core, hold = P[P.regime_set == "core"], P[P.regime_set == "holdout"]
    n_core = int(core.n_cells.max()) if len(core) else 0
    n_hold = int(hold.n_cells.max()) if len(hold) else 0
    rows = [r"Family & order & \multicolumn{4}{c}{core} & \multicolumn{3}{c}{held-out} \\",
            r"\cmidrule(lr){3-6}\cmidrule(lr){7-9}",
            r" & & $\rhoV$ & med.\ err & q25--q75 & win vs last & $\rhoV$ & med.\ err & win vs last \\", r"\midrule"]
    sec = "order ladders (Phase 0b)"
    for fam in families:
        fc = core[core.family == fam]
        if fc.empty:
            continue
        best = fc.loc[fc.med_error.idxmin()] if fc.med_error.notna().any() else None
        for i, (_, r) in enumerate(fc.iterrows()):
            h = hold[(hold.family == fam) & (hold.variant == r.variant)]
            h = h.iloc[0] if len(h) else None
            marker = r" \textbullet" if int(r.is_roster) else ""
            lab = order_label(r.order_label) + marker
            hc = ([f3(float(h.valid_rate)), f4(float(h.med_error)), f3(float(h.win_rate_vs_last))] if h is not None else ["--"] * 3)
            rows.append(f"{fam_label(fam) if i == 0 else ''} & {lab} & {f3(float(r.valid_rate))} & {f4(float(r.med_error))} & "
                        f"{f4(float(r.q25_error))}--{f4(float(r.q75_error))} & {f3(float(r.win_rate_vs_last))} & " + " & ".join(hc) + r" \\")
        rows.append(r"\addlinespace[2pt]")
        if best is not None:
            fact(sec, f"{fam}: order with the lowest core median error (noise pooled, uncapped cells, conditional on validity)",
                 f"{best.order_label}{' (roster: ' + best.roster_name + ')' if int(best.is_roster) else ' (ladder only)'}: med. err {best.med_error:.5f}, valid rate {best.valid_rate:.3f}",
                 "results/phase0b/order_ladders_panels.csv", f"family == {fam}, regime_set == core, noise == pooled", "argmin med_error")
        ros = fc[fc.is_roster == 1]
        fact(sec, f"{fam}: the roster orders' core median error and valid rate",
             "; ".join(f"{r.roster_name} ({r.order_label}): {r.med_error:.5f}, valid {r.valid_rate:.3f}" for _, r in ros.iterrows()) or "none",
             "results/phase0b/order_ladders_panels.csv", f"family == {fam}, regime_set == core, noise == pooled, is_roster == 1", "med_error, valid_rate")
        if fam == "neville":
            fact(sec, "neville: valid rate and core median error by degree",
                 "; ".join(f"{r.order_label}: valid {r.valid_rate:.3f}, med. err {r.med_error:.5f}" for _, r in fc.iterrows()),
                 "results/phase0b/order_ladders_panels.csv", "family == neville, regime_set == core, noise == pooled", "valid_rate, med_error per degree")
    frag(name, "ll" + "rrrr" + "rrr", rows,
         ["results/phase0b/order_ladders_panels.csv (Phase 0b: every order of every family with an order parameter, evaluated on Phase 1's "
          "grid at the headline stratum; noise pooled; uncapped cells; the roster orders reproduce phase1_records.csv exactly, "
          "results/phase0b/order_ladders_agreement.txt)"],
         f"regime_set core ({n_core} cells) | held-out ({n_hold} cells), noise == pooled; bullet = roster method; rho_V = valid rate over all "
         "records; med. err, q25--q75 over the valid records (conditional on validity); win vs last = fraction of all records where the "
         "variant is valid and below the last-value error",
         description)


def f18():
    _ladder_fragment("f18_ladders_classical.tex", LADDERS_CLASSICAL,
                     "Order ladders of the classical families (Phase 0b): every order next to the roster orders, core and held-out")


def f19():
    _ladder_fragment("f19_ladders_fits.tex", LADDERS_FITS,
                     "Order ladders of the fits (Phase 0b): Richardson terms, fixed exponents and the parametric models, core and held-out")


# ═════════════════════════════════════════════════════════════════════════════
# F20  roster (Table tab:roster)
# ═════════════════════════════════════════════════════════════════════════════
def f20():
    from src.accelerators import METHOD_NAMES
    from src.evaluation import FAMILY, USES_FUTURE_X
    from src.trivial import TRIVIAL_METHOD_NAMES
    fams = list(dict.fromkeys(FAMILY[m] for m in METHOD_NAMES if m not in TRIVIAL_METHOD_NAMES))
    rows = [r"Family & type & $K$ & members \\", r"\midrule"]
    sec = "roster"
    total = 0
    for fam in fams:
        members = [m for m in METHOD_NAMES if FAMILY[m] == fam and m not in TRIVIAL_METHOD_NAMES]
        types = {("TE" if USES_FUTURE_X[m] else "LE") for m in members}
        t = "mixed" if len(types) > 1 else (r"\TE" if "TE" in types else r"\LE")
        total += len(members)
        rows.append(f"{fam_label(fam)} & {t} & {len(members)} & " + ", ".join(mth(m) for m in members) + r" \\")
        fact(sec, f"{fam}: K and type", f"K = {len(members)}, type {t.replace(chr(92), '')}; members {', '.join(members)}",
             "src/accelerators.py (METHODS), src/evaluation.py (FAMILY, USES_FUTURE_X)", "-", "registry order")
    rows.append(r"\midrule")
    rows.append(rf"\textbf{{accelerators}} & & {total} & {len(fams)} families \\")
    rows.append(r"\addlinespace[2pt]")
    triv = list(TRIVIAL_METHOD_NAMES)
    rows.append(f"trivial comparators (not counted) & & {len(triv)} & " + ", ".join(mth(m) for m in triv) + r" \\")
    assert total == N_ACC
    fact(sec, "totals", f"{total} accelerators in {len(fams)} families; {len(triv)} trivial comparators ({len(METHOD_NAMES)} registered methods)",
         "src/accelerators.py", "-", "len(ACCEL_METHODS), len(TRIVIAL_METHOD_NAMES), len(METHOD_NAMES)")
    frag("f20_roster.tex", "llrp{0.62\\linewidth}", rows,
         ["src/accelerators.py (METHODS registry)", "src/evaluation.py (FAMILY, USES_FUTURE_X)"],
         "families in order of first appearance in the registry; type TE = evaluates at the target index (USES_FUTURE_X), LE = estimates the "
         "limit, mixed = a family with both; K = accelerators in the family; the trivial comparators are listed as comparators and not counted",
         "Method roster by family (Table tab:roster), with type and member count, derived from the registry")


# ═════════════════════════════════════════════════════════════════════════════
# Facts for §7.1 (Phase 2) and Phase 3 that need no fragment of their own
# ═════════════════════════════════════════════════════════════════════════════
CASCADE_RULES = [("log_log_slope", ">", -0.10, "rational_fit"), ("richardson_r2", "<", 0.50, "rational_fit")]


def facts_phase2():
    sec = "Richardson failure (§7.1, Phase 2)"
    path = os.path.join(RES, "phase2", f"phase2_correlations_g{gname(HEADLINE_G)}.csv")
    if os.path.exists(path):
        Cr = pd.read_csv(path)
        if "spearman_vs_RR" in Cr.columns:
            for _, r in Cr[Cr.regime == "ALL"].iterrows():
                fact(sec, f"pooled Spearman correlation of {r.feature} with richardson_1's R_R_med / log median error at g={gname(HEADLINE_G)}",
                     f"{r.spearman_vs_RR:.4f} (p {r.p_vs_RR:.3f}) / {r.spearman_vs_log_err:.4f} (p {r.p_vs_log_err:.3f}); n_cells {int(r.n_cells)}, dropped {int(r.n_dropped_nan)}",
                     rel(path), f"regime == ALL, feature == {r.feature}", "spearman_vs_RR, spearman_vs_log_err over uncapped cells (features averaged over seeds)")
        else:
            print("  (phase2_correlations lacks the descriptive columns: §7.1 correlation facts skipped)")
    path = os.path.join(RES, "phase2", f"phase2_rules_g{gname(HEADLINE_G)}.csv")
    if os.path.exists(path):
        Ru = pd.read_csv(path)
        if "lower_error_frac_cells" in Ru.columns:
            for feat, op, thr, alt in CASCADE_RULES:
                r = Ru[(Ru.feature == feat) & (Ru.operator == op) & (Ru.threshold == thr) & (Ru.alternative == alt)]
                if len(r):
                    r = r.iloc[0]
                    fact(sec, f"adopted cascade rule {feat} {op} {thr} -> {alt} at g={gname(HEADLINE_G)}: fire rate; cells fired; lower-error fraction of cells / of records; median relative change; r1 / alt valid rate on the fired records",
                         f"{r.fire_rate:.4f}; {int(r.n_cells_fired)} of {int(r.n_cells_total)}; {r.lower_error_frac_cells:.4f} / {r.lower_error_frac_records:.4f}; "
                         f"{r.median_rel_change:+.4f}; {r.r1_valid_rate:.4f} / {r.alt_valid_rate:.4f}",
                         rel(path), f"feature == {feat}, operator == {op}, threshold == {thr}, alternative == {alt}", "src.panels.rule_panel over the uncapped cells with a finite feature")
    path = os.path.join(RES, "phase2", "phase2_denominator_counts.csv")
    if os.path.exists(path):
        D = pd.read_csv(path)
        fact(sec, "zero-denominator branch of R_R = E_R / E_last (E_last <= SKILL_EPS): records and cells where it fired, per horizon",
             "; ".join(f"g={r.target_g:g}: {int(r.n_records_zero_denominator)} of {int(r.n_records_total)} records, {int(r.n_cells_affected)} of {int(r.n_cells_total)} cells" for _, r in D.iterrows()),
             rel(path), "all rows (uncapped cells)", "src.panels.zero_denominator_flags")


def facts_phase3():
    SC = read("phase3", "phase3_selector_comparison.csv")
    if "valid_rate" not in SC.columns:
        return
    sec = "selectors (Phase 3)"
    for g in STRATA:
        sg = SC[SC.target_g == g]
        for _, r in sg.iterrows():
            fact(sec, f"{r.selector} at g={gname(g)}: valid_rate; n_valid/n_total; median error; mean error (conditional on validity)",
                 f"{r.valid_rate:.4f}; {int(r.n_valid)}/{int(r.n_total)}; {r.med_error:.5f}; {r.mean_error:.5f}",
                 "results/phase3/phase3_selector_comparison.csv", f"selector == {r.selector}, target_g == {g}", "src.panels.error_panel over the chosen records")
        if len(sg):
            spread = float(sg.valid_rate.max() - sg.valid_rate.min())
            fact(sec, f"g={gname(g)}: does the validity rate differ across selectors?",
                 f"{'YES' if spread > 0.001 else 'no'} (spread {spread:.4f}; the Phase-3 warning fires above 0.001)",
                 "results/phase3/phase3_selector_comparison.csv", f"target_g == {g}", "max - min of valid_rate over selectors")


def facts_classifier():
    path = os.path.join(RES, "phase3", "phase3_regime_classifier.csv")
    if not os.path.exists(path):
        return
    K = pd.read_csv(path)
    if "protocol" not in K.columns:
        return
    sec = "regime classifier (§7.2)"
    for _, r in K[K.regime == "__OVERALL__"].iterrows():
        fact(sec, f"{r.protocol}: n_samples; n_correct; accuracy",
             f"{int(r.n_samples)}; {int(r.n_correct)}; {r.accuracy:.3f}",
             rel(path), f"regime == __OVERALL__, protocol == {r.protocol}", f"accuracy = n_correct / n_samples; {r.split_unit}")
    n_regimes = K[K.regime != "__OVERALL__"].regime.nunique()
    fact(sec, "chance level", f"{1 / n_regimes:.4f} (1 / {n_regimes} regimes)",
         rel(path), "regime != __OVERALL__", "1 / number of distinct regimes in the file")


# ═════════════════════════════════════════════════════════════════════════════
# Named facts that need no fragment
# ═════════════════════════════════════════════════════════════════════════════
def named_facts():
    sec = "dangerous set"
    fact(sec, "excluded accelerators ('dangerous' is the legacy implementation name; derived from the full Phase 1)", ", ".join(sorted(DANGEROUS)) or "none",
         "results/phase1/dangerous_methods.json", "dangerous_methods", ARTIFACT["criterion"])
    fact(sec, "criterion", f"{ARTIFACT['criterion']}; floor RANK_MIN_VALID = {ARTIFACT.get('rank_min_valid')} (config.RANK_MIN_VALID = {C.RANK_MIN_VALID}); "
         "no composite score enters the exclusion",
         "results/phase1/dangerous_methods.json", "criterion, rank_min_valid", "valid_rate = mean over uncapped core cells of the per-cell valid rate (equal cell weights)")
    fact(sec, "identical to the legacy hard-coded set?", f"{'YES' if SAME_AS_LEGACY else 'NO'}: +{ADDED_VS_LEGACY} -{REMOVED_VS_LEGACY}",
         "results/phase1/dangerous_methods.json vs src/config.py LEGACY_DANGEROUS_METHODS", "-", "set difference")
    tab = pd.DataFrame(ARTIFACT["table"]).sort_values(["valid_rate", "method"])
    fact(sec, "pooled valid rate of each excluded method (cat_rate, median error alongside)",
         "; ".join(f"{r.method} valid {r.valid_rate:.4f} (cat {r.cat_rate:.4f}, med. err {r.med_error:.4f})" for _, r in tab[tab.dangerous == 1].iterrows()) or "none",
         "results/phase1/dangerous_methods.json", "table, dangerous == 1", "valid_rate; cat_rate and med_error are descriptive, not part of the criterion")
    kept = tab[tab.dangerous == 0]
    if len(kept):
        safe = kept.iloc[0]
        fact(sec, "lowest valid rate among the non-excluded accelerators", f"{safe.method} valid {safe.valid_rate:.4f} (cat {safe.cat_rate:.4f}, med. err {safe.med_error:.4f})",
             "results/phase1/dangerous_methods.json", "table, dangerous == 0", "min valid_rate")
    fact(sec, "scope of the derivation", f"Phase 1 only: obs_idx = {C.PHASE1['full']['obs_idx']}, noise {C.PHASE1['full']['noise_levels']}, "
         f"{C.PHASE1['full']['n_seeds']} seeds, the core regimes, pooled over the three strata with equal cell weights, capped cells excluded",
         "src/config.py PHASE1; scripts/derive_dangerous.py", "-", "the derivation reads phase1_aggregated.csv and nothing else; no depth other than the Phase-1 depth enters it")
    fact(sec, "artifact provenance", f"git_head {ARTIFACT.get('git_head')}, created {ARTIFACT.get('created')}, source {ARTIFACT.get('source')}, pool {ARTIFACT.get('pool')} (n_pool {ARTIFACT.get('n_pool')})",
         "results/phase1/dangerous_methods.json", "-", "header fields")

    # sigma = 0 cancellation NaNs in the difference-based families (committed aggregate)
    sec = "sigma = 0 cancellation NaNs"
    A = AGG[(AGG.is_oracle == 0)]
    for fam in CLASSICAL_FAMILIES:
        s = A[A.family == fam]
        by = s.groupby("noise").valid_rate.mean()
        fact(sec, f"{fam}: invalid rate by noise (Phase 1, all cells)", ", ".join(f"sigma={n:g}: {100 * (1 - v):.2f}%" for n, v in by.items()),
             "results/phase1/phase1_aggregated.csv", f"family == {fam}, is_oracle == 0 (core + held-out, capped included; {int(AGG.n_total.max())} seeds per cell)",
             "100 * (1 - mean over cells of valid_rate) per noise level")
    s = A[A.family.isin(CLASSICAL_FAMILIES)]
    by = s.groupby("noise").valid_rate.mean()
    fact(sec, f"all {len(CLASSICAL)} classical (difference-based) variants: invalid rate by noise (Phase 1)",
         ", ".join(f"sigma={n:g}: {100 * (1 - v):.2f}%" for n, v in by.items()),
         "results/phase1/phase1_aggregated.csv", f"family in the {len(CLASSICAL_FAMILIES)} classical families", "100 * (1 - mean valid_rate) per noise level")
    nd = A[(~A.method.isin(DANGEROUS)) & (A.is_trivial == 0)]
    by = nd.groupby("noise").valid_rate.mean()
    fact(sec, "all non-dangerous accelerators: invalid rate by noise (Phase 1)",
         ", ".join(f"sigma={n:g}: {100 * (1 - v):.2f}%" for n, v in by.items()),
         "results/phase1/phase1_aggregated.csv", "method not in the dangerous set, is_trivial == 0", "100 * (1 - mean valid_rate) per noise level")
    fact(sec, "mechanism", "exact-zero differences on noise-free plateaus give zero denominators (DENOM_TOL) in the Shanks/Wynn/Levin/Brezinski/Weniger/Anderson transforms; the estimate is NaN, the record invalid",
         "src/accelerators.py", "-", "-")

    # richardson_3 by depth: committed aggregate when present (Prompt 5A), else the raw file
    sec = "richardson_3 validity by depth"
    vpath = os.path.join(RES, "phase5a", "phase5a_validity_by_depth.csv")
    p1_depth = int(C.PHASE1["full"]["obs_idx"])
    r3_at_p1_depth = "not recomputed (phase5a_validity_by_depth.csv absent)"
    if os.path.exists(vpath):
        V = pd.read_csv(vpath)
        r3 = V[V.method == "richardson_3"].groupby("obs_idx").apply(lambda g: (g.valid_rate * g.n).sum() / g.n.sum())
        fact(sec, "richardson_3 valid rate by observation depth (committed aggregate)",
             ", ".join(f"obs {int(d)}: {v:.3f}" for d, v in r3.items()),
             "results/phase5a/phase5a_validity_by_depth.csv", "method == richardson_3", "n-weighted mean of valid_rate over noise levels per obs_idx")
        if p1_depth in set(int(d) for d in r3.index):
            r3_at_p1_depth = f"{100 * float(r3.loc[p1_depth]):.0f} % valid (phase5a_validity_by_depth.csv)"
        else:
            r3_at_p1_depth = f"not recomputed (obs_idx {p1_depth} is not in phase5a_validity_by_depth.csv)"
        low = (V[(V.is_trivial == 0) & (V.obs_idx == V.obs_idx.min())].groupby("method")
               .apply(lambda g: (g.valid_rate * g.n).sum() / g.n.sum()).sort_values().head(8))
        fact(sec, f"least valid accelerators at obs {int(V.obs_idx.min())} (committed aggregate)",
             ", ".join(f"{m} {v:.3f}" for m, v in low.items()), "results/phase5a/phase5a_validity_by_depth.csv",
             f"obs_idx == {int(V.obs_idx.min())}, is_trivial == 0", "n-weighted valid_rate per method")
    G1 = G_CORE[G_CORE.target_g == HEADLINE_G]
    if "win_rate_vs_assumed" in G1.columns:
        for m in ("log_linear", "rational_fit", "richardson_1", "single_exp_fit"):
            r = G1[G1.method == m]
            if len(r):
                r = r.iloc[0]
                fact("trivial baseline", f"{m} g={gname(HEADLINE_G)} core: fixed-reference win rates (vs assumed / last / wmean / wmin)",
                     f"{r.win_rate_vs_assumed:.3f} / {r.win_rate_vs_last:.3f} / {r.win_rate_vs_wmean:.3f} / {r.win_rate_vs_wmin:.3f}; "
                     f"median skill vs assumed {r.med_skill_vs_assumed:.3f}, vs last {r.med_skill_vs_last:.3f}",
                     SRC_G[0], f"method == {m}, target_g == {HEADLINE_G}", "win_rate_vs_* = mean over cells of the per-cell win rate; med_skill_vs_* = median over cells")
    cpath = os.path.join(RES, "phase5b", "phase5b_sweep1_consumers.csv")
    if os.path.exists(cpath):
        Cn = pd.read_csv(cpath)
        for m in ("log_linear", "rational_fit", "richardson_1", "constant_assumed"):
            sub = Cn[(Cn.method == m) & (Cn.regime_set == "core") & (Cn.target_g == HEADLINE_G)]
            if len(sub):
                fact("sweep 1 (assumed asymptote)", f"sweep 1b, {m} at g={gname(HEADLINE_G)} core: median error by L_hat mode",
                     ", ".join(f"{r.assumed_mode}: {r.med_error:.4f} (skill {r.med_skill:.2f}, win vs assumed {r.win_rate_vs_assumed:.2f})" for _, r in sub.iterrows()),
                     "results/phase5b/phase5b_sweep1_consumers.csv", f"method == {m}, regime_set == core, target_g == {HEADLINE_G}", "med_error / med_skill / win_rate_vs_assumed per assumed_mode")
    # The raw-derived facts (phase5a_raw.csv, then phase1_records.csv) are computed by
    # scripts/derive_raw_facts.py into results/raw_facts.csv and emitted from there, in
    # the stored order, at the two places where this script used to read the raw files.
    emitted = emit_raw_facts("results/phase5a/phase5a_raw.csv")
    if not emitted:
        fact(sec, "richardson_3 valid rate by observation depth (Phase 5a)", "not recomputed (raw file absent)",
             "results/phase5a/phase5a_raw.csv (git-ignored)", "method == richardson_3", "mean(valid) per obs_idx")
    fact(sec, "why the artifact does not flag richardson_3", f"the excluded set is derived at obs_idx = {p1_depth} only, where richardson_3 is {r3_at_p1_depth}; "
         "its 7-parameter curve_fit (src/accelerators.py, _fit_richardson, n_terms = 3, maxfev = 3000) does not converge on 30-60-point windows and returns NaN",
         "src/accelerators.py; src/config.py PHASE1; results/phase5a/phase5a_validity_by_depth.csv", "-", "n-weighted valid_rate of richardson_3 at the Phase-1 depth")

    emit_raw_facts("results/phase1/phase1_records.csv")

    # method roster notes (Prompt 5A / 5B)
    sec = "method roster (Weniger, Levin, pool)"
    fact(sec, "Weniger delta: root cause of the pre-5A degeneracy",
         "with the remainder estimate w_n = s_n the numerator sum_j (-1)^j C(k,j) beta_j s_j / w_j collapses to sum_j (-1)^j C(k,j) beta_j, "
         "the k-th difference of a degree-(k-1) polynomial, identically 0 for k = 1, 2: weniger_d1/d2 returned 0 for every input "
         "(= constant_assumed under L_hat = 0)", "src/accelerators.py::_weniger_delta (docstring)", "-", "-")
    fact(sec, "Weniger delta: fix (Prompt 5A)", "forward-difference remainder w_n = s_{n+1} - s_n on the last order+2 window values, Pochhammer weight (n0+j+1)_(k-1) via scipy.special.poch; "
         "recovers the limit of 0.3 + 0.5 * 0.9^n to 1.2e-15 / 1.4e-15 (weniger_d1 / d2)",
         "src/accelerators.py::_weniger_delta; tests/test_accelerators.py::test_weniger_recovers_geometric_limit", "-", "-")
    fact(sec, "Weniger delta: identity with Levin t and removal (Prompt 5B)",
         "this codebase's Levin 't' uses the same forward-difference remainder and the power weight (n0+j+1)^(k-1), which equals the Pochhammer weight for k <= 2, "
         "so the corrected weniger_d1/d2 == levin_t1/t2 to machine precision on the 96 audit windows (max |diff| 0); the pair is retired from the roster "
         f"(RETIRED_METHODS; {N_ACC} accelerators in {len(set(FAM[m] for m in ACCEL_METHODS))} families remain) and kept tested",
         "src/accelerators.py (RETIRED_METHODS); tests/test_accelerators.py::test_weniger_equals_levin_t", "-", "-")
    fact(sec, "Levin remainder naming (docstring correction, no numerical change)",
         "'t' = forward-difference remainder w_n = Delta s_n (Weniger's d~-type; not Levin's backward-difference t); 'u' = (n+1) Delta s_n; 'v' = the ratio-of-differences form",
         "src/accelerators.py::_levin_transform (docstring)", "-", "-")
    fact(sec, f"{len(PHASE2_POOL)}-method Phase 2/3/4 pool (eight accelerators + last_value)", ", ".join(PHASE2_POOL) + " (weniger_d2 replaced by levin_t2, to which the corrected weniger_d2 is identical; the pool is defined once in src.pipeline.PHASE2_POOL)",
         "src/pipeline.py::PHASE2_POOL", "-", "-")
    fact(sec, "L_hat consumers (accelerators whose output depends on the assumed asymptote)",
         ", ".join(LHAT_CONSUMERS) + f" ({len(LHAT_CONSUMERS)}; measured on the audit windows of tests/test_input_dependence.py)",
         "tests/test_input_dependence.py; phases/phase5b.py::LHAT_CONSUMERS", "output differs between L_hat = 0 and 0.5 * min(window) on >= 1 window", "-")

    # win rates vs each fixed trivial for the top-10 accelerators per stratum, core and held-out
    if "win_rate_vs_assumed" in G_CORE.columns:
        for key, G, RK in (("core", G_CORE, RANK_CORE), ("held-out", G_HOLD, RANK_HOLD)):
            for g in STRATA:
                top = sorted(RK[g], key=RK[g].get)[:10]
                parts = []
                for m in top:
                    r = row_of(G, g, m)
                    parts.append(f"{RK[g][m]}. {m}: {r.win_rate_vs_assumed:.2f}/{r.win_rate_vs_last:.2f}/{r.win_rate_vs_wmean:.2f}/{r.win_rate_vs_wmin:.2f}")
                fact("top-10 win rates (vs assumed / last / wmean / wmin)", f"g={gname(g)} {key}", "; ".join(parts),
                     SRC_G[0] if key == "core" else SRC_G[1], f"target_g == {g}, rank-eligible accelerators ranked by med_error, top 10",
                     "win_rate_vs_<ref> = mean over uncapped cells of the per-cell fraction of seeds with error below the reference's")

    # pipeline provenance
    sec = "pipeline provenance"
    M = load_manifest(RES)
    manifest_rule = "written by reproduce_all.py at the start of a run, after every step and at the end"
    if M is not None:
        head = M.get("git_head", {})
        steps = "; ".join(f"{name} {secs:,.0f} s" for name, secs in M.get("steps", {}).items())
        total = M.get("total_seconds")     # null inside a run: the manifest's final rewrite sets it
        fact(sec, "run (results/run_manifest.json)",
             f"mode {M.get('mode')}; code {head.get('short')} ({head.get('full')}); started {M.get('started')}, finished {M.get('finished') or 'in progress'}; "
             f"jobs {M.get('jobs')}; total {'in progress' if total is None else f'{total:,.0f} s'}; steps: {steps}",
             "results/run_manifest.json", "-", manifest_rule)
    else:
        fact(sec, "run (results/run_manifest.json)", "manifest absent", "results/run_manifest.json", "-", manifest_rule)
    from reproduce_all import plan
    rows = plan("full")
    fact(sec, "evaluation counts (central), derived from the config grids",
         "; ".join(f"{label} {n:,} ({note})" for label, n, note in rows) + f"; TOTAL {sum(n for _, n, _ in rows):,}",
         "python reproduce_all.py --plan", "-", "reproduce_all.plan('full')")
    rp = os.path.join(RES, "real_data", "real_data_roster_provenance.json")
    if os.path.exists(rp):
        with open(rp, encoding="utf-8") as fh:
            P = json.load(fh)
        cfg = P.get("configuration", {})
        fact(sec, "roster evaluation of the recorded curves (results/real_data/real_data_roster_provenance.json)",
             f"ran separately after the full run: {P.get('script')} at code {P.get('git_head', {}).get('short')} ({P.get('git_head', {}).get('full')}), "
             f"tracked tree clean {P.get('tracked_tree_clean')}; started {P.get('started')}, finished {P.get('finished')}, {P.get('seconds')} s; "
             f"library versions equal the run manifest's: {P.get('versions_equal_run_manifest')}; {P.get('cells')} cells x {P.get('methods')} methods = {P.get('rows')} rows; "
             f"RIDGE {cfg.get('RIDGE')}, DENOM_TOL {cfg.get('DENOM_TOL')}, validity [{cfg.get('MIN_VALID')}, {cfg.get('MAX_VALID')}], CAT_MULT {cfg.get('CAT_MULT')}, "
             f"window {cfg.get('window_len')}, L_hat mode {cfg.get('assumed_mode')}",
             "results/real_data/real_data_roster_provenance.json", "-", "header fields; written by scripts/run_real_data.py --roster-only")
    sp = os.path.join(RES, "phase1", "phase1_selection_provenance.json")
    if os.path.exists(sp):
        with open(sp, encoding="utf-8") as fh:
            P = json.load(fh)
        fact(sec, "selection by trial on the Phase 1 records (results/phase1/phase1_selection_provenance.json)",
             f"ran separately after the full run: {P.get('script')} at code {P.get('git_head', {}).get('short')} ({P.get('git_head', {}).get('full')}), "
             f"tracked tree clean {P.get('tracked_tree_clean')}; started {P.get('started')}, finished {P.get('finished')}, {P.get('seconds')} s; "
             f"library versions equal the run manifest's: {P.get('versions_equal_run_manifest')}; records {P.get('records_rows'):,} rows, sha256 {P.get('records_sha256')}; "
             f"cells per design {P.get('cells')}, trials per design {P.get('trials')}; pools: " + "; ".join(f"{k} ({len(v)}) {', '.join(v)}" for k, v in P.get("pools", {}).items())
             + f"; {P.get('note')}",
             "results/phase1/phase1_selection_provenance.json", "-", "header fields; written by scripts/derive_selection.py")
    fact(sec, "design constants", f"L_true log-uniform on {C.L_TRUE_RANGE} per (regime, seed); L_hat mode {C.ASSUMED_L_MODE}; strata g = {STRATA} (headline {HEADLINE_G}); "
         f"n_obs = {C.OBS_IDX}, window {C.WINDOW_LEN}; horizon cap {C.HORIZON_N_CAP}; rank floor valid_rate >= {C.RANK_MIN_VALID} "
         f"(also the exclusion criterion of the dangerous artifact); CAT_MULT {C.CAT_MULT}; no composite score", "src/config.py", "-", "-")

    # effective sample: the families and cells entering the pooled statistics, per set and stratum
    sec = "effective sample"
    for hold, lab, G in ((0, "core", G_CORE), (1, "held-out", G_HOLD)):
        for g in STRATA:
            S = AGG[(AGG.is_holdout == hold) & (AGG.target_g == g)]
            unc = S[S.capped == 0]
            n_fam, n_cells = int(unc.regime.nunique()), int(unc.drop_duplicates(["regime", "noise"]).shape[0])
            capped = sorted(set(S.regime) - set(unc.regime))
            sg = G[G.target_g == g]
            assert n_fam == int(sg.n_regimes.max()) and n_cells == int(sg.n_cells.max()), (g, hold, n_fam, n_cells)
            fact(sec, f"g={gname(g)} {lab}: families and cells entering the pooled statistics; families excluded as capped",
                 f"{n_fam} families, {n_cells} cells; capped: {', '.join(capped) or 'none'}",
                 "results/phase1/phase1_aggregated.csv", f"is_holdout == {hold}, target_g == {g}",
                 "regimes and (regime, noise) cells with capped == 0; regimes with capped == 1 (equal to n_regimes / n_cells of the pooled table)")

    # order ladders under noise: per set and noise level, the classical ladders against the last value
    facts_ladders_under_noise()


def emit_raw_facts(raw_file):
    """Emit the stored raw-derived FACTS rows whose file column names raw_file
    (results/raw_facts.csv, written by scripts/derive_raw_facts.py), in the
    stored order; returns the number emitted (0 when the file is absent)."""
    if RAW_FACTS is None:
        return 0
    rows = RAW_FACTS[RAW_FACTS.file.str.startswith(raw_file)]
    for _, r in rows.iterrows():
        fact(r.section, r.fact, r.value, r.file, r["filter"], r.formula)
    return len(rows)


LADDERS_CLASSICAL_NOISE = [f for f in ("shanks", "wynn_eps", "wynn_rho", "levin_t", "levin_u", "levin_v", "brezinski_theta", "anderson")]


def facts_ladders_under_noise():
    """From order_ladders_panels.csv, per set and noise level (not pooled), for the
    ladders of the classical families (LADDERS_CLASSICAL without neville and pade):
    the variants with med_error below NOOP_BAND[0] x the last value's, the variants
    with a win rate above WIN_BAND[1], and the variant with the lowest median error."""
    path = os.path.join(RES, "phase0b", "order_ladders_panels.csv")
    if not os.path.exists(path):
        return
    P = pd.read_csv(path)
    P = P[P.noise.astype(str) != "pooled"].copy()
    P["sigma"] = P.noise.astype(float)
    fams = [f for f in LADDERS_CLASSICAL if f not in ("neville", "pade")]
    assert fams == LADDERS_CLASSICAL_NOISE, fams
    sec = "order ladders under noise (Phase 0b)"
    note = f"sigma = 0 includes the intrinsic-noise families ({', '.join(INTRINSIC_NOISE)}) with their own noise"
    for rs in ("core", "holdout"):
        for sigma in sorted(P.sigma.unique()):
            sub = P[(P.regime_set == rs) & (P.sigma == sigma) & (P.family.isin(fams))]
            last = P[(P.regime_set == rs) & (P.sigma == sigma) & (P.variant == "trivial:last_value")]
            if sub.empty or last.empty or sub.med_error.notna().sum() == 0:
                continue
            e_last = float(last.med_error.iloc[0])
            below = sub[sub.med_error < NOOP_BAND[0] * e_last]
            high = sub[sub.win_rate_vs_last > WIN_BAND[1]]
            best = sub.loc[sub.med_error.idxmin()]
            lab = "core" if rs == "core" else "held-out"
            value = (f"{len(below)} of {len(sub)}" + (": " + ", ".join(f"{r.family} {r.order_label} ({r.med_error:.5f})" for _, r in below.iterrows()) if len(below) else "")
                     + f"; {len(high)} of {len(sub)}" + (": " + ", ".join(f"{r.family} {r.order_label} ({r.win_rate_vs_last:.3f})" for _, r in high.iterrows()) if len(high) else "")
                     + f"; lowest: {best.family} {best.order_label} med. err {best.med_error:.5f}, valid {best.valid_rate:.3f}, win vs last {best.win_rate_vs_last:.3f}"
                     + (f" ({note})" if sigma == 0 else ""))
            fact(sec, f"{lab} sigma={sigma:g}: classical ladder variants with median error below {NOOP_BAND[0]} x the last value's ({e_last:.5f}); "
                      f"variants with win rate vs last_value above {WIN_BAND[1]}; the variant with the lowest median error",
                 value,
                 "results/phase0b/order_ladders_panels.csv", f"regime_set == {rs}, noise == {sigma:g}, family in {fams}",
                 f"count(med_error < {NOOP_BAND[0]} * med_error of trivial:last_value); count(win_rate_vs_last > {WIN_BAND[1]}); argmin med_error (uncapped cells, conditional on validity)")


# ═════════════════════════════════════════════════════════════════════════════
# R9d Part C: the new fragments (section 5.2)
# ═════════════════════════════════════════════════════════════════════════════
def _noop_class_stats(S):
    """The classical no-op statistics of one (set, stratum, noise class): S = the
    uncapped AGG rows of that class for every method.  Per variant: MI = median
    over the class's cells of med_improve, win = mean of win_rate_vs_last, error
    = median of med_error; plus the last value's and rational_fit's."""
    by = S[S.method.isin(CLASSICAL)].groupby("method")
    mi = by.med_improve.median().reindex(CLASSICAL)
    win = by.win_rate_vs_last.mean().reindex(CLASSICAL)
    err = by.med_error.median().reindex(CLASSICAL)
    rf = S[S.method == "rational_fit"]
    return dict(n_fam=int(S.regime.nunique()), n_cells=int(S.drop_duplicates(["regime", "noise"]).shape[0]),
                mi=mi, win=win, err=err,
                inband=int(((mi >= NOOP_BAND[0]) & (mi <= NOOP_BAND[1])).sum()),
                wband=int(((win >= WIN_BAND[0]) & (win <= WIN_BAND[1])).sum()),
                mi_min=float(mi.min()), mi_med=float(mi.median()), mi_max=float(mi.max()),
                win_mean=float(win.mean()), win_max=float(win.max()),
                err_last=float(S[S.method == "last_value"].med_error.median()),
                err_cls=float(err.median()), err_best=float(err.min()),
                err_rf=float(rf.med_error.median()) if len(rf) else float("nan"),
                win_rf=float(rf.win_rate_vs_last.mean()) if len(rf) else float("nan"))


def _class_cells(hold, g):
    """The uncapped (regime, noise) cells of a set and stratum with their noise class."""
    S = AGG[(AGG.is_holdout == hold) & (AGG.target_g == g) & (AGG.capped == 0)]
    cells = S.drop_duplicates(["regime", "noise"])[["regime", "noise"]].copy()
    cells["cls"] = [noise_class(r, n) for r, n in zip(cells.regime, cells.noise)]
    return S, cells


def _class_rows(S, cells, cls):
    sel = cells if cls == "all" else cells[cells.cls == cls]
    keys = set(zip(sel.regime, sel.noise))
    return S[[(r, n) in keys for r, n in zip(S.regime, S.noise)]]


NOISE_CLASS_DEF = (f"noise class of a (family, noise) cell: noise-free = sigma 0 and no intrinsic noise; intrinsic noise only = sigma 0 and "
                   f"one of the intrinsic-noise families ({', '.join(INTRINSIC_NOISE)}); otherwise sigma = the level")


def f21():
    """The classical no-op by noise class: one row per (set, stratum, noise class)
    plus an 'all cells' row per (set, stratum); the 'all cells' row reproduces
    the band count of f04 (core) and f04b (held-out) at every stratum."""
    N = len(CLASSICAL)
    head = [r"Set & $g$ & noise class & families & cells & in band & \multicolumn{3}{c}{MI over the variants} & \multicolumn{2}{c}{win vs last} & "
            r"\multicolumn{4}{c}{median error} & \meth{rational\_fit} \\",
            r"\cmidrule(lr){7-9}\cmidrule(lr){10-11}\cmidrule(lr){12-15}",
            r" & & & & & $k/N$ & min & median & max & mean & max & last value & classical & best classical & \meth{rational\_fit} & win vs last \\",
            r"\midrule"]
    rows = list(head)
    sec = "classical no-op by noise class"
    for hold, setlab, key in ((0, "core", "core"), (1, "held-out", "holdout")):
        first_set = True
        for g in STRATA:
            S, cells = _class_cells(hold, g)
            if cells.empty:
                continue
            classes = sorted(cells.cls.unique(), key=noise_class_order)
            for cls in classes + ["all"]:
                st = _noop_class_stats(_class_rows(S, cells, cls))
                if cls == "all":
                    assert st["inband"] == NOOP_BAND_COUNTS[key][g], (key, g, st["inband"], NOOP_BAND_COUNTS[key][g])
                lab = "all cells" if cls == "all" else noise_class_label(cls)
                rows.append(f"{setlab if first_set else ''} & {gname(g) if cls == classes[0] else ''} & {lab} & {st['n_fam']} & {st['n_cells']} & "
                            f"{st['inband']}/{N} & {f3(st['mi_min'])} & {f3(st['mi_med'])} & {f3(st['mi_max'])} & {f3(st['win_mean'])} & {f3(st['win_max'])} & "
                            f"{f4(st['err_last'])} & {f4(st['err_cls'])} & {f4(st['err_best'])} & {f4(st['err_rf'])} & {f3(st['win_rf'])} \\\\")
                first_set = False
                fact(sec, f"{setlab} g={gname(g)} {'all cells' if cls == 'all' else noise_class_label(cls, tex=False)}: families / cells / variants in the MI band / "
                          "MI min, median, max / win rate mean, max / median error of last_value, classical (median over variants), best classical, rational_fit / rational_fit win rate",
                     f"{st['n_fam']} / {st['n_cells']} / {st['inband']} of {N} / {st['mi_min']:.3f}, {st['mi_med']:.3f}, {st['mi_max']:.3f} / "
                     f"{st['win_mean']:.3f}, {st['win_max']:.3f} / {st['err_last']:.4f}, {st['err_cls']:.4f}, {st['err_best']:.4f}, {st['err_rf']:.4f} / {st['win_rf']:.3f}",
                     "results/phase1/phase1_aggregated.csv",
                     f"is_holdout == {hold}, target_g == {g}, capped == 0, " + ("all cells" if cls == "all" else f"noise class {cls}"),
                     "per variant: MI = median over the class's cells of med_improve, win = mean of win_rate_vs_last, error = median of med_error; "
                     f"in band = count(MI in [{NOOP_BAND[0]}, {NOOP_BAND[1]}]); last_value / rational_fit = the same over their cells")
            rows.append(r"\addlinespace[2pt]")
        if hold == 0:
            rows.append(r"\midrule")
    while rows and rows[-1] in (r"\addlinespace[2pt]", r"\midrule"):
        rows.pop()
    frag("f21_noop_by_noise.tex", "lllrrr" + "r" * 10, rows,
         ["results/phase1/phase1_aggregated.csv (per (method, regime, noise, g) cell; capped == 0)"],
         f"capped == 0; {NOISE_CLASS_DEF}; families = distinct curve families in the class, cells = (family, noise) cells; "
         f"for a classical variant and a class: MI = median over the class's cells of med_improve, win = mean over cells of win_rate_vs_last, "
         f"error = median over cells of med_error; in band = variants with MI in [{NOOP_BAND[0]}, {NOOP_BAND[1]}] (k/N, N = {N}); MI and win "
         "summarised over the variants; median error of the last value, of the classical variants (median over variants), of the best variant "
         "and of rational_fit; the 'all cells' row reproduces the band count of f04 (core) and f04b (held-out)",
         "Classical no-op by noise class: the classical variants against the +/-10 % band in each noise class, per set and stratum, "
         "with the last value and rational_fit alongside")


def _f21b(hold, key, name):
    N = len(CLASSICAL)
    S, cells = _class_cells(hold, HEADLINE_G)
    if cells.empty:
        print(f"  ({name}: no uncapped cell at the headline stratum; skipped)")
        return
    classes = sorted(cells.cls.unique(), key=noise_class_order)
    stats = {cls: _noop_class_stats(_class_rows(S, cells, cls)) for cls in classes}
    head = [r"Family & Method & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{noise_class_label(c)}}}" for c in classes) + r" \\",
            "".join(rf"\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}" for i in range(len(classes))),
            r" & & " + " & ".join("MI & win vs last" for _ in classes) + r" \\", r"\midrule"]
    rows = list(head)
    last_fam = None
    sec = "classical no-op by noise class"
    setlab = "core" if hold == 0 else "held-out"
    for m in CLASSICAL:
        fam = FAM[m]
        if last_fam is not None and fam != last_fam:
            rows.append(r"\addlinespace[2pt]")
        last_fam = fam
        cells_ = []
        for c in classes:
            mi, w = float(stats[c]["mi"][m]), float(stats[c]["win"][m])
            ib = math.isfinite(mi) and NOOP_BAND[0] <= mi <= NOOP_BAND[1]
            wb = math.isfinite(w) and WIN_BAND[0] <= w <= WIN_BAND[1]
            cells_ += [(r"\textbf{" + f3(mi) + "}") if ib else f3(mi), (r"\textbf{" + f3(w) + "}") if wb else f3(w)]
        rows.append(f"{fam_label(fam)} & {dag(m)}{mth(m)} & " + " & ".join(cells_) + r" \\")
    rows.append(r"\midrule")
    rows.append(r"\multicolumn{2}{l}{in the $\pm 10\%$ band (MI in $[0.9, 1.1]$)} & " + " & ".join(f"{stats[c]['inband']}/{N} & " for c in classes) + r" \\")
    rows.append(r"\multicolumn{2}{l}{win rate vs last in $[0.4, 0.6]$ (coin flip)} & " + " & ".join(f" & {stats[c]['wband']}/{N}" for c in classes) + r" \\")
    for c in classes:
        st = stats[c]
        fact(sec, f"{setlab} g={gname(HEADLINE_G)} {noise_class_label(c, tex=False)}: every classical variant's MI / win rate vs last_value",
             "; ".join(f"{m} {float(st['mi'][m]):.3f} / {float(st['win'][m]):.3f}" for m in CLASSICAL)
             + f"; in band {st['inband']} of {N}; win rate in [{WIN_BAND[0]}, {WIN_BAND[1]}] {st['wband']} of {N}",
             "results/phase1/phase1_aggregated.csv", f"is_holdout == {hold}, target_g == {HEADLINE_G}, capped == 0, noise class {c}",
             "MI = median over the class's cells of med_improve; win = mean over cells of win_rate_vs_last")
    frag(name, "ll" + "rr" * len(classes), rows,
         ["results/phase1/phase1_aggregated.csv (per (method, regime, noise, g) cell; capped == 0)"],
         f"is_holdout == {hold}, target_g == {gname(HEADLINE_G)}, capped == 0; {NOISE_CLASS_DEF}; MI = median over the class's cells of med_improve, "
         f"bold = inside [{NOOP_BAND[0]}, {NOOP_BAND[1]}]; win vs last = mean over the class's cells of win_rate_vs_last, bold = inside [{WIN_BAND[0]}, {WIN_BAND[1]}]; "
         "variants in the order of f04",
         f"Classical no-op by noise class, {setlab} families, headline stratum: every classical variant's median improvement factor and win rate "
         "against the last value in each noise class")


def f21b():
    _f21b(0, "core", "f21b_noop_by_noise_variants_core.tex")
    _f21b(1, "holdout", "f21b_noop_by_noise_variants_holdout.tex")


def f22():
    """rational_fit and single_exp_fit family by family, one fragment per stratum."""
    sec = "per family (rational_fit, single_exp_fit)"
    NAMED = (DEFAULT_METHOD, CONSERVATIVE_METHOD)
    for g in STRATA:
        rows = [rf"Set & curve family & cells & last value & \multicolumn{{4}}{{c}}{{{mth(DEFAULT_METHOD)}}} & \multicolumn{{2}}{{c}}{{{mth(CONSERVATIVE_METHOD)}}} & "
                r"\multicolumn{2}{c}{best rank-eligible accelerator} \\",
                r"\cmidrule(lr){5-8}\cmidrule(lr){9-10}\cmidrule(lr){11-12}",
                r" & & & med.\ err & med.\ err & win vs last & $\rhoV$ & $\rhoC$ & med.\ err & win vs last & method & med.\ err \\", r"\midrule"]
        foot = []
        for hold, setlab, RK in ((0, "core", RANK_CORE), (1, "held-out", RANK_HOLD)):
            T = FAMILY_TABLES[(g, hold)]
            eligible = sorted(RK[g], key=RK[g].get)
            n_unc = 0
            stat = {m: dict(n_win=0, losers=[], worse=[]) for m in NAMED}
            for reg in SET_REGIMES[hold]:
                if reg not in T.index.get_level_values("regime") or DEFAULT_METHOD not in T.loc[reg].index:
                    rows.append(f"{setlab} & {tt(reg)} & 0 & " + " & ".join(["capped"] * 9) + r" \\")
                    continue
                sub = T.loc[reg]
                last, rf = sub.loc["last_value"], sub.loc[DEFAULT_METHOD]
                se = sub.loc[CONSERVATIVE_METHOD] if CONSERVATIVE_METHOD in sub.index else None
                cand = sub.loc[[m for m in eligible if m in sub.index]].reset_index().dropna(subset=["med_error"])
                best = cand.sort_values(["med_error", "method"]).iloc[0] if len(cand) else None
                n_unc += 1
                for m, r in ((DEFAULT_METHOD, rf), (CONSERVATIVE_METHOD, se)):
                    if r is None:
                        continue
                    if r.win > 0.5:
                        stat[m]["n_win"] += 1
                    else:
                        stat[m]["losers"].append((reg, float(r.win)))
                    if r.med_error > last.med_error:
                        stat[m]["worse"].append((reg, float(r.med_error), float(last.med_error)))
                se_cells = f"{f4(float(se.med_error))} & {f3(float(se.win))}" if se is not None else "-- & --"
                rows.append(f"{setlab} & {tt(reg)} & {int(rf.n)} & {f4(float(last.med_error))} & {f4(float(rf.med_error))} & {f3(float(rf.win))} & "
                            f"{f3(float(rf.valid))} & {f3(float(rf.cat_rate))} & {se_cells} & "
                            + (f"{mth(best.method)} & {f4(float(best.med_error))}" if best is not None else "-- & --") + r" \\")
            rows.append(r"\addlinespace[2pt]")
            foot.append((setlab, hold, n_unc, stat))
        rows.pop()
        rows.append(r"\midrule")
        for setlab, hold, n_unc, stat in foot:
            for m in NAMED:
                n_win, losers, worse = stat[m]["n_win"], stat[m]["losers"], stat[m]["worse"]
                rows.append(rf"\multicolumn{{12}}{{l}}{{{setlab}, {mth(m)}: its win rate against the last value exceeds 0.5 in {n_win} of {n_unc} uncapped families"
                            + ("; not in " + ", ".join(f"{tt(r)} ({w:.3f})" for r, w in losers) if losers else "")
                            + "; its median error exceeds the last value's in " + (", ".join(f"{tt(r)} ({e:.4f} vs {l:.4f})" for r, e, l in worse) if worse else "no family") + r"} \\")
                fact(sec, f"g={gname(g)} {setlab}: uncapped families where {m}'s family win rate vs last_value exceeds 0.5; the families where it does not (win rate); "
                          "the families where its family median error exceeds the last value's (both)",
                     f"{n_win} of {n_unc}; " + (", ".join(f"{r} {w:.3f}" for r, w in losers) if losers else "none") + "; "
                     + (", ".join(f"{r} {e:.4f} vs {l:.4f}" for r, e, l in worse) if worse else "none"),
                     "results/phase1/phase1_aggregated.csv", f"is_holdout == {hold}, target_g == {g}, capped == 0, method in ({m}, last_value)",
                     "per family: win = mean over the family's cells of win_rate_vs_last; median error = median over cells of med_error")
        frag(f"f22_per_family_g{gname(g)}.tex", "llrr" + "rrrr" + "rr" + "lr", rows,
             ["results/phase1/phase1_aggregated.csv (per (method, regime, noise, g) cell; capped == 0)",
              "results/phase1/phase1_global.csv, results/phase1/phase1_global_holdout.csv (rank_eligible == 1: the candidates of the last two columns)"],
             f"target_g == {gname(g)}; one row per curve family (core, then held-out); cells = the family's uncapped (noise) cells; {FAMILY_STAT_NOTE}; "
             f"{CONSERVATIVE_METHOD} = the conservative alternative the paper names (median error and win rate vs the last value, the same statistics); "
             "best rank-eligible accelerator = the lowest family median error among the accelerators that are rank-eligible in the pooled table of that set "
             "and stratum (ties by method name); a family with no uncapped cell prints 'capped'; the footers give, per set and named method, the uncapped "
             "families where its win rate against the last value exceeds 0.5 (the family win rate = mean over the family's cells of the per-cell win rate)",
             f"rational_fit and single_exp_fit family by family at g = {gname(g)}: the last value, rational_fit (median error, win rate, validity, catastrophe rate), "
             "single_exp_fit (median error, win rate) and the best rank-eligible accelerator per curve family, with the families where each named method loses")


def f23():
    """rational_fit against every other member of LEAD and the last value: families
    and cells where it has the strictly lower median error."""
    comps = [m for m in LEAD if m != "rational_fit"] + ["last_value"]
    head = [r"Comparator & " + " & ".join(rf"\multicolumn{{4}}{{c}}{{$g = {gname(g)}$}}" for g in STRATA) + r" \\",
            "".join(rf"\cmidrule(lr){{{2 + 4 * i}-{5 + 4 * i}}}" for i in range(len(STRATA))),
            r" & " + " & ".join(r"core fam. & core cells & h.-o. fam. & h.-o. cells" for _ in STRATA) + r" \\", r"\midrule"]
    rows = list(head)
    sec = "family head-to-head (rational_fit)"
    for m in comps:
        cells_, parts = [], []
        for g in STRATA:
            for hold, lab in ((0, "core"), (1, "held-out")):
                fam = FAMILY_TABLES[(g, hold)].med_error.unstack("method")
                cel = CELL_TABLES[(g, hold)]
                kf = nf = kc = nc = 0
                if "rational_fit" in fam.columns and m in fam.columns:
                    both = fam[["rational_fit", m]].dropna()
                    kf, nf = int((both["rational_fit"] < both[m]).sum()), int(len(both))
                if "rational_fit" in cel.columns and m in cel.columns:
                    both = cel[["rational_fit", m]].dropna()
                    kc, nc = int((both["rational_fit"] < both[m]).sum()), int(len(both))
                cells_ += [f"{kf}/{nf}", f"{kc}/{nc}"]
                parts.append(f"g={gname(g)} {lab}: families {kf}/{nf}, cells {kc}/{nc}")
        rows.append(f"{mth(m)} & " + " & ".join(cells_) + r" \\")
        fact(sec, f"rational_fit vs {m}: families / cells where rational_fit has the strictly lower median error (k/n, n = both finite)",
             "; ".join(parts), "results/phase1/phase1_aggregated.csv", f"capped == 0, methods rational_fit and {m}, per regime_set and target_g",
             "families: median over the family's cells of med_error; cells: the cell med_error; count(rational_fit < comparator) over the pairs where both are finite")
    fact(sec, "LEAD (the leading methods)", ", ".join(LEAD), "results/phase1/phase1_global.csv; results/phase1/phase1_global_holdout.csv",
         f"target_g == {HEADLINE_G}, is_trivial == 0, rank_eligible == 1", LEAD_RULE)
    frag("f23_family_head_to_head.tex", "l" + "r" * (4 * len(STRATA)), rows,
         ["results/phase1/phase1_aggregated.csv (per (method, regime, noise, g) cell; capped == 0)"],
         f"capped == 0; comparators = every other member of LEAD ({LEAD_RULE}) and last_value; fam. = curve families where rational_fit's "
         "family median error (median over the family's cells of med_error) is strictly below the comparator's, k/n with n = families where both are "
         "finite; cells = the same over the (family, noise) cells with the cell med_error",
         "Family head-to-head: curve families and cells where rational_fit has the strictly lower median error than each other leading method "
         "and the last value, per stratum, core and held-out")


def roster_table():
    """real_data_roster_v2.csv and its provenance (None, None when absent)."""
    path = os.path.join(RES, "real_data", "real_data_roster_v2.csv")
    if not os.path.exists(path):
        return None, None
    prov = None
    pp = os.path.join(RES, "real_data", "real_data_roster_provenance.json")
    if os.path.exists(pp):
        with open(pp, encoding="utf-8") as fh:
            prov = json.load(fh)
    return pd.read_csv(path), prov


def roster_header(prov):
    """The extra comment line of the roster-based fragments."""
    if prov is None:
        return ["roster  : results/real_data/real_data_roster_provenance.json absent"]
    return [f"roster  : results/real_data/real_data_roster_provenance.json (evaluation started {prov.get('started')} on code "
            f"{prov.get('git_head', {}).get('short')}, {prov.get('rows')} rows; added after the full run)"]


def _real_block(sub):
    """valid k/n, wins k/n, catastrophic k/n and the median error ratio to the last value (median of skill_vs_last over the valid rows)."""
    n = int(len(sub))
    ok = sub[sub.valid == 1]
    return dict(n=n, valid=int(sub.valid.sum()), win=int(sub.win_vs_last.sum()), cat=int(sub.catastrophic.sum()),
                ratio=float(ok.skill_vs_last.median()) if len(ok) else float("nan"),
                mi=float(ok.improve_ratio.median()) if len(ok) else float("nan"))


def f24():
    """The leading methods side by side: core, held-out, L_hat sensitivity, recorded curves."""
    Cn = sweep1_consumers()
    R, prov = roster_table()
    head = [r"Method & \multicolumn{5}{c}{core} & \multicolumn{5}{c}{held-out} & $\hat L$ sens. & \multicolumn{3}{c}{recorded curves} \\",
            r"\cmidrule(lr){2-6}\cmidrule(lr){7-11}\cmidrule(lr){13-15}",
            r" & $r$ & med.\ err & win vs last & $\rhoC$ & $\rhoV$ & $r$ & med.\ err & win vs last & $\rhoC$ & $\rhoV$ & "
            r"& wins pre & wins post & ratio pre \\", r"\midrule"]
    rows = list(head)
    sec = "leading fits"
    for m in LEAD:
        cells_, parts = [], []
        for G, RK, lab in ((G_CORE, RANK_CORE, "core"), (G_HOLD, RANK_HOLD, "held-out")):
            r = row_of(G, HEADLINE_G, m)
            rk = RK[HEADLINE_G].get(m, float("nan"))
            cells_ += [fint(rk), f4(float(r.med_error)), f3(float(r.win_rate_vs_last)), f3(float(r.cat_rate)), f3(float(r.valid_rate))]
            parts.append(f"{lab}: rank {fint(rk)}, med. err {float(r.med_error):.4f}, win vs last {float(r.win_rate_vs_last):.3f}, "
                         f"cat {float(r.cat_rate):.3f}, valid {float(r.valid_rate):.3f}")
        if m in LHAT_CONSUMERS:
            ch = lhat_change(Cn, m, "core", HEADLINE_G) if Cn is not None else None
            sens = pct(ch[1] / 100, 1) if ch is not None and math.isfinite(ch[1]) else "--"
            parts.append(f"L_hat sensitivity {ch[1]:.1f} %" if ch is not None else "L_hat sensitivity --")
        else:
            sens = "indep."
            parts.append("L_hat independent")
        cells_.append(sens)
        if R is not None and m in set(R.method):
            pre, post = _real_block(R[(R.method == m) & (R.post_min_target == 0)]), _real_block(R[(R.method == m) & (R.post_min_target == 1)])
            cells_ += [f"{pre['win']}/{pre['n']}", f"{post['win']}/{post['n']}", f3(pre["ratio"])]
            parts.append(f"recorded curves: wins pre {pre['win']}/{pre['n']}, post {post['win']}/{post['n']}, median error ratio pre {pre['ratio']:.3f}")
        else:
            cells_ += ["--"] * 3
        rows.append(f"{dag(m)}{mth(m)} & " + " & ".join(cells_) + r" \\")
        fact(sec, f"{m} at g={gname(HEADLINE_G)}", "; ".join(parts),
             "results/phase1/phase1_global.csv; results/phase1/phase1_global_holdout.csv; results/phase5b/phase5b_sweep1_consumers.csv; results/real_data/real_data_roster_v2.csv",
             f"target_g == {HEADLINE_G}; method == {m}",
             "rank / med_error / win_rate_vs_last / cat_rate / valid_rate of the pooled tables; L_hat sensitivity = 100 * (max - min) / min of med_error "
             "across the assumed modes (core); wins = count(win_vs_last) over the pre- / post-minimum cells of the roster file; ratio = median of skill_vs_last over the valid pre-minimum rows")
    frag("f24_leading_fits.tex", "l" + "r" * 10 + "r" + "rrr", rows,
         ["results/phase1/phase1_global.csv (core regimes, all three strata, capped cells excluded)",
          "results/phase1/phase1_global_holdout.csv (held-out regimes)",
          "results/phase5b/phase5b_sweep1_consumers.csv (Phase 5b sweep 1b; the assumed-asymptote sensitivity)",
          "results/real_data/real_data_roster_v2.csv (the recorded curves, every method under the benchmark protocol)"],
         f"target_g == {gname(HEADLINE_G)}; members of LEAD ({LEAD_RULE}); r = rank among the rank-eligible accelerators; L_hat sens. = max change of the "
         "core median error across the five assumed-asymptote modes as a percentage of the minimum (the sweep-1b invariance rows), 'indep.' for a method "
         "whose output does not depend on the assumed asymptote; recorded curves: wins vs the last value before and after the curve's minimum (k/n) and "
         "the median ratio of the method's error to the last value's over the valid pre-minimum cells",
         "The leading fits side by side at the headline stratum: core and held-out rank, median error, win rate vs the last value, catastrophe and validity "
         "rates, sensitivity to the assumed asymptote, and the recorded curves",
         extra=roster_header(prov))


def f25():
    """Every method on the recorded curves (real_data_roster_v2.csv)."""
    R, prov = roster_table()
    if R is None:
        print("  (real_data_roster_v2.csv absent: fragment f25 skipped; produced by scripts/run_real_data.py --roster-only)")
        return
    from src.accelerators import METHOD_NAMES
    from src.trivial import TRIVIAL_METHOD_NAMES
    fams = list(dict.fromkeys(FAMILY[m] for m in METHOD_NAMES if m not in TRIVIAL_METHOD_NAMES))     # the family order of f20
    accs = [m for fam in fams for m in ACCEL_METHODS if FAMILY[m] == fam and m in set(R.method)]
    trivs = [m for m in SKILL_REFERENCE_METHODS if m in set(R.method)]
    head = [r"Method & family & T & \multicolumn{4}{c}{before the minimum} & \multicolumn{4}{c}{after the minimum} & \multicolumn{2}{c}{all cells} \\",
            r"\cmidrule(lr){4-7}\cmidrule(lr){8-11}\cmidrule(lr){12-13}",
            r" & & & valid & wins & catastr. & ratio & valid & wins & catastr. & ratio & wins & strictly valid \\", r"\midrule"]
    rows = list(head)
    sec = "recorded curves, roster"
    blocks = {}
    last_fam = None
    for m in accs + trivs:
        if m == trivs[0]:
            rows.append(r"\midrule")
        elif last_fam is not None and FAMILY[m] != last_fam:
            rows.append(r"\addlinespace[2pt]")
        last_fam = FAMILY[m]
        sub = R[R.method == m]
        pre, post, allc = _real_block(sub[sub.post_min_target == 0]), _real_block(sub[sub.post_min_target == 1]), _real_block(sub)
        strict = int(sub.valid_strict.sum())
        blocks[m] = dict(pre=pre, post=post, all=allc, strict=strict)
        rows.append(f"{dag(m)}{mth(m)} & {fam_label(FAMILY[m])} & {TYPE_MACRO[METHOD_TYPE[m]]} & "
                    f"{pre['valid']}/{pre['n']} & {pre['win']}/{pre['n']} & {pre['cat']}/{pre['n']} & {f3(pre['ratio'])} & "
                    f"{post['valid']}/{post['n']} & {post['win']}/{post['n']} & {post['cat']}/{post['n']} & {f3(post['ratio'])} & "
                    f"{allc['win']}/{allc['n']} & {strict}/{allc['n']} \\\\")
    n_pre = blocks[accs[0]]["pre"]["n"] if accs else 0
    every = [m for m in accs if blocks[m]["pre"]["win"] == n_pre and n_pre > 0]
    fact(sec, "accelerators that win against the last value on every pre-minimum cell", f"{len(every)} of {len(accs)}: " + (", ".join(every) or "none"),
         "results/real_data/real_data_roster_v2.csv", "post_min_target == 0, is_trivial == 0", f"count(win_vs_last) == {n_pre} (the pre-minimum cells)")
    if "rational_fit" in blocks:
        for split, lab in (("pre", "before the minimum"), ("post", "after the minimum"), ("all", "over all cells")):
            ref = blocks["rational_fit"][split]["win"]
            more = [m for m in accs if blocks[m][split]["win"] > ref]
            fact(sec, f"accelerators with more wins against the last value than rational_fit {lab} (rational_fit: {ref}/{blocks['rational_fit'][split]['n']})",
                 f"{len(more)} of {len(accs)}: " + (", ".join(f"{m} {blocks[m][split]['win']}" for m in more) or "none"),
                 "results/real_data/real_data_roster_v2.csv", {"pre": "post_min_target == 0", "post": "post_min_target == 1", "all": "all rows"}[split] + ", is_trivial == 0",
                 "count(win_vs_last) per method, compared with rational_fit's")
    dis = R[R.valid != R.valid_strict]
    per_m = dis.groupby("method").size().sort_values(ascending=False)
    fact(sec, "records on which the two validity rules disagree (benchmark: finite and in [MIN_VALID, MAX_VALID]; legacy strict: finite and in [0, 2 x window max])",
         f"{len(dis)} of {len(R)} records; per method: " + (", ".join(f"{m} {n}" for m, n in per_m.items()) or "none"),
         "results/real_data/real_data_roster_v2.csv", "valid != valid_strict", "count per method")
    frag("f25_real_roster.tex", "lll" + "rrrr" * 2 + "rr", rows,
         ["results/real_data/real_data_roster_v2.csv (the six recorded XGBoost curves on the (depth x target) grid; every accelerator and the four deployable "
          "trivial predictors under the benchmark's configuration and per-record scoring, src.evaluation)"],
         f"all rows; accelerators in the family order of f20, then the four trivial predictors as comparators; dagger = the excluded set; T = method type; "
         f"before / after the minimum = post_min_target 0 / 1 (target round beyond the recorded curve's argmin); valid = benchmark validity (finite and in "
         f"[{C.MIN_VALID:g}, {C.MAX_VALID:g}]), k/n; wins = cells with win_vs_last == 1 (an invalid record never wins); catastr. = invalid or error above "
         f"{C.CAT_MULT:g} x the last value's; ratio = median over the valid cells of the error ratio to the last value's (skill_vs_last); strictly valid = the "
         "legacy rule (finite and in [0, 2 x window max]) over all cells",
         "The recorded curves, every method: validity, wins against the last value, catastrophes and the median error ratio before and after the "
         "curve minimum, wins over all cells and the strict-validity count",
         extra=roster_header(prov))


def f26():
    """The classical variants on the recorded curves."""
    R, prov = roster_table()
    if R is None:
        print("  (real_data_roster_v2.csv absent: fragment f26 skipped)")
        return
    splits = [("pre-minimum", 0), ("post-minimum", 1), ("all", None)]
    head = [r"Family & Method & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{lab} cells}}" for lab, _ in splits) + r" \\",
            "".join(rf"\cmidrule(lr){{{3 + 3 * i}-{5 + 3 * i}}}" for i in range(len(splits))),
            r" & & " + " & ".join("MI & wins & valid" for _ in splits) + r" \\", r"\midrule"]
    rows = list(head)
    sec = "recorded curves, classical variants"
    N = len(CLASSICAL)
    stats = {lab: {} for lab, _ in splits}
    last_fam = None
    for m in CLASSICAL:
        fam = FAM[m]
        if last_fam is not None and fam != last_fam:
            rows.append(r"\addlinespace[2pt]")
        last_fam = fam
        cells_ = []
        for lab, flag in splits:
            sub = R[R.method == m] if flag is None else R[(R.method == m) & (R.post_min_target == flag)]
            b = _real_block(sub)
            stats[lab][m] = b
            ib = math.isfinite(b["mi"]) and NOOP_BAND[0] <= b["mi"] <= NOOP_BAND[1]
            cells_ += [(r"\textbf{" + f3(b["mi"]) + "}") if ib else f3(b["mi"]), f"{b['win']}/{b['n']}", f"{b['valid']}/{b['n']}"]
        rows.append(f"{fam_label(fam)} & {dag(m)}{mth(m)} & " + " & ".join(cells_) + r" \\")
    rows.append(r"\midrule")
    foot = {}
    for lab, _ in splits:
        st = stats[lab]
        n = max((b["n"] for b in st.values()), default=0)
        foot[lab] = dict(
            inband=int(sum(1 for b in st.values() if math.isfinite(b["mi"]) and NOOP_BAND[0] <= b["mi"] <= NOOP_BAND[1])),
            wband=int(sum(1 for b in st.values() if b["n"] and WIN_BAND[0] <= b["win"] / b["n"] <= WIN_BAND[1])),
            win_mean=float(np.mean([b["win"] / b["n"] for b in st.values() if b["n"]])) if n else float("nan"), n=n)
    rows.append(r"\multicolumn{2}{l}{in the $\pm 10\%$ band (MI in $[0.9, 1.1]$)} & " + " & ".join(f"{foot[lab]['inband']}/{N} & & " for lab, _ in splits) + r" \\")
    rows.append(r"\multicolumn{2}{l}{win rate vs last in $[0.4, 0.6]$ (coin flip)} & " + " & ".join(f" & {foot[lab]['wband']}/{N} & " for lab, _ in splits) + r" \\")
    rows.append(r"\multicolumn{2}{l}{mean win rate vs last} & " + " & ".join(f" & {f3(foot[lab]['win_mean'])} & " for lab, _ in splits) + r" \\")
    pre = R[R.post_min_target == 0]
    per_ds = []
    for d in sorted(pre.dataset.unique()):
        sub = pre[pre.dataset == d]
        k = 0
        for m in CLASSICAL:
            b = _real_block(sub[sub.method == m])
            k += int(math.isfinite(b["mi"]) and NOOP_BAND[0] <= b["mi"] <= NOOP_BAND[1])
        per_ds.append((d, k, int(sub.drop_duplicates(["obs_depth", "target_round"]).shape[0])))
    rows.append(r"\multicolumn{11}{l}{pre-minimum cells, variants in the band per dataset: " +
                "; ".join(f"{tt(d)} {k}/{N} ({n} cells)" for d, k, n in per_ds) + r"} \\")
    for lab, _ in splits:
        f_ = foot[lab]
        fact(sec, f"{lab} cells ({f_['n']}): classical variants with MI in [{NOOP_BAND[0]}, {NOOP_BAND[1]}]; with a win rate vs last_value in [{WIN_BAND[0]}, {WIN_BAND[1]}]; mean win rate",
             f"{f_['inband']} of {N}; {f_['wband']} of {N}; {f_['win_mean']:.3f}", "results/real_data/real_data_roster_v2.csv",
             {"pre-minimum": "post_min_target == 0", "post-minimum": "post_min_target == 1", "all": "all rows"}[lab] + f", family in {CLASSICAL_FAMILIES}",
             "MI = median over the valid rows of improve_ratio; win rate = count(win_vs_last) / cells; mean over the variants")
    fact(sec, "pre-minimum cells: classical variants in the MI band per dataset (datasets with at least one pre-minimum cell)",
         "; ".join(f"{d} {k} of {N} ({n} cells)" for d, k, n in per_ds), "results/real_data/real_data_roster_v2.csv",
         f"post_min_target == 0, per dataset, family in {CLASSICAL_FAMILIES}", "count(MI in the band) per dataset; MI = median over the valid rows of improve_ratio")
    frag("f26_real_classical.tex", "ll" + "rrr" * len(splits), rows,
         ["results/real_data/real_data_roster_v2.csv (the recorded curves, every method under the benchmark protocol)"],
         f"family in {CLASSICAL_FAMILIES} (the {N} classical variants); pre-minimum / post-minimum / all cells (post_min_target 0 / 1 / any); MI = median "
         f"over the valid rows of improve_ratio (last-value error / method error), bold = inside [{NOOP_BAND[0]}, {NOOP_BAND[1]}]; wins = cells with "
         "win_vs_last == 1 (k/n); valid = benchmark validity (k/n); footers: variants in the MI band, variants with a win rate inside the coin-flip band, "
         "the mean win rate, and the in-band count per dataset over the pre-minimum cells",
         "The classical variants on the recorded curves: median improvement factor, wins against the last value and validity before and after the "
         "curve minimum and over all cells",
         extra=roster_header(prov))



# ═════════════════════════════════════════════════════════════════════════════
# R9f Part B: choosing a method by trial (f27, f27b, f28) and the named methods
# on the recorded curves (f29, f30)
# ═════════════════════════════════════════════════════════════════════════════
POOL_LABEL = {"lead": lambda n: f"the leading fits ({n})", "named": lambda n: "the two named methods",
              "classical": lambda n: f"the classical variants ({n})", "all": lambda n: f"all accelerators ({n})"}
DESIGN_LABEL = {"one_pilot": "one pilot run", "many_pilots": "many pilot runs"}
LAST = "last_value"


def pool_label(name):
    try:
        return POOL_LABEL[name](len(POOLS[name]))
    except KeyError:
        raise KeyError(f"no printed label for the candidate pool {name!r}; add it to POOL_LABEL") from None


def selection_tables():
    """phase1_selection_global.csv and its provenance (None, None when absent).  The pools
    recorded in the provenance must equal the current definitions (scripts.selection_defs),
    otherwise the committed aggregate is stale: rerun scripts/derive_selection.py."""
    path = os.path.join(RES, "phase1", "phase1_selection_global.csv")
    if not os.path.exists(path):
        return None, None
    S = pd.read_csv(path, float_precision="round_trip")
    prov = None
    pp = os.path.join(RES, "phase1", "phase1_selection_provenance.json")
    if os.path.exists(pp):
        with open(pp, encoding="utf-8") as fh:
            prov = json.load(fh)
        for name in POOL_NAMES:
            assert prov["pools"].get(name) == POOLS[name], \
                f"the committed selection pool {name!r} differs from the current definition; rerun scripts/derive_selection.py"
    return S, prov


def selection_header(prov):
    """The extra comment line of the selection-based fragments."""
    if prov is None:
        return ["selection: results/phase1/phase1_selection_provenance.json absent"]
    return [f"selection: results/phase1/phase1_selection_provenance.json (analysis started {prov.get('started')} on code "
            f"{prov.get('git_head', {}).get('short')} from {prov.get('records_rows'):,} records; cells per design {prov.get('cells')}, "
            f"trials per design {prov.get('trials')}; added after the full run, evaluates no method)"]


def _kn(k, n):
    return f"{int(k)}/{int(n)}"


def _f27(design, name):
    S, prov = selection_tables()
    if S is None:
        print(f"  (phase1_selection_global.csv absent: fragment {name} skipped; produced by scripts/derive_selection.py)")
        return
    S = S[S.design == design]
    sec = "selection by trial (synthetic families)"
    design_text = (prov or {}).get("designs", {}).get(design, design)
    head = [r"Set & noise class & candidate pool & families & chosen valid & win vs default & default chosen & win vs last & "
            r"\multicolumn{2}{c}{families with win rate $> 0.5$} & \multicolumn{3}{c}{median error} & ratio to default \\",
            r"\cmidrule(lr){9-10}\cmidrule(lr){11-13}",
            r" & & & & & & & & vs default & vs last & chosen & default & last value & \\", r"\midrule"]
    rows = list(head)
    cols = ["chosen_valid_rate", "win_vs_default", "default_chosen", "win_vs_last"]

    def emit_fact(r, g, setlab):
        cls = noise_class_label(r.noise_class, tex=False)
        fact(sec, f"{design}, {setlab} g={gname(g)} {cls}, pool {r.pool}: families / cells / chosen valid / win vs default / default chosen / win vs last / "
                  "families with win rate > 0.5 vs default, vs last (k/n) / median error chosen, default, last value / median ratio to the default",
             f"{int(r.families)} / {int(r.cells)} / {r.chosen_valid_rate:.3f} / {r.win_vs_default:.3f} / {r.default_chosen:.3f} / {r.win_vs_last:.3f} / "
             f"{_kn(r.families_win_default_k, r.families_win_default_n)}, {_kn(r.families_win_last_k, r.families_win_last_n)} / "
             f"{r.med_error_chosen:.4f}, {r.med_error_default:.4f}, {r.med_error_last:.4f} / {r.med_ratio_vs_default:.3f}",
             "results/phase1/phase1_selection_global.csv",
             f"design == {design}, regime_set == {r.regime_set}, target_g == {g}, noise_class == {r.noise_class}, pool == {r.pool}",
             "means over the class's cells of the per-cell shares; medians over the cells of the per-cell medians (scripts/derive_selection.py)")

    for rs, setlab in (("core", "core"), ("holdout", "held-out")):
        Sg = S[(S.regime_set == rs) & (S.target_g == HEADLINE_G)]
        if Sg.empty:
            continue
        classes = sorted(Sg.noise_class.unique(), key=noise_class_order)
        first_set = True
        for cls in classes:
            Sc = Sg[Sg.noise_class == cls]
            for i, pool in enumerate([p for p in POOL_NAMES if p in set(Sc.pool)]):
                r = Sc[Sc.pool == pool].iloc[0]
                rows.append(f"{setlab if first_set else ''} & {noise_class_label(cls) if i == 0 else ''} & {pool_label(pool)} & {int(r.families)} & "
                            + " & ".join(f3(float(r[c])) for c in cols) + f" & {_kn(r.families_win_default_k, r.families_win_default_n)} & "
                            f"{_kn(r.families_win_last_k, r.families_win_last_n)} & {f4(float(r.med_error_chosen))} & {f4(float(r.med_error_default))} & "
                            f"{f4(float(r.med_error_last))} & {f3(float(r.med_ratio_vs_default))} \\\\")
                first_set = False
                emit_fact(r, HEADLINE_G, setlab)
            rows.append(r"\addlinespace[2pt]")
        if rs == "core":
            rows.append(r"\midrule")
    while rows and rows[-1] in (r"\addlinespace[2pt]", r"\midrule"):
        rows.pop()
    for g in [g for g in STRATA if g != HEADLINE_G]:
        for rs, setlab in (("core", "core"), ("holdout", "held-out")):
            Sg = S[(S.regime_set == rs) & (S.target_g == g) & (S.noise_class == NOISE_CLASS_ALL)]
            for pool in [p for p in POOL_NAMES if p in set(Sg.pool)]:
                emit_fact(Sg[Sg.pool == pool].iloc[0], g, setlab)
    frag(name, "lll" + "r" * 11, rows,
         ["results/phase1/phase1_selection_global.csv (scripts/derive_selection.py on the git-ignored phase1_records.csv: the (family, noise, g) cells "
          "with capped == 0, all seeds; one row per set, stratum, noise class, candidate pool and design)"],
         f"design == {design} ({design_text}); target_g == {gname(HEADLINE_G)}; {NOISE_CLASS_DEF}; 'all cells' pools the classes; candidate pools: "
         f"the leading fits = LEAD ({LEAD_RULE}: {', '.join(POOLS['lead'])}), the two named methods = {DEFAULT_METHOD} (the default) and "
         f"{CONSERVATIVE_METHOD}, the classical variants = the {len(POOLS['classical'])} members of the families {CLASSICAL_FAMILIES}, all accelerators = "
         f"the {len(POOLS['all'])} accelerators of the registry; a trial = one (pilot, final) choice on one cell; chosen valid = mean over the class's cells "
         f"of the share of trials whose chosen record is valid on the final run; win vs default (vs last) = mean over cells of the share of trials where "
         f"the chosen record is valid and its error is strictly below {DEFAULT_METHOD}'s (the last value's) on the same final run (an invalid chosen record "
         f"never wins; a valid one wins against an invalid reference); default chosen = mean over cells of the share of trials where the chosen method is "
         f"{DEFAULT_METHOD} itself (never a strict win); families with win rate > 0.5 = the families whose mean over their cells of the win share exceeds "
         f"0.5, k/n; median error = median over the class's cells of the cell median over the trials (chosen: over the trials where it is valid; default "
         f"and last value: over the final runs of the trials, so the 'all cells' values reproduce the pooled Phase 1 medians); ratio to default = median "
         f"over cells of the cell median of chosen error / {DEFAULT_METHOD} error over the trials where both are valid",
         f"Choosing a method by trial on the synthetic families, design {DESIGN_LABEL[design]} ({design}), headline stratum: per set, noise class and "
         f"candidate pool, how often the method chosen on the pilot run(s) is valid on the final run, beats the default and the last value there, and "
         f"the median errors of the chosen method, the default and the last value",
         extra=selection_header(prov))


def f27():
    _f27("one_pilot", "f27_selection_one_pilot.tex")
    _f27("many_pilots", "f27b_selection_many_pilots.tex")


def _score(sub, m):
    """A method's score on a set of recorded-curve cells: the median of skill_vs_last over the
    cells, an invalid record counted as +infinity (NaN when the method has no record there)."""
    s = sub[sub.method == m]
    if s.empty:
        return float("nan")
    v = np.where(s.valid.to_numpy() == 1, s.skill_vs_last.to_numpy(dtype=float), np.inf)
    return float(np.median(v))


def _choose(sub, members):
    """The member with the lowest score on the cells (ties: registry order); (method, score)."""
    best = None
    for m in sorted(members, key=registry_index):
        sc = _score(sub, m)
        if best is None or sc < best[1]:
            best = (m, sc)
    return best


def fscore(v):
    """A score: finite -> 3 decimals; +infinity (the median record invalid) -> the infinity sign."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "--"
    return r"$\infty$" if math.isinf(v) else f"{v:.3f}"


def f28():
    """Choosing a method by trial on the recorded curves (pre-minimum cells)."""
    R, prov = roster_table()
    if R is None:
        print("  (real_data_roster_v2.csv absent: fragment f28 skipped)")
        return
    pre = R[R.post_min_target == 0]
    datasets = [d for d in REAL_DATASETS if d in set(pre.dataset)] + sorted(set(pre.dataset) - set(REAL_DATASETS))
    designs = [("depth", "same curve, leave one depth out"), ("dataset", "another dataset, leave one dataset out")]
    sec = "selection by trial (recorded curves)"
    folds = {}
    for pool in POOL_NAMES:
        members = POOLS[pool]
        fd = []
        for d in datasets:
            sd = pre[pre.dataset == d]
            for ob in sorted(sd.obs_depth.unique()):
                train, test = sd[sd.obs_depth != ob], sd[sd.obs_depth == ob]
                if train.empty:
                    continue
                m, _ = _choose(train, members)
                fd.append(dict(dataset=d, depth=int(ob), chosen=m, chosen_score=_score(test, m), default_score=_score(test, DEFAULT_METHOD)))
        folds[("depth", pool)] = fd
        fs = []
        for d in datasets:
            train, test = pre[pre.dataset != d], pre[pre.dataset == d]
            if train.empty:
                continue
            m, _ = _choose(train, members)
            fs.append(dict(dataset=d, depth=None, chosen=m, chosen_score=_score(test, m), default_score=_score(test, DEFAULT_METHOD)))
        folds[("dataset", pool)] = fs
    head = [r"Design & candidate pool & folds & \multicolumn{3}{c}{chosen score vs default's} & chosen $< 1$ & \multicolumn{2}{c}{median score over folds} & methods chosen \\",
            r"\cmidrule(lr){4-6}\cmidrule(lr){8-9}",
            r" & & & below & equal & above & & chosen & default & \\", r"\midrule"]
    rows = list(head)
    for key, dlabel in designs:
        for i, pool in enumerate(POOL_NAMES):
            fd = folds[(key, pool)]
            n = len(fd)
            cs = np.array([f["chosen_score"] for f in fd], dtype=float)
            ds = np.array([f["default_score"] for f in fd], dtype=float)
            below, equal, above = int((cs < ds).sum()), int((cs == ds).sum()), int((cs > ds).sum())
            beats_last = int((cs < 1.0).sum())
            counts = {}
            for f in fd:
                counts[f["chosen"]] = counts.get(f["chosen"], 0) + 1
            chosen_txt = ", ".join(f"{mth(m)} ({k})" for m, k in sorted(counts.items(), key=lambda kv: (-kv[1], registry_index(kv[0]))))
            med_c = float(np.median(cs)) if n else float("nan")
            med_d = float(np.median(ds)) if n else float("nan")
            rows.append(f"{dlabel if i == 0 else ''} & {pool_label(pool)} & {n} & {below} & {equal} & {above} & {beats_last} & {fscore(med_c)} & {fscore(med_d)} & {chosen_txt} \\\\")
            fact(sec, f"{dlabel}, pool {pool}: folds; folds with the chosen score below / equal to / above the default's; folds with the chosen score below 1; "
                      "median over folds of the chosen score and of the default's; methods chosen (counts)",
                 f"{n}; {below} / {equal} / {above}; {beats_last}; {med_c:.4f} / {med_d:.4f}; " + ", ".join(f"{m} {k}" for m, k in sorted(counts.items(), key=lambda kv: (-kv[1], registry_index(kv[0])))),
                 "results/real_data/real_data_roster_v2.csv", f"post_min_target == 0; design {key}; pool {pool} ({len(POOLS[pool])} members)",
                 "score = median of skill_vs_last over the cells, an invalid record counted as +inf; chosen = argmin score on the training cells (ties: registry order)")
            for d in datasets:
                fds = [f for f in fd if f["dataset"] == d]
                if not fds:
                    continue
                if key == "depth":
                    val = "; ".join(f"depth {f['depth']}: {f['chosen']} (chosen {f['chosen_score']:.4f}, default {f['default_score']:.4f})" for f in fds)
                else:
                    f = fds[0]
                    val = f"{f['chosen']} (chosen {f['chosen_score']:.4f}, default {f['default_score']:.4f})"
                fact(sec, f"{dlabel}, pool {pool}, {d}: the chosen method(s) with the chosen and the default's score on the left-out cells", val,
                     "results/real_data/real_data_roster_v2.csv", f"post_min_target == 0, dataset == {d}; design {key}; pool {pool}",
                     "chosen on the pre-minimum cells of the other depths of the dataset (depth design) / of the other datasets (dataset design); scored on the left-out cells")
        if key == designs[0][0]:
            rows.append(r"\midrule")
    n_pre = int(pre.drop_duplicates(["dataset", "obs_depth", "target_round"]).shape[0])
    frag("f28_selection_recorded.tex", "llrrrrrrrl", rows,
         ["results/real_data/real_data_roster_v2.csv (the recorded curves, every method under the benchmark protocol)"],
         f"post_min_target == 0 (the {n_pre} cells before the curve minimum, datasets {', '.join(datasets)}); a method's score on a set of cells = the median of "
         f"skill_vs_last (error / the last value's error) over those cells with an invalid record counted as +infinity; same curve, leave one depth out = for "
         f"each dataset and observation depth, the chosen method is the pool member with the lowest score on the pre-minimum cells of the other depths of that "
         f"dataset (ties: registry order), evaluated by its score on the left-out depth's pre-minimum cells next to {DEFAULT_METHOD}'s score there; another "
         f"dataset, leave one dataset out = chosen on the pre-minimum cells of the other datasets, evaluated on the left-out dataset; below / equal / above = "
         f"folds where the chosen score is below / equal to / above the default's; chosen < 1 = folds where the chosen method beats the last value at the "
         f"median; candidate pools as in f27 (lead: {', '.join(POOLS['lead'])}; named: {DEFAULT_METHOD}, {CONSERVATIVE_METHOD}; classical: the "
         f"{len(POOLS['classical'])} variants; all: {len(POOLS['all'])} accelerators)",
         "Choosing a method by trial on the recorded curves, cells before the minimum: leave one depth out and leave one dataset out, per candidate pool, "
         "the chosen method's score against the default's and against the last value",
         extra=roster_header(prov))


def f29():
    """rational_fit and single_exp_fit on the recorded curves, per dataset and pooled, before and after the minimum."""
    R, prov = roster_table()
    if R is None:
        print("  (real_data_roster_v2.csv absent: fragment f29 skipped)")
        return
    NAMED = (DEFAULT_METHOD, CONSERVATIVE_METHOD)
    datasets = [d for d in REAL_DATASETS if d in set(R.dataset)] + sorted(set(R.dataset) - set(REAL_DATASETS))
    head = [r"Dataset & \multicolumn{2}{c}{cells} & " + " & ".join(rf"\multicolumn{{4}}{{c}}{{{mth(m)}}}" for m in NAMED) + r" \\",
            r"\cmidrule(lr){2-3}\cmidrule(lr){4-7}\cmidrule(lr){8-11}",
            r" & before & after & wins before & ratio before & wins after & ratio after & wins before & ratio before & wins after & ratio after \\", r"\midrule"]
    rows = list(head)
    sec = "named methods on the recorded curves"
    for d in datasets + ["pooled"]:
        sub = R if d == "pooled" else R[R.dataset == d]
        n_split = {flag: int(sub[sub.post_min_target == flag].drop_duplicates(["dataset", "obs_depth", "target_round"]).shape[0]) for flag in (0, 1)}
        cells_, parts = [str(n_split[0]), str(n_split[1])], []
        for m in NAMED:
            for flag, lab in ((0, "before"), (1, "after")):
                b = _real_block(sub[(sub.method == m) & (sub.post_min_target == flag)])
                if b["n"] == 0:
                    cells_ += ["--", "--"]
                    parts.append(f"{m} {lab}: no cell")
                else:
                    cells_ += [f"{b['win']}/{b['n']}", f3(b["ratio"])]
                    parts.append(f"{m} {lab}: wins {b['win']}/{b['n']}, ratio {b['ratio']:.4f} ({b['valid']}/{b['n']} valid)")
        if d == "pooled":
            rows.append(r"\midrule")
        rows.append(f"{tt(d) if d != 'pooled' else chr(92) + 'textbf{pooled}'} & " + " & ".join(cells_) + r" \\")
        fact(sec, f"{d}: cells before / after the minimum; per named method, before and after: wins vs the last value (k/n) and the median of skill_vs_last over the valid cells",
             f"{n_split[0]} / {n_split[1]}; " + "; ".join(parts), "results/real_data/real_data_roster_v2.csv",
             ("all datasets" if d == "pooled" else f"dataset == {d}") + f", method in {NAMED}, post_min_target 0 / 1",
             "wins = count(win_vs_last == 1); ratio = median over the valid rows of skill_vs_last (error / the last value's error)")
    frag("f29_real_named_by_dataset.tex", "l" + "rr" + "rrrr" * len(NAMED), rows,
         ["results/real_data/real_data_roster_v2.csv (the recorded curves, every method under the benchmark protocol)"],
         f"method in {NAMED}; one row per dataset and a pooled row; before / after = post_min_target 0 / 1 (target round beyond the recorded curve's argmin); "
         "cells = (depth, target) cells of the split; wins = cells with win_vs_last == 1 (k/n; an invalid record never wins); ratio = median over the valid "
         "cells of skill_vs_last (the method's error / the last value's); a split with no cell prints --",
         "The two named methods on the recorded curves, per dataset and pooled: cells, wins against the last value and the median error ratio before and "
         "after the curve minimum",
         extra=roster_header(prov))


def f30():
    """The datasets whose curve minimum is at the final recorded round: error as a share of the final loss by observation depth."""
    R, prov = roster_table()
    if R is None:
        print("  (real_data_roster_v2.csv absent: fragment f30 skipped)")
        return
    t_max = int(R.target_round.max())
    curves = read("real_data", "real_data_curves.csv")
    assert len(curves) == t_max, (len(curves), t_max)        # the largest target round is the final recorded round (rounds are 1-based, len(curves) of them)
    ds = [d for d in REAL_DATASETS if d in set(R[R.argmin_round == t_max].dataset)] + sorted(set(R[R.argmin_round == t_max].dataset) - set(REAL_DATASETS))
    if not ds:
        print("  (no recorded curve has its minimum at the final round: fragment f30 skipped)")
        return
    METHODS = (LAST, DEFAULT_METHOD, CONSERVATIVE_METHOD)
    sub = R[R.dataset.isin(ds) & (R.target_round == t_max) & R.method.isin(METHODS)]
    depths = sorted(sub.obs_depth.unique())
    head = [r"$\nobs$ & share of rounds observed & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{mth(m)}}}" for m in METHODS) + r" \\",
            "".join(rf"\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}" for i in range(len(METHODS))),
            r" & & " + " & ".join("median & max" for _ in METHODS) + r" \\", r"\midrule"]
    rows = list(head)
    sec = "final loss by depth (recorded curves)"
    for ob in depths:
        s = sub[sub.obs_depth == ob]
        cells_, parts = [f"{100 * ob / t_max:.0f}\\%"], []
        for m in METHODS:
            sm = s[s.method == m]
            ok = sm[sm.valid == 1]
            rel = 100 * ok.error / ok.true_val
            med, mx = (float(rel.median()), float(rel.max())) if len(ok) else (float("nan"), float("nan"))
            suffix = f" ({len(ok)}/{len(sm)} valid)" if len(ok) < len(sm) else ""
            cells_ += [(f"{med:.2f}\\%" if math.isfinite(med) else "--") + suffix, (f"{mx:.2f}\\%" if math.isfinite(mx) else "--") + suffix]
            parts.append(f"{m}: median {med:.3f} %, max {mx:.3f} %" + (f" ({len(ok)}/{len(sm)} valid)" if len(ok) < len(sm) else ""))
        rows.append(f"{int(ob)} & " + " & ".join(cells_) + r" \\")
        fact(sec, f"depth {int(ob)} (share of rounds observed {100 * ob / t_max:.1f} %): median and max over the datasets of 100 x error / true_val, per method",
             "; ".join(parts), "results/real_data/real_data_roster_v2.csv",
             f"dataset in ({', '.join(ds)}) (argmin_round == {t_max}, the final recorded round), target_round == {t_max}, obs_depth == {int(ob)}, method in {METHODS}, valid == 1",
             "100 * error / true_val per dataset; median and max over the datasets")
    rows.append(r"\midrule")
    rows.append(rf"\multicolumn{{{2 + 2 * len(METHODS)}}}{{l}}{{datasets whose minimum is at the final recorded round ({t_max}): " + ", ".join(tt(d) for d in ds)
                + f"; target = round {t_max}; share of rounds observed = $\\nobs / {t_max}$" + r"} \\")
    frag("f30_real_final_loss_by_depth.tex", "lr" + "rr" * len(METHODS), rows,
         ["results/real_data/real_data_roster_v2.csv (the recorded curves, every method under the benchmark protocol)",
          "results/real_data/real_data_curves.csv (the recorded curves: the final round)"],
         f"datasets whose curve minimum is at the final recorded round ({', '.join(ds)}: argmin_round == {t_max} == the largest target round == the last "
         f"recorded round); target_round == {t_max}; one row per observation depth; share of rounds observed = depth / {t_max}; per method the median and "
         f"the maximum over those {len(ds)} datasets of 100 x error / true_val (the recorded loss at the target round), over the valid records; when a record "
         "is invalid the count of valid ones is printed beside the value",
         f"Final loss by observation depth on the {len(ds)} recorded curves whose minimum is at the final round ({', '.join(ds)}): the error of the last value, "
         f"{DEFAULT_METHOD} and {CONSERVATIVE_METHOD} as a percentage of the final loss, median and maximum over the datasets",
         extra=roster_header(prov))


# ═════════════════════════════════════════════════════════════════════════════
# FACTS.md and the fragment index
# ═════════════════════════════════════════════════════════════════════════════
def write_facts(answer_lines):
    order = ["roster", "trivial baseline", "ranking", "family bootstrap (§5.4)", "skill summary", "classical no-op",
             "classical no-op by noise class", "per family (rational_fit, single_exp_fit)", "family head-to-head (rational_fit)", "leading fits",
             "selection by trial (synthetic families)",
             "order ladders (Phase 0b)", "order ladders under noise (Phase 0b)", "generalisation", "dangerous set",
             "richardson_3 validity by depth", "sigma = 0 cancellation NaNs", "invalid rates", "capped cells", "effective sample",
             "sweep 1 (assumed asymptote)", "Richardson failure (§7.1, Phase 2)", "selectors (Phase 3)",
             "regime classifier (§7.2)", "ensembles (Phase 5a)", "diagnostics (Phase 4)",
             "real data (v2)", "real data fixed methods (§8)", "real data (v2) perturbation diagnostic",
             "recorded curves, roster", "recorded curves, classical variants",
             "selection by trial (recorded curves)", "named methods on the recorded curves", "final loss by depth (recorded curves)",
             "real data (legacy 18-cell run)", "pipeline provenance"]
    secs = list(dict.fromkeys(order + [f["section"] for f in FACTS]))
    with open(ARGS.facts, "w", encoding="utf-8", newline="\n") as f:
        f.write("# FACTS.md -- headline numbers of the redesign-v2 results, with provenance\n\n")
        f.write(f"{PROV}.  Regenerate with `python scripts/make_paper_tables.py`.  "
                "Every row names the file, the filter applied to it and the formula; numbers in the paper come from here or from the "
                "fragments in `paper_fragments/`, never from hand-typing.  Facts drawn from the git-ignored raw per-record files are "
                "marked as such.\n\n")
        f.write("## Named facts (the review asked for these explicitly)\n\n")
        f.write("- **richardson_3 validity by depth**: see section *richardson_3 validity by depth* "
                "and the Phase-1-depth-only scope of the exclusion derivation under *dangerous set*.\n")
        f.write("- **sigma = 0 cancellation NaNs**: section *sigma = 0 cancellation NaNs*.\n")
        if SAME_AS_LEGACY:
            f.write("- **The excluded set equals the eight methods excluded under the previous criterion**: section *dangerous set*.\n")
        else:
            f.write(f"- **The excluded set differs from the previous eight: added {ADDED_VS_LEGACY}, removed {REMOVED_VS_LEGACY}**: "
                    "section *dangerous set*.\n")
        f.write("- **Capped-cell inventory**: section *capped cells* and fragment `f11_capped_block.tex`.\n")
        f.write("- **Oracle vs best deployable trivial vs best method, per stratum, core and held-out**: section *trivial baseline* and fragment `f01_trivial_baseline.tex`.\n")
        if answer_lines:
            f.write("- **Real-data perturbation diagnostic (fragment f08)**: " + answer_lines[0] + "\n")
        f.write("\n")
        for sec in secs:
            rows = [x for x in FACTS if x["section"] == sec]
            if not rows:
                continue
            f.write(f"## {sec}\n\n| fact | value | file | filter | formula |\n|---|---|---|---|---|\n")
            for x in rows:
                cells = [str(x[k]).replace("|", "\\|").replace("\n", " ") for k in ("fact", "value", "file", "filter", "formula")]
                f.write("| " + " | ".join(cells) + " |\n")
            f.write("\n")
    print(f"  wrote {rel(ARGS.facts)} ({len(FACTS)} facts)")


def write_index():
    """The fragment index paper_fragments/README.md: every fragment in the folder
    (scripts.analyze_by_ltrue.write_fragment_index, shared with the tercile script)."""
    n = write_fragment_index(OUT, RES)
    print(f"  wrote {rel(os.path.join(OUT, 'README.md'))} ({n} fragments in the folder)")


def main():
    print(f"make_paper_tables.py (redesign v2): results = {rel(RES)}, out = {rel(OUT)}, facts = {rel(ARGS.facts)}")
    print(f"  {PROV}; dangerous = {sorted(DANGEROUS)}")
    print(f"  raw-derived facts: {'results/raw_facts.csv (' + str(len(RAW_FACTS)) + ' rows)' if RAW_FACTS is not None else 'results/raw_facts.csv absent'}")
    print(f"  LEAD = {LEAD}")
    print(f"  pools: " + "; ".join(f"{k} ({len(v)})" for k, v in POOLS.items()))
    f01(); f02(); f03(); f04(); f04b(); f05(); f05b(); f06(); f07()
    answer = f08()
    f09(); f10(); f11(); f12()
    f14(); f15(); f16(); f18(); f19(); f20()
    f21(); f21b(); f22(); f23(); f24(); f25(); f26()
    f27(); f28(); f29(); f30()
    facts_phase2(); facts_phase3(); facts_classifier()
    facts_lhat_invariance_full_precision()
    named_facts()
    write_facts(answer)
    write_index()
    print(f"\nDONE: {len(FRAGMENTS)} fragments in {rel(OUT)}, {len(FACTS)} facts in {rel(ARGS.facts)}")


if __name__ == "__main__":
    main()
