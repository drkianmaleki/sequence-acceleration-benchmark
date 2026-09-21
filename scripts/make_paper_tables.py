#!/usr/bin/env python3
"""
make_paper_tables.py  (v3, redesign v2)
=======================================
Turn the committed results/ of the redesign-v2 pipeline into the LaTeX table
fragments of the paper and into FACTS.md, the list of every headline number
with its provenance (file, filter, formula).  Read-only over results/.

    python scripts/make_paper_tables.py                      # defaults below
    python scripts/make_paper_tables.py --results results --out paper_fragments --facts FACTS.md
    python scripts/make_paper_tables.py --no-raw             # skip the git-ignored raw files

Inputs are the per-stratum files (phase1_global.csv keyed by target_g,
phase2_*_g{g}.csv, phase5b_sweep*_global.csv keyed by target_g, ...).  The
legacy-named headline copies phase2_correlations.csv and phase2_rules.csv
are never read.  The three git-ignored raw per-record files
(phase1_records.csv, phase4_raw.csv, phase5a_raw.csv) are read only for
the facts that no committed aggregate can supply (richardson_3 validity by
depth; the record-level skill split of the fixed default); when they are
absent those facts are marked "not recomputed" in FACTS.md.

Fragments (paper_fragments/, one tabular per file, booktabs, no \\begin{table})
------------------------------------------------------------------------------
  f01_trivial_baseline.tex          trivial comparators vs accelerators, per stratum, core | held-out
  f02_ranking_g{0.5,0.1,0.02}.tex   main ranking (accelerators above the validity floor) + unranked block
  f03_skill_summary.tex             fraction of cells with median skill < 1 per family, per stratum, core | held-out
  f04_classical_noop.tex            the 23 classical variants vs the +/-10 % band, improvement factor and skill
  f05_sweep1_modes.tex              assumed-asymptote mode sweep, cascade rows (Phase 5b sweep 1a; clamped features)
  f05b_sweep1_consumers_{core,holdout}.tex   sweep 1b: L_hat-consuming accelerators + constant_assumed under every mode (when the file exists)
  f06_generalisation.tex            held-out vs core rank shift per method
  f07_real_data_v2.tex              real-data re-evaluation over the (depth x target) grid, with skill
  f07b_real_data_legacy18.tex       the preserved 18-cell legacy real-data run
  f08_real_perturb_diagnostic.tex   perturb_iqr of the routed method vs cell failure: all cells, pre-/post-minimum targets, by routed method (AUC)
  f09a_selectors_phase3.tex         Phase 3 selectors, mean stability per stratum + leave-one-regime-out
  f09b_ensemble_phase5a_core.tex    Phase 5a selectors / ensembles, core, error and skill per stratum
  f09c_ensemble_phase5a_holdout.tex Phase 5a, held-out regimes
  f09d_ablation_phase5a.tex         Phase 5a ablation comparisons per stratum
  f10a_diagnostics_correlations.tex Phase 4 diagnostic-error correlations, core pooled | held-out
  f10b_diagnostics_depth.tex        Phase 4 perturb_iqr correlation by observation depth (headline stratum)
  f10c_diagnostics_ensemble.tex     Phase 4 selectors with trivial references, error and skill
  f10d_diagnostics_filter.tex       Phase 4 cascade + perturb_iqr screen
  f11_capped_block.tex              every capped (regime x stratum) cell with achieved_g; depth-grid counts

Macros used (defined in the paper preamble): \\meth{}, \\diag{}, \\Stab,
\\rhoC, \\rhoV, \\TE, \\LE, \\nobs, \\nf.
"""

import argparse
import datetime as _dt
import json
import math
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import src.config as C                                    # noqa: E402
from src.pipeline import ACCEL_METHODS, TRIVIAL_NON_ORACLE   # noqa: E402
from src.trivial import ORACLE_METHODS, SKILL_REFERENCE_METHODS  # noqa: E402

STRATA = list(C.HORIZON_GAP_FRACTIONS)          # [0.5, 0.1, 0.02]
HEADLINE_G = float(C.HEADLINE_G)                # 0.1
ORACLE = "constant_oracle"
DEPLOYABLE = list(SKILL_REFERENCE_METHODS)      # constant_assumed, last_value, window_mean, window_min
CLASSICAL_FAMILIES = ("shanks", "wynn_eps", "wynn_rho", "levin", "brezinski", "weniger", "anderson")
NOOP_BAND = (0.9, 1.1)                          # median improvement factor within +/-10 % of 1
SKILL_NOOP = 0.9                                # median skill >= 0.9: no better than 10 % over the best trivial
FAMILY_ORDER = ["richardson", "parametric", "pade", "neville", "baseline", "ensemble",
                "shanks", "wynn_eps", "wynn_rho", "levin", "brezinski", "weniger", "anderson", "trivial"]
TYPE_MACRO = {"trajectory": r"\TE", "limit": r"\LE"}
REAL_DATASETS = ["adult", "bank_marketing", "covertype", "higgs", "jannis", "miniboone"]


# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Paper table fragments + FACTS.md (redesign v2)")
    p.add_argument("--results", default=os.path.join(_ROOT, "results"))
    p.add_argument("--out", default=os.path.join(_ROOT, "paper_fragments"))
    p.add_argument("--facts", default=os.path.join(_ROOT, "FACTS.md"))
    p.add_argument("--no-raw", action="store_true",
                   help="do not read the git-ignored raw per-record files")
    return p.parse_args()


ARGS = parse_args()
RES = ARGS.results
OUT = ARGS.out
os.makedirs(OUT, exist_ok=True)


def _git(*cmd):
    try:
        return subprocess.check_output(["git"] + list(cmd), cwd=_ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


CODE_HEAD = _git("rev-parse", "--short", "HEAD") + ("-dirty" if _git("status", "--porcelain") else "")
RESULTS_HEAD = _git("log", "-1", "--format=%h", "--", "results")
STAMP = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
PROV = (f"generated by scripts/make_paper_tables.py at code {CODE_HEAD}, "
        f"results as of commit {RESULTS_HEAD}")
PROV_STAMPED = PROV + f", {STAMP}"


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


def frag(name, colspec, rows, sources, filters, description, notes=()):
    """Write one booktabs tabular with a provenance comment header."""
    path = os.path.join(OUT, name)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("% AUTO-GENERATED -- do not hand-edit.  " + PROV + "\n")
        f.write("% " + description + "\n")
        for s in sources:
            f.write(f"% source : {s}\n")
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
FAM = G_CORE.drop_duplicates("method").set_index("method")["family"].to_dict()
TYP = G_CORE.drop_duplicates("method").set_index("method")["method_type"].to_dict()
assert set(ACCEL_METHODS) == set(G_CORE[G_CORE.is_trivial == 0].method), "accelerator roster mismatch"
assert len(ACCEL_METHODS) == 51 and len(DEPLOYABLE) == 4
CLASSICAL = sorted([m for m in ACCEL_METHODS if FAM[m] in CLASSICAL_FAMILIES],
                   key=lambda m: (CLASSICAL_FAMILIES.index(FAM[m]), m))
assert len(CLASSICAL) == 23, len(CLASSICAL)

SRC_G = ["results/phase1/phase1_global.csv (core regimes, all three strata, capped cells excluded)",
         "results/phase1/phase1_global_holdout.csv (held-out regimes)"]


def dag(m):
    return r"$\dagger$" if m in DANGEROUS else ""


def rank_accelerators(G):
    """Rank among the 51 accelerators only, per stratum, med_error ascending,
    over rank-eligible rows (valid_rate >= RANK_MIN_VALID, finite metric)."""
    out = {}
    for g in STRATA:
        s = G[(G.target_g == g) & (G.is_trivial == 0) & (G.rank_eligible == 1)]
        s = s.sort_values(["med_error", "method"])
        out[g] = {m: i + 1 for i, m in enumerate(s.method)}
    return out


RANK_CORE = rank_accelerators(G_CORE)
RANK_HOLD = rank_accelerators(G_HOLD)


def row_of(G, g, m):
    s = G[(G.target_g == g) & (G.method == m)]
    return s.iloc[0] if len(s) else None


# ═════════════════════════════════════════════════════════════════════════════
# F01  trivial baseline -- THE PAPER'S FIRST RESULT
# ═════════════════════════════════════════════════════════════════════════════
def f01():
    rows = [r"Stratum / row & \multicolumn{4}{c}{core (18 regimes)} & \multicolumn{4}{c}{held-out (6 regimes)} \\",
            r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
            r" & med.\ err & med.\ skill & $\rhoV$ & $\rhoC$ & med.\ err & med.\ skill & $\rhoV$ & $\rhoC$ \\",
            r"\midrule"]
    for g in STRATA:
        core = G_CORE[G_CORE.target_g == g]
        hold = G_HOLD[G_HOLD.target_g == g]
        rows.append(mid(rf"\textit{{$g = {gname(g)}$}}" + (r" (headline)" if g == HEADLINE_G else ""), 9))
        dep_c = core[core.method.isin(DEPLOYABLE)].sort_values("med_error")
        for m in dep_c.method:
            rc, rh = row_of(G_CORE, g, m), row_of(G_HOLD, g, m)
            rows.append(f"{mth(m)} & {f4(rc.med_error)} & {f3(rc.med_skill)} & {f3(rc.valid_rate)} & {f3(rc.cat_rate)} & "
                        f"{f4(rh.med_error)} & {f3(rh.med_skill)} & {f3(rh.valid_rate)} & {f3(rh.cat_rate)} \\\\")
        rc, rh = row_of(G_CORE, g, ORACLE), row_of(G_HOLD, g, ORACLE)
        rows.append(f"{mth(ORACLE)} \\textit{{(reference, not deployable)}} & {f4(rc.med_error)} & {f3(rc.med_skill)} & "
                    f"{f3(rc.valid_rate)} & {f3(rc.cat_rate)} & {f4(rh.med_error)} & {f3(rh.med_skill)} & "
                    f"{f3(rh.valid_rate)} & {f3(rh.cat_rate)} \\\\")
        rows.append(r"\addlinespace[2pt]")
        stats = {}
        for label, G, RK in (("core", G_CORE, RANK_CORE), ("holdout", G_HOLD, RANK_HOLD)):
            s = G[G.target_g == g]
            acc = s[s.is_trivial == 0]
            dep = s[s.method.isin(DEPLOYABLE)]
            best_dep_m = dep.loc[dep.med_error.idxmin(), "method"]
            best_dep_e = float(dep.med_error.min())
            best_m = min(RK[g], key=RK[g].get)
            stats[label] = dict(
                best_dep_m=best_dep_m, best_dep_e=best_dep_e, best_m=best_m,
                oracle_e=float(s[s.method == ORACLE].med_error.iloc[0]),
                n_beat_err=int((acc.med_error < best_dep_e).sum()),
                n_beat_skill=int((acc.med_skill < 1.0).sum()),
                n_eligible=int(acc.rank_eligible.sum()),
                med_e=float(acc.med_error.median()), med_s=float(acc.med_skill.median()),
                med_v=float(acc.valid_rate.median()), med_c=float(acc.cat_rate.median()))
        for label, key in (("core", "core"), ("held-out", "holdout")):
            m = stats[key]["best_m"]
            rc, rh = row_of(G_CORE, g, m), row_of(G_HOLD, g, m)
            rows.append(f"best accelerator, {label} rank 1: {mth(m)} & {f4(rc.med_error)} & {f3(rc.med_skill)} & "
                        f"{f3(rc.valid_rate)} & {f3(rc.cat_rate)} & {f4(rh.med_error)} & {f3(rh.med_skill)} & "
                        f"{f3(rh.valid_rate)} & {f3(rh.cat_rate)} \\\\")
        sc, sh = stats["core"], stats["holdout"]
        rows.append(f"median accelerator (51) & {f4(sc['med_e'])} & {f3(sc['med_s'])} & {f3(sc['med_v'])} & {f3(sc['med_c'])} & "
                    f"{f4(sh['med_e'])} & {f3(sh['med_s'])} & {f3(sh['med_v'])} & {f3(sh['med_c'])} \\\\")
        rows.append(f"accelerators beating best deployable trivial & {sc['n_beat_err']}/51 & {sc['n_beat_skill']}/51 & & & "
                    f"{sh['n_beat_err']}/51 & {sh['n_beat_skill']}/51 & & \\\\")
        if g != STRATA[-1]:
            rows.append(r"\midrule")
        for key, label in (("core", "core"), ("holdout", "held-out")):
            st = stats[key]
            sec = "trivial baseline"
            flt = f"target_g == {g}, regime_set == {key}, capped cells excluded (pooled table)"
            fil = SRC_G[0] if key == "core" else SRC_G[1]
            fact(sec, f"g={gname(g)} {label}: best deployable trivial", f"{st['best_dep_m']} (med. err {st['best_dep_e']:.4f})",
                 fil, flt, "argmin med_error over {constant_assumed, last_value, window_mean, window_min}")
            fact(sec, f"g={gname(g)} {label}: constant_oracle med. err", f"{st['oracle_e']:.4f}", fil, flt, "med_error row of constant_oracle")
            bm = st["best_m"]
            bro = row_of(G_CORE if key == "core" else G_HOLD, g, bm)
            fact(sec, f"g={gname(g)} {label}: best accelerator", f"{bm} (med. err {bro.med_error:.4f}, med. skill {bro.med_skill:.3f})",
                 fil, flt + ", is_trivial == 0, rank_eligible == 1", "argmin med_error over rank-eligible accelerators")
            fact(sec, f"g={gname(g)} {label}: accelerators with med. err below best deployable trivial",
                 f"{st['n_beat_err']} of 51 ({st['n_eligible']} rank-eligible)", fil, flt + ", is_trivial == 0",
                 "count(med_error < min med_error of the four deployable trivials)")
            fact(sec, f"g={gname(g)} {label}: accelerators with pooled med. skill < 1", f"{st['n_beat_skill']} of 51", fil,
                 flt + ", is_trivial == 0", "count(med_skill < 1); skill = hindsight best-of-four (strict): err / best-of-four trivial error per cell, pooled median")
            fact(sec, f"g={gname(g)} {label}: median accelerator", f"med. err {st['med_e']:.4f}, med. skill {st['med_s']:.3f}",
                 fil, flt + ", is_trivial == 0", "median over the 51 accelerators of med_error / med_skill")
    frag("f01_trivial_baseline.tex", "l" + "r" * 8, rows, SRC_G,
         "per stratum; oracle excluded from every count; skill = hindsight best-of-four (strict): err / best-of-four deployable trivial per cell",
         "Trivial comparators vs accelerators: the four deployable trivials, the oracle constant as a labelled reference, "
         "the rank-1 accelerator of each regime set, the median accelerator, and how many of the 51 accelerators beat the best "
         "deployable trivial (count by pooled median error / count with pooled median skill < 1)",
         notes=["'accelerators beating best deployable trivial': first number = med_error below the best single deployable "
                "trivial's med_error; second = pooled med_skill < 1 (beats the per-cell best-of-four on the median cell)"])


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
            rows.append(f"{RANK_CORE[g][m]} & {dag(m)}{mth(m)} & {esc(FAM[m])} & {TYPE_MACRO[TYP[m]]} & "
                        f"{f4(rc.med_error)} & {f3(rc.med_skill)} & {f3(rc.cat_rate)} & {f3(rc.valid_rate)} & "
                        f"{fint(rh_rank)} & {f4(rh.med_error)} & {f3(rh.med_skill)} & {f3(rh.cat_rate)} & {f3(rh.valid_rate)} \\\\")
        below = G_CORE[(G_CORE.target_g == g) & (G_CORE.is_trivial == 0) & (G_CORE.rank_eligible == 0)]
        below = below.sort_values(["valid_rate", "method"], ascending=[False, True])
        rows.append(r"\midrule")
        rows.append(mid(rf"\textit{{Unranked: below the validity floor $\rhoV < {C.RANK_MIN_VALID:g}$ on the core regimes "
                        rf"({len(below)} methods; shown, never ranked)}}", 13))
        for _, rc in below.iterrows():
            m = rc.method
            rh = row_of(G_HOLD, g, m)
            rh_rank = RANK_HOLD[g].get(m, float("nan"))
            rows.append(f"-- & {dag(m)}{mth(m)} & {esc(FAM[m])} & {TYPE_MACRO[TYP[m]]} & "
                        f"{f4(rc.med_error)} & {f3(rc.med_skill)} & {f3(rc.cat_rate)} & {f3(rc.valid_rate)} & "
                        f"{fint(rh_rank)} & {f4(rh.med_error)} & {f3(rh.med_skill)} & {f3(rh.cat_rate)} & {f3(rh.valid_rate)} \\\\")
        frag(f"f02_ranking_g{gname(g)}.tex", "rlll" + "rrrr" + "rrrrr", rows, SRC_G,
             f"target_g == {g}; rank r = position by med_error among the 51 accelerators with rank_eligible == 1 "
             f"(valid_rate >= {C.RANK_MIN_VALID}), core and held-out ranked separately; trivial comparators not ranked "
             f"(see f01); dagger = in the dangerous artifact",
             f"Main ranking at g = {gname(g)}: rank pool = accelerators above the validity floor, core and held-out side by side, "
             f"unranked block appended")
        fact("ranking", f"g={gname(g)}: rank-eligible accelerators (core / held-out)",
             f"{len(RANK_CORE[g])} / {len(RANK_HOLD[g])} of 51", SRC_G[0], f"target_g == {g}, is_trivial == 0",
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
        label = esc(fam) + (r" \textit{(4 deployable)}" if fam == "trivial" else "")
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
    rows.append(r"\textbf{all accelerators} & 51 & & " + " & ".join(vals) + r" \\")
    for g in STRATA:
        for h, lab in ((0, "core"), (1, "held-out")):
            sub = acc[(acc.target_g == g) & (acc.is_holdout == h)]
            fact("skill summary", f"all accelerators: cells with median skill < 1, g={gname(g)}, {lab}",
                 f"{100 * sub.beats.mean():.1f}% ({int(sub.beats.sum())}/{len(sub)} cells)",
                 "results/phase1/phase1_aggregated.csv",
                 f"is_trivial == 0, target_g == {g}, is_holdout == {h}, capped == 0",
                 "mean over (method, regime, noise) cells of [med_skill < 1]")
    frag("f03_skill_summary.tex", "lrl" + "r" * 6, rows,
         ["results/phase1/phase1_aggregated.csv (per (method, regime, noise, g) cell: median skill over 30 seeds)"],
         "capped == 0, is_oracle == 0; a cell counts when its median-seed skill is < 1 (NaN median skill counts as not < 1)",
         "Skill summary: fraction of (method x regime x noise) cells whose median skill is below 1 (the method beat the "
         "hindsight best-of-four deployable trivial (strict) on the median seed), per family, per stratum, core vs held-out",
         notes=["K = methods in the family; T = method type (\\TE trajectory extrapolator / \\LE limit estimator)"])


# ═════════════════════════════════════════════════════════════════════════════
# F04  classical no-op table
# ═════════════════════════════════════════════════════════════════════════════
def f04():
    head = [r"Family & Method & \multicolumn{2}{c}{$g = 0.5$} & \multicolumn{2}{c}{$g = 0.1$} & \multicolumn{2}{c}{$g = 0.02$} \\",
            r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
            r" & & MI & med.\ skill & MI & med.\ skill & MI & med.\ skill \\", r"\midrule"]
    rows = list(head)
    inband = {g: 0 for g in STRATA}
    snoop = {g: 0 for g in STRATA}
    last_fam = None
    for m in CLASSICAL:
        fam = FAM[m]
        cells = []
        for g in STRATA:
            r = row_of(G_CORE, g, m)
            mi, sk = float(r.med_improve), float(r.med_skill)
            ib = math.isfinite(mi) and NOOP_BAND[0] <= mi <= NOOP_BAND[1]
            sn = math.isfinite(sk) and sk >= SKILL_NOOP
            inband[g] += int(ib)
            snoop[g] += int(sn)
            cells.append((r"\textbf{" + f3(mi) + "}") if ib else f3(mi))
            cells.append((r"\textbf{" + f3(sk) + "}") if sn else f3(sk))
        if last_fam is not None and fam != last_fam:
            rows.append(r"\addlinespace[2pt]")
        last_fam = fam
        rows.append(f"{esc(fam)} & {dag(m)}{mth(m)} & " + " & ".join(cells) + r" \\")
    rows.append(r"\midrule")
    rows.append(r"\multicolumn{2}{l}{in the $\pm 10\%$ band (MI in $[0.9, 1.1]$)} & " +
                " & ".join(f"{inband[g]}/23 & " for g in STRATA).rstrip(" &") + r" \\")
    rows.append(r"\multicolumn{2}{l}{median skill $\geq 0.9$} & " +
                " & ".join(f" & {snoop[g]}/23" for g in STRATA) + r" \\")
    for g in STRATA:
        fact("classical no-op", f"g={gname(g)} core: classical variants with MI in [0.9, 1.1]", f"{inband[g]} of 23", SRC_G[0],
             f"target_g == {g}, family in {CLASSICAL_FAMILIES}", "count(0.9 <= med_improve <= 1.1); MI = median of curr_err / err")
        fact("classical no-op", f"g={gname(g)} core: classical variants with median skill >= 0.9", f"{snoop[g]} of 23", SRC_G[0],
             f"target_g == {g}, family in {CLASSICAL_FAMILIES}", "count(med_skill >= 0.9)")
    frag("f04_classical_noop.tex", "ll" + "rr" * 3, rows, [SRC_G[0]],
         "core regimes, capped excluded; the 23 classical variants = families shanks, wynn_eps, wynn_rho, levin, brezinski, "
         "weniger, anderson; MI = med_improve (median over seeds and cells of current-value error / method error); "
         "bold MI = inside the +/-10 % band; bold skill = median skill >= 0.9",
         "Classical no-op table by stratum: median improvement factor vs the +/-10 % band, and median skill")


# ═════════════════════════════════════════════════════════════════════════════
# F05  assumed-asymptote mode sweep (Phase 5b sweep 1)
# ═════════════════════════════════════════════════════════════════════════════
def f05():
    S = read("phase5b", "phase5b_sweep1_global.csv")
    gs = [g for g in STRATA if g in set(S.target_g)]
    head = [r"Regime set & $\hat{L}$ mode & $\sigma$ & " +
            " & ".join(rf"\multicolumn{{4}}{{c}}{{$g = {gname(g)}$}}" for g in gs) + r" \\",
            "".join(rf"\cmidrule(lr){{{4 + 4 * i}-{7 + 4 * i}}}" for i in range(len(gs))),
            r" & & & " + " & ".join(r"fire & prec. & recall & gain" for _ in gs) + r" \\", r"\midrule"]
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
                        cells += [f3(r.fire_rate), f3(r.precision), f3(r.recall), signed(float(r.mean_gain))]
                    else:
                        cells += ["--"] * 4
                lab_rs = ("core" if rs == "core" else "held-out") if (i == 0 and j == 0) else ""
                lab_mode = (esc(mode) + (r" \textit{(oracle)}" if mode == "oracle" else "")) if j == 0 else ""
                rows.append(f"{lab_rs} & {lab_mode} & {nz:g} & " + " & ".join(cells) + r" \\")
            if mode != modes[-1]:
                rows.append(r"\addlinespace[1pt]")
        if rs == "core":
            rows.append(r"\midrule")
    core_h = S[(S.regime_set == "core") & (S.target_g == HEADLINE_G)]
    piv = core_h.pivot_table(index="noise", columns="assumed_mode", values="mean_gain")
    spread = float((piv.max(axis=1) - piv.min(axis=1)).max())
    nonzero = [m for m in modes if m != "zero"]
    piv_nz = piv[nonzero]
    spread_nz = float((piv_nz.max(axis=1) - piv_nz.min(axis=1)).max())
    fact("sweep 1 (assumed asymptote)", f"g={gname(HEADLINE_G)} core: max spread of cascade mean gain across the five L_hat modes",
         f"{spread:.6f} (across the four non-zero modes {spread_nz:.6f})", "results/phase5b/phase5b_sweep1_global.csv",
         f"regime_set == core, target_g == {HEADLINE_G}", "max over noise of (max - min over assumed_mode of mean_gain)")
    frag("f05_sweep1_modes.tex", "lll" + "rrrr" * len(gs), rows,
         ["results/phase5b/phase5b_sweep1_global.csv (Phase 5b sweep 1, pooled over regimes, capped excluded)"],
         "all rows; fire = cascade fire rate, prec./recall = precision/recall of the rational_fit rule against "
         "'rational_fit beats richardson_1', gain = mean error saved when the cascade fires",
         "Assumed-asymptote mode sweep: Phase-2 cascade metrics under each L_hat mode, core and held-out, per stratum",
         notes=["the cascade features clamp L0 = max(0, min(L_hat, 0.5 * min(window))), so the four non-zero modes coincide "
                "whenever L_hat >= 0.5 * min(window); see FACTS.md"])


# ═════════════════════════════════════════════════════════════════════════════
# F06  held-out vs core generalisation
# ═════════════════════════════════════════════════════════════════════════════
def f05b():
    path = os.path.join(RES, "phase5b", "phase5b_sweep1_consumers.csv")
    if not os.path.exists(path):
        print("  (phase5b_sweep1_consumers.csv absent: fragment f05b skipped; produced by a run at or after Prompt 5A)")
        return
    Cn = pd.read_csv(path)
    from phases.phase5b import SWEEP1_METHODS
    modes = [m for m in C.ASSUMED_L_MODES if m in set(Cn.assumed_mode)]
    gs = [g for g in STRATA if g in set(Cn.target_g)]
    head = [r"Method & mode & " + " & ".join(rf"\multicolumn{{4}}{{c}}{{$g = {gname(g)}$}}" for g in gs) + r" \\",
            "".join(rf"\cmidrule(lr){{{3 + 4 * i}-{6 + 4 * i}}}" for i in range(len(gs))),
            r" & & " + " & ".join(r"med.\ err & skill & win/assumed & $\rhoC$" for _ in gs) + r" \\", r"\midrule"]
    for rs in ("core", "holdout"):
        rows = list(head)
        for m in SWEEP1_METHODS:
            for i, mode in enumerate(modes):
                cells = []
                for g in gs:
                    r = Cn[(Cn.method == m) & (Cn.assumed_mode == mode) & (Cn.regime_set == rs) & (Cn.target_g == g)]
                    if len(r):
                        r = r.iloc[0]
                        cells += [f4(float(r.med_error)), f3(float(r.med_skill)), f3(float(r.win_rate_vs_assumed)), f3(float(r.cat_rate))]
                    else:
                        cells += ["--"] * 4
                rows.append(f"{mth(m) if i == 0 else ''} & {esc(mode)} & " + " & ".join(cells) + r" \\")
            rows.append(r"\addlinespace[1pt]")
        frag(f"f05b_sweep1_consumers_{rs}.tex", "ll" + "rrrr" * len(gs), rows,
             ["results/phase5b/phase5b_sweep1_consumers.csv (Phase 5b sweep 1b; capped excluded; pooled over noise)"],
             f"regime_set == {rs}; skill = hindsight best-of-four (strict); win/assumed = win rate vs constant_assumed under the same mode",
             f"Assumed-asymptote sweep 1b, {rs} regimes: the L_hat-consuming accelerators and constant_assumed under every mode")


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
    rhos = []
    for g in STRATA:
        both = [m for m in ACCEL_METHODS if m in RANK_CORE[g] and m in RANK_HOLD[g]]
        rho = spearmanr([RANK_CORE[g][m] for m in both], [RANK_HOLD[g][m] for m in both]).correlation
        rhos.append((g, rho, len(both)))
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
         "ranks among the 51 accelerators (rank-eligible only; -- = below the validity floor in that regime set); "
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
    fact("real data (v2)", "cells on the grid", f"{len(S)} = 6 datasets x 5 depths x 3 targets",
         "results/real_data/real_data_summary_v2.csv", "all rows", "row count")
    fact("real data (v2)", "median cascade skill over the 90 cells", f"{S.cascade_skill.median():.3f}",
         "results/real_data/real_data_summary_v2.csv", "all rows", "median(cascade_skill); skill = hindsight best-of-four (strict): cascade_err / best-of-four trivial error")
    fact("real data (v2)", "cells with cascade skill >= 1 (cascade no better than the best trivial)", f"{n_fail} of 90",
         "results/real_data/real_data_summary_v2.csv", "all rows", "count(cascade_skill >= 1)")
    fact("real data (v2)", "cells with negative improvement over the current value", f"{n_neg} of 90",
         "results/real_data/real_data_summary_v2.csv", "all rows", "count(improvement < 0); improvement = (current_err - cascade_err) / current_err")
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
         "all 90 cells; routed: R1 = richardson_1, RF = rational_fit; err = |cascade prediction - recorded value at the "
         "target round|; skill = hindsight best-of-four (strict): err / best-of-four trivial error on the same cell, bold when >= 1; a trailing + marks a post-minimum target (target round beyond the recorded curve's argmin)",
         "Real-data re-evaluation over the (depth x target) grid: routed method, cascade error and skill per cell")

    # legacy 18-cell run, preserved as its own fragment
    L = read("real_data", "real_data_results.csv")
    L["improvement"] = (L.current_err - L.cascade_err) / L.current_err
    rows = [r"Dataset & $\nobs$ & routed & predicted & cascade err & current err & improv. \\", r"\midrule"]
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
    fact("real data (legacy 18-cell run)", "reduction in mean error, all 18 cells", f"{red_all:+.1f}%",
         "results/real_data/real_data_results.csv", "all rows", "100 * (mean current_err - mean cascade_err) / mean current_err")
    fact("real data (legacy 18-cell run)", "cells worse than the current value", f"{int((L.improvement < 0).sum())} of 18",
         "results/real_data/real_data_results.csv", "all rows", "count(cascade_err > current_err)")
    rows.append(r"\multicolumn{7}{l}{reduction in mean error: " + ", ".join(parts) + f"; all cells ${red_all:+.1f}$\\%" + r"} \\")
    frag("f07b_real_data_legacy18.tex", "lrlrrrr", rows,
         ["results/real_data/real_data_results.csv (the pre-redesign 18-cell run: fixed target = final recorded round, "
          "legacy assumed asymptote 0.01; kept for the record, regenerable with scripts/analyze_real_diagnostics_legacy.py)"],
         "all 18 rows; bold = cascade worse than the current value",
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
        res[dexpr] = auc_row("all 90", S, fail, dlabel, dexpr)
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
        answer = (f"On the 90-cell grid the perturb_iqr of the routed method {verdict} failing cells "
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
             answer, "results/real_data/real_data_summary_v2.csv", "all 90 cells, perturb_iqr finite",
             "AUC = Mann-Whitney U(fail, succ) / (n_fail * n_succ); failure = cascade_skill >= 1 or improvement < 0")
    frag("f08_real_perturb_diagnostic.tex", "llrrrrrrl", rows,
         ["results/real_data/real_data_summary_v2.csv (perturb_iqr = IQR of the routed method's prediction over 5 "
          "evaluations on 2 %-perturbed windows, crc32 seed per (dataset, depth))"],
         "all 90 cells, then pre-minimum (target round <= argmin round of the recorded curve) and post-minimum targets, then by routed method; "
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
    CV = read("phase3", "phase3_cv_results.csv")
    cv_mean = CV.groupby("selector").mean_stability.mean()
    NAME3 = {"oracle": "oracle (regime known)", "phase2_cascade": "Phase-2 cascade", "enhanced_cascade": "enhanced cascade",
             "fixed_rational": r"fixed \meth{rational\_fit}", "fixed_richardson": r"fixed \meth{richardson\_1}",
             "fixed_single_exp": r"fixed \meth{single\_exp\_fit}", "fixed_current": "current value (baseline)"}
    order3 = ["oracle", "phase2_cascade", "enhanced_cascade", "fixed_rational", "fixed_richardson", "fixed_single_exp", "fixed_current"]
    rows = [r"Selector & " + " & ".join(rf"$\Stab$ ($g = {gname(g)}$)" for g in STRATA) + r" & $n$ & capped excl. & LORO $\Stab$ \\",
            r"\midrule"]
    for s in order3:
        vals = []
        for g in STRATA:
            r = SC[(SC.selector == s) & (SC.target_g == g)]
            vals.append(f4(float(r.mean_stability.iloc[0])) if len(r) else "--")
        rh = SC[(SC.selector == s) & (SC.target_g == HEADLINE_G)].iloc[0]
        cv = cv_mean.get(s, float("nan"))
        rows.append(f"{NAME3[s]} & " + " & ".join(vals) + f" & {int(rh.n)} & {int(rh.n_capped_excluded)} & {f4(float(cv)) if math.isfinite(cv) else '--'} \\\\")
    for s in ("oracle", "phase2_cascade", "fixed_rational", "fixed_richardson"):
        r = SC[(SC.selector == s) & (SC.target_g == HEADLINE_G)].iloc[0]
        fact("selectors (Phase 3)", f"{s}: mean achieved stability at g={gname(HEADLINE_G)}", f"{r.mean_stability:.4f} (n = {int(r.n)})",
             "results/phase3/phase3_selector_comparison.csv", f"selector == {s}, target_g == {HEADLINE_G}", "mean_stability column")
    frag("f09a_selectors_phase3.tex", "l" + "r" * (len(STRATA) + 3), rows,
         ["results/phase3/phase3_selector_comparison.csv (core regimes, capped cells excluded)",
          "results/phase3/phase3_cv_results.csv (leave-one-regime-out; mean over held-out regimes)"],
         "all selectors; n and capped-excluded count at the headline stratum; LORO = mean over the 18 leave-one-regime-out folds "
         "(selectors without a fold entry show --)",
         "Phase 3 selectors: mean achieved stability per stratum and under leave-one-regime-out cross-validation")

    # b/c) Phase 5a ensembles
    ORDER5 = [("oracle_51", "oracle over the 51-method pool"), ("constant_oracle", r"\meth{constant\_oracle} \textit{(reference)}"),
              ("oracle_9", "oracle over the 9-method pool"),
              ("fixed_rational", r"fixed \meth{rational\_fit}"), ("fixed_richardson", r"fixed \meth{richardson\_1}"),
              ("phase2_cascade", "Phase-2 cascade"),
              ("equal_ensemble_9", "equal ensemble (9)"), ("diag_ensemble_9", "diagnostic-weighted ensemble (9)"),
              ("equal_ensemble_51", "equal ensemble (51)"), ("diag_ensemble_51", "diagnostic-weighted ensemble (51)"),
              ("capped_diag_51", "capped diagnostic-weighted (51)"),
              ("equal_ensemble_safe", "equal ensemble (safe)"), ("diag_ensemble_safe", "diagnostic-weighted (safe)"),
              ("capped_diag_safe", "capped diagnostic-weighted (safe)"),
              ("threshold_ens_010", "threshold ensemble, IQR $\\leq 0.10$"), ("threshold_ens_050", "threshold ensemble, IQR $\\leq 0.50$"),
              ("threshold_ens_safe", "threshold ensemble, no threshold"),
              ("current_value", r"\meth{current\_value}"), ("constant_assumed", r"\meth{constant\_assumed}"),
              ("last_value", r"\meth{last\_value}"), ("window_min", r"\meth{window\_min}"), ("window_mean", r"\meth{window\_mean}")]
    for fname, path, label in (("f09b_ensemble_phase5a_core.tex", "phase5a_ensemble.csv", "core"),
                               ("f09c_ensemble_phase5a_holdout.tex", "phase5a_ensemble_holdout.csv", "held-out")):
        E = read("phase5a", path)
        head = [r"Selector & \multicolumn{2}{c}{$g = 0.5$} & \multicolumn{3}{c}{$g = 0.1$} & \multicolumn{2}{c}{$g = 0.02$} \\",
                r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}\cmidrule(lr){7-8}",
                r" & med.\ err & med.\ skill & mean err & med.\ err & med.\ skill & med.\ err & med.\ skill \\", r"\midrule"]
        rows = list(head)
        for i, (sel, name) in enumerate(ORDER5):
            cells = []
            for g in STRATA:
                r = E[(E.selector == sel) & (E.target_g == g)]
                if len(r):
                    r = r.iloc[0]
                    if g == HEADLINE_G:
                        cells += [f4(float(r.mean_error)), f4(float(r.median_error)), f3(float(r.med_skill))]
                    else:
                        cells += [f4(float(r.median_error)), f3(float(r.med_skill))]
                else:
                    cells += ["--"] * (3 if g == HEADLINE_G else 2)
            if sel in ("fixed_rational", "equal_ensemble_9", "current_value"):
                rows.append(r"\addlinespace[2pt]")
            rows.append(f"{name} & " + " & ".join(cells) + r" \\")
        for sel in ("oracle_51", "constant_oracle", "oracle_9", "fixed_rational", "fixed_richardson", "phase2_cascade", "equal_ensemble_9", "constant_assumed"):
            r = E[(E.selector == sel) & (E.target_g == HEADLINE_G)]
            if len(r):
                r = r.iloc[0]
                fact("ensembles (Phase 5a)", f"{sel} ({label}) at g={gname(HEADLINE_G)}",
                     f"mean err {r.mean_error:.6f}, median err {r.median_error:.6f}, med. skill {r.med_skill:.4f} (n = {int(r.n)})",
                     f"results/phase5a/{path}", f"selector == {sel}, target_g == {HEADLINE_G}",
                     "mean/median of the selector's absolute error over records; med_skill = hindsight best-of-four (strict), median over records of err / best-of-four trivial")
        frag(fname, "l" + "rr" + "rrr" + "rr", rows,
             [f"results/phase5a/{path} ({label} regimes; obs 30/60/90/120 x 3 noise x 20 seeds; capped cells excluded)"],
             "all selectors; error = |prediction - true value at n_f|; skill = hindsight best-of-four (strict): err / best-of-four trivial error per record, median over records",
             f"Phase 5a selectors and ensembles, {label} regimes: median error and median skill per stratum "
             f"(mean error at the headline stratum), oracle_51 vs constant_oracle vs the fixed defaults")

    # d) ablation
    AB = read("phase5a", "phase5a_ablation.csv")
    comps = list(dict.fromkeys(AB.comparison))
    rows = [r"Comparison (positive = first is better) & " + " & ".join(rf"$g = {gname(g)}$" for g in STRATA) + r" & $n$ \\", r"\midrule"]
    for cmp in comps:
        vals = []
        for g in STRATA:
            r = AB[(AB.comparison == cmp) & (AB.target_g == g)]
            vals.append(signed(float(r.mean_improvement.iloc[0]), 4) if len(r) else "--")
        n = int(AB[(AB.comparison == cmp) & (AB.target_g == HEADLINE_G)].n.iloc[0])
        rows.append(f"{esc(cmp)} & " + " & ".join(vals) + f" & {n} \\\\")
    frag("f09d_ablation_phase5a.tex", "l" + "r" * (len(STRATA) + 1), rows,
         ["results/phase5a/phase5a_ablation.csv (core regimes, capped excluded)"],
         "all comparisons; mean_improvement = mean over records of (error of the second selector - error of the first)",
         "Phase 5a ablation: paired mean error differences between selectors per stratum")


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
         "(shift_IQR is undefined for current_value, weniger_d2 and anderson_1 in the stored table)",
         "Phase 4 diagnostics: correlation of perturb_IQR and shift_IQR with error, core pooled and held-out")

    OB = read("phase4", "phase4_obs_reliability.csv")
    ob = OB[(OB.diagnostic == "perturb_iqr") & (OB.target_g == HEADLINE_G)]
    depths = sorted(ob.obs_idx.unique())
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

    E4 = read("phase4", "phase4_ensemble.csv")
    rows = [r"Selector & mean err & med.\ err & med.\ skill & $n$ \\", r"\midrule"]
    for _, r in E4.iterrows():
        lab = mth(r.selector) if r.is_trivial else esc(r.selector)
        rows.append(f"{lab}{' (trivial)' if r.is_trivial else ''} & {f4(float(r.mean_error))} & {f4(float(r.median_error))} & "
                    f"{f3(float(r.med_skill))} & {int(r.n)} \\\\")
    frag("f10c_diagnostics_ensemble.tex", "lrrrr", rows,
         ["results/phase4/phase4_ensemble.csv (core regimes, headline stratum, capped excluded)"],
         "all rows; phase2_proxy = the Phase-2 cascade evaluated inside Phase 4; diag_ensemble = perturb_IQR-weighted ensemble of the 9 methods",
         "Phase 4 selectors with the trivial references: error and skill at the headline stratum")

    CF = read("phase4", "phase4_cascade_filter.csv")
    rows = [r"Screen & mean err & med.\ err & $n$ \\", r"\midrule"]
    for _, r in CF.iterrows():
        rows.append(f"{esc(r['filter'])} & {f4(float(r.mean_error))} & {f4(float(r.median_error))} & {int(r.n)} \\\\")
    base = float(CF[CF['filter'] == 'no_filter'].mean_error.iloc[0])
    best = CF.loc[CF.mean_error.idxmin()]
    fact("diagnostics (Phase 4)", "perturb_IQR screen on the cascade: best threshold vs no filter (mean error)",
         f"{best['filter']}: {best.mean_error:.6f} vs no_filter {base:.6f} ({100 * (best.mean_error - base) / base:+.2f}%)",
         "results/phase4/phase4_cascade_filter.csv", "all rows", "mean_error column")
    frag("f10d_diagnostics_filter.tex", "lrrr", rows,
         ["results/phase4/phase4_cascade_filter.csv (core regimes, headline stratum)"],
         "all rows; a cell whose routed prediction has perturb_IQR above the threshold falls back to the current value",
         "Phase 4: the perturb_IQR screen applied to the Phase-2 cascade")


# ═════════════════════════════════════════════════════════════════════════════
# F11  capped block
# ═════════════════════════════════════════════════════════════════════════════
def f11():
    cells = (CAPPED1.groupby(["regime", "is_holdout", "target_g"], as_index=False)
             .agg(n_f=("n_f", "first"), achieved_g=("achieved_g", "first"), n_methods=("method", "nunique"), n_cells=("n", "first")))
    hz = HORIZONS.set_index(["regime", "target_g"])
    rows = [r"Regime & set & $g$ & $\nf$ & achieved $g$ & gap$(\nobs)$ & gap$(\nf)$ & note \\", r"\midrule"]
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
    rows.append(r"\multicolumn{8}{l}{Capped cells in the depth grids (regime $\times$ depth $\times$ $g$): " +
                "; ".join(f"{ph}: {n} over {dep} depths" for ph, n, dep, _ in counts) +
                f"; Phase 3: {len(c3)} (regime, depth, noise, $g$) cells" + r"} \\")
    rows.append(r"\multicolumn{8}{l}{Regimes ever capped in the depth grids: " +
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
# Named facts that need no fragment
# ═════════════════════════════════════════════════════════════════════════════
def named_facts():
    sec = "dangerous set"
    fact(sec, "dangerous accelerators (redesign v2, derived from the full Phase 1)", ", ".join(sorted(DANGEROUS)),
         "results/phase1/dangerous_methods.json", "dangerous_methods", ARTIFACT["criterion"])
    fact(sec, "identical to the legacy hard-coded set?", f"{'YES' if DANGEROUS == LEGACY else 'NO'}: +{sorted(DANGEROUS - LEGACY)} -{sorted(LEGACY - DANGEROUS)}",
         "results/phase1/dangerous_methods.json vs src/config.py LEGACY_DANGEROUS_METHODS", "-", "set difference")
    tab = pd.DataFrame(ARTIFACT["table"]).sort_values("stability")
    fact(sec, "pooled S of the dangerous eight", "; ".join(f"{r.method} {r.stability:+.4f}" for _, r in tab[tab.dangerous == 1].iterrows()),
         "results/phase1/dangerous_methods.json", "table, dangerous == 1", "S = valid_rate - 2.0 cat_rate + 0.4 beats_rate")
    safe = tab[tab.dangerous == 0].iloc[0]
    fact(sec, "nearest non-dangerous accelerator", f"{safe.method} S = {safe.stability:+.4f}", "results/phase1/dangerous_methods.json", "table, dangerous == 0", "min S")
    fact(sec, "scope of the derivation", f"Phase 1 only: obs_idx = {C.PHASE1['full']['obs_idx']}, noise {C.PHASE1['full']['noise_levels']}, "
         f"{C.PHASE1['full']['n_seeds']} seeds, 18 core regimes, pooled over the three strata, capped cells excluded",
         "src/config.py PHASE1; scripts/derive_dangerous.py", "-", "the derivation reads phase1_aggregated.csv and nothing else; no depth other than 90 enters it")
    fact(sec, "artifact provenance", f"git_head {ARTIFACT.get('git_head')}, created {ARTIFACT.get('created')}, source {ARTIFACT.get('source')}, pool {ARTIFACT.get('pool')} (n_pool {ARTIFACT.get('n_pool')})",
         "results/phase1/dangerous_methods.json", "-", "header fields")

    # sigma = 0 cancellation NaNs in the difference-based families (committed aggregate)
    sec = "sigma = 0 cancellation NaNs"
    A = AGG[(AGG.is_oracle == 0)]
    for fam in CLASSICAL_FAMILIES:
        s = A[A.family == fam]
        by = s.groupby("noise").valid_rate.mean()
        fact(sec, f"{fam}: invalid rate by noise (Phase 1, all cells)", ", ".join(f"sigma={n:g}: {100 * (1 - v):.2f}%" for n, v in by.items()),
             "results/phase1/phase1_aggregated.csv", f"family == {fam}, is_oracle == 0 (core + held-out, capped included; 30 seeds per cell)",
             "100 * (1 - mean over cells of valid_rate) per noise level")
    s = A[A.family.isin(CLASSICAL_FAMILIES)]
    by = s.groupby("noise").valid_rate.mean()
    fact(sec, "all 23 classical (difference-based) variants: invalid rate by noise (Phase 1)",
         ", ".join(f"sigma={n:g}: {100 * (1 - v):.2f}%" for n, v in by.items()),
         "results/phase1/phase1_aggregated.csv", "family in the seven classical families", "100 * (1 - mean valid_rate) per noise level")
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
    if os.path.exists(vpath):
        V = pd.read_csv(vpath)
        r3 = V[V.method == "richardson_3"].groupby("obs_idx").apply(lambda g: (g.valid_rate * g.n).sum() / g.n.sum())
        fact(sec, "richardson_3 valid rate by observation depth (committed aggregate)",
             ", ".join(f"obs {int(d)}: {v:.3f}" for d, v in r3.items()),
             "results/phase5a/phase5a_validity_by_depth.csv", "method == richardson_3", "n-weighted mean of valid_rate over noise levels per obs_idx")
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
    raw = os.path.join(RES, "phase5a", "phase5a_raw.csv")
    if not ARGS.no_raw and os.path.exists(raw):
        df = pd.read_csv(raw, usecols=["obs_idx", "noise", "method", "valid", "is_holdout", "target_g", "capped", "skill", "error", "ref_error", "regime"])
        r3 = df[df.method == "richardson_3"]
        by = r3.groupby("obs_idx").valid.mean()
        fact(sec, "richardson_3 valid rate by observation depth (Phase 5a, all regimes, strata and noise levels)",
             ", ".join(f"obs {int(d)}: {v:.3f}" for d, v in by.items()),
             "results/phase5a/phase5a_raw.csv (git-ignored; regenerated by scripts/run_phase5a.py --full)",
             "method == richardson_3", "mean(valid) per obs_idx; the 2026-09-21 run gives 0.481 / 0.583 / 0.981 / 0.999")
        byn = r3.groupby("noise").valid.mean()
        fact(sec, "richardson_3 valid rate by noise (Phase 5a)", ", ".join(f"sigma={n:g}: {v:.3f}" for n, v in byn.items()),
             "results/phase5a/phase5a_raw.csv", "method == richardson_3", "mean(valid) per noise")
        w = r3[r3.obs_idx == 30].groupby("regime").valid.mean().sort_values().head(5)
        fact(sec, "richardson_3 at obs 30: least valid regimes", ", ".join(f"{k} {v:.2f}" for k, v in w.items()),
             "results/phase5a/phase5a_raw.csv", "method == richardson_3, obs_idx == 30", "mean(valid) per regime")
        nd = df[(~df.method.isin(DANGEROUS)) & (df.method != ORACLE)]
        fact("invalid rates", "Phase 5a invalid rate outside the dangerous set (all records)", f"{100 * (1 - nd.valid.mean()):.3f}%",
             "results/phase5a/phase5a_raw.csv", "method not dangerous, not the oracle", "100 * mean(valid == 0)")
        fact("invalid rates", "Phase 5a invalid rate outside the dangerous set by noise",
             ", ".join(f"sigma={n:g}: {100 * (1 - v):.3f}%" for n, v in nd.groupby('noise').valid.mean().items()),
             "results/phase5a/phase5a_raw.csv", "method not dangerous, not the oracle", "per noise")
        fact("invalid rates", "Phase 5a invalid rate outside the dangerous set by depth",
             ", ".join(f"obs {int(d)}: {100 * (1 - v):.3f}%" for d, v in nd.groupby('obs_idx').valid.mean().items()),
             "results/phase5a/phase5a_raw.csv", "method not dangerous, not the oracle", "per obs_idx")
        cl = nd[nd.method.isin(CLASSICAL)]
        fact("sigma = 0 cancellation NaNs", "the 23 classical variants: invalid rate by noise (Phase 5a, four depths)",
             ", ".join(f"sigma={n:g}: {100 * (1 - v):.3f}%" for n, v in cl.groupby('noise').valid.mean().items()),
             "results/phase5a/phase5a_raw.csv", "family in the seven classical families", "per noise")
        rf = df[(df.method == "rational_fit") & (df.target_g == HEADLINE_G) & (df.is_holdout == 0) & (df.capped == 0)]
        n_lt, n_gt = int((rf.skill < 1).sum()), int((rf.skill > 1).sum())
        fact("ensembles (Phase 5a)", f"fixed rational_fit, core g={gname(HEADLINE_G)}: records with skill < 1 / > 1",
             f"{n_lt} / {n_gt} of {len(rf)} ({100 * n_lt / len(rf):.1f}% beat the hindsight best-of-four trivial); this is why its median skill prints as 1.0000",
             "results/phase5a/phase5a_raw.csv", f"method == rational_fit, target_g == {HEADLINE_G}, is_holdout == 0, capped == 0",
             "count(skill < 1), count(skill > 1)")
    else:
        fact(sec, "richardson_3 valid rate by observation depth (Phase 5a)", "not recomputed (raw file absent); the 2026-09-21 full run (REPORT_3B.md) gives obs 30: 0.481, 60: 0.583, 90: 0.981, 120: 0.999",
             "results/phase5a/phase5a_raw.csv (git-ignored)", "method == richardson_3", "mean(valid) per obs_idx")
    fact(sec, "why the artifact does not flag richardson_3", f"the dangerous set is derived at obs_idx = {C.PHASE1['full']['obs_idx']} only, where richardson_3 is 98 % valid; "
         "its 7-parameter curve_fit (src/accelerators.py, _fit_richardson, n_terms = 3, maxfev = 3000) does not converge on 30-60-point windows and returns NaN",
         "src/accelerators.py; src/config.py PHASE1", "-", "-")

    p1 = os.path.join(RES, "phase1", "phase1_records.csv")
    if not ARGS.no_raw and os.path.exists(p1):
        df = pd.read_csv(p1, usecols=["method", "valid", "is_oracle", "noise", "is_holdout", "capped"])
        nd = df[(~df.method.isin(DANGEROUS)) & (df.is_oracle == 0)]
        fact("invalid rates", "Phase 1 invalid rate outside the dangerous set (all records)", f"{100 * (1 - nd.valid.mean()):.3f}%",
             "results/phase1/phase1_records.csv (git-ignored)", "method not dangerous, not the oracle", "100 * mean(valid == 0)")
        fact("invalid rates", "Phase 1 invalid rate outside the dangerous set by noise",
             ", ".join(f"sigma={n:g}: {100 * (1 - v):.3f}%" for n, v in nd.groupby('noise').valid.mean().items()),
             "results/phase1/phase1_records.csv", "method not dangerous, not the oracle", "per noise")

    # method implementation notes (committed aggregate)
    sec = "method implementation notes"
    ca = AGG[AGG.method == "constant_assumed"].set_index(["regime", "noise", "target_g"]).med_error
    for m in ("weniger_d1", "weniger_d2"):
        w = AGG[AGG.method == m].set_index(["regime", "noise", "target_g"]).med_error
        j = pd.concat([ca.rename("ca"), w.rename("w")], axis=1).dropna()
        same = int(np.isclose(j.ca, j.w, rtol=0, atol=1e-9).sum())
        fact(sec, f"{m}: cells whose median error equals constant_assumed's (L_hat = 0)", f"{same} of {len(j)} cells",
             "results/phase1/phase1_aggregated.csv", f"method in ({m}, constant_assumed), matched on (regime, noise, target_g)",
             "count(|med_error_w - med_error_ca| <= 1e-9); with w_n = s_n the Weniger numerator sum_j (-1)^j C(k,j) beta_j is identically 0 "
             "for k = 1, 2 (src/accelerators.py _weniger_delta), so the transform returns 0 = the assumed asymptote")
    # pipeline provenance
    sec = "pipeline provenance"
    fact(sec, "results commit", RESULTS_HEAD, "git log -1 -- results", "-", "-")
    fact(sec, "full run", "2026-09-20 23:35 to 2026-09-21 03:58, 15,783 s; Phase 1 1,839 s, Phase 2 979 s, Phase 4 1,797 s, Phase 5a 8,868 s (--jobs 7, 5 perturbation trials), Phase 5b 2,287 s",
         "REPORT_3B_full_run.txt; REPORT_3B.md", "-", "-")
    fact(sec, "evaluation counts (central)", "Phase 1 362,880; Phase 2 772,200; Phase 4 224,640 (+1,555,200 diagnostic calls); Phase 5a 967,680 (+4,406,400 perturbation calls); Phase 5b 532,800; real data 630",
         "python reproduce_all.py --plan", "-", "-")
    fact(sec, "design constants", f"L_true log-uniform on {C.L_TRUE_RANGE} per (regime, seed); L_hat mode {C.ASSUMED_L_MODE}; strata g = {STRATA} (headline {HEADLINE_G}); "
         f"n_obs = {C.OBS_IDX}, window {C.WINDOW_LEN}; horizon cap {C.HORIZON_N_CAP}; rank floor valid_rate >= {C.RANK_MIN_VALID}; "
         f"CAT_MULT {C.CAT_MULT}; S = valid - {C.W_CAT} cat + {C.W_BEATS} beats", "src/config.py", "-", "-")


# ═════════════════════════════════════════════════════════════════════════════
# FACTS.md and the fragment index
# ═════════════════════════════════════════════════════════════════════════════
def write_facts(answer_lines):
    order = ["trivial baseline", "ranking", "skill summary", "classical no-op", "generalisation", "dangerous set",
             "richardson_3 validity by depth", "sigma = 0 cancellation NaNs", "invalid rates", "capped cells",
             "sweep 1 (assumed asymptote)", "selectors (Phase 3)", "ensembles (Phase 5a)", "diagnostics (Phase 4)",
             "real data (v2)", "real data (v2) perturbation diagnostic", "real data (legacy 18-cell run)", "pipeline provenance"]
    secs = list(dict.fromkeys(order + [f["section"] for f in FACTS]))
    with open(ARGS.facts, "w", encoding="utf-8", newline="\n") as f:
        f.write("# FACTS.md -- headline numbers of the redesign-v2 results, with provenance\n\n")
        f.write(f"{PROV}.  Regenerate with `python scripts/make_paper_tables.py`.  "
                "Every row names the file, the filter applied to it and the formula; numbers in the paper come from here or from the "
                "fragments in `paper_fragments/`, never from hand-typing.  Facts drawn from the git-ignored raw per-record files are "
                "marked as such.\n\n")
        f.write("## Named facts (the review asked for these explicitly)\n\n")
        f.write("- **richardson_3 validity by depth**: see section *richardson_3 validity by depth* (0.48 / 0.58 / 0.98 / 1.00 at obs 30 / 60 / 90 / 120) "
                "and the obs-90-only scope of the dangerous derivation under *dangerous set*.\n")
        f.write("- **sigma = 0 cancellation NaNs**: section *sigma = 0 cancellation NaNs*.\n")
        f.write("- **The dangerous set is IDENTICAL to the legacy eight under the redesign**: section *dangerous set*.\n")
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
    with open(os.path.join(OUT, "README.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("# paper_fragments/\n\nLaTeX table fragments generated from `results/` by `scripts/make_paper_tables.py` "
                f"({PROV}).  Each file is one `tabular` (booktabs) with a comment header naming its sources and filters; "
                "wrap it in `table`/`caption` in the paper.  The header's code hash is HEAD when the generator ran (`-dirty` = uncommitted "
                "changes present; the generator's own changes land in the following commit); the results hash is the last commit that touched `results/`.  Macros used: `\\meth`, `\\diag`, `\\Stab`, `\\rhoC`, `\\rhoV`, "
                "`\\TE`, `\\LE`, `\\nobs`, `\\nf`.\n\n| fragment | content | sources |\n|---|---|---|\n")
        for name, desc, src in FRAGMENTS:
            f.write(f"| `{name}` | {desc} | {'; '.join(s.split(' (')[0] for s in src)} |\n")
    print(f"  wrote {rel(os.path.join(OUT, 'README.md'))} ({len(FRAGMENTS)} fragments)")


def main():
    print(f"make_paper_tables.py (redesign v2): results = {rel(RES)}, out = {rel(OUT)}, facts = {rel(ARGS.facts)}")
    print(f"  code {CODE_HEAD}, results as of {RESULTS_HEAD}; dangerous = {sorted(DANGEROUS)}")
    f01(); f02(); f03(); f04(); f05(); f05b(); f06(); f07()
    answer = f08()
    f09(); f10(); f11()
    named_facts()
    write_facts(answer)
    write_index()
    print(f"\nDONE: {len(FRAGMENTS)} fragments in {rel(OUT)}, {len(FACTS)} facts in {rel(ARGS.facts)}")


if __name__ == "__main__":
    main()
