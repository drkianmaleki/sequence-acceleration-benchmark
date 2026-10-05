#!/usr/bin/env python3
"""
analyze_by_ltrue.py  (redesign v2; analysis only)
============================================================
Do the Phase-1 conclusions depend on where the hidden asymptote L_true sits?
L_true is log-uniform on [0.005, 0.5] (median ~0.05) while the six recorded
real curves have floors of 0.14-0.69, and predict-L_hat (L_hat = 0) is the
strongest deployable trivial at g <= 0.1.

Two paths, and the script prints which one it took:

  records path  results/phase1/phase1_records.csv is present and --from-csv
                is not given: reads the git-ignored records, excludes capped
                cells, bins every record by L_true into the log-scale
                terciles of [0.005, 0.5],

                    T1 [0.005, 0.0232)   T2 [0.0232, 0.1077)   T3 [0.1077, 0.5]

                aggregates per stratum g in {0.5, 0.1, 0.02}, per tercile,
                core and out-of-design separately, for the four deployable
                trivials, constant_oracle (reference, never ranked), the
                top-10 accelerators by pooled median error at g = 0.1 (core
                ranking of phase1_global.csv) and the classical variants
                pooled as one row (plus each variant individually, for the
                in-band count):

                    n_records, n_valid, med_error (median over valid records),
                    win_rate_vs_last, win_rate_vs_assumed (mean of the 0/1 win
                    indicators over all records; an invalid estimate never wins),
                    med_skill_vs_last (median over valid records), mi (median improve_ratio)

                and writes results/phase1/phase1_by_Ltrue.csv;
  reader path   the records file is absent, or --from-csv is given: reads the
                committed results/phase1/phase1_by_Ltrue.csv instead and
                writes nothing under results/.

Both paths write the same fragments and FACTS section from that table:

    paper_fragments/f13_by_Ltrue_g{0.5,0.1,0.02}.tex   (one fragment per stratum;
                                                       a stale combined f13_by_Ltrue.tex is removed)
    the section "## L_true terciles" of FACTS.md, replaced in place, with three
    facts: the rank-1 accelerator per tercile at g = 0.1 core; the classical
    in-band (MI in [0.9, 1.1]) count per tercile; predict-L_hat vs last_value
    win rate per tercile.  scripts/make_paper_tables.py does not know about
    this section: regenerating FACTS.md drops it until this script is run again.

and rewrite the fragment index paper_fragments/README.md (every fragment in
the folder; write_fragment_index below, shared with make_paper_tables.py).

    python scripts/analyze_by_ltrue.py [--results results] [--out paper_fragments] [--facts FACTS.md] [--print-g 0.1] [--from-csv]

It runs after make_paper_tables.py in reproduce_all.py because it replaces
its own section of the FACTS.md that the table generator writes.

provenance() and load_manifest() below build the header of every generated
file from results/run_manifest.json; make_paper_tables.py imports them from
here (this module does nothing at import, that one reads results/).
"""

import argparse
import json
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import src.config as C                                   # noqa: E402
from src.evaluation import FAMILY, METHOD_TYPE           # noqa: E402  (the registry's family and type maps)
from src.pipeline import ACCEL_METHODS                   # noqa: E402
from src.trivial import SKILL_REFERENCE_METHODS          # noqa: E402

EDGES = [0.005, 0.0232, 0.1077, 0.5]
TERCILES = ["T1", "T2", "T3"]
TERCILE_LABEL = {"T1": "[0.005, 0.0232)", "T2": "[0.0232, 0.1077)", "T3": "[0.1077, 0.5]"}
CLASSICAL_FAMILIES = ("shanks", "wynn_eps", "wynn_rho", "levin", "brezinski", "anderson")
NOOP_BAND = (0.9, 1.1)
ORACLE = "constant_oracle"
DEPLOYABLE = list(SKILL_REFERENCE_METHODS)   # constant_assumed, last_value, window_mean, window_min
MANIFEST = "run_manifest.json"               # written by reproduce_all.py at the start of a run, after every step and at the end
COMBINED_F13 = "f13_by_Ltrue.tex"            # the earlier combined fragment (one file for every stratum); removed when found
INDEX_MACROS = "`\\meth`, `\\diag`, `\\rhoC`, `\\rhoV`, `\\TE`, `\\LE`, `\\nobs`, `\\nf`"


def tercile(L):
    return pd.cut(L, EDGES, right=False, include_lowest=True, labels=TERCILES).astype(str)


def _git(*cmd):
    try:
        return subprocess.check_output(["git"] + list(cmd), cwd=_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def load_manifest(results_dir):
    """
    The run manifest of a results tree as it stands (None when the file is
    absent).  reproduce_all.py writes it at the start of a run and rewrites
    it after every step, so a generator running inside a run reads its own
    run, with finished and total_seconds still null.
    """
    path = os.path.join(results_dir, MANIFEST)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def provenance(script, results_dir, code_head=None):
    """
    The provenance phrase of every generated header (fragments, their README,
    FACTS.md): the run the results come from, read from the run manifest --
    start time, code commit, mode, wall time, worker count.  Inside a run
    the wall time is not known yet and reads "in progress"; the manifest's
    final rewrite records it.  The commit that holds the results is the one
    the files sit in; it cannot be known when they are generated, so it is
    not named.  Without a manifest the phrase says so and names the commit
    the generator ran at (code_head; HEAD when not given).
    """
    M = load_manifest(results_dir)
    if M is None:
        head = code_head or _git("rev-parse", "--short", "HEAD")
        return f"generated by {script} from results without a run manifest; generator at code {head}"
    total = M.get("total_seconds")
    return (f"generated by {script} from the run started {M.get('started')} on code "
            f"{M.get('git_head', {}).get('short')} ({M.get('mode')}; "
            f"{'in progress' if total is None else f'{total:,.0f} s'}; {M.get('jobs')} workers)")


# ── the fragment index (shared with make_paper_tables.py) ────────────────────
def list_fragments(out_dir):
    """Every fragment file of the folder (f<nn>*.tex), in name order."""
    return sorted(f for f in os.listdir(out_dir) if re.fullmatch(r"f\d+[^/\\]*\.tex", f))


def fragment_header(path):
    """(description, sources, filter) read from the comment header of a fragment:
    line 1 is the provenance, line 2 the description, then the source, filter
    and note lines."""
    with open(path, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    desc = lines[1][2:] if len(lines) > 1 and lines[1].startswith("% ") else ""
    sources, filt = [], ""
    for ln in lines[2:]:
        if not ln.startswith("%"):
            break
        if ln.startswith("% source : "):
            sources.append(ln[len("% source : "):])
        elif ln.startswith("% filter : "):
            filt = ln[len("% filter : "):]
    return desc, sources, filt


def write_fragment_index(out_dir, results_dir):
    """paper_fragments/README.md: one row per fragment in the folder (name order),
    with the description and the source files read from the fragment headers.
    Rewritten by make_paper_tables.py and by this script; returns the count."""
    prov = provenance("the table generators", results_dir)
    names = list_fragments(out_dir)
    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("# paper_fragments/\n\n"
                "LaTeX table fragments generated from `results/` by `scripts/make_paper_tables.py` and, for `f13`, "
                f"`scripts/analyze_by_ltrue.py` ({prov}); this index lists every fragment in the folder and is rewritten by both "
                "scripts; `all_tables.tex` (`scripts/build_tables_document.py`) inputs every fragment into one standalone document, "
                "and `scripts/check_tables.py` regenerates everything from the committed `results/` and compares.  Each file is one "
                "`tabular` (booktabs) with a comment header naming its sources and filters; wrap it in `table`/`caption` in the paper.  "
                "The header names the run the results come from (`results/run_manifest.json`: start time, code commit, mode, wall time, "
                "workers); the commit that holds the results is the one these files sit in.  "
                f"Macros used: {INDEX_MACROS}.\n\n| fragment | content | sources |\n|---|---|---|\n")
        for name in names:
            desc, src, _ = fragment_header(os.path.join(out_dir, name))
            f.write(f"| `{name}` | {desc} | {'; '.join(s.split(' (')[0] for s in src)} |\n")
    return len(names)


# ── the per-tercile table ────────────────────────────────────────────────────
def stats(sub):
    ok = sub["valid"] == 1
    return dict(
        n_records=int(len(sub)), n_valid=int(ok.sum()),
        med_error=float(sub.loc[ok, "error"].median()) if ok.any() else float("nan"),
        win_rate_vs_last=float(sub["win_vs_last"].mean()) if len(sub) else float("nan"),
        win_rate_vs_assumed=float(sub["win_vs_assumed"].mean()) if len(sub) else float("nan"),
        med_skill_vs_last=float(sub.loc[ok, "skill_vs_last"].median()) if ok.any() else float("nan"),
        mi=float(sub.loc[ok, "improve_ratio"].median()) if ok.any() else float("nan"),
    )


def top10_and_classical(global_csv):
    G = pd.read_csv(global_csv)
    top10 = (G[(G.target_g == C.HEADLINE_G) & (G.is_trivial == 0) & (G.rank_eligible == 1)]
             .sort_values("med_error").head(10).method.tolist())
    classical = sorted([m for m in ACCEL_METHODS if FAMILY[m] in CLASSICAL_FAMILIES and METHOD_TYPE[m] == "limit"])
    assert len(classical) == sum(1 for m in ACCEL_METHODS if FAMILY[m] in CLASSICAL_FAMILIES), classical
    return top10, classical


def compute_from_records(records_path, top10, classical):
    """The per-tercile table from the git-ignored records (capped cells excluded)."""
    cols = ["regime", "is_holdout", "noise", "seed", "L_true", "target_g", "capped", "method", "family",
            "is_trivial", "is_oracle", "error", "valid", "improve_ratio",
            "skill_vs_last", "win_vs_last", "win_vs_assumed"]
    df = pd.read_csv(records_path, usecols=cols)
    n_all = len(df)
    df = df[df["capped"] == 0].copy()
    df["tercile"] = tercile(df["L_true"])
    print(f"records {n_all:,}; capped excluded -> {len(df):,}; (regime, seed) pairs per tercile: "
          + ", ".join(f"{t} {df[df.tercile == t].drop_duplicates(['regime', 'seed']).shape[0]}" for t in TERCILES))
    classical_row = f"classical_{len(classical)}"
    rows = []
    strata = sorted(df.target_g.unique(), reverse=True)
    for g in strata:
        for t in TERCILES:
            for hold, rs in ((0, "core"), (1, "holdout")):
                base = df[(df.target_g == g) & (df.tercile == t) & (df.is_holdout == hold)]
                groups = ([(m, "trivial") for m in DEPLOYABLE] + [(ORACLE, "oracle")]
                          + [(m, "top10") for m in top10] + [(m, "classical_variant") for m in classical])
                for m, grp in groups:
                    sub = base[base.method == m]
                    rows.append(dict(target_g=g, tercile=t, tercile_range=TERCILE_LABEL[t], regime_set=rs,
                                     row=m, group=grp, **stats(sub)))
                sub = base[base.method.isin(classical)]
                rows.append(dict(target_g=g, tercile=t, tercile_range=TERCILE_LABEL[t], regime_set=rs,
                                 row=classical_row, group="classical_pooled", **stats(sub)))
    out = pd.DataFrame(rows)
    for c in ("med_error", "win_rate_vs_last", "win_rate_vs_assumed", "med_skill_vs_last", "mi"):
        out[c] = out[c].round(6)
    return out


def main():
    p = argparse.ArgumentParser(description="Phase-1 results by L_true tercile (analysis only)")
    p.add_argument("--results", default=os.path.join(_ROOT, "results"),
                   help="results tree: reads phase1/phase1_records.csv (or phase1/phase1_by_Ltrue.csv) and phase1/phase1_global.csv")
    p.add_argument("--out", default=os.path.join(_ROOT, "paper_fragments"), help="fragment directory (f13_by_Ltrue_g*.tex, README.md)")
    p.add_argument("--facts", default=os.path.join(_ROOT, "FACTS.md"))
    p.add_argument("--print-g", type=float, default=C.HEADLINE_G)
    p.add_argument("--from-csv", action="store_true",
                   help="the reader's path: use the committed phase1/phase1_by_Ltrue.csv even when the records file is present; writes nothing under results/")
    args = p.parse_args()
    args.records = os.path.join(args.results, "phase1", "phase1_records.csv")
    args.global_csv = os.path.join(args.results, "phase1", "phase1_global.csv")
    args.out_csv = os.path.join(args.results, "phase1", "phase1_by_Ltrue.csv")

    top10, classical = top10_and_classical(args.global_csv)
    classical_row = f"classical_{len(classical)}"
    print("top-10 accelerators by pooled median error at g = %g (core): %s" % (C.HEADLINE_G, ", ".join(top10)))

    if not args.from_csv and os.path.exists(args.records):
        print(f"path: records ({os.path.relpath(args.records, _ROOT)} -> {os.path.relpath(args.out_csv, _ROOT)})")
        out = compute_from_records(args.records, top10, classical)
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        out.to_csv(args.out_csv, index=False)
        print(f"  wrote {os.path.relpath(args.out_csv, _ROOT)} ({len(out)} rows)")
    else:
        if not os.path.exists(args.out_csv):
            print(f"ERROR: neither {args.records} nor {args.out_csv} found")
            return 1
        why = "--from-csv" if args.from_csv else "the records file is absent"
        print(f"path: reader ({why}): reading the committed {os.path.relpath(args.out_csv, _ROOT)}; nothing under results/ is written")
        out = pd.read_csv(args.out_csv, float_precision="round_trip")
        got = set(out[out.group == "top10"].row)
        assert got == set(top10), f"the committed table's top-10 rows {sorted(got)} differ from the ranking's {top10}"
        assert set(out[out.group == "classical_variant"].row) == set(classical), "the committed table's classical variants differ from the registry's"
    strata = sorted(out.target_g.unique(), reverse=True)

    # ── the three facts ──────────────────────────────────────────────────────
    facts = []
    gh = C.HEADLINE_G
    rank1 = {}
    for t in TERCILES:
        s = out[(out.target_g == gh) & (out.tercile == t) & (out.regime_set == "core") & (out.group == "top10")]
        r = s.sort_values("med_error").iloc[0]
        rank1[t] = (r.row, r.med_error, r.win_rate_vs_last, r.win_rate_vs_assumed)
    facts.append(("rank-1 accelerator per L_true tercile at g = %g core (among the top-10 by pooled median error)" % gh,
                  "; ".join(f"{t} {TERCILE_LABEL[t]}: {m} (med. err {e:.4f}, win vs last {wl:.3f}, win vs L_hat {wa:.3f})"
                            for t, (m, e, wl, wa) in rank1.items()),
                  "results/phase1/phase1_by_Ltrue.csv", f"target_g == {gh}, regime_set == core, group == top10",
                  "argmin med_error (median over valid records) per tercile"))
    inband = {}
    for t in TERCILES:
        s = out[(out.target_g == gh) & (out.tercile == t) & (out.regime_set == "core") & (out.group == "classical_variant")]
        inband[t] = int(((s.mi >= NOOP_BAND[0]) & (s.mi <= NOOP_BAND[1])).sum())
    facts.append(("classical variants inside the +/-10 %% MI band per L_true tercile at g = %g core" % gh,
                  "; ".join(f"{t}: {inband[t]} of {len(classical)}" for t in TERCILES),
                  "results/phase1/phase1_by_Ltrue.csv", f"target_g == {gh}, regime_set == core, group == classical_variant",
                  "count(0.9 <= mi <= 1.1); mi = median improve_ratio (current-value error / method error) over valid records"))
    wl = {}
    for t in TERCILES:
        for rs in ("core", "holdout"):
            s = out[(out.target_g == gh) & (out.tercile == t) & (out.regime_set == rs)]
            wl[(t, rs)] = (float(s[s.row == "constant_assumed"].win_rate_vs_last.iloc[0]),
                           float(s[s.row == "last_value"].win_rate_vs_assumed.iloc[0]),
                           float(s[s.row == "constant_assumed"].med_error.iloc[0]),
                           float(s[s.row == "last_value"].med_error.iloc[0]))
    facts.append(("predict-L_hat (constant_assumed, L_hat = 0) vs last_value per L_true tercile at g = %g" % gh,
                  "; ".join(f"{t} {rs}: L_hat wins {a:.3f} (last wins {b:.3f}); med. err {e1:.4f} vs {e2:.4f}"
                            for (t, rs), (a, b, e1, e2) in wl.items()),
                  "results/phase1/phase1_by_Ltrue.csv", f"target_g == {gh}, rows constant_assumed / last_value",
                  "win_rate_vs_last of constant_assumed; win_rate_vs_assumed of last_value; med_error of each"))

    # ── FACTS.md section (replaced in place; the same text on both paths) ────
    prov = provenance("scripts/analyze_by_ltrue.py", args.results)
    head = "## L_true terciles (scripts/analyze_by_ltrue.py, not part of make_paper_tables.py)"
    body = [head, "",
            f"{prov[0].upper()}{prov[1:]}, from results/phase1/phase1_by_Ltrue.csv, the per-tercile aggregate of the git-ignored "
            "results/phase1/phase1_records.csv (written by this script when the records file is present, read as committed otherwise); "
            "capped cells excluded; log-scale terciles of L_true in "
            "[0.005, 0.5]: T1 [0.005, 0.0232), T2 [0.0232, 0.1077), T3 [0.1077, 0.5].  Regenerating FACTS.md with "
            "make_paper_tables.py drops this section; re-run this script afterwards.", "",
            "| fact | value | file | filter | formula |", "|---|---|---|---|---|"]
    for f in facts:
        body.append("| " + " | ".join(str(x).replace("|", "\\|") for x in f) + " |")
    body.append("")
    if os.path.exists(args.facts):
        txt = open(args.facts, encoding="utf-8").read()
        if head in txt:
            i = txt.index(head)
            j = txt.find("\n## ", i + len(head))
            txt = txt[:i] + "\n".join(body) + (txt[j + 1:] if j != -1 else "")
        else:
            txt = txt.rstrip("\n") + "\n\n" + "\n".join(body)
        open(args.facts, "w", encoding="utf-8", newline="\n").write(txt)
        print(f"  updated {os.path.relpath(args.facts, _ROOT)} (section '{head[3:]}', {len(facts)} facts)")

    # ── fragments f13, one per stratum ───────────────────────────────────────
    def esc(s):
        return str(s).replace("_", r"\_")

    def mth(s):
        return r"\meth{" + esc(s) + "}"

    def f4(v):
        return "--" if not np.isfinite(v) else f"{v:.4f}"

    def f3(v):
        return "--" if not np.isfinite(v) else f"{v:.3f}"

    order = ([(m, "trivial") for m in DEPLOYABLE] + [(ORACLE, "oracle")] + [(m, "top10") for m in top10]
             + [(classical_row, "classical_pooled")])
    labels = {"constant_assumed": r"predict $\hat L$ (\meth{constant\_assumed})", ORACLE: mth(ORACLE) + r" \textit{(ref.)}",
              classical_row: f"{len(classical)} classical variants (pooled)"}
    os.makedirs(args.out, exist_ok=True)
    stale = os.path.join(args.out, COMBINED_F13)
    if os.path.exists(stale):
        os.remove(stale)
        print(f"  removed the stale combined {os.path.relpath(stale, _ROOT)} (superseded by one fragment per stratum)")
    for g in strata:
        lines = ["% AUTO-GENERATED -- do not hand-edit.  " + prov,
                 f"% Phase-1 results by L^* tercile at g = {g:g}: the deployable trivials, the oracle reference, the top-10 accelerators "
                 f"(by pooled median error at g = {gh:g}, core) and the {len(classical)} classical variants pooled; core | out-of-design",
                 "% source : results/phase1/phase1_by_Ltrue.csv (the per-tercile aggregate this fragment prints; the aggregate of the git-ignored "
                 "results/phase1/phase1_records.csv, capped == 0)",
                 f"% filter : target_g == {g:g}; log-scale terciles of L^* (L_true) in [0.005, 0.5]: T1 [0.005, 0.0232), T2 [0.0232, 0.1077), T3 [0.1077, 0.5]; "
                 "med. err = median over valid records; win vs X = mean of the per-record win indicator (invalid never wins); "
                 "skill vs last = median over valid records of err / err(last_value)",
                 r"\begin{tabular}{ll" + "r" * 12 + "}", r"\toprule",
                 r"$g$ / row & & \multicolumn{4}{c}{T1: $L^{*} \in [0.005, 0.0232)$} & "
                 r"\multicolumn{4}{c}{T2: $[0.0232, 0.1077)$} & \multicolumn{4}{c}{T3: $[0.1077, 0.5]$} \\",
                 r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}\cmidrule(lr){11-14}",
                 r" & set & med.\ err & win/last & win/$\hat L$ & skill/last & med.\ err & win/last & win/$\hat L$ & skill/last "
                 r"& med.\ err & win/last & win/$\hat L$ & skill/last \\", r"\midrule"]
        lines.append(r"\multicolumn{14}{l}{\textit{$g = " + f"{g:g}" + "$}" + (" (headline)" if g == gh else "") + r"} \\")
        for m, grp in order:
            for hold, rs, lab in ((0, "core", "core"), (1, "holdout", "out-of-design")):
                cells = []
                for t in TERCILES:
                    r = out[(out.target_g == g) & (out.tercile == t) & (out.regime_set == rs) & (out.row == m)]
                    if len(r):
                        r = r.iloc[0]
                        wl_ = "--" if m == "last_value" else f3(r.win_rate_vs_last)
                        wa_ = "--" if m == "constant_assumed" else f3(r.win_rate_vs_assumed)
                        sl_ = "--" if m == "last_value" else f3(r.med_skill_vs_last)
                        cells += [f4(r.med_error), wl_, wa_, sl_]
                    else:
                        cells += ["--"] * 4
                name = labels.get(m, mth(m)) if hold == 0 else ""
                lines.append(f"{name} & {lab} & " + " & ".join(cells) + r" \\")
            if grp in ("oracle", "top10") and m in (ORACLE, top10[-1]):
                lines.append(r"\addlinespace[2pt]")
        lines += [r"\bottomrule", r"\end{tabular}"]
        out_tex = os.path.join(args.out, f"f13_by_Ltrue_g{g:g}.tex")
        open(out_tex, "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
        print(f"  wrote {os.path.relpath(out_tex, _ROOT)} ({len(lines)} lines)")
    n_idx = write_fragment_index(args.out, args.results)
    print(f"  wrote {os.path.relpath(os.path.join(args.out, 'README.md'), _ROOT)} ({n_idx} fragments in the folder)")

    # ── console table at --print-g ───────────────────────────────────────────
    gp = args.print_g
    for rs in ("core", "holdout"):
        print(f"\n  g = {gp:g}, {rs}: median error / win vs last / win vs L_hat / median skill vs last")
        print(f"  {'row':<22}" + "".join(f"  {t + ' ' + TERCILE_LABEL[t]:>30}" for t in TERCILES))
        for m, grp in order:
            cells = []
            for t in TERCILES:
                r = out[(out.target_g == gp) & (out.tercile == t) & (out.regime_set == rs) & (out.row == m)]
                r = r.iloc[0] if len(r) else None
                cells.append("--" if r is None else f"{r.med_error:.4f}/{r.win_rate_vs_last:.3f}/{r.win_rate_vs_assumed:.3f}/{r.med_skill_vs_last:.3f}")
            print(f"  {m:<22}" + "".join(f"  {c:>30}" for c in cells))
    print("\n  FACTS:")
    for f in facts:
        print(f"    - {f[0]}: {f[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
