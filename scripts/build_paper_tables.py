#!/usr/bin/env python3
"""
The table builder of the manuscript
===================================
Version 6 (release, 2026-10-05).  One file, two places: beside the manuscript (with the folders
fragments/ and tables/), and in the repository as scripts/build_paper_tables.py, where it reads
paper_fragments/ and writes paper_tables_main/.  It builds the tables of the main text (Sections 4
to 8) and, beside the manuscript, the numbers of the supplementary tables (tables/supp_refs.tex,
read from supplement.tex).  The curve families outside the design are labelled "out-of-design".

Every table that asoc_paper.tex inputs (tables/<name>.tex) is built here as a
row and column subset of ONE generated fragment (fragments/f*.tex, copied
unchanged from the repository's paper_fragments/).  Nothing numeric is typed
in this file: a table is described by which fragment it comes from, which
columns it keeps and which rows it keeps.  After building, every number that
appears in a table is checked to occur in its source fragment; a table that
fails the check is not written and the script exits with status 1.

    python <this file>              build every table listed in TABLES
    python <this file> --selftest   also pass every fragment through the
                                         engine unchanged and compare row by row
    python <this file> --list       list the tables and their sources

Layout (the script runs from its own folder; no absolute path anywhere):

    <this file>               the builder
    fragments/f*.tex          the generated fragments (inputs, never edited)
    supplement.tex            read for the order of its tables (input, never edited)
    tables/<name>.tex         the tables the manuscript inputs (outputs)
    tables/supp_refs.tex      \\Sref{<fragment>} -> the number of that supplementary table (output)

Only the Python standard library is needed.

How to add a table: append one entry to TABLES.

    name     output file name (tables/<name>.tex)
    source   fragment name without .tex
    cols     source columns to keep, 0-based, increasing (None = all)
    rows     function(row) -> bool choosing body rows (None = all); a row has
             row.text[c]   the cell printed in source column c ('' if empty or
                           covered by a multi-column cell that starts earlier)
             row.sticky[c] the last non-empty cell seen in column c (group labels
                           that the fragment prints only on a group's first row)
             row.block     number of \\midrule lines above the row in the body
             row.i         index of the row among the body rows
    fill     source columns whose group label is re-printed on the first kept
             row of each group (default: none)
    header   list of LaTeX lines replacing the fragment's header (default: the
             fragment's own header, mapped to the kept columns)
    colspec  column specification replacing the derived one
    full     False to write a plain tabular instead of a full-width tabular*
"""

import argparse
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SELF = os.path.basename(__file__)
if os.path.isdir(os.path.join(HERE, "fragments")):
    # beside the manuscript
    FRAGMENTS = os.path.join(HERE, "fragments")
    OUT = os.path.join(HERE, "tables")
else:
    # in the repository, under scripts/
    FRAGMENTS = os.path.join(os.path.dirname(HERE), "paper_fragments")
    OUT = os.path.join(os.path.dirname(HERE), "paper_tables_main")

# ── the tables of the manuscript ─────────────────────────────────────────────
# One entry per table, in the order of the paper.  Stage 1: the roster.  Stage 2: Sections 5 and 6.
# Stage 3: Sections 7 and 8.
RAGGED = r">{\raggedright\arraybackslash}"


def _footer(row):
    """A full-width note line of a fragment (one multi-column cell spanning the table)."""
    return len(row.cells) == 1 and row.cells[0][1] is not None


TABLES = [
    # Section 4: the roster, the whole fragment; the two text columns wrap
    dict(name="tab_roster", source="f20_roster",
         colspec=RAGGED + r"p{0.20\linewidth}lr" + RAGGED + r"p{0.56\linewidth}"),
    # Section 5.1: trivial predictors -- median error, win rate vs the last value and strict skill, both sets
    dict(name="tab_trivial", source="f01_trivial_baseline", cols=[0, 1, 4, 6, 7, 10, 12],
         colspec=RAGGED + r"p{0.36\linewidth}rrrrrr"),
    # Section 5.2: the classical variants by noise class, headline stratum
    dict(name="tab_noop_noise", source="f21_noop_by_noise", cols=[0, 2, 3, 5, 7, 8, 9, 11, 12, 14],
         rows=lambda r: r.sticky[1] == "0.1", fill=[0]),
    # Section 5.3: skill by method family, the whole fragment
    dict(name="tab_skill", source="f03_skill_summary"),
    # Section 5.4: the leading fits -- the synthetic columns and the sensitivity to the assumed asymptote
    dict(name="tab_leading", source="f24_leading_fits", cols=list(range(0, 12)),
         header=[r"Method & \multicolumn{5}{c}{core} & \multicolumn{5}{c}{out-of-design} & $\hat L$ \\",
                 r"\cmidrule(lr){2-6}\cmidrule(lr){7-11}",
                 r" & $r$ & err & win & $\rhoC$ & $\rhoV$ & $r$ & err & win & $\rhoC$ & $\rhoV$ & sens. \\"]),
    # Section 5.5: the two named fits family by family, headline stratum (note lines are quoted in the text)
    dict(name="tab_family", source="f22_per_family_g0.1", cols=[0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11],
         rows=lambda r: not _footer(r),
         header=[r"Set & curve family & last & \multicolumn{4}{c}{\meth{rational\_fit}} & \multicolumn{2}{c}{\meth{single\_exp\_fit}} "
                 r"& \multicolumn{2}{c}{best} \\",
                 r"\cmidrule(lr){4-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}",
                 r" & & err & err & win & $\rhoV$ & $\rhoC$ & err & win & method & err \\"]),
    # Section 5.6: the family bootstrap -- point values and percentile intervals
    dict(name="tab_boot", source="f15_bootstrap", cols=[0, 1, 2, 3, 4]),
    # Section 6.2: core against out-of-design, the whole fragment
    dict(name="tab_gen", source="f16_generalisation_summary",
         header=[r"$g$ & \multicolumn{3}{c}{eligible accelerators} & Spearman & \multicolumn{2}{c}{rank, core / out-of-design} \\",
                 r"\cmidrule(lr){2-4}\cmidrule(lr){6-7}",
                 r" & core & out-of-design & both & $\rho$ & \meth{rational\_fit} & \meth{log\_linear} \\"]),
    # Section 6.4: validity by depth (the note line is quoted in the caption) and the capped cells
    dict(name="tab_depth", source="f12_validity_by_depth", rows=lambda r: not _footer(r)),
    dict(name="tab_capped", source="f11_capped_block", cols=[0, 1, 2, 3, 4, 7], rows=lambda r: not _footer(r),
         colspec=r"lllrr" + RAGGED + r"p{0.36\linewidth}"),
    # Section 7.3: the two perturbation diagnostics against the error, the whole fragment
    dict(name="tab_diagcorr", source="f10a_diagnostics_correlations"),
    # Section 7.4: selectors and ensembles on the core families, headline stratum
    dict(name="tab_ensembles", source="f09b_ensemble_phase5a_core", cols=[0, 5, 6, 7, 8, 9]),
    # Section 8.2: the fixed methods and the cascade on the recorded curves, pooled block
    dict(name="tab_realmethods", source="f14_real_fixed_methods", cols=[1, 2, 3, 4, 5, 6, 7, 8], rows=lambda r: r.block == 0),
    # Section 8.3: the leading fits on the recorded curves (the recorded-curve columns of the leading-fits fragment)
    dict(name="tab_realleading", source="f24_leading_fits", cols=[0, 12, 13, 14],
         header=[r"Method & wins before & wins after & error ratio before \\"]),
    # Section 8.3: the two named fits by dataset, the whole fragment
    dict(name="tab_realnamed", source="f29_real_named_by_dataset",
         header=[r"Dataset & \multicolumn{2}{c}{cells} & \multicolumn{4}{c}{\meth{rational\_fit}} & \multicolumn{4}{c}{\meth{single\_exp\_fit}} \\",
                 r"\cmidrule(lr){2-3}\cmidrule(lr){4-7}\cmidrule(lr){8-11}",
                 r" & before & after & wins & ratio & wins & ratio & wins & ratio & wins & ratio \\",
                 r" & & & before & before & after & after & before & before & after & after \\"]),
    # Section 8.4: how accurately the final loss is predicted, by depth (the note line is quoted in the caption)
    dict(name="tab_early", source="f30_real_final_loss_by_depth", rows=lambda r: not _footer(r)),
    # Section 8.5: choosing by trial on the synthetic families, the two noise levels, three pools; one pilot and 29 pilots
    dict(name="tab_trial_one", source="f27_selection_one_pilot", cols=[0, 1, 2, 4, 5, 6, 7, 8, 9, 12, 13, 14],
         rows=lambda r: r.sticky[1].startswith(r"$\sigma") and not r.text[2].startswith("the two named"), fill=[0, 1],
         header=[r"Set & noise & candidate pool & valid & \multicolumn{2}{c}{catastrophe rate} & win vs & default & win vs & \multicolumn{3}{c}{median error} \\",
                 r"\cmidrule(lr){5-6}\cmidrule(lr){10-12}",
                 r" & & & & chosen & default & default & chosen & last & chosen & default & last \\"]),
    dict(name="tab_trial_many", source="f27b_selection_many_pilots", cols=[0, 1, 2, 4, 5, 6, 7, 8, 9, 12, 13, 14],
         rows=lambda r: r.sticky[1].startswith(r"$\sigma") and not r.text[2].startswith("the two named"), fill=[0, 1],
         header=[r"Set & noise & candidate pool & valid & \multicolumn{2}{c}{catastrophe rate} & win vs & default & win vs & \multicolumn{3}{c}{median error} \\",
                 r"\cmidrule(lr){5-6}\cmidrule(lr){10-12}",
                 r" & & & & chosen & default & default & chosen & last & chosen & default & last \\"]),
    # Section 8.5: choosing by trial on the recorded curves (the lists of chosen methods stay in the supplement)
    dict(name="tab_trial_rec", source="f28_selection_recorded", cols=[0, 1, 2, 3, 4, 5, 6, 7, 8],
         colspec=RAGGED + r"p{0.21\linewidth}lrrrrrrr",
         header=[r"Design & candidate pool & folds & \multicolumn{3}{c}{chosen score vs default's} & chosen & \multicolumn{2}{c}{median score} \\",
                 r"\cmidrule(lr){4-6}\cmidrule(lr){8-9}",
                 r" & & & below & equal & above & $< 1$ & chosen & default \\"]),
]

RULES = ("\\toprule", "\\midrule", "\\bottomrule", "\\addlinespace")
MULTI = re.compile(r"^\\multicolumn\{(\d+)\}\{([^{}]*)\}\{(.*)\}$", re.S)
CMID = re.compile(r"\\cmidrule(\([^)]*\))?\{(\d+)-(\d+)\}")
NUMBER = re.compile(r"\d+(?:\.\d+)?")


class BuildError(Exception):
    pass


# ── parsing ──────────────────────────────────────────────────────────────────
def split_cells(text):
    """Split a table row at the '&' signs that are not escaped and not inside braces."""
    cells, cur, depth, i = [], [], 0, 0
    while i < len(text):
        ch = text[i]
        if ch == "\\" and i + 1 < len(text):
            cur.append(text[i:i + 2])
            i += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if ch == "&" and depth == 0:
            cells.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
        i += 1
    cells.append("".join(cur))
    return [c.strip() for c in cells]


def parse_colspec(spec):
    """'llrp{0.62\\linewidth}' -> ['l', 'l', 'r', 'p{0.62\\linewidth}'] (one token per column)."""
    tokens, i = [], 0
    while i < len(spec):
        ch = spec[i]
        if ch in "lrc":
            tokens.append(ch)
            i += 1
        elif ch in "pmb":
            j = spec.index("}", i)
            tokens.append(spec[i:j + 1])
            i = j + 1
        elif ch.isspace():
            i += 1
        else:
            raise BuildError(f"column specification {spec!r}: unsupported token at {spec[i:]!r}")
    return tokens


class Row:
    """One table row: cells = [(span, align or None, text)] in source order."""

    def __init__(self, cells, ncols, where):
        self.cells = cells
        self.text = [""] * ncols
        c = 0
        for span, _align, text in cells:
            if c < ncols:
                self.text[c] = text
            c += span
        if c != ncols:
            raise BuildError(f"{where}: the row spans {c} columns, the table has {ncols}")
        self.sticky = list(self.text)
        self.block = 0
        self.i = -1

    def render(self, keep=None):
        out, c = [], 0
        keepset = None if keep is None else set(keep)
        for span, align, text in self.cells:
            k = span if keepset is None else sum(1 for j in range(c, c + span) if j in keepset)
            if k:
                out.append(text if align is None else f"\\multicolumn{{{k}}}{{{align}}}{{{text}}}")
            c += span
        return " & ".join(out) + " \\\\"


def parse_row(line, ncols, where):
    body = line[:-2].rstrip()
    cells = []
    for cell in split_cells(body):
        m = MULTI.match(cell)
        cells.append((int(m.group(1)), m.group(2), m.group(3)) if m else (1, None, cell))
    return Row(cells, ncols, where)


class Fragment:
    def __init__(self, name):
        self.name = name
        path = os.path.join(FRAGMENTS, name + ".tex")
        if not os.path.exists(path):
            raise BuildError(f"fragments/{name}.tex is missing (copy the repository's paper_fragments/*.tex into fragments/)")
        lines = [l.rstrip() for l in open(path, encoding="utf-8").read().splitlines()]
        self.comments = [l for l in lines if l.startswith("%")]
        code = [l.strip() for l in lines if l.strip() and not l.startswith("%")]
        begins = [l for l in code if l.startswith("\\begin{tabular}")]
        if len(begins) != 1 or not code[0].startswith("\\begin{tabular}") or code[-1] != "\\end{tabular}":
            raise BuildError(f"{name}: expected exactly one tabular environment")
        self.spec = parse_colspec(code[0][len("\\begin{tabular}{"):-1])
        self.ncols = len(self.spec)
        self.source_text = "\n".join(code[1:-1])
        self.header, self.body = [], []          # items: ('rule', text) | ('cmid', text) | ('row', Row)
        target, seen_top, n_rows, block = self.header, False, 0, 0
        sticky = [""] * self.ncols
        for k, line in enumerate(code[1:-1]):
            where = f"{name}, line {k + 2} of the table"
            if line == "\\toprule":
                seen_top = True
                continue
            if line == "\\bottomrule":
                continue
            if line.startswith("\\midrule") and target is self.header:
                target = self.body
                continue
            if line.startswith(RULES):
                target.append(("rule", line))
                if line.startswith("\\midrule"):
                    block += 1
                continue
            if line.startswith("\\cmidrule"):
                target.append(("cmid", line))
                continue
            if not line.endswith("\\\\"):
                raise BuildError(f"{where}: not a rule and not a row ending in \\\\: {line[:60]!r}")
            row = parse_row(line, self.ncols, where)
            if target is self.body:
                for c in range(self.ncols):
                    if row.text[c]:
                        sticky[c] = row.text[c]
                row.sticky, row.block, row.i = list(sticky), block, n_rows
                n_rows += 1
            target.append(("row", row))
        if not seen_top or target is self.header:
            raise BuildError(f"{name}: no \\toprule ... \\midrule header found")

    def provenance(self):
        return self.comments[0] if self.comments else "% (no provenance line)"


# ── building ─────────────────────────────────────────────────────────────────
def map_cmid(text, keep):
    """Re-number the \\cmidrule ranges of a header line for the kept columns."""
    pos = {c: i + 1 for i, c in enumerate(keep)}          # source column (0-based) -> new column (1-based)
    out = []
    for m in CMID.finditer(text):
        cols = [pos[c] for c in range(int(m.group(2)) - 1, int(m.group(3))) if c in pos]
        if cols:
            out.append(f"\\cmidrule{m.group(1) or ''}{{{min(cols)}-{max(cols)}}}")
    return "".join(out)


def tidy(items):
    """Drop separators at the start and the end and collapse runs of separators (a \\midrule wins)."""
    out = []
    for kind, val in items:
        if kind == "rule":
            if not out:
                continue
            if out[-1][0] == "rule":
                if val.startswith("\\midrule"):
                    out[-1] = (kind, val)
                continue
        out.append((kind, val))
    while out and out[-1][0] == "rule":
        out.pop()
    return out


def numeric_tokens(text):
    text = re.sub(r"\\multicolumn\{\d+\}", r"\\multicolumn", text)
    text = CMID.sub("", text)
    text = re.sub(r"\\addlinespace\[[^\]]*\]", "", text)
    return set(NUMBER.findall(text))


def build(spec):
    frag = Fragment(spec["source"])
    keep = list(range(frag.ncols)) if spec.get("cols") is None else list(spec["cols"])
    if keep != sorted(set(keep)) or not keep or keep[-1] >= frag.ncols or keep[0] < 0:
        raise BuildError(f"{spec['name']}: cols must be increasing column numbers between 0 and {frag.ncols - 1}")
    choose = spec.get("rows")
    fill = [c for c in (spec.get("fill") or []) if c in keep]

    if spec.get("header") is not None:
        header = list(spec["header"])
    else:
        header = []
        for kind, val in frag.header:
            if kind == "row":
                header.append(val.render(keep))
            elif kind == "cmid":
                mapped = map_cmid(val, keep)
                if mapped:
                    header.append(mapped)
            else:
                header.append(val)

    items, printed = [], {}
    for kind, val in frag.body:
        if kind != "row":
            items.append((kind, val))
            continue
        if choose is not None and not choose(val):
            continue
        if fill:
            cells, c, changed = [], 0, False
            for span, align, text in val.cells:
                if align is None and c in fill:
                    label = val.sticky[c]
                    if changed or printed.get(c) != label:
                        text, changed = label, True
                        printed[c] = label
                    else:
                        text = ""
                cells.append((span, align, text))
                c += span
            val = Row(cells, frag.ncols, spec["name"])
        items.append(("row", val.render(keep)))
    items = tidy(items)
    if not any(kind == "row" for kind, _ in items):
        raise BuildError(f"{spec['name']}: no row was kept")
    body = [val for _kind, val in items]

    colspec = spec.get("colspec") or "".join(frag.spec[c] for c in keep)
    if spec.get("full", True):
        begin, end = f"\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep{{\\fill}}}}{colspec}@{{}}}}", "\\end{tabular*}"
    else:
        begin, end = f"\\begin{{tabular}}{{{colspec}}}", "\\end{tabular}"

    stray = numeric_tokens("\n".join(header + body)) - numeric_tokens(frag.source_text)
    if stray:
        raise BuildError(f"{spec['name']}: numbers not present in fragments/{frag.name}.tex: {sorted(stray)}")

    lines = [f"% built by {SELF} from the fragment {frag.name}.tex -- do not edit; edit TABLES in {SELF}",
             frag.provenance(), begin, "\\toprule"] + header + ["\\midrule"] + body + ["\\bottomrule", end]
    return frag, "\n".join(lines) + "\n", sum(1 for kind, _ in items if kind == "row")


def write_supp_refs():
    r"""tables/supp_refs.tex: \Sref{<fragment>} prints the number of the supplementary table that shows
    that fragment, read from the order of the \label{stab:...} lines of supplement.tex."""
    path = os.path.join(HERE, "supplement.tex")
    if not os.path.exists(path):
        if os.path.exists(os.path.join(HERE, "asoc_paper.tex")):
            raise BuildError(f"supplement.tex is missing beside {SELF} (needed for the supplementary table numbers)")
        return None                      # in the repository there is no supplement to number
    labels = re.findall(r"\\label\{stab:([^}]+)\}", open(path, encoding="utf-8").read())
    if not labels or len(labels) != len(set(labels)):
        raise BuildError("supplement.tex: no table label found, or a label occurs twice")
    lines = [f"% built by {SELF} from supplement.tex -- do not edit",
             "% \\Sref{<fragment>} prints the number of the supplementary table that shows that fragment",
             "\\makeatletter"]
    lines += [f"\\@namedef{{stab@{name}}}{{S{k}}}" for k, name in enumerate(labels, 1)]
    lines += ["\\newcommand{\\Sref}[1]{\\@ifundefined{stab@#1}{\\textbf{S??}\\@latex@warning{No supplementary table for #1}}{\\@nameuse{stab@#1}}}",
              "\\makeatother"]
    with open(os.path.join(OUT, "supp_refs.tex"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    return labels


def selftest():
    """Every fragment, passed through the engine unchanged, must give back its own rows and rules."""
    names = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(FRAGMENTS, "f*.tex")))
    bad = 0
    for name in names:
        try:
            frag, text, _ = build(dict(name="selftest", source=name, full=False))
            got = [l for l in text.splitlines() if not l.startswith("%")]
            want = ["\\begin{tabular}{" + "".join(frag.spec) + "}"] + frag.source_text.splitlines() + ["\\end{tabular}"]
            norm = lambda ls: [re.sub(r"\s+", " ", l).strip() for l in ls if l.strip()]
            # the engine collapses a run of separators into one (a \midrule wins) and drops a separator
            # that directly precedes \bottomrule; apply exactly that to the fragment before comparing
            g, raw, w = norm(got), norm(want), []
            sep = ("\\addlinespace", "\\midrule")
            for l in raw:
                if l.startswith(sep) and w and w[-1].startswith(sep):
                    if l.startswith("\\midrule"):
                        w[-1] = l
                    continue
                if l == "\\bottomrule" and w and w[-1].startswith(sep):
                    w.pop()
                w.append(l)
            if g != w:
                bad += 1
                k = next((i for i, (a, b) in enumerate(zip(g, w)) if a != b), min(len(g), len(w)))
                print(f"  SELFTEST DIFFERS  {name}: line {k}: built {g[k][:70] if k < len(g) else None!r} | fragment {w[k][:70] if k < len(w) else None!r}")
        except BuildError as exc:
            bad += 1
            print(f"  SELFTEST ERROR    {exc}")
    print(f"  selftest: {len(names)} fragments, {len(names) - bad} reproduced, {bad} not")
    return bad == 0 and bool(names)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build the manuscript's tables from the generated fragments.")
    ap.add_argument("--selftest", action="store_true", help="also pass every fragment through the engine unchanged")
    ap.add_argument("--list", action="store_true", help="list the tables and stop")
    ap.add_argument("--fragments", help="folder with the generated fragments (default: found from the script's place)")
    ap.add_argument("--out", help="folder to write the tables to (default: found from the script's place)")
    args = ap.parse_args(argv)
    global FRAGMENTS, OUT
    if args.fragments:
        FRAGMENTS = os.path.abspath(args.fragments)
    if args.out:
        OUT = os.path.abspath(args.out)
    if args.list:
        for t in TABLES:
            print(f"  tables/{t['name']}.tex  <-  fragments/{t['source']}.tex")
        return 0
    if not glob.glob(os.path.join(FRAGMENTS, "f*.tex")):
        print(f"ERROR: no fragment f*.tex in {FRAGMENTS}")
        return 1
    os.makedirs(OUT, exist_ok=True)
    ok, runs = True, set()
    for t in TABLES:
        try:
            frag, text, n = build(t)
        except BuildError as exc:
            ok = False
            print(f"  FAILED  {exc}")
            continue
        with open(os.path.join(OUT, t["name"] + ".tex"), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        runs.add(frag.provenance())
        print(f"  wrote tables/{t['name']}.tex  ({n} rows from fragments/{frag.name}.tex)")
    if len(runs) > 1:
        ok = False
        print("  FAILED  the fragments used come from different runs:")
        for r in sorted(runs):
            print("    " + r[:150])
    elif runs:
        print("  fragments: " + next(iter(runs)).lstrip("% ")[:170])
    try:
        labels = write_supp_refs()
        if labels is None:
            labels = []
            print("  no supplement.tex beside the script: the supplementary table numbers are not written")
        else:
            print(f"  wrote tables/supp_refs.tex  ({len(labels)} supplementary tables)")
        used = set()
        if os.path.exists(os.path.join(HERE, "asoc_paper.tex")):
            with open(os.path.join(HERE, "asoc_paper.tex"), encoding="utf-8") as fh:
                text = re.sub(r"(?<!\\)%.*", "", fh.read())          # comments do not count
            used = set(re.findall(r"\\Sref\{([^}]+)\}", text))
        missing = sorted(used - set(labels)) if labels else []
        if missing:
            ok = False
            print("  FAILED  the manuscript cites supplementary tables that supplement.tex does not have: " + ", ".join(missing))
    except BuildError as exc:
        ok = False
        print(f"  FAILED  {exc}")
    if args.selftest:
        ok = selftest() and ok
    print("TABLE BUILD: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
