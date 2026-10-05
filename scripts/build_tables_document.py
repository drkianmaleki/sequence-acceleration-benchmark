#!/usr/bin/env python3
"""
build_tables_document.py  (redesign v2)
===================================================
paper_fragments/all_tables.tex: a self-contained `article` that inputs every
fragment of the folder, in name order, each inside a landscape table scaled
to the page (adjustbox, max width and max total height), with the fragment's
description line as the caption.  The preamble provides the macros the
fragments use (\\meth, \\diag, \\rhoV, \\rhoC, \\TE, \\LE, \\nobs, \\nf) with
\\providecommand, so the file compiles on its own:

    python scripts/build_tables_document.py [--out paper_fragments] [--results results]
    cd paper_fragments && pdflatex -interaction=nonstopmode -halt-on-error all_tables.tex   (twice, for the list of tables)

Pipeline step after the terciles step (reproduce_all.py); scripts/check_tables.py
regenerates it with the fragments.
"""

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts.analyze_by_ltrue import fragment_header, list_fragments, provenance   # noqa: E402

DOCUMENT = "all_tables.tex"
MACROS = [
    (r"\meth", r"[1]{\texttt{#1}}"),
    (r"\diag", r"[1]{\textsf{#1}}"),
    (r"\rhoV", r"{\rho_{\mathrm{v}}}"),
    (r"\rhoC", r"{\rho_{\mathrm{c}}}"),
    (r"\TE", r"{\textsc{te}}"),
    (r"\LE", r"{\textsc{le}}"),
    (r"\nobs", r"{n_{\mathrm{obs}}}"),
    (r"\nf", r"{n_f}"),
]
PACKAGES = ["booktabs", "array", "amsmath", "amssymb", "adjustbox", "pdflscape"]

_TEX_SPECIAL = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
                "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def tex_escape(s):
    return "".join(_TEX_SPECIAL.get(ch, ch) for ch in str(s))


def build(out_dir, results_dir):
    names = list_fragments(out_dir)
    prov = provenance("scripts/build_tables_document.py", results_dir)
    L = [r"\documentclass[10pt]{article}",
         r"% AUTO-GENERATED -- do not hand-edit.  " + prov,
         r"% Every fragment of this folder, input unchanged, in name order; compile twice (list of tables).",
         r"\usepackage[T1]{fontenc}",
         r"\usepackage[utf8]{inputenc}",
         r"\usepackage[a4paper,margin=15mm]{geometry}"]
    L += [rf"\usepackage{{{p}}}" for p in PACKAGES]
    L += [rf"\providecommand{{{m}}}{body}" for m, body in MACROS]
    L += [r"\setlength{\tabcolsep}{4pt}",
          r"\begin{document}",
          r"\section*{Generated tables of the sequence-acceleration benchmark}",
          tex_escape(prov[0].upper() + prov[1:]) + ".  Each table is one fragment of \\texttt{paper\\_fragments/}, input unchanged "
          "and scaled to the page; the caption is the fragment's own description line, and the fragment's comment header names its "
          "sources and filters.  The index \\texttt{README.md} maps every fragment to its sources.",
          r"\listoftables",
          r"\clearpage"]
    for name in names:
        desc, _, _ = fragment_header(os.path.join(out_dir, name))
        label = "tab:" + os.path.splitext(name)[0]
        L += [r"\begin{landscape}",
              r"\begin{table}[p]",
              r"\centering",
              rf"\caption{{\texttt{{{tex_escape(name)}}}: {tex_escape(desc)}}}",
              rf"\label{{{label}}}",
              r"\begin{adjustbox}{max width=\linewidth, max totalheight=0.86\textheight}",
              rf"\input{{{name}}}",
              r"\end{adjustbox}",
              r"\end{table}",
              r"\end{landscape}"]
    L.append(r"\end{document}")
    path = os.path.join(out_dir, DOCUMENT)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(L) + "\n")
    return path, len(names)


def main(argv=None):
    p = argparse.ArgumentParser(description="paper_fragments/all_tables.tex: every fragment in one standalone document")
    p.add_argument("--out", default=os.path.join(_ROOT, "paper_fragments"), help="the fragment folder (input and output)")
    p.add_argument("--results", default=os.path.join(_ROOT, "results"), help="results tree (for the provenance header)")
    args = p.parse_args(argv)
    path, n = build(args.out, args.results)
    print(f"  wrote {os.path.relpath(path, _ROOT)} ({n} fragments input)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
