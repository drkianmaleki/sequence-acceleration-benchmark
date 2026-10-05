"""
tests/test_fragment_text.py
===========================
The printed text of every fragment follows the paper's terminology: no
printed (non-comment) line contains "dangerous", "current" or "regime" in
any case, nor L_{\\mathrm{true}}, nor a '<' or '>' outside math mode, nor an
escaped-underscore identifier outside \\meth{}, \\texttt{} or \\diag{}.
Comment lines (starting with %) are not checked; the standalone document
all_tables.tex is not a fragment.
"""

import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
FRAG_DIR = os.path.join(_ROOT, "paper_fragments")

FORBIDDEN_WORDS = ("dangerous", "current", "regime")
_MATH = re.compile(r"\$[^$]*\$")
_WRAPPED = re.compile(r"\\(meth|texttt|diag)\{[^{}]*\}")


def fragments():
    return sorted(f for f in os.listdir(FRAG_DIR) if re.fullmatch(r"f\d+.*\.tex", f))


def printed_lines(name):
    with open(os.path.join(FRAG_DIR, name), encoding="utf-8") as fh:
        for i, line in enumerate(fh.read().splitlines(), 1):
            if not line.startswith("%"):
                yield i, line


def problems(name):
    out = []
    for i, line in printed_lines(name):
        low = line.lower()
        for w in FORBIDDEN_WORDS:
            if w in low:
                out.append(f"{name}:{i}: contains '{w}': {line.strip()}")
        if r"L_{\mathrm{true}}" in line:
            out.append(f"{name}:{i}: prints L_true: {line.strip()}")
        no_math = _MATH.sub("", line)
        if "<" in no_math or ">" in no_math:
            out.append(f"{name}:{i}: '<' or '>' outside math mode: {line.strip()}")
        unwrapped = _WRAPPED.sub("", line)
        if r"\_" in unwrapped:
            out.append(f"{name}:{i}: escaped-underscore identifier outside \\meth / \\texttt / \\diag: {line.strip()}")
    return out


def test_fragments_exist():
    names = fragments()
    assert len(names) >= 30, names


def test_printed_text_terminology():
    bad = []
    for name in fragments():
        bad += problems(name)
    assert not bad, "\n".join(bad)
