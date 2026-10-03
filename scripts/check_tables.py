#!/usr/bin/env python3
"""
check_tables.py  (redesign v2, R9d Part C)
==========================================
The reader's check: regenerate every table from the committed results/ and
compare with what is committed.

    python scripts/check_tables.py

Runs the generators into a temporary folder --
scripts/make_paper_tables.py, scripts/analyze_by_ltrue.py --from-csv (the
reader's path: the committed phase1_by_Ltrue.csv, never the git-ignored
records) and scripts/build_tables_document.py -- compares every file of the
temporary folder with paper_fragments/ and FACTS.md byte for byte after LF
normalisation, lists the committed files that were not regenerated, runs
the code fingerprint check (scripts/run_code_fingerprint.py), prints one
line per file and a verdict, and exits 0 only if everything is identical.
It writes nothing under results/ (the temporary folder is removed at the end).
"""

import os
import shutil
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts.run_code_fingerprint import FINGERPRINT, check_fingerprint, path_set_from_tree   # noqa: E402

GENERATORS = [
    ["scripts/make_paper_tables.py", "--results", "{results}", "--out", "{out}", "--facts", "{facts}"],
    ["scripts/analyze_by_ltrue.py", "--results", "{results}", "--out", "{out}", "--facts", "{facts}", "--from-csv"],
    ["scripts/build_tables_document.py", "--out", "{out}", "--results", "{results}"],
]


def _norm(path):
    with open(path, "rb") as fh:
        return fh.read().replace(b"\r\n", b"\n")


def _run(cmd):
    r = subprocess.run([sys.executable] + cmd, cwd=_ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(f"  GENERATOR FAILED: {' '.join(cmd)} (exit {r.returncode})")
        print(r.stdout[-4000:])
        print(r.stderr[-4000:])
    return r.returncode == 0


def main(results_dir=None, fragments_dir=None, facts_path=None) -> int:
    results_dir = results_dir or os.path.join(_ROOT, "results")
    fragments_dir = fragments_dir or os.path.join(_ROOT, "paper_fragments")
    facts_path = facts_path or os.path.join(_ROOT, "FACTS.md")
    tmp = tempfile.mkdtemp(prefix="check_tables_")
    out, facts = os.path.join(tmp, "paper_fragments"), os.path.join(tmp, "FACTS.md")
    os.makedirs(out)
    print(f"check_tables.py: regenerating from {os.path.relpath(results_dir, _ROOT)} into a temporary folder, "
          f"comparing with {os.path.relpath(fragments_dir, _ROOT)}/ and {os.path.relpath(facts_path, _ROOT)}")
    ok = True
    try:
        for cmd in GENERATORS:
            cmd = [c.format(results=results_dir, out=out, facts=facts) for c in cmd]
            if not _run(cmd):
                ok = False
                break
        n_same = n_diff = n_missing = n_extra = 0
        if ok:
            gen = sorted(os.listdir(out))
            com = sorted(f for f in os.listdir(fragments_dir) if os.path.isfile(os.path.join(fragments_dir, f)))
            for name in gen:
                p_new, p_old = os.path.join(out, name), os.path.join(fragments_dir, name)
                if not os.path.exists(p_old):
                    print(f"  NOT COMMITTED   paper_fragments/{name}")
                    n_extra += 1
                elif _norm(p_new) == _norm(p_old):
                    print(f"  identical       paper_fragments/{name}")
                    n_same += 1
                else:
                    print(f"  DIFFERS         paper_fragments/{name}")
                    n_diff += 1
            for name in com:
                if name not in gen:
                    print(f"  NOT REGENERATED paper_fragments/{name}")
                    n_missing += 1
            if _norm(facts) == _norm(facts_path):
                print("  identical       FACTS.md")
                n_same += 1
            else:
                print("  DIFFERS         FACTS.md")
                n_diff += 1
            ok = n_diff == 0 and n_missing == 0 and n_extra == 0
            print(f"  {n_same} identical, {n_diff} differ, {n_missing} not regenerated, {n_extra} not committed")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("fingerprint check:")
    fp_ok = check_fingerprint(_ROOT, path_set_from_tree(_ROOT), os.path.join(_ROOT, FINGERPRINT)) == 0
    ok = ok and fp_ok
    print("TABLES CHECK: " + ("PASS -- every fragment, the index, all_tables.tex and FACTS.md regenerate identically from the committed results, "
                              "and the code fingerprint matches" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
