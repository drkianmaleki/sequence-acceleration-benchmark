"""
tests/test_tables_regenerate.py
===============================
The reader's check (scripts/check_tables.py): every fragment, the index,
all_tables.tex and FACTS.md regenerate byte for byte (after LF
normalisation) from the committed results/ alone, through the reader's
path of the tercile script, and the code fingerprint matches.  It must pass
in a tree without the git-ignored raw files: no generator reads them.
"""

import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)


def test_tables_regenerate_from_the_committed_results():
    r = subprocess.run([sys.executable, os.path.join(_ROOT, "scripts", "check_tables.py")],
                       cwd=_ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert r.returncode == 0, r.stdout[-6000:] + r.stderr[-3000:]
    assert "TABLES CHECK: PASS" in r.stdout
    assert "DIFFERS" not in r.stdout and "NOT REGENERATED" not in r.stdout and "NOT COMMITTED" not in r.stdout
