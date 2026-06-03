"""
run_phase0_tests.py
===================
Phase 0 entry point.

Usage
-----
    cd sequence-acceleration-benchmark/
    python scripts/run_phase0_tests.py

Output (saved to ./results/)
----------------------------
    phase0_unit_tests.csv   ~200 rows, ~30 KB
    phase0_unit_tests.txt   human-readable PASS / fail / INVALID report
"""

import os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tests.test_accelerators import run_all_tests, write_report

if __name__ == "__main__":
    df = run_all_tests()
    write_report(df, "results")
