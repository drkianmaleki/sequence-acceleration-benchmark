"""
reproduce_all.py
================
Single entry point that reproduces all tables in the main text of:

    Maleki, K. (2026). Finite-Horizon Learning-Curve Prediction for Gradient
    Boosting: Regime Dependence, Failure Detection, and Conservative
    Extrapolation Rules. Machine Learning (submitted).

Usage
-----
    pip install -r requirements.txt
    python reproduce_all.py

    # Faster sanity-check run (reduced seeds / noise levels):
    python reproduce_all.py --quick

Runtime (full mode, modern laptop / workstation)
------------------------------------------------
    Phase 0  — unit tests             :  ~1 min
    Phase 1  — synthetic benchmark    :  ~20-40 min
    Phase 2  — failure detection      :  ~25-40 min
    Phase 3  — adaptive selection     :  ~10-20 min
    Phase 4  — perturbation diag.     :  ~15-30 min
    Phase 5a — ensemble ablation      :  ~10-15 min
    Phase 5b — sensitivity sweeps     :  ~10-20 min
    Real data — XGBoost transfer      :  ~15-30 min (requires internet + OpenML)
    Total                             :  ~2-4 hours

Table / figure mapping
----------------------
    Phase 0  → Appendix A.4 (unit-test results)
    Phase 1  → Tables 4, 5, 22, 23; Figures 1-6
    Phase 2  → Tables 7, 8; Figures (phase2)
    Phase 3  → Tables 9, 10, 11; Figures (phase3)
    Phase 4  → Tables 19, 20, 21; Figures (phase4)
    Phase 5a → Table 11 (corrected); Figures (phase5a)
    Phase 5b → Tables 12, 13, 14, 15, 29, 30, 31; Figures (phase5b)
    Real data → Tables 16, 17, 18; Figures (real_data)

All output is written to results/<phase>/ and results/real_data/.
The results/ directory ships empty; every table and figure is generated here.
Results are only meaningful alongside the commit that produced them, so
regenerate the full set rather than mixing output from different commits.
"""

import argparse
import os
import subprocess
import sys
import time


SCRIPTS = [
    ("Phase 0 — unit tests",           [sys.executable, "scripts/run_phase0_tests.py"]),
    ("Phase 1 — synthetic benchmark",  [sys.executable, "scripts/run_phase1.py", "--full"]),
    ("Phase 2 — failure detection",    [sys.executable, "scripts/run_phase2.py",  "--full"]),
    ("Phase 3 — adaptive selection",   [sys.executable, "scripts/run_phase3.py",  "--full"]),
    ("Phase 4 — perturbation diag.",   [sys.executable, "scripts/run_phase4.py",  "--full"]),
    ("Phase 5a — ensemble ablation",   [sys.executable, "scripts/run_phase5a.py", "--full"]),
    ("Phase 5b — sensitivity sweeps",  [sys.executable, "scripts/run_phase5b.py", "--full"]),
    ("Real data — XGBoost transfer",   [sys.executable, "scripts/run_real_data.py"]),
]

SCRIPTS_QUICK = [
    ("Phase 0 — unit tests",           [sys.executable, "scripts/run_phase0_tests.py"]),
    ("Phase 1 — synthetic benchmark",  [sys.executable, "scripts/run_phase1.py",  "--quick"]),
    ("Phase 2 — failure detection",    [sys.executable, "scripts/run_phase2.py",  "--quick"]),
    ("Phase 3 — adaptive selection",   [sys.executable, "scripts/run_phase3.py",  "--quick"]),
    ("Phase 4 — perturbation diag.",   [sys.executable, "scripts/run_phase4.py",  "--quick"]),
    ("Phase 5a — ensemble ablation",   [sys.executable, "scripts/run_phase5a.py", "--quick"]),
    ("Phase 5b — sensitivity sweeps",  [sys.executable, "scripts/run_phase5b.py", "--quick"]),
    ("Real data — XGBoost transfer",   [sys.executable, "scripts/run_real_data.py"]),
]


def parse_args():
    p = argparse.ArgumentParser(description="Reproduce all tables in the paper.")
    p.add_argument("--quick", action="store_true",
                   help="Reduced seeds/noise for a fast sanity-check run.")
    p.add_argument("--skip-real-data", action="store_true",
                   help="Skip the OpenML download step (requires internet).")
    return p.parse_args()


def run_step(label: str, cmd: list, step: int, total: int) -> bool:
    bar = "=" * 72
    print(f"\n{bar}")
    print(f"  [{step}/{total}]  {label}")
    print(bar)
    t0 = time.time()
    # Phase scripts print box-drawing characters; force UTF-8 in the child so
    # that redirecting output to a file cannot fail on a non-UTF-8 console.
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)),
                            env=env)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n  ERROR: {label} failed (exit code {result.returncode}).")
        print(f"  Elapsed: {elapsed:.0f}s\n")
        return False
    print(f"\n  Completed in {elapsed:.0f}s")
    return True


def main():
    args = parse_args()
    plan = SCRIPTS_QUICK if args.quick else SCRIPTS

    if args.skip_real_data:
        plan = [(label, cmd) for label, cmd in plan
                if "real data" not in label.lower()]

    mode = "QUICK" if args.quick else "FULL"
    print("=" * 72)
    print(f"  Reproducing all paper tables  [{mode} mode]")
    print("=" * 72)
    print(f"  {len(plan)} phases to run.")
    print("  Output: results/<phase>/\n")

    failures = []
    for i, (label, cmd) in enumerate(plan, 1):
        ok = run_step(label, cmd, i, len(plan))
        if not ok:
            failures.append(label)

    print("\n" + "=" * 72)
    if not failures:
        print("  All phases completed successfully.")
        print("  Results are in results/")
    else:
        print(f"  {len(failures)} phase(s) failed:")
        for f in failures:
            print(f"    - {f}")
        sys.exit(1)
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
