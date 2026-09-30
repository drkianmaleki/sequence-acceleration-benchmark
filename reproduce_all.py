"""
reproduce_all.py  (redesign v2)
===============================
Single entry point that reproduces every result of the redesign-v2 pipeline
in the required order:

    Phase 0   analytic unit tests of the accelerator roster (src.pipeline.ACCEL_METHODS)
    Phase 1   main benchmark (24 regimes x 30 seeds x 3 noise x 3 gap strata x every
              registered method; the counts are printed by --plan)
    Phase 0b  order ladders: every order of every family with an order parameter,
              on Phase 1's grid at the headline stratum, with an exact-agreement
              check against phase1_records.csv (scripts/order_ladders.py)
    derive    excluded-method re-derivation  ->  results/phase1/dangerous_methods.json
              (pooled validity below config.RANK_MIN_VALID; 'dangerous' is the legacy name)
    Phase 2   Richardson failure characterisation (13 depths, core regimes)
    Phase 3   adaptive selection (analysis of Phase 2)
    Phase 4   perturbation / shift diagnostics (4 depths)
    Phase 5a  full-pool ensemble ablation (4 depths; (obs_idx x noise) blocks
              evaluated in cpu_count() - 1 worker processes; byte-identical output
              for any job count, see scripts/run_phase5a.py --jobs)
    Phase 5b  sensitivity sweeps (assumed asymptote, window length)
    real data re-evaluation of the recorded XGBoost curves on the (depth x target) grid
    tables    scripts/make_paper_tables.py  ->  paper_fragments/*.tex + FACTS.md
    terciles  scripts/analyze_by_ltrue.py   ->  results/phase1/phase1_by_Ltrue.csv, f13, its FACTS section
              (after the tables step, because it replaces its own section of FACTS.md)
    manifest  results/run_manifest.json: mode, git head, start / end, worker count, per-step
              and total wall seconds, the evaluation plan of the mode, library versions
              (written at the end of every run; the committed copy is the full run's)

The excluded-method set is derived from Phase-1 output into
results/phase1/dangerous_methods.json; phases 2-5 refuse to run without
that artifact, so the ordering is enforced by construction and executed
explicitly here.  A failed Phase 1 or derivation stops the run; any other
failure is reported in the summary table and the remaining steps still run.

Usage
-----
    python reproduce_all.py                # full run (hours; the wall time of the last full run
                                           # is in results/run_manifest.json and README.md)
    python reproduce_all.py --quick        # 2 seeds, 2 regimes per group, 2 noise
                                           # levels: every phase end to end (~2-3 min)
    python reproduce_all.py --plan         # print the evaluation counts per phase
    python reproduce_all.py --skip-real-data

Afterwards:
    python scripts/check_dangerous.py      # artifact still matches Phase 1

Per-phase grids live in src/config.py (PHASE1 ... PHASE5B, REAL_DATA).
Results are only meaningful alongside the commit that produced them, so
regenerate the full set rather than mixing output from different commits.
"""

import argparse
import datetime as _dt
import json
import os
import platform
import subprocess
import sys
import time

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

MANIFEST = os.path.join("results", "run_manifest.json")   # written at the end of every run (quick or full)


# (label, script, takes_mode_flag)
STEPS = [
    ("Phase 0 — analytic unit tests",              "scripts/run_phase0_tests.py", False),
    ("Phase 1 — main benchmark",                   "scripts/run_phase1.py",       True),
    ("Phase 0b — order ladders",                   "scripts/order_ladders.py",    True),
    ("Dangerous re-derivation (Phase 1 -> artifact)", "scripts/derive_dangerous.py", False),
    ("Phase 2 — failure detection",                "scripts/run_phase2.py",       True),
    ("Phase 3 — adaptive selection",               "scripts/run_phase3.py",       True),
    ("Phase 4 — perturbation diagnostics",         "scripts/run_phase4.py",       True),
    ("Phase 5a — ensemble ablation",               "scripts/run_phase5a.py",      True),
    ("Phase 5b — sensitivity sweeps",              "scripts/run_phase5b.py",      True),
    ("Real data — recorded-curve re-evaluation",   "scripts/run_real_data.py",    True),
    ("Paper tables and FACTS",                     "scripts/make_paper_tables.py", False),
    ("L_true terciles (FACTS section, f13)",       "scripts/analyze_by_ltrue.py",  False),
]


def parse_args():
    p = argparse.ArgumentParser(description="Reproduce all results (redesign v2).")
    p.add_argument("--quick", action="store_true",
                   help="2 seeds, 2 regimes per group, 2 noise levels; every phase runs.")
    p.add_argument("--plan", action="store_true",
                   help="Print planned evaluation counts per phase (full and quick) and exit.")
    p.add_argument("--skip-real-data", action="store_true",
                   help="Skip the real-data re-evaluation step.")
    return p.parse_args()


def plan(mode: str) -> list:
    """Planned evaluation counts per phase from the config grids."""
    import src.config as C
    from src.accelerators import METHOD_NAMES
    from src.pipeline import resolve_regimes, ACCEL_METHODS, TRIVIAL_NON_ORACLE
    from phases.phase2 import PHASE2_METHODS
    from phases.phase4 import PHASE4_METHODS, EVAL_METHODS as P4_EVAL
    from phases.phase5a import PERTURB_METHODS as P5A_PERTURB

    from tests.test_accelerators import TEST_CASES
    from src.trivial import SKILL_REFERENCE_METHODS

    rows = []
    # Phase 0: the accelerator roster x the analytic cases of the harness
    rows.append(("Phase 0", len(ACCEL_METHODS) * len(TEST_CASES),
                 f"{len(ACCEL_METHODS)} accelerators x {len(TEST_CASES)} analytic cases (trivials excluded)"))

    c = C.PHASE1[mode]
    r = resolve_regimes(c["core_regimes"], c["holdout_regimes"], True)
    n1 = len(r) * c["n_seeds"] * len(c["noise_levels"]) * len(c["gap_fractions"]) * len(METHOD_NAMES)
    rows.append(("Phase 1", n1, f"{len(r)} regimes x {c['n_seeds']} seeds x {len(c['noise_levels'])} noise x "
                                f"{len(c['gap_fractions'])} strata x {len(METHOD_NAMES)} methods"))
    from scripts.order_ladders import all_variants, build_ladders, COMPARATORS
    n_lad, n_all = len(build_ladders()), len(all_variants())
    n0b = n_all * len(r) * c["n_seeds"] * len(c["noise_levels"])
    rows.append(("Phase 0b", n0b, f"{n_lad} ladder variants + {len(COMPARATORS)} comparators x {len(r)} regimes x "
                                  f"{c['n_seeds']} seeds x {len(c['noise_levels'])} noise at g = {C.HEADLINE_G} "
                                  f"(+ agreement check against phase1_records.csv)"))
    rows.append(("Dangerous derivation", 0, "reads phase1_aggregated.csv"))

    c = C.PHASE2[mode]
    r = resolve_regimes(c["core_regimes"], include_holdout=False)
    cells2 = len(c["obs_idx_list"]) * len(c["noise_list"]) * c["n_seeds"] * len(r) * len(c["gap_fractions"])
    rows.append(("Phase 2", cells2 * len(PHASE2_METHODS),
                 f"{len(c['obs_idx_list'])} depths x {len(c['noise_list'])} noise x {c['n_seeds']} seeds x "
                 f"{len(r)} core regimes x {len(c['gap_fractions'])} strata x {len(PHASE2_METHODS)} methods "
                 f"(+2 skill-reference calls per cell)"))
    rows.append(("Phase 3", 0, "analysis of Phase 2 output"))

    c = C.PHASE4[mode]
    r = resolve_regimes(c["core_regimes"], c["holdout_regimes"], True)
    cells4 = len(c["obs_idx_list"]) * len(c["noise_list"]) * c["n_seeds"] * len(r) * len(c["gap_fractions"])
    central4 = cells4 * len(P4_EVAL)
    diag4 = cells4 * len(PHASE4_METHODS) * (len(c["shifts"]) + c["perturb_trials"])
    rows.append(("Phase 4", central4, f"central evaluations ({len(r)} regimes, {len(P4_EVAL)} methods); "
                                      f"+ {diag4:,} diagnostic calls"))

    c = C.PHASE5A[mode]
    r = resolve_regimes(c["core_regimes"], c["holdout_regimes"], True)
    cells5 = len(c["obs_idx_list"]) * len(c["noise_list"]) * c["n_seeds"] * len(r) * len(c["gap_fractions"])
    central5 = cells5 * len(METHOD_NAMES)
    pert5 = cells5 * len(P5A_PERTURB) * c["perturb_trials"]
    rows.append(("Phase 5a", central5, f"central evaluations ({len(r)} regimes, {len(METHOD_NAMES)} methods); "
                                       f"+ {pert5:,} perturbation calls"))

    c = C.PHASE5B[mode]
    r = resolve_regimes(c["core_regimes"], c["holdout_regimes"], True)
    base = len(c["noise_list"]) * c["n_seeds"] * len(r) * len(c["gap_fractions"])
    from phases.phase5b import CASCADE_EVAL_METHODS, SWEEP1_METHODS
    n_extra_refs = len(set(SKILL_REFERENCE_METHODS) - set(SWEEP1_METHODS))   # trivial references sweep 1b adds
    s1 = len(c["assumed_modes"]) * base * len(CASCADE_EVAL_METHODS)
    s1b = len(c["assumed_modes"]) * base * (len(SWEEP1_METHODS) + n_extra_refs)
    s2 = len(c["window_lengths"]) * base * len(CASCADE_EVAL_METHODS)
    rows.append(("Phase 5b", s1 + s1b + s2, f"sweep1a {s1:,} + sweep1b {s1b:,} + sweep2 {s2:,} "
                                           f"({len(r)} regimes, {len(c['gap_fractions'])} strata)"))

    from src.datasets import DATASET_IDS
    from src.trajectories import REAL_DATA_METHODS
    rd = C.REAL_DATA
    pairs = [(d, t) for d in rd["depths"] for t in rd["targets"] if d < t]
    n_ds, n_m = len(DATASET_IDS), len(REAL_DATA_METHODS)
    rows.append(("Real data", n_ds * len(pairs) * n_m,
                 f"{n_ds} datasets x {len(pairs)} (depth, target) pairs x {n_m} methods"))
    return rows


def print_plan():
    for mode in ("full", "quick"):
        rows = plan(mode)
        total = sum(n for _, n, _ in rows)
        print("=" * 78)
        print(f"  PLANNED EVALUATIONS  [{mode.upper()}]")
        print("=" * 78)
        for label, n, note in rows:
            print(f"  {label:<22} {n:>12,}   {note}")
        print("  " + "-" * 74)
        print(f"  {'TOTAL (central)':<22} {total:>12,}")
        print()


def _git_head() -> dict:
    """Full and short hash of the checked-out commit ("unknown" outside a git checkout)."""
    out = {}
    for key, args in (("full", ["rev-parse", "HEAD"]), ("short", ["rev-parse", "--short", "HEAD"])):
        try:
            out[key] = subprocess.check_output(["git"] + args, cwd=_ROOT, stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            out[key] = "unknown"
    return out


def _versions() -> dict:
    """Python and the numerical libraries the run depends on ("not installed" when absent)."""
    out = {"python": platform.python_version()}
    for name in ("numpy", "scipy", "pandas", "sklearn", "xgboost"):
        try:
            out[name] = __import__(name).__version__
        except Exception:
            out[name] = "not installed"
    return out


def write_manifest(mode: str, outcomes: list, started: str, finished: str, jobs: int) -> str:
    """
    results/run_manifest.json: the provenance of the run that produced the
    results tree -- mode, git head, start and end time, the worker count the
    parallel steps used, every step's wall time in order, the total, the
    evaluation plan of that mode and the library versions.  The committed
    copy is the one of the full run; a quick run overwrites it in a
    worktree only.
    """
    path = os.path.join(_ROOT, MANIFEST)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "mode": mode,
        "git_head": _git_head(),
        "started": started,
        "finished": finished,
        "jobs": jobs,
        "steps": {label: round(elapsed, 1) for label, _ok, elapsed in outcomes},
        "failed_steps": [label for label, ok, _ in outcomes if not ok],
        "total_seconds": round(sum(elapsed for _, _, elapsed in outcomes), 1),
        "plan": [{"step": label, "evaluations": n, "note": note} for label, n, note in plan(mode)],
        "versions": _versions(),
    }
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=2)
    return path


def run_step(label: str, cmd: list, step: int, total: int):
    bar = "=" * 72
    print(f"\n{bar}")
    print(f"  [{step}/{total}]  {label}")
    print(f"  $ {' '.join(os.path.relpath(c, _ROOT) if os.path.isabs(c) else c for c in cmd[1:])}")
    print(bar, flush=True)
    t0 = time.time()
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    result = subprocess.run(cmd, cwd=_ROOT, env=env)
    elapsed = time.time() - t0
    ok = result.returncode == 0
    if ok:
        print(f"\n  Completed in {elapsed:.0f}s", flush=True)
    else:
        print(f"\n  ERROR: {label} failed (exit code {result.returncode}).")
        print(f"  Elapsed: {elapsed:.0f}s\n", flush=True)
    return ok, elapsed


def main():
    args = parse_args()
    if args.plan:
        print_plan()
        return

    mode_flag = "--quick" if args.quick else "--full"
    steps = [s for s in STEPS if not (args.skip_real_data and s[1].endswith("run_real_data.py"))]

    mode = "QUICK" if args.quick else "FULL"
    print("=" * 72)
    print(f"  Reproducing all results  [{mode} mode, redesign v2]")
    print("=" * 72)
    print(f"  {len(steps)} steps.  Order: Phase 0 -> Phase 1 -> Phase 0b (ladders) -> "
          f"excluded-set re-derivation -> Phases 2-5 -> real data -> tables -> L_true terciles")
    print("  Output: results/<phase>/\n")

    outcomes = []
    t_start = time.time()
    started = _dt.datetime.now().isoformat(timespec="seconds")
    for i, (label, script, takes_mode) in enumerate(steps, 1):
        cmd = [sys.executable, script] + ([mode_flag] if takes_mode else [])
        ok, elapsed = run_step(label, cmd, i, len(steps))
        outcomes.append((label, ok, elapsed))
        if not ok and script.endswith(("run_phase1.py", "derive_dangerous.py")):
            print("  Stopping: later phases depend on this step.")
            break
    finished = _dt.datetime.now().isoformat(timespec="seconds")

    # The parallel steps (Phase 5a, Phase 0b) run with their scripts' default
    # worker count; record it with the run.
    from phases.phase5a import default_jobs
    manifest = write_manifest("quick" if args.quick else "full", outcomes, started, finished, default_jobs())

    print("\n" + "=" * 72)
    print(f"  PIPELINE SUMMARY  [{mode}]")
    print("=" * 72)
    print(f"  {'#':>2}  {'step':<48} {'status':<8} {'time':>8}")
    print("  " + "-" * 70)
    for i, (label, ok, elapsed) in enumerate(outcomes, 1):
        print(f"  {i:>2}  {label:<48} {'OK' if ok else 'FAILED':<8} {elapsed:>7.0f}s")
    print("  " + "-" * 70)
    print(f"  {'':>2}  {'total':<48} {'':<8} {time.time() - t_start:>7.0f}s")
    print(f"  Manifest: {os.path.relpath(manifest, _ROOT)}")
    failures = [label for label, ok, _ in outcomes if not ok]
    if not failures and len(outcomes) == len(steps):
        print("  All steps completed successfully.  Results are in results/")
    else:
        print(f"  {len(failures)} step(s) failed: {failures}")
        print("=" * 72 + "\n")
        sys.exit(1)
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
