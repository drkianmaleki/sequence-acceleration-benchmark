"""
phases/phase1.py
================
Phase 1 — Full Synthetic Benchmark.

This module is a thin wrapper that exposes the Phase 1 entry point from
src/evaluation.py under the same phases/ package structure used by
phases/phase2.py through phases/phase5b.py.

The core evaluation logic lives in src/evaluation.py so that it can be
imported by both this module and the run scripts without circular imports.

Usage (direct)
--------------
    from phases.phase1 import run as run_phase1
    results = run_phase1(n_seeds=30, noise_levels=[0.0, 0.001, 0.005], ...)

Usage (via script)
------------------
    python scripts/run_phase1.py --full
"""

from src.evaluation import run_phase1 as run      # re-export under canonical name
from src.evaluation import build_cfg, stability_score, is_valid

__all__ = ["run", "build_cfg", "stability_score", "is_valid"]
