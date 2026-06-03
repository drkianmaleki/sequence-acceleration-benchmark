# sequence-acceleration-benchmark

Benchmark code for the paper:

> Maleki, K. (2026). **Finite-Horizon Learning-Curve Prediction for Gradient Boosting: Regime Dependence, Failure Detection, and Conservative Extrapolation Rules.** *Machine Learning* (submitted).

51 sequence-acceleration methods from 13 families are evaluated across 18 synthetic convergence regimes, 3 noise levels, 3 prediction horizons, 30 seeds, and 10 observation depths (2.478 M evaluations total).

---

## Installation

```bash
git clone https://github.com/drkianmaleki/sequence-acceleration-benchmark.git
cd sequence-acceleration-benchmark
pip install -r requirements.txt
```

Python 3.9 or later is required.

---

## Reproducing all tables

A single script runs every phase in sequence and writes all output to `results/`:

```bash
python reproduce_all.py          # full run (~2-4 hours)
python reproduce_all.py --quick  # reduced run for a sanity check (~15 min)
```

`--quick` uses fewer seeds and noise levels; all phases still run end-to-end.

The real-data phase (Section 9) downloads datasets from OpenML and requires an
internet connection. To skip it:

```bash
python reproduce_all.py --skip-real-data
```

Pre-computed results for all phases are already committed to `results/` and
match the numbers in the paper.

---

## Running individual phases

| Script | Phase | Paper section | Runtime (full) |
|--------|-------|---------------|----------------|
| `scripts/run_phase0_tests.py` | Unit tests | Appendix A | ~1 min |
| `scripts/run_phase1.py --full` | Synthetic benchmark | §4–5 | ~20–40 min |
| `scripts/run_phase2.py --full` | Failure detection | §6 | ~25–40 min |
| `scripts/run_phase3.py --full` | Adaptive selection | §7 | ~10–20 min |
| `scripts/run_phase4.py --full` | Perturbation diagnostics | §10 | ~15–30 min |
| `scripts/run_phase5a.py --full` | Ensemble ablation | §7.3 | ~10–15 min |
| `scripts/run_phase5b.py --full` | Sensitivity sweeps | §8 | ~10–20 min |
| `scripts/run_real_data.py` | Real-data transfer check | §9 | ~15–30 min |

---

## Repository structure

```
sequence-acceleration-benchmark/
├── src/
│   ├── accelerators.py   All 51 method implementations
│   ├── config.py         Central hyperparameter configuration
│   ├── evaluation.py     Phase 1 evaluation loop
│   ├── generators.py     18 synthetic convergence-regime generators
│   ├── diagnostics.py    Perturbation and shift IQR diagnostics
│   ├── trajectories.py   Feature extraction and cascade logic
│   ├── datasets.py       OpenML dataset loading and XGBoost training
│   └── plots.py          Publication figure generation
├── phases/
│   ├── phase1.py         Thin wrapper around src/evaluation.py
│   ├── phase2.py         Richardson failure-condition mapping
│   ├── phase3.py         Adaptive method selection
│   ├── phase4.py         Perturbation diagnostic study
│   ├── phase5a.py        Ensemble weight ablation
│   └── phase5b.py        Sensitivity sweeps (L∞, window length, γ)
├── scripts/              Entry points (one per phase, see table above)
├── tests/
│   └── test_accelerators.py   Unit tests on four analytic sequences
├── results/              Pre-computed output figures and tables
├── data/                 Downloaded datasets (populated at runtime; see data/README.md)
├── requirements.txt
└── reproduce_all.py      Single script reproducing all paper tables
```

---

## Table / figure mapping

| Paper table or figure | Produced by |
|-----------------------|-------------|
| Tables 4, 22 (global stability ranking) | `run_phase1.py --full` → `results/phase1/phase1_global.csv` |
| Tables 5, 23 (per-regime best method) | `run_phase1.py --full` → `results/phase1/phase1_regime_best.csv` |
| Tables 7, 8 (failure detection) | `run_phase2.py --full` → `results/phase2/phase2_correlations.csv`, `phase2_rules.csv` |
| Tables 9, 10, 11 (adaptation limits) | `run_phase3.py --full` → `results/phase3/` |
| Tables 12–15, 29–31 (sensitivity) | `run_phase5b.py --full` → `results/phase5b/` |
| Tables 16–18 (real-data transfer) | `run_real_data.py` → `results/real_data/` |
| Tables 19–21 (perturbation diagnostics) | `run_phase4.py --full` → `results/phase4/` |
| Appendix A.4 (unit-test results) | `run_phase0_tests.py` → `results/phase0_unit_tests.txt` |

---

## Data availability

- **Synthetic sequences** are generated deterministically from `src/generators.py`
  and `src/config.py`; no download is required.
- **Real XGBoost curves** (Section 9) are derived from the
  [OpenML CC-18 benchmark suite](https://www.openml.org), downloaded
  automatically by `scripts/run_real_data.py`.

---

## Citation

```bibtex
@article{maleki2026finite,
  title   = {Finite-Horizon Learning-Curve Prediction for Gradient Boosting:
             Regime Dependence, Failure Detection, and Conservative Extrapolation Rules},
  author  = {Maleki, Kian},
  journal = {Machine Learning},
  year    = {2026},
  note    = {Submitted}
}
```
