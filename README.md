# sequence-acceleration-benchmark

Benchmark code for the paper:

> Maleki, K. (2026). **Finite-Horizon Learning-Curve Prediction for Gradient Boosting: Why Sequence Acceleration Fails and Simple Curve Fits Suffice.** *Machine Learning* (submitted).

Given the validation-loss trajectory of a gradient boosting run, predict the loss at a specified future round. This repository evaluates **51 methods from 13 families** — Richardson extrapolation, Padé approximants, parametric curve fits, and the classical algebraic accelerators (Shanks, Wynn, Levin, Brezinski, Anderson mixing) — across **18 synthetic convergence regimes**, 3 noise levels, 3 prediction horizons, 30 seeds, and 10 observation depths (**2.478 M evaluations**), plus a transfer check on six real XGBoost curves from OpenML CC-18.

**Headline findings.**
- The classical acceleration toolbox is a **no-op** on this task: 21 of 23 classical methods deliver median improvement within ±10 % of the do-nothing baseline, despite being exact on clean textbook sequences. They help only at long horizon with exactly zero noise — a cell real validation curves never occupy.
- Trajectory curve fits earn 2–4× median improvement but carry **both tails** of the ranking; method choice is risk management, and the paper reports median error and catastrophe rate as separate axes.
- Adaptive selection hits hard limits (regime classification 35.6 %, ensembles lose to the best fixed method). The recommended deployment is a fixed default (`rational_fit`) plus a perturbation-based reliability alarm.
- The real-data check reproduces the synthetic structure without calibration (58.7 % reduction in mean prediction error).

---

## Installation

```bash
git clone https://github.com/drkianmaleki/sequence-acceleration-benchmark.git
cd sequence-acceleration-benchmark
pip install -r requirements.txt
```

Python 3.9+ is required. The code is tested with pandas 2.x and 3.x (dataset loading uses copy-safe array conversion, so pandas 3's copy-on-write semantics are handled).

---

## Reproducing all results

A single script runs every phase in sequence and writes all output to `results/`:

```bash
python reproduce_all.py            # full run (~2–4 hours)
python reproduce_all.py --quick    # reduced run for a sanity check (~15 min)
python reproduce_all.py --skip-real-data   # skip the OpenML download phase
```

`--quick` uses fewer seeds and noise levels; all phases still run end-to-end. The real-data phase downloads datasets from OpenML and requires an internet connection.

`results/` is committed with the exact output of the corrected runs behind every number in the paper; the commands above regenerate it from scratch (matching up to the documented real-data drift). Don't mix output from different commits.

---

## Reproducing the paper tables

After a full run, one script converts `results/` into every LaTeX table fragment used in the paper, plus a `REPORT.md` with the derived headline numbers:

```bash
python scripts/make_paper_tables.py results results/real_data paper_tables/
```

This emits 28 fragments (`t03` … `t31`, matching the paper's tables) and is the authoritative path from raw CSVs to the manuscript: no table value in the paper is hand-typed.

---

## Running individual phases

| Script | Phase | Paper section | Runtime (full) |
|--------|-------|---------------|----------------|
| `scripts/run_phase0_tests.py` | Unit tests on analytic sequences | Appendix A | ~1 min |
| `scripts/run_phase1.py --full` | Synthetic benchmark | §4–5 | ~20–40 min |
| `scripts/run_phase2.py --full` | Failure detection | §6 | ~25–40 min |
| `scripts/run_phase3.py --full` | Adaptive selection | §7 | ~10–20 min |
| `scripts/run_phase4.py --full` | Perturbation diagnostics | §10 | ~15–30 min |
| `scripts/run_phase5a.py --full` | Ensemble ablation | §7.3 | ~10–15 min |
| `scripts/run_phase5b.py --full` | Sensitivity sweeps | §8 | ~10–20 min |
| `scripts/run_real_data.py` | Real-data transfer check | §9 | ~15–30 min |

**Real-data reproducibility note.** `run_real_data.py` retrains XGBoost from scratch on six OpenML datasets. The structural results — regime mapping, cascade routing, failure locations — are stable across XGBoost versions; individual per-case error percentages can drift in the third decimal, which is expected and does not affect the paper's aggregates.

---

## Tests

```bash
pip install pytest
pytest tests/
```

`test_generators.py` checks the invariants each synthetic regime must satisfy — most importantly that `TRUTH[regime](n)` equals the noiseless output of `GENERATORS[regime]` at every `n`. A mismatch there is silent: the run completes and the numbers look plausible, but every method in that regime is scored against a target its sequence never attains. `test_accelerators.py` verifies the 51 method implementations on four analytic sequences with known limits (Appendix A of the paper reports the results).

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
├── phases/               Phase implementations (phase1 … phase5b)
├── scripts/              Entry points (one per phase, see table above)
│   ├── check_dangerous.py       Verifies the derived dangerous-method set
│   └── make_paper_tables.py     Regenerates every paper table from results/
├── tests/
│   ├── test_accelerators.py     Unit tests on four analytic sequences
│   └── test_generators.py       Regime definition invariants
├── results/              Generated output (empty until a phase is run)
├── data/                 Downloaded datasets (populated at runtime)
├── requirements.txt
└── reproduce_all.py      Single script reproducing all paper results
```

---

## Paper table mapping

| Paper tables | Produced from |
|--------------|---------------|
| 4–8, 24, 25 (method landscape: no-op, global ranking, champions, dangerous set) | `run_phase1.py --full` → `results/phase1/` |
| 9, 10, 26, 27 (failure detection, feature correlations, rules) | `run_phase2.py --full` → `results/phase2/` |
| 11–13, 28 (classifier, selectors, ensembles) | `run_phase3.py --full` → `results/phase3/` |
| 14–17, 30–32 (sensitivity sweeps) | `run_phase5b.py --full` → `results/phase5b/` |
| 13, 29 (corrected ensemble ablation, reliability ranking) | `run_phase5a.py --full` → `results/phase5a/` |
| 18–20 (real-data transfer) | `run_real_data.py` → `results/real_data/` |
| 21–23 (perturbation diagnostics) | `run_phase4.py --full` → `results/phase4/` |
| Appendix A.4 (unit-test results) | `run_phase0_tests.py` → `results/phase0_unit_tests.txt` |

All fragments are emitted in one pass by `scripts/make_paper_tables.py` (see above).

---

## Code corrections

Three defects found during internal review were corrected before the reported runs, and are disclosed in Appendix A of the paper: the `staircase` generator emitted its steps in reversed order, the `delayed_plateau` generator applied its defining stall outside the noiseless trend, and the dangerous-method exclusion list was re-synchronised with the Phase 1 criterion (S < 0). All results in the paper come from complete re-runs under the corrected code; the corrected generators are covered by `tests/test_generators.py`.

---

## Data availability

- **Synthetic sequences** are generated deterministically from `src/generators.py` and `src/config.py`; no download is required.
- **Real XGBoost curves** (§9) are derived from the [OpenML CC-18 benchmark suite](https://www.openml.org), downloaded automatically by `scripts/run_real_data.py`. The exact outputs used in the paper are committed under `results/`, including the real-data curves and per-case results in `results/real_data/`.

---

## Citation

```bibtex
@article{maleki2026finite,
  title   = {Finite-Horizon Learning-Curve Prediction for Gradient Boosting:
             Why Sequence Acceleration Fails and Simple Curve Fits Suffice},
  author  = {Maleki, Kian},
  journal = {Machine Learning},
  year    = {2026},
  note    = {Submitted}
}
```
