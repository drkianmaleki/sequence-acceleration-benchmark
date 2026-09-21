# sequence-acceleration-benchmark

Benchmark code for the paper:

> Maleki, K. (2026). **Finite-Horizon Learning-Curve Prediction for Gradient Boosting: Why Sequence Acceleration Fails and Simple Curve Fits Suffice.** *Machine Learning* (in revision; title as submitted).

Given the validation-loss trajectory of a gradient-boosting run, predict the loss at a specified future round. The repository evaluates **51 sequence-acceleration methods from 13 families** — Richardson extrapolation, Padé approximants, parametric curve fits, and the classical algebraic accelerators (Shanks, Wynn ε/ρ, Levin, Brezinski, Weniger, Anderson) — **against five trivial comparators**, on **18 core and 6 held-out synthetic convergence regimes** with hidden, heterogeneous asymptotes, three noise levels, three gap-defined prediction horizons and 30 seeds (**362,880 evaluations** in the main benchmark; **2.86 M** central evaluations across all phases, plus 6 M diagnostic calls), and re-evaluates six recorded real XGBoost curves (OpenML ids 1590, 1461, 1596, 23512, 41150, 41168) on a 90-cell (depth × target) grid.

This is **redesign v2** (2026-09). The design of the originally submitted version was rejected in review because (a) every regime shared one true asymptote L* = 0.01, (b) that exact value was handed to every method as the "assumed" asymptote, and (c) the headline horizons sat where the target had already converged, so a predictor that simply returned the constant beat every method. Each ingredient is removed here; the numbers behind the revised paper are in `FACTS.md` and `paper_fragments/`, generated from the committed `results/` by one script.

---

## Design (redesign v2)

**Hidden heterogeneous asymptotes.** Each (regime, seed) draws its own true asymptote `L_true` log-uniformly from (0.005, 0.5) (`src/generators.py`, `config.ASYMPTOTE_MODE = "hetero"`). No method ever sees `L_true`; only the evaluation harness and the oracle comparator do (`tests/test_redesign.py` enforces the separation). The old shared-asymptote design survives only as `config.ASYMPTOTE_MODE = "legacy"` for the regression test that locks in the referee's finding.

**Assumed asymptote L̂ (`src/asymptote.py`).** What a method is told about the asymptote is a separate, explicit choice: `zero` (L̂ = 0, the deployment-honest default and the only mode of the main run), `half`, `oracle` (L̂ = L_true, labelled as such), `double`, and `winmin` (0.9 × the window minimum, data-driven). Every accelerator, feature extractor and cascade input reads `cfg["L_inf"] = L̂`; Phase 5b sweep 1 varies the mode.

**Trivial comparators and skill (`src/trivial.py`).** Five trivial predictors are registered as first-class methods: `constant_assumed` (returns L̂), `last_value`, `window_mean`, `window_min`, and the evaluation-only `constant_oracle` (returns `L_true`; flagged `is_oracle`, shown in tables, never ranked, never in a pool). Every record carries two kinds of skill. `skill = err(method) / err(best of the four deployable trivials on that cell)` is the **hindsight best-of-four (strict)** bar — the denominator needs the truth to pick the trivial, so skill < 1 means the method beat every non-oracle trivial predictor there, including the one only hindsight could have chosen; it is aggregated as `med_skill`. The **fixed-reference** columns `skill_vs_{assumed,last,wmean,wmin}` (error ratio against each deployable trivial separately) and `win_vs_{...}` (1 when the method's error is below that trivial's) are aggregated as `med_skill_vs_*` and `win_rate_vs_*` in every table that carries `med_skill` (Phase 1 aggregated / global / held-out, Phase 2, Phase 4 and 5a selector tables, real data v2).

**Gap-defined horizons with capping (`src/horizons.py`).** Horizons are defined by the remaining gap, not by index: for a target fraction g ∈ {0.5, 0.1, 0.02} (headline g = 0.1), `n_f(regime, n_obs, g)` is the first n > n_obs at which the noiseless gap to `L_true` has shrunk to g × gap(n_obs), searched forward and capped at 50,000. When the cap binds, the achieved fraction is recorded and the cell is **excluded from every pooled statistic**; capped cells are reported in their own `*_capped.csv` block (fragment `f11`).

**Core and held-out regimes (`src/generators.py`).** 18 core regimes (`single_exp` … `broken_power_law`) are used everywhere; 6 held-out regimes (`stretched_exp`, `logistic_tail`, `inv_sqrt_log`, `random_knots`, `real_boot_a`, `real_boot_b`) are evaluation-only: nothing is trained, tuned or derived on them, and every pooled table exists in a core and a held-out version.

**Validity floor and ranking (`src/pipeline.py`).** Pooled tables are sorted by median error over uncapped cells; a method is rank-eligible only when its pooled valid rate is ≥ 0.9 (`config.RANK_MIN_VALID`) and it is not the oracle. Below-floor methods stay visible in an `*_unranked.csv` block with their valid rate.

**Dangerous-method artifact (`src/dangerous.py`, `scripts/derive_dangerous.py`).** The set of methods excluded from ensembles ("dangerous": pooled stability `S = valid − 2·cat + 0.4·beats < 0` over the core regimes at n_obs = 90, capped cells excluded, accelerators only) is **re-derived from the Phase-1 output** into `results/phase1/dangerous_methods.json`; Phases 2–5 read that artifact and refuse to run without it. `config.LEGACY_DANGEROUS_METHODS` is kept only for the record; `scripts/check_dangerous.py` verifies that the artifact matches the committed Phase-1 table. (Under the redesign the derived set is identical to the legacy eight — `FACTS.md`, section *dangerous set*.)

**Provenance.** Every real-data summary row records the Phase-2 feature file that produced the regime centroids and the commit hash; the artifact records its source (repo-relative), commit and time; every fragment and every `FACTS.md` row names its source file, filter and formula.

---

## Installation

```bash
git clone https://github.com/drkianmaleki/sequence-acceleration-benchmark.git
cd sequence-acceleration-benchmark
pip install -r requirements.txt
```

Python 3.9+; tested with pandas 2.x and 3.x. No data download is needed: the synthetic regimes are generated deterministically and the real-data step re-evaluates the recorded curves committed under `results/real_data/`.

---

## Reproducing all results

```bash
python reproduce_all.py --plan     # evaluation counts per phase (full and quick)
python reproduce_all.py            # full run: 4.4 h on 8 cores (measured 2026-09-21)
python reproduce_all.py --quick    # 2 seeds, 2 regimes per group, 2 noise levels: every phase end to end (~2-3 min)
python reproduce_all.py --skip-real-data
```

Order: Phase 0 → Phase 1 → dangerous re-derivation → Phases 2, 3, 4, 5a, 5b → real data. Phase 5a evaluates its (depth × noise) blocks in `cpu_count() − 1` worker processes; the output is byte-identical for any job count (`scripts/run_phase5a.py --jobs N`). Wall times of the full run: Phase 1 31 min, Phase 2 16 min, Phase 4 30 min, Phase 5a 2.5 h (7 workers), Phase 5b 38 min, everything else seconds.

`results/` is committed with the exact output of the full run (commit `30de724`, log in `REPORT_3B_full_run.txt`); three per-record files (`phase1_records.csv`, `phase4_raw.csv`, `phase5a_raw.csv`, 409 MB together) are git-ignored and regenerated by the run. After a re-run, `python scripts/check_dangerous.py` confirms that the artifact still matches Phase 1. Don't mix output from different commits.

---

## Reproducing the paper tables

```bash
python scripts/make_paper_tables.py            # -> paper_fragments/*.tex + FACTS.md
python scripts/make_paper_tables.py --no-raw   # same, without reading the git-ignored raw files
```

The script reads the per-stratum result files (never the legacy-named headline copies) and writes 20 booktabs `tabular` fragments to `paper_fragments/` (index in `paper_fragments/README.md`) plus `FACTS.md`, the list of every headline number with its file, filter and formula. Fragment `f01_trivial_baseline.tex` — the four deployable trivials, the oracle reference, the best and median accelerator, and how many accelerators beat the best deployable trivial, per stratum, core and held-out — is the paper's first result. No table value in the paper is hand-typed.

---

## Running individual phases

| Script | Phase | Grid (full) | Wall (full run) |
|---|---|---|---|
| `scripts/run_phase0_tests.py` | analytic unit tests | 51 accelerators × 4 sequences | 2 s |
| `scripts/run_phase1.py --full` | main benchmark | 24 regimes × 30 seeds × 3 noise × 3 strata × 56 methods (n_obs = 90) | 31 min |
| `scripts/derive_dangerous.py` | dangerous artifact | reads `phase1_aggregated.csv` | 1 s |
| `scripts/run_phase2.py --full` | failure detection | 13 depths × 5 noise × 20 seeds × 18 core regimes × 3 strata × 11 methods | 16 min |
| `scripts/run_phase3.py --full` | adaptive selection | analysis of Phase 2 | 5 s |
| `scripts/run_phase4.py --full` | perturbation / shift diagnostics | 4 depths × 3 noise × 20 seeds × 24 regimes × 3 strata × 13 methods (+1.56 M diagnostic calls) | 30 min |
| `scripts/run_phase5a.py --full [--jobs N]` | ensemble ablation | 4 depths × 3 noise × 20 seeds × 24 regimes × 3 strata × 56 methods (+4.4 M perturbation calls) | 2.5 h |
| `scripts/run_phase5b.py --full` | sensitivity sweeps | sweep 1a cascade vs assumed asymptote (clamped features, labelled), sweep 1b the 10 L_hat-consuming accelerators + `constant_assumed` under the 5 modes, window length, CAT_MULT; 2 strata | 38 min (+ sweep 1b, not yet timed) |
| `scripts/run_real_data.py` | recorded-curve re-evaluation | 6 datasets × 5 depths × 3 targets × 7 methods; every summary three ways (all / pre-minimum / post-minimum targets) | 5 s |

`--quick` is accepted by every runner. `scripts/run_real_data.py --retrain` is the legacy path that downloads the OpenML datasets and retrains XGBoost to regenerate the recorded curves; it is not part of `reproduce_all.py`.

---

## Tests

```bash
pip install pytest
pytest tests/          # 342 tests
```

`test_generators.py` checks the invariants each synthetic regime must satisfy (in particular that the noiseless generator output equals the truth function at every n; a mismatch there is silent at run time). `test_accelerators.py` verifies the 51 method implementations on four analytic sequences with known limits. `test_redesign.py` locks in the redesign: hidden heterogeneous asymptotes, the `L_true` / `L̂` separation (no method reads `cfg["L_true"]`), gap-defined horizons and capping, the trivial comparators and skill, the held-out split, the rank floor, and a regression test that reproduces the referee's finding under `ASYMPTOTE_MODE = "legacy"`. `test_pipeline_v2.py` covers the shared pipeline helpers, the dangerous artifact, the Phase-5a serial-vs-parallel byte identity and the real-data provenance columns. `test_input_dependence.py` is the permanent guard against degenerate accelerators: on 96 generated windows every accelerator must react to a 1 % window perturbation on ≥ 90 % of the windows where it is finite and may coincide with a deployable trivial on ≤ 10 % of them (`current_value` exempt); it also measures the L_hat-consumer list that Phase 5b sweep 1b evaluates. It was added after `weniger_d1/d2` were found to return 0 for every input (fixed in `src/accelerators.py::_weniger_delta`, d-type remainder `w_n = Δs_n` with Pochhammer weights).

---

## Repository structure

```
sequence-acceleration-benchmark/
├── src/
│   ├── accelerators.py     51 accelerator implementations (+ the trivial comparators registered as methods)
│   ├── trivial.py          the five trivial comparators, best_reference_error, skill_score, skill_vs_table (fixed-reference skill)
│   ├── asymptote.py        assumed-asymptote modes (zero / half / oracle / double / winmin)
│   ├── generators.py       18 core + 6 held-out regimes, hidden per-(regime, seed) L_true
│   ├── horizons.py         gap-defined horizons n_f(regime, n_obs, g) with the 50,000 cap
│   ├── pipeline.py         shared phase helpers: regime sets, capped exclusion, rank floor, skill table, provenance
│   ├── dangerous.py        derivation, artifact I/O and loading of the dangerous-method set
│   ├── evaluation.py       Phase-1 evaluation loop and pooled tables
│   ├── trajectories.py     window features, Phase-2 cascade, recorded-curve re-evaluation (real data)
│   ├── diagnostics.py      perturbation and shift IQR diagnostics
│   ├── datasets.py         OpenML loading and XGBoost training (legacy --retrain path only)
│   ├── plots.py            figure helpers
│   └── config.py           every constant: modes, strata, cap, floor, per-phase grids
├── phases/                 phase1.py ... phase5b.py (one module per phase)
├── scripts/
│   ├── run_phase0_tests.py ... run_phase5b.py, run_real_data.py   entry points (see table above)
│   ├── derive_dangerous.py         Phase 1 -> results/phase1/dangerous_methods.json
│   ├── check_dangerous.py          verifies the artifact against the committed Phase-1 table
│   ├── make_paper_tables.py        results/ -> paper_fragments/*.tex + FACTS.md
│   ├── make_paper_figures.py       optional summary figures (paper_figures/, git-ignored)
│   └── analyze_real_diagnostics_legacy.py   verifies the stored pre-redesign 18-cell real-data run
├── tests/                  test_generators, test_accelerators, test_redesign, test_pipeline_v2
├── results/                committed output of the full run (see results/README.md for the layout)
├── paper_fragments/        generated LaTeX table fragments (f01 ... f11) + index
├── FACTS.md                generated headline numbers with provenance
├── REPORT_1.md ... REPORT_3B.md, REPORT_*_run.txt   development reports and console logs of the redesign
├── data/                   OpenML downloads (legacy --retrain path only; git-ignored)
├── requirements.txt
└── reproduce_all.py        single script reproducing every result in order
```

---

## Where the numbers are

- `FACTS.md`: every headline number with file / filter / formula, grouped by topic (trivial baseline, ranking, skill summary, classical no-op, generalisation, dangerous set, richardson_3 validity by depth, σ = 0 cancellation NaNs, invalid rates, capped cells, sweep 1, selectors, ensembles, diagnostics, real data, pipeline provenance).
- `paper_fragments/`: the tables, one file each (`paper_fragments/README.md` maps fragments to sources).
- `REPORT_3B.md`: the full-run report (wall times, evaluation counts, invalid rates per phase, the derived dangerous set, the results inventory).

---

## Legacy material

The pre-redesign artefacts are kept on purpose and clearly labelled: `results/real_data/real_data_results.csv` with `figure_rd_01..03` (the original 18-cell real-data run at the legacy assumed asymptote 0.01; fragment `f07b`), `scripts/analyze_real_diagnostics_legacy.py` (verifies that run and its perturbation-diagnostic AUC; not part of `reproduce_all.py`), `config.LEGACY_DANGEROUS_METHODS` and `config.ASYMPTOTE_MODE = "legacy"` (regression test only). Earlier code corrections (the `staircase` and `delayed_plateau` generators, the dangerous-set synchronisation) are covered by `tests/test_generators.py` and by the artifact step.

---

## Data availability

- **Synthetic sequences** are generated deterministically from `src/generators.py` and `src/config.py`; no download is required.
- **Real XGBoost curves** were recorded from six [OpenML](https://www.openml.org) datasets (ids 1590, 1461, 1596, 23512, 41150, 41168) and are committed as `results/real_data/real_data_curves.csv`; the default real-data step re-evaluates these recorded curves without retraining, so it is exactly reproducible. `scripts/run_real_data.py --retrain` regenerates the curves (XGBoost-version drift in the third decimal is expected).

---

## Citation

```bibtex
@article{maleki2026finite,
  title   = {Finite-Horizon Learning-Curve Prediction for Gradient Boosting:
             Why Sequence Acceleration Fails and Simple Curve Fits Suffice},
  author  = {Maleki, Kian},
  journal = {Machine Learning},
  year    = {2026},
  note    = {In revision}
}
```
