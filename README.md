# sequence-acceleration-benchmark

The code and results of the paper named in `CITATION.cff`:

> Maleki, K. (2026). **Sequence Acceleration versus Trajectory Fitting for Early Learning-Curve Prediction: A Benchmark with Hidden Asymptotes and Trivial Baselines.** Submitted to *Applied Soft Computing*.

**The question.** Given the first rounds of a learning curve, how low will the loss be later: do sequence-acceleration methods from numerical analysis, which estimate the limit a sequence approaches, or simple curve fits, which extend a function fitted to the observed values, forecast it better than trivial predictors such as repeating the last observed value?

**The answer.** On synthetic curves with hidden final levels and on recorded XGBoost validation curves, the 21 classical limit-estimating methods help only on noise-free curves; simple curve fits do beat the trivial predictors, a three-parameter rational curve fitted to the observed rounds is the robust default, choosing a fit from the observed rounds alone (by rules, classification or ensembles) did not improve on it, and trying the leading fits on a completed run of the same problem and keeping the winner does.

The benchmark evaluates 52 sequence-acceleration methods from 12 families (`len(src.pipeline.ACCEL_METHODS)`) against five trivial comparators on 24 synthetic curve families (18 in the design, 6 out-of-design), three noise levels, three gap-defined horizons and 30 seeds, and re-evaluates six recorded XGBoost curves on a 90-cell (depth × target) grid. Every number in the paper comes from `FACTS.md` or from a fragment in `paper_fragments/`, generated from `results/` by the scripts of this repository; nothing is hand-typed.

---

## Check it in a few minutes (nothing is rerun)

```bash
git clone https://github.com/drkianmaleki/sequence-acceleration-benchmark.git
cd sequence-acceleration-benchmark
pip install -r requirements.txt            # lower bounds; or requirements-lock.txt for the exact tested versions
pytest -q                                  # 415 tests
python scripts/check_tables.py             # every table regenerates from the committed results
python scripts/build_paper_tables.py --selftest   # the main-text tables build from the fragments
```

- `pytest -q` runs the 415 tests: the generators, the accelerator roster on analytic sequences, the redesign's invariants (hidden asymptotes, the separation of the true and the assumed asymptote, gap-defined horizons, the validity floor), the pipeline helpers, the paired perturbation diagnostic, the order ladders, the selection-by-trial analysis on hand-made records, the code fingerprint, the printed terminology of every fragment, and the two checks below as tests (`tests/test_tables_regenerate.py`, `tests/test_paper_tables.py`). Two tests pin fitted numbers exactly and run only in the tested environment (see *The tested environment*); on another platform they are skipped and say why.
- `python scripts/check_tables.py` regenerates every fragment, the fragment index, `paper_fragments/all_tables.tex` and `FACTS.md` from the committed `results/` into a temporary folder, compares them byte for byte with the committed files, and runs the code-fingerprint check. Exit 0 means that every table in the repository is exactly what the committed results produce.
- `python scripts/build_paper_tables.py --selftest` builds the tables of the paper's main text from the fragments as row and column subsets, checks every number against its fragment, and passes every one of the 49 fragments through the same engine unchanged; it prints `TABLE BUILD: PASS` and writes the tables to `paper_tables_main/` (git-ignored).

---

## Rerunning the benchmark

```bash
python reproduce_all.py --plan     # the evaluation counts of the full and the quick run
python reproduce_all.py --quick    # 2 seeds, 2 curve families per group, 2 noise levels: every phase end to end, about 4 minutes
python reproduce_all.py            # the full run
python reproduce_all.py --skip-real-data
```

The reported full run (`results/run_manifest.json`) took 29,256 s (8.1 h) with 7 worker processes for the parallel steps (`--jobs N`; the output is byte-identical for any job count). Its wall times per step, from the manifest: Phase 1 43 min, Phase 0b 11 min, Phase 2 23 min, Phase 4 51 min, Phase 5a 4.3 h, Phase 5b 1.7 h, everything else seconds. `python reproduce_all.py --plan` prints the evaluation counts: 369,360 evaluations in the main benchmark, 257,040 in the order ladders, 2.86 M central evaluations in all, plus 6.1 M diagnostic calls.

Order of the steps: Phase 0 (analytic unit tests) → Phase 1 (main benchmark) → Phase 0b (order ladders, ending with the exact-agreement check against Phase 1) → the excluded-set derivation → Phases 2, 3, 4, 5a, 5b → real data (the reported methods, then the roster evaluation) → raw-derived facts (`scripts/derive_raw_facts.py`) → selection by trial (`scripts/derive_selection.py`) → paper tables (`scripts/make_paper_tables.py`) → L_true terciles (`scripts/analyze_by_ltrue.py`) → the standalone tables document (`scripts/build_tables_document.py`). In full mode the driver writes `results/run_code_fingerprint.json` at the start, next to the first manifest write. Five per-record files (`phase1/phase1_records.csv`, `phase0b/order_ladders_records.csv`, `phase2/phase2_records.csv`, `phase4/phase4_raw.csv`, `phase5a/phase5a_raw.csv`) are git-ignored and regenerated by the run; every table regenerates without them. After a rerun, `python scripts/check_dangerous.py` confirms that the excluded-method artifact still matches Phase 1. Do not mix output from different runs.

---

## From the results to the tables

1. **Result files.** `results/` holds the output of the reported run, one folder per phase (`results/README.md` is the inventory; the raw per-record files are git-ignored).
2. **Fragments and facts.** `python scripts/make_paper_tables.py` writes the LaTeX fragments `paper_fragments/f*.tex` (one `tabular` each, with a comment header naming its sources and filters) and `FACTS.md`, every headline number with its file, filter and formula. `python scripts/analyze_by_ltrue.py` adds the three `f13` fragments and their `FACTS.md` section (from the committed `results/phase1/phase1_by_Ltrue.csv` when the records file is absent), and `python scripts/build_tables_document.py` writes `paper_fragments/all_tables.tex`, a standalone document that inputs every fragment with its description line as the caption (`pdflatex` it twice in `paper_fragments/`). `paper_fragments/README.md` is the index of every fragment with its sources.
3. **Main-text tables.** `python scripts/build_paper_tables.py` builds the tables of the paper's main text as row and column subsets of the fragments, checking every number it writes against the fragment it came from. `--selftest` also passes every fragment through the engine unchanged; `--list` prints the tables.
4. **The check.** `python scripts/check_tables.py` redoes step 2 into a temporary folder and compares.

The definitions shared by the table script and the selection analysis (the noise class of a cell, the classical variants, the leading-method rule, the default `rational_fit` and the conservative alternative `single_exp_fit`, the candidate pools) live in `scripts/selection_defs.py`. Printed text follows the paper's terminology ("curve family", "out-of-design", "the excluded set", "last value"; `tests/test_fragment_text.py` checks it); file names, column names, stored values and FACTS filters keep the code identifiers (`regime`, `holdout`, `dangerous`).

---

## Where each table of the paper comes from

Table 1 and Table A.1 (the symbols and the generator definitions) are typed in the manuscript. Every other table is built from one fragment:

| Table | Fragment |
|---|---|
| 2 | `f20_roster` |
| 3 | `f01_trivial_baseline` |
| 4 | `f21_noop_by_noise` |
| 5 | `f03_skill_summary` |
| 6 | `f24_leading_fits` (synthetic columns) |
| 7 | `f22_per_family_g0.1` |
| 8 | `f15_bootstrap` |
| 9 | `f16_generalisation_summary` |
| 10 | `f12_validity_by_depth` |
| 11 | `f11_capped_block` |
| 12 | `f10a_diagnostics_correlations` |
| 13 | `f09b_ensemble_phase5a_core` |
| 14 | `f14_real_fixed_methods` |
| 15 | `f24_leading_fits` (recorded-curve columns) |
| 16 | `f29_real_named_by_dataset` |
| 17 | `f30_real_final_loss_by_depth` |
| 18 | `f27_selection_one_pilot` |
| 19 | `f27b_selection_many_pilots` |
| 20 | `f28_selection_recorded` |

Supplementary tables, in order: S1 `f18_ladders_classical`, S2 `f19_ladders_fits`, S3 `f11_capped_block`, S4 `f01_trivial_baseline`, S5 `f02_ranking_g0.5`, S6 `f02_ranking_g0.1`, S7 `f02_ranking_g0.02`, S8 `f06_generalisation`, S9 `f13_by_Ltrue_g0.5`, S10 `f13_by_Ltrue_g0.1`, S11 `f13_by_Ltrue_g0.02`, S12 `f04_classical_noop`, S13 `f04b_classical_noop_holdout`, S14 `f21_noop_by_noise`, S15 `f21b_noop_by_noise_variants_core`, S16 `f21b_noop_by_noise_variants_holdout`, S17 `f24_leading_fits`, S18 `f22_per_family_g0.5`, S19 `f22_per_family_g0.1`, S20 `f22_per_family_g0.02`, S21 `f23_family_head_to_head`, S22 `f15_bootstrap`, S23 `f05_sweep1_modes`, S24 `f05b_sweep1_consumers_core`, S25 `f05b_sweep1_consumers_holdout`, S26 `f09a_selectors_phase3`, S27 `f09b_ensemble_phase5a_core`, S28 `f09c_ensemble_phase5a_holdout`, S29 `f09d_ablation_phase5a`, S30 `f10b_diagnostics_depth`, S31 `f10c_diagnostics_ensemble`, S32 `f10d_diagnostics_filter`, S33 `f14_real_fixed_methods`, S34 `f25_real_roster`, S35 `f26_real_classical`, S36 `f07_real_data_v2`, S37 `f08_real_perturb_diagnostic`, S38 `f07b_real_data_legacy18`, S39 `f27_selection_one_pilot`, S40 `f27b_selection_many_pilots`, S41 `f27c_selection_by_family`, S42 `f28_selection_recorded`.

---

## The tested environment

`requirements.txt` gives the lower bounds; `requirements-lock.txt` records the exact environment in which the reported results were produced and this release was tested (Python 3.12 on Windows, and the versions of numpy, scipy, pandas, matplotlib, scikit-learn, xgboost, openml and pytest). The library versions of the run itself are also in `results/run_manifest.json` and in the provenance files under `results/real_data/`.

Least-squares fits with several parameters can differ in their last digits between platforms and library versions. Two tests therefore assert exact equality of fitted numbers only in the tested environment (`tests/test_perturbations.py::test_shift_iqr_is_unchanged_pinned_values` and `tests/test_real_roster.py::test_legacy_path_reproduces_the_committed_results_on_a_subset`, guarded by `tests/tested_environment.py`: the same operating system, Python major and minor version, and numpy and scipy versions as `requirements-lock.txt`); elsewhere they are skipped with that reason. Every other check is platform-independent: the tests, `check_tables.py` and `build_paper_tables.py --selftest` compare committed files with files regenerated from committed files. The workflow in `.github/workflows/tests.yml` runs all of them on every push.

---

## Provenance

- `results/run_manifest.json` describes the reported run: mode, code commit, start and end time, worker count, the wall time of every step, the evaluation plan and the library versions.
- `results/run_code_fingerprint.json` holds the SHA-256 (CRLF-normalised) of every code file that produced a result, as the file was at the run; `python scripts/run_code_fingerprint.py --check` verifies the working tree against it. A file deliberately changed after the run carries its current hash and one sentence saying what changed and which test guarantees the old outputs; the fingerprint lists these recorded changes, and any other difference is a mismatch.
- The run's commit is not part of the public history, which holds one commit per release; the fingerprint is what ties the committed results to the code.
- Every fragment and every `FACTS.md` row names its source file, filter and formula, and the header of every generated file names the run it comes from. Results added after the run carry their own provenance file (`results/real_data/real_data_roster_provenance.json`, `results/phase1/phase1_selection_provenance.json`).

---

## What the benchmark does

**Hidden heterogeneous asymptotes.** Each (curve family, seed) draws its own true asymptote `L_true` log-uniformly from (0.005, 0.5) (`src/generators.py`, `config.ASYMPTOTE_MODE = "hetero"`). No method ever sees `L_true`; only the evaluation harness and the oracle comparator do (`tests/test_redesign.py` enforces the separation). The shared-asymptote design of the originally submitted version survives only as `config.ASYMPTOTE_MODE = "legacy"` for the regression test that reproduces the referee's finding.

**Assumed asymptote L̂ (`src/asymptote.py`).** What a method is told about the asymptote is an explicit choice: `zero` (L̂ = 0, the deployment-honest default and the only mode of the main run), `half`, `oracle` (L̂ = L_true, labelled as such), `double` and `winmin` (0.9 × the window minimum). Every accelerator, feature extractor and cascade input reads `cfg["L_inf"] = L̂`; Phase 5b sweep 1 varies the mode.

**Trivial comparators and skill (`src/trivial.py`).** Five trivial predictors are registered as first-class methods: `constant_assumed` (returns L̂), `last_value`, `window_mean`, `window_min`, and the evaluation-only `constant_oracle` (returns `L_true`; flagged `is_oracle`, shown in tables, never ranked, never in a pool). `skill = err(method) / err(best of the four deployable trivials on that cell)` is the hindsight best-of-four (strict) bar, aggregated as `med_skill`; the fixed-reference columns `skill_vs_{assumed,last,wmean,wmin}` and `win_vs_{...}` compare against each deployable trivial separately and are aggregated as `med_skill_vs_*` and `win_rate_vs_*`.

**Gap-defined horizons with capping (`src/horizons.py`).** For a target fraction g ∈ {0.5, 0.1, 0.02} (headline g = 0.1), `n_f(regime, n_obs, g)` is the first n > n_obs at which the noiseless gap to `L_true` has shrunk to g × gap(n_obs), searched forward and capped at 50,000. When the cap binds, the cell is excluded from every pooled statistic and reported in its own `*_capped.csv` block (fragment `f11`).

**Curve families in and outside the design (`src/generators.py`).** 18 families (`single_exp` … `broken_power_law`; `regime` in the code) are used everywhere; 6 out-of-design families (`stretched_exp`, `logistic_tail`, `inv_sqrt_log`, `random_knots`, `real_boot_a`, `real_boot_b`) are evaluated but nothing is trained, tuned or derived on them, and every pooled table exists in a core and an out-of-design version (`is_holdout`, `regime_set`, `*_holdout.csv`). The two `real_boot` families are smoothing-spline lower envelopes of XGBoost validation curves trained on two OpenML datasets outside the recorded-curve test set, `electricity` (id 151) and `nomao` (id 1486), built by `scripts/make_real_boot_sources.py` into `results/real_data/real_boot_sources.csv` under three pre-specified inclusion criteria, with the measured values and verdicts in `real_boot_sources_provenance.json`.

**Validity floor, ranking and the excluded set (`src/pipeline.py`, `src/dangerous.py`).** Pooled tables are sorted by median error over uncapped cells; a method is rank-eligible only when its pooled valid rate is at least 0.9 (`config.RANK_MIN_VALID`) and it is not the oracle; below-floor methods stay visible in an `*_unranked.csv` block. The same floor is the exclusion criterion of the ensembles: an accelerator whose pooled valid rate over the core families at the Phase-1 depth is below it is excluded, the set is derived from the Phase-1 output into `results/phase1/dangerous_methods.json` (schema `dangerous_methods/v3`; `dangerous` is the legacy implementation name, the paper says "the excluded set"), and Phases 2 to 5 read that artifact.

**Descriptive reporting (`src/panels.py`).** No composite score is formed anywhere. Every set of records is summarised by one descriptive panel (`error_panel`, used by Phases 1, 2, 3 and 5b): `n_total`, `n_valid`, `valid_rate`, `cat_rate` over all records; `mean_error`, `sd_error`, `med_error`, `q25_error`, `q75_error`, `p90_error` over the valid records only (conditional on validity, always shown next to the validity rate); and `win_rate_vs_last`. Rule tables carry the rule panel (`rule_panel`: what happened when a rule fired). Phase 3 scores a selector on every record of every cell it chose a method for; an invalid choice stays invalid. The fixed-threshold cascades need no cross-validation: the per-family panel (`phase3_regime_results.csv`) is the per-family evidence and the out-of-design families of Phase 5a are the out-of-sample test.

**Regime classifier (`phases/phase3.py`).** One row is the seed-averaged feature vector of one (curve family, obs_idx, noise) cell. Three protocols are reported side by side and never combined: `grouped_by_depth` (primary; every fold holds out all cells at one observation depth), `grouped_by_noise` and `stratified_5fold_legacy` (comparison only).

**Order ladders (Phase 0b, `scripts/order_ladders.py`).** For every family with an order parameter, the ladder is the set of variants obtained by changing only that parameter in the family's own constructor in `src/accelerators.py`; the orders that are roster methods are the registry functions themselves. Every variant is evaluated on Phase 1's grid at the headline stratum with exactly Phase 1's windows, noise streams and horizons, and at the end the script compares every roster-marked variant's records with `results/phase1/phase1_records.csv` (exact equality) and exits non-zero on any mismatch (`order_ladders_agreement.txt`).

**Paired perturbations (`src/diagnostics.py`).** The perturbation diagnostic of Phases 4 and 5a draws one factor array per window from `RandomState(perturbation_key(regime, seed, obs_idx, sigma))`, and every method at every horizon is applied to the same perturbed windows, so a method's diagnostic does not depend on which other methods are evaluated or on the order of the families. Phase 4's rejection rules are the precision and recall of a detector of a pre-defined event (an invalid prediction, or an error above `CAT_MULT` times the last-value error).

**Choosing a method by trial (`scripts/derive_selection.py`).** On the Phase 1 records, a method chosen because it had the lowest error on one pilot seed (`one_pilot`) or the lowest median error over the other 29 seeds of the cell (`many_pilots`) is scored on another seed of the same cell against the default and the last value, per set, noise class and candidate pool, with the catastrophe rate of the chosen method next to the default's (fragments `f27`, `f27b`, `f27c`); `f28` does the same on the recorded curves' pre-minimum cells.

---

## Running individual phases

| Script | Phase | Grid (full) | Wall (reported run) |
|---|---|---|---|
| `scripts/run_phase0_tests.py` | analytic unit tests | the accelerator roster × 4 sequences | 2 s |
| `scripts/run_phase1.py --full` | main benchmark | 24 curve families × 30 seeds × 3 noise × 3 strata × every registered method (52 accelerators + 5 trivial comparators; n_obs = 90) | 43 min |
| `scripts/order_ladders.py --full [--jobs N]` | Phase 0b: order ladders | every order of every family with an order parameter + the trivial comparators × 24 families × 30 seeds × 3 noise at g = 0.1, then the exact-agreement check | 11 min |
| `scripts/derive_dangerous.py` | exclusion artifact | reads `phase1_aggregated.csv` | 1 s |
| `scripts/run_phase2.py --full` | Richardson failure characterisation | 13 depths × 5 noise × 20 seeds × 18 core families × 3 strata × the Phase-2 pool + `constant_assumed` + `constant_oracle` | 23 min |
| `scripts/run_phase3.py --full` | adaptive selection | analysis of Phase 2 (per-record selector panels; three classifier protocols) | 8 s |
| `scripts/run_phase4.py --full` | paired perturbation / shift diagnostics | 4 depths × 3 noise × 20 seeds × 24 families × 3 strata × the pool + 3 trivial references (+1.56 M diagnostic calls) | 51 min |
| `scripts/run_phase5a.py --full [--jobs N]` | ensemble ablation (paired perturbations) | 4 depths × 3 noise × 20 seeds × 24 families × 3 strata × every registered method (+4.6 M perturbation calls) | 4.3 h |
| `scripts/run_phase5b.py --full` | sensitivity sweeps | sweep 1a cascade vs assumed asymptote (rule panel), sweep 1b the L̂-consuming accelerators + `constant_assumed` under the 5 modes, sweep 2 window length; 2 strata | 1.7 h |
| `scripts/run_real_data.py` | recorded-curve re-evaluation | 6 datasets × 5 depths × 3 targets × the 7 reported methods; then the roster evaluation of the same 90 cells with every accelerator and the four deployable trivial predictors (`--roster-only` writes those two files alone) | 8 s + 32 s |
| `scripts/make_real_boot_sources.py` | `real_boot` generator sources | downloads OpenML 151 and 1486, trains XGBoost as the recorded curves were, checks the inclusion criteria | minutes (network) |

`--quick` is accepted by every runner. `scripts/run_real_data.py --retrain` is the legacy path that downloads the OpenML datasets and retrains XGBoost to regenerate the recorded curves; it is not part of `reproduce_all.py`.

---

## Tests

`pytest -q` runs 415 tests. `test_generators.py` checks the invariants of every synthetic family (the noiseless generator output equals the truth function at every n; the `real_boot` sources are outside the recorded-curve test set). `test_accelerators.py` verifies the roster on four analytic sequences with known limits. `test_redesign.py` locks in the redesign: hidden heterogeneous asymptotes, the `L_true` / L̂ separation, gap-defined horizons and capping, the trivial comparators and skill, the out-of-design split, the rank floor, and the regression test under `ASYMPTOTE_MODE = "legacy"`. `test_pipeline_v2.py` covers the pipeline helpers, the explicit accelerator roster and the registry's class of every method, the exclusion artifact, the Phase-1 panel schema, the Phase-2 and Phase-3 outputs and the Phase-5a serial-vs-parallel byte identity. `test_panels.py`, `test_phase3_classifier.py`, `test_perturbations.py`, `test_order_ladders.py`, `test_input_dependence.py` (every accelerator must react to a 1 % window perturbation), `test_review_edge_cases.py` and `test_selection.py` check the panels, the classifier protocols, the pairing of the diagnostic, the ladders, the degeneracy guard, the edge cases of the definitions and the selection-by-trial analysis on hand-made records with known answers. `test_run_code_fingerprint.py` checks the fingerprint's write and check logic and that the committed fingerprint passes on the tree; `test_real_roster.py` checks the roster evaluation of the recorded curves; `test_provenance_header.py` the header of every generated file; `test_tables_regenerate.py` runs `scripts/check_tables.py`; `test_fragment_text.py` checks the printed text of every fragment against the paper's terminology; `test_paper_tables.py` runs `scripts/build_paper_tables.py --selftest`.

---

## Repository structure

```
sequence-acceleration-benchmark/
├── src/
│   ├── accelerators.py     the accelerator roster (52 implementations; + the trivial comparators registered as methods)
│   ├── panels.py           the descriptive error panel, the rule panel and the zero-denominator mask
│   ├── trivial.py          the five trivial comparators, best_reference_error, skill_score, skill_vs_table
│   ├── asymptote.py        assumed-asymptote modes (zero / half / oracle / double / winmin)
│   ├── generators.py       18 core + 6 out-of-design curve families, hidden per-(regime, seed) L_true
│   ├── horizons.py         gap-defined horizons n_f(regime, n_obs, g) with the 50,000 cap
│   ├── pipeline.py         shared phase helpers: family sets, capped exclusion, rank floor, skill table, provenance
│   ├── dangerous.py        derivation, artifact I/O and loading of the excluded-method set ('dangerous' is the legacy name)
│   ├── evaluation.py       Phase-1 evaluation loop, pooled tables and the method registry's family and class maps
│   ├── trajectories.py     window features, Phase-2 cascade, recorded-curve re-evaluation (real data)
│   ├── diagnostics.py      paired perturbation factors and shift IQR diagnostics
│   ├── datasets.py         OpenML loading and XGBoost training (real_boot sources; legacy --retrain path)
│   ├── plots.py            figure helpers
│   └── config.py           every constant: modes, strata, cap, floor, per-phase grids
├── phases/                 phase1.py ... phase5b.py (one module per phase)
├── scripts/
│   ├── run_phase0_tests.py ... run_phase5b.py, run_real_data.py   entry points (see the table above)
│   ├── order_ladders.py            Phase 0b: order ladders on the Phase-1 grid + exact-agreement check
│   ├── derive_dangerous.py         Phase 1 -> results/phase1/dangerous_methods.json (validity criterion)
│   ├── check_dangerous.py          verifies the artifact against the committed Phase-1 table
│   ├── make_real_boot_sources.py   OpenML 151 + 1486 -> results/real_data/real_boot_sources.csv (+ provenance, criteria)
│   ├── run_code_fingerprint.py     results/run_code_fingerprint.json: write and check the hashes of the result-producing code
│   ├── derive_raw_facts.py         the FACTS rows that need the git-ignored raw files -> results/raw_facts.csv
│   ├── selection_defs.py           the definitions shared by the table script and the selection analysis
│   ├── derive_selection.py         choosing a method by trial on the Phase 1 records -> results/phase1/phase1_selection_*.csv + provenance
│   ├── make_paper_tables.py        results/ -> paper_fragments/*.tex + FACTS.md (reads no raw file)
│   ├── analyze_by_ltrue.py         Phase-1 results by L_true tercile -> f13 per stratum, its FACTS section
│   ├── build_tables_document.py    paper_fragments/all_tables.tex: every fragment in one standalone document
│   ├── build_paper_tables.py       the main-text tables of the paper as subsets of the fragments, every number checked
│   ├── check_tables.py             the reader's check: regenerate every table from the committed results/ and compare
│   ├── make_paper_figures.py       optional summary figures (paper_figures/, git-ignored)
│   ├── dev/compare_quick_runs.py   development tool: before/after regression check of two --quick results trees
│   └── analyze_real_diagnostics_legacy.py   verifies the stored pre-redesign 18-cell real-data run
├── tests/                  the test modules listed above; tested_environment.py guards the two exact-equality tests
├── results/                committed output of the reported run (results/README.md is the inventory)
├── paper_fragments/        generated LaTeX table fragments (f01 ... f30, f13 per stratum) + index + all_tables.tex
├── FACTS.md                generated headline numbers with provenance
├── data/                   OpenML downloads (legacy --retrain path only; git-ignored)
├── .github/workflows/tests.yml   the checks above, on every push and pull request
├── requirements.txt        lower bounds
├── requirements-lock.txt   the tested environment
├── CITATION.cff, LICENSE
└── reproduce_all.py        single script reproducing every result in order
```

---

## Data availability

- **Synthetic sequences** are generated deterministically from `src/generators.py` and `src/config.py`; no download is required.
- **Real XGBoost curves** were recorded from six [OpenML](https://www.openml.org) datasets (ids 1590, 1461, 1596, 23512, 41150, 41168) and are committed as `results/real_data/real_data_curves.csv`; the default real-data step re-evaluates these recorded curves without retraining, so it is exactly reproducible. `scripts/run_real_data.py --retrain` regenerates the curves (XGBoost-version drift in the third decimal is expected).
- **`real_boot` generator sources** are the validation curves of two further OpenML datasets, `electricity` (id 151) and `nomao` (id 1486), committed as `results/real_data/real_boot_sources.csv` with `real_boot_sources_provenance.json`; they are generator inputs of the out-of-design `real_boot_a` / `real_boot_b` families and are never scored.
- **Legacy material** is kept and labelled: `results/real_data/real_data_results.csv` with `figure_rd_01..03` (the original 18-cell real-data run at the legacy assumed asymptote 0.01; fragment `f07b`), `scripts/analyze_real_diagnostics_legacy.py`, `config.LEGACY_DANGEROUS_METHODS` and `config.ASYMPTOTE_MODE = "legacy"` (regression test only).

---

## Licence and citation

The code and the results are released under the MIT License (`LICENSE`). To cite the software or its results, use `CITATION.cff` (GitHub's "Cite this repository") or:

```bibtex
@article{maleki2026sequence,
  title   = {Sequence Acceleration versus Trajectory Fitting for Early Learning-Curve
             Prediction: A Benchmark with Hidden Asymptotes and Trivial Baselines},
  author  = {Maleki, Kian},
  journal = {Applied Soft Computing},
  year    = {2026},
  note    = {Submitted}
}
```
