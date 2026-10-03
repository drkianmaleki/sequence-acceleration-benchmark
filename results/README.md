# results/

This directory holds generated output. It is populated by running the
benchmark; nothing here is written by hand.

```
python reproduce_all.py               # everything, in order
python scripts/run_phase1.py --full   # one phase at a time
```

Each phase writes to its own subdirectory (`phase0b/`, `phase1/`, `phase2/`,
... , `real_data/`), creating it if needed.

## Provenance

Results belong to the run recorded in `run_manifest.json`, and
`run_code_fingerprint.json` records the code of that run: the SHA-256 (CRLF
normalised) of every code file that produces a result -- `src/`, `phases/`,
the result-producing step scripts and the test modules they import -- as the
files were at the run. Any change to `src/generators.py`,
`src/accelerators.py`, `src/evaluation.py`, `src/panels.py` or the validity
window in `src/config.py` invalidates every stored table, so regenerate the
whole set rather than mixing output from different commits; a result added
later must carry its own provenance file (the roster evaluation of the
recorded curves does, `real_data/real_data_roster_provenance.json`, and so
does the selection analysis, `phase1/phase1_selection_provenance.json`), the
file it changes must be recorded in the fingerprint with one sentence and the
test that guarantees the old outputs, and
`python scripts/run_code_fingerprint.py --check` must pass on the tree. The
committed tree is the output of the previous full run
(code 842ddb9) minus the outputs of removed code paths (Phase 5b sweep 3,
the Phase 3 leave-one-regime-out step, the stability-score figure and the
Phase 2 phase-diagram files, all still available at their last commits) plus
the `real_boot` source files; the full run of the current code regenerates
it.

After re-running Phase 1, check that the derived excluded-method set is
still correct and declares the validity criterion:

```
python scripts/check_dangerous.py
```

## Git-ignored raw files

Five large per-record files are excluded from version control (see
`.gitignore`) and regenerated on each run: `phase1/phase1_records.csv`
(one row per (regime, noise, seed, g, method); carries `E_last`),
`phase0b/order_ladders_records.csv` (one row per ladder variant and window at
the headline stratum), `phase2/phase2_records.csv` (one row per Phase-2
record: `estimate`, `error`, `valid`, `catastrophic`, `E_last`, `capped`,
`L_true`, `L_hat`, `n_f`, `achieved_g`; Phase 3 reads it), `phase4/phase4_raw.csv`
and `phase5a/phase5a_raw.csv`. Everything else under `results/`, including
`phase2/phase2_features.csv` and `phase2/phase2_sweep_aggregated.csv`, is
committed. The ten `FACTS.md` rows that only the raw files can supply are
computed once by `scripts/derive_raw_facts.py` into the committed
`raw_facts.csv`, so the table generator reads no raw file and every table
regenerates from the committed tree (`python scripts/check_tables.py`).
The selection-by-trial aggregates `phase1/phase1_selection_cells.csv`,
`phase1/phase1_selection_global.csv` and `phase1/phase1_selection_provenance.json`
(`scripts/derive_selection.py`) are, like `raw_facts.csv`, derived from the
git-ignored `phase1/phase1_records.csv` and committed; the analysis evaluates
no method.

## Descriptive panel

Every per-method or per-selector table carries the same *descriptive panel*
(`src/panels.py::error_panel`): `n_total`, `n_valid`, `valid_rate`,
`cat_rate` over all records; `mean_error`, `sd_error`, `med_error`,
`q25_error`, `q75_error`, `p90_error` over the valid records only
(conditional on validity, to be read next to the validity rate); and
`win_rate_vs_last`, the fraction of all records where the method is valid and
its error is below the last-value error. Rule tables (Phase 2 rules, the
Phase 5b cascade) carry the *rule panel* (`rule_panel`): `n_cells_total`,
`n_cells_fired`, `fire_rate`, the panel of `richardson_1` (`r1_*`) and of the
routed method (`alt_*`) over the fired cells, `lower_error_frac_records`,
`lower_error_frac_cells`, `median_rel_change`, and the two panels over the
not-fired cells (`nf_*`). No composite score appears anywhere.

## Layout (redesign v2)

Every phase writes per-stratum tables keyed by `target_g` (the gap fraction
g in {0.5, 0.1, 0.02}; headline g = 0.1) and separates core from held-out
regimes (`is_holdout` / `regime_set`). Pooled tables exclude capped cells
and carry `rank` / `rank_eligible`; the below-floor methods form an
`*_unranked.csv` block and the capped cells a `*_capped.csv` block.

| path | content |
|---|---|
| `run_manifest.json` | provenance of the run that produced this tree, written by `reproduce_all.py` at the start of the run, again after every step and at the end (so the generators inside the run read their own run; the final rewrite sets the total): mode, git head (full and short), start and end time, the worker count of the parallel steps, every step's wall seconds in order, the total, the evaluation plan of that mode and the library versions (Python, numpy, scipy, pandas, scikit-learn, xgboost); the committed copy is the full run's |
| `run_code_fingerprint.json` | the code fingerprint of that run (`scripts/run_code_fingerprint.py`): purpose, the run (start, full commit hash, mode), the algorithm, and per code file of the path set its SHA-256 at the run; a file deliberately changed or added after the run also carries `sha256_now` and `changed_after_run` (one sentence: what changed and which test guarantees the old outputs); `--check` compares the working tree with it |
| `raw_facts.csv` | the ten FACTS rows derived from the git-ignored raw files (`scripts/derive_raw_facts.py`; columns section, fact, value, file, filter, formula), emitted by the table generator in place of reading the raw files |
| `phase0_unit_tests.csv/.txt` | analytic unit tests of the accelerator roster |
| `phase0b/order_ladders_panels.csv` | Phase 0b: every order of every family with an order parameter (and the trivial comparators) on Phase 1's grid at the headline stratum; per (family, variant, regime set, noise or `pooled`) the descriptive panel, `n_cells`, `n_cells_capped_excluded`; `is_roster` / `roster_name` mark the orders that are roster methods |
| `phase0b/order_ladders_agreement.txt` | the exact-agreement check of every roster-marked ladder variant against `phase1_records.csv` (variants checked, records compared, verdict) |
| `phase1/phase1_global.csv`, `phase1_global_holdout.csv` | pooled ranking per stratum, core / held-out (median error, validity floor) |
| `phase1/phase1_aggregated.csv` | one row per (method, regime, noise, g): the descriptive panel of the seeds, `med_improve`, `med_skill` (hindsight best-of-four, strict), `med_skill_vs_*` / `win_rate_vs_*` against each deployable trivial |
| `phase1/phase1_unranked.csv`, `phase1_capped.csv`, `phase1_horizons.csv`, `phase1_heatmap_g*.csv`, `phase1_heatmap_valid_g*.csv`, `phase1_regime_best.csv` | floor block, capped block, horizon search, method x regime median error and valid rate per stratum, per-regime best by skill and best by median error at the floor |
| `phase1/phase1_by_Ltrue.csv` | Phase-1 results by L_true tercile (`scripts/analyze_by_ltrue.py`, the last pipeline step) |
| `phase1/phase1_selection_cells.csv`, `phase1_selection_global.csv`, `phase1_selection_provenance.json` | choosing a method by trial on the Phase 1 records (`scripts/derive_selection.py`; derived from the git-ignored `phase1_records.csv`, evaluates no method): per uncapped (set, regime, noise, g) cell, candidate pool (lead = the leading methods, named = `rational_fit` and `single_exp_fit`, classical = the classical variants, all = every accelerator) and design (`one_pilot`: every ordered pair of seeds, the member with the lowest error on the pilot seed; `many_pilots`: every seed as the final run, the member with the lowest median error over the other seeds among those above the validity floor), the share of trials whose chosen record is valid, beats the default and beats the last value, how often the default itself is chosen, the median errors of the chosen method, the default and the last value and the median error ratio, and (appended after those columns) `cat_rate_chosen`, the share of trials whose chosen record on the final seed carries the Phase 1 `catastrophic` flag as recorded (invalid, or error above `CAT_MULT` times the last value's; a trial without a choice counts as catastrophic, as it counts as invalid), and `cat_rate_default`, the same for `rational_fit`'s record on the same final seed; the pooled table per set, stratum, noise class (and `all`), pool and design, whose `all` rows reproduce the pooled Phase 1 medians of the default and the last value (checked) and whose `cat_rate_default` (the mean over cells) reproduces the pooled Phase 1 `cat_rate` of `rational_fit` to four decimals (checked, as is every cell's against `phase1_aggregated.csv`); the provenance file records the script, commit, tree state, timing, library versions, the records file's row count and SHA-256, the pools, the designs, cells and trials per design, the two agreement checks and the definition of the two catastrophe-rate columns |
| `phase1/dangerous_methods.json` | the excluded-method artifact read by Phases 2-5 (schema `dangerous_methods/v3`: pooled `valid_rate < RANK_MIN_VALID`; `dangerous` is the legacy implementation name) |
| `phase2/phase2_sweep_aggregated.csv`, `phase2_features.csv`, `phase2_capped.csv` | per-cell descriptive panels of the Phase-2 pool, window features, capped block |
| `phase2/phase2_richardson_targets_g*.csv`, `phase2_denominator_counts.csv` | per cell: `richardson_1`'s error normalised by the last-value error (`R_R_med`) and the log of its median error, with validity counts; per horizon the records and cells where the `E_last <= 1e-12` branch of the ratio fired |
| `phase2/phase2_correlations_g*.csv`, `phase2_rules_g*.csv` | Spearman correlations of the window features with both targets; every candidate rule's rule panel (`phase2_rules.csv` / `phase2_correlations.csv` are copies of the headline stratum) |
| `phase3/phase3_selector_comparison.csv`, `phase3_regime_results.csv`, `phase3_capped_cells.csv` | selectors scored per record on the chosen method's records (no fallback): the panel per stratum and per regime (the per-family evidence of the fixed-threshold cascades), capped cells with the chosen method |
| `phase3/phase3_regime_classifier.csv` | regime classification accuracy under three protocols side by side (`protocol` column: `grouped_by_depth` primary, `grouped_by_noise`, `stratified_5fold_legacy`), one row per regime and protocol plus `__OVERALL__`; the `split_unit` column states what a row is and what a fold holds out, `note` the fold count |
| `phase4/*` | shift and paired perturbation diagnostics: correlations (core pooled + `_holdout`), rejection rules (precision / recall of a detector of the catastrophe event), depth reliability, selectors, cascade screen, capped block |
| `phase5a/*` | ensemble ablation with paired perturbations: selectors (core + `_holdout`), by sigma, per regime, ablation, method weights, oracle gap, capped block, `phase5a_validity_by_depth.csv` (method x depth x noise valid rate) |
| `phase5b/*` | sweeps 1a (cascade vs assumed asymptote, rule panel; clamped features, labelled), 1b (`phase5b_sweep1_consumers*.csv`: the L_hat-consuming accelerators + `constant_assumed` under every mode), 2 (window length, rule panel); the former sweep 3 (CAT_MULT) was removed with the composite score (outputs at commit 842ddb9) |
| `real_data/real_boot_sources.csv`, `real_boot_sources_provenance.json` | the `real_boot_a` / `real_boot_b` generator inputs: XGBoost validation curves of OpenML `electricity` (151) and `nomao` (1486), outside the recorded-curve test set, with the inclusion criteria and their verdicts (`scripts/make_real_boot_sources.py`); never scored |
| `real_data/real_data_results_v2.csv`, `real_data_summary_v2.csv`, `real_data_curve_minima_v2.csv`, `real_data_strata_v2.csv`, `figure_rd_v2_01_skill.png` | re-evaluation of the recorded curves on the 6 x 15 (depth, target) grid; per-curve argmin round and rise from the minimum; every summary three ways (all / pre-minimum / post-minimum targets) |
| `real_data/real_data_roster_v2.csv`, `real_data_roster_provenance.json` | the roster evaluation of the same 90 cells: every accelerator and the four deployable trivial predictors (`src.trajectories.ROSTER_REAL_METHODS`, no oracle) scored under the benchmark's own configuration and per-record definitions (`src.evaluation`: `build_cfg`, `is_valid`, the Phase-1 record loop), one row per (dataset, depth, target, method) with `prediction_raw`, both validity rules (`valid`, `valid_strict`), error, `E_last`, catastrophic, improve ratio, skill and the fixed-reference skill / win columns; the provenance file records the script, commit, tree state, timing, library versions (and whether they equal the run manifest's), the numerical configuration, datasets, cells, methods and rows -- this evaluation was added after the full run (`scripts/run_real_data.py --roster-only`) |
| `real_data/real_data_curves.csv`, `real_data_results.csv`, `figure_rd_0{1,2,3}_*.png` | the recorded curves (the real-data test set) and the preserved pre-redesign 18-cell run |

Removed with their code paths and no longer written: `phase5b/phase5b_sweep3_*.csv` and `figure_p5b_03_catmult.png` (at 842ddb9), `phase3/phase3_cv_results.csv` and `figure_p3_05_cv_summary.png`, `phase1/figure_01_stability_ranking.png` (now `figure_01_error_ranking.png`), `phase2/phase2_phase_diagram_g*.csv` (at d6beb53).

The paper tables are generated from these files by
`python scripts/make_paper_tables.py` (fragments in `paper_fragments/`,
headline numbers with provenance in `FACTS.md`; the raw-derived rows from
`raw_facts.csv`, written by `python scripts/derive_raw_facts.py` after a run),
followed by `python scripts/analyze_by_ltrue.py` (the tercile fragments and
their FACTS section, from `phase1/phase1_by_Ltrue.csv` when the records
file is absent) and `python scripts/build_tables_document.py`
(`paper_fragments/all_tables.tex`); with `python scripts/derive_selection.py`
(the selection aggregates, after a run) the five run as the last five steps of
`reproduce_all.py`, and `python scripts/check_tables.py` regenerates all of
it from the committed tree and compares.
