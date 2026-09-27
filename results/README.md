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

Results are only meaningful alongside the commit that produced them. Any
change to `src/generators.py`, `src/accelerators.py`, `src/evaluation.py`,
`src/panels.py` or the validity window in `src/config.py` invalidates every
stored table, so regenerate the whole set rather than mixing output from
different commits. The committed tree is the output of the previous full run
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
committed.

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
| `phase0_unit_tests.csv/.txt` | analytic unit tests of the accelerator roster |
| `phase0b/order_ladders_panels.csv` | Phase 0b: every order of every family with an order parameter (and the trivial comparators) on Phase 1's grid at the headline stratum; per (family, variant, regime set, noise or `pooled`) the descriptive panel, `n_cells`, `n_cells_capped_excluded`; `is_roster` / `roster_name` mark the orders that are roster methods |
| `phase0b/order_ladders_agreement.txt` | the exact-agreement check of every roster-marked ladder variant against `phase1_records.csv` (variants checked, records compared, verdict) |
| `phase1/phase1_global.csv`, `phase1_global_holdout.csv` | pooled ranking per stratum, core / held-out (median error, validity floor) |
| `phase1/phase1_aggregated.csv` | one row per (method, regime, noise, g): the descriptive panel of the seeds, `med_improve`, `med_skill` (hindsight best-of-four, strict), `med_skill_vs_*` / `win_rate_vs_*` against each deployable trivial |
| `phase1/phase1_unranked.csv`, `phase1_capped.csv`, `phase1_horizons.csv`, `phase1_heatmap_g*.csv`, `phase1_heatmap_valid_g*.csv`, `phase1_regime_best.csv` | floor block, capped block, horizon search, method x regime median error and valid rate per stratum, per-regime best by skill and best by median error at the floor |
| `phase1/phase1_by_Ltrue.csv` | Phase-1 results by L_true tercile (`scripts/analyze_by_ltrue.py`, the last pipeline step) |
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
| `real_data/real_data_curves.csv`, `real_data_results.csv`, `figure_rd_0{1,2,3}_*.png` | the recorded curves (the real-data test set) and the preserved pre-redesign 18-cell run |

Removed with their code paths and no longer written: `phase5b/phase5b_sweep3_*.csv` and `figure_p5b_03_catmult.png` (at 842ddb9), `phase3/phase3_cv_results.csv` and `figure_p3_05_cv_summary.png`, `phase1/figure_01_stability_ranking.png` (now `figure_01_error_ranking.png`), `phase2/phase2_phase_diagram_g*.csv` (at d6beb53).

The paper tables are generated from these files by
`python scripts/make_paper_tables.py` (fragments in `paper_fragments/`,
headline numbers with provenance in `FACTS.md`), followed by
`python scripts/analyze_by_ltrue.py`; both run as the last two steps of
`reproduce_all.py`.
