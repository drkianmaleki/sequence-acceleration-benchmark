# results/

This directory holds generated output. It is populated by running the
benchmark; nothing here is written by hand.

```
python reproduce_all.py            # everything
python scripts/run_phase1.py --full   # one phase at a time
```

Each phase writes to its own subdirectory (`phase1/`, `phase2/`, ... ,
`real_data/`), creating it if needed.

## Provenance

Results are only meaningful alongside the commit that produced them. Any
change to `src/generators.py`, `src/accelerators.py`, `src/evaluation.py`,
or the validity window in `src/config.py` invalidates every stored table,
so regenerate the whole set rather than mixing output from different
commits.

After re-running Phase 1, check that the derived dangerous-method set is
still correct:

```
python scripts/check_dangerous.py
```

Three large per-record files are deliberately excluded from version control
(see `.gitignore`) and regenerated on each run: `phase1/phase1_records.csv`
(362,880 rows, 91 MB), `phase4/phase4_raw.csv` (224,640 rows, 54 MB) and
`phase5a/phase5a_raw.csv` (967,680 rows, 264 MB).  Everything else under
`results/`, including `phase2/phase2_features.csv` and
`phase2/phase2_sweep_aggregated.csv`, is committed.

## Layout (redesign v2)

Every phase writes per-stratum tables keyed by `target_g` (the gap fraction
g in {0.5, 0.1, 0.02}; headline g = 0.1) and separates core from held-out
regimes (`is_holdout` / `regime_set`).  Pooled tables exclude capped cells
and carry `rank` / `rank_eligible`; the below-floor methods form an
`*_unranked.csv` block and the capped cells a `*_capped.csv` block.

| path | content |
|---|---|
| `phase0_unit_tests.csv/.txt` | analytic unit tests of the 49 accelerators |
| `phase1/phase1_global.csv`, `phase1_global_holdout.csv` | pooled ranking per stratum, core / held-out |
| `phase1/phase1_aggregated.csv` | one row per (method, regime, noise, g): valid / cat / beats rate, median error, improvement, `med_skill` (hindsight best-of-four, strict), `med_skill_vs_*` / `win_rate_vs_*` against each deployable trivial |
| `phase1/phase1_unranked.csv`, `phase1_capped.csv`, `phase1_horizons.csv`, `phase1_heatmap_g*.csv`, `phase1_regime_best.csv` | floor block, capped block, horizon search, per-regime stability maps, per-regime champions |
| `phase1/dangerous_methods.json` | the dangerous-method artifact read by Phases 2-5 (`scripts/derive_dangerous.py`) |
| `phase2/phase2_phase_diagram_g*.csv`, `phase2_rules_g*.csv`, `phase2_correlations_g*.csv` | failure detection per stratum (`phase2_rules.csv` / `phase2_correlations.csv` are copies of the headline stratum) |
| `phase2/phase2_features.csv`, `phase2_sweep_aggregated.csv`, `phase2_capped.csv` | window features, per-cell method table, capped block |
| `phase3/*` | selector comparison, per-regime results, leave-one-regime-out CV, regime classifier, capped cells |
| `phase4/*` | diagnostic correlations (core pooled + `_holdout`), rejection rules, depth reliability, selectors, cascade screen, capped block |
| `phase5a/*` | ensemble ablation: selectors (core + `_holdout`), by sigma, per regime, ablation, method weights, oracle gap, capped block, `phase5a_validity_by_depth.csv` (method × depth × noise valid rate) |
| `phase5b/*` | sweeps 1a (cascade vs assumed asymptote; clamped features, labelled), 1b (`phase5b_sweep1_consumers*.csv`: the L_hat-consuming accelerators + `constant_assumed` under every mode), 2 (window length), 3 (CAT_MULT) with the sweep-3 unranked block |
| `real_data/real_data_results_v2.csv`, `real_data_summary_v2.csv`, `real_data_curve_minima_v2.csv`, `real_data_strata_v2.csv`, `figure_rd_v2_01_skill.png` | re-evaluation of the recorded curves on the 6 x 15 (depth, target) grid; per-curve argmin round and rise from the minimum; every summary three ways (all / pre-minimum / post-minimum targets) |
| `real_data/real_data_curves.csv`, `real_data_results.csv`, `figure_rd_0{1,2,3}_*.png` | the recorded curves (input) and the preserved pre-redesign 18-cell run |

The paper tables are generated from these files by
`python scripts/make_paper_tables.py` (fragments in `paper_fragments/`,
headline numbers with provenance in `FACTS.md`).
