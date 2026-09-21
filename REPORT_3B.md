**Report back**

# Redesign v2, Prompt 3B: the full pipeline run, results committed and pushed

Date: 2026-09-20 23:35 to 2026-09-21 03:58 (run), report 2026-09-21. Branch `redesign-v2` in `code/`, code at `0375f24` (unchanged; no method or pipeline code was touched). Results commit `30de724`, pushed to `origin/redesign-v2` (section 7); this report is committed on top. Complete console log in `REPORT_3B_full_run.txt` (1,340 lines).

Pre-flight: tree clean at `0375f24`; `python -m pytest tests/ -q`: 342 passed in 28.4 s; no other Python process on the machine; 1.4 TB free.

## 1. `--plan` evaluation counts (printed before the run)

```
  PLANNED EVALUATIONS  [FULL]
  Phase 0                         204   51 accelerators x 4 analytic cases (trivials excluded)
  Phase 1                     362,880   24 regimes x 30 seeds x 3 noise x 3 strata x 56 methods
  Dangerous derivation              0   reads phase1_aggregated.csv
  Phase 2                     772,200   13 depths x 5 noise x 20 seeds x 18 core regimes x 3 strata x 11 methods (+2 skill-reference calls per cell)
  Phase 3                           0   analysis of Phase 2 output
  Phase 4                     224,640   central evaluations (24 regimes, 13 methods); + 1,555,200 diagnostic calls
  Phase 5a                    967,680   central evaluations (24 regimes, 56 methods); + 4,406,400 perturbation calls
  Phase 5b                    532,800   sweep1 28,800 + sweep2 28,800 + sweep3 475,200 (24 regimes, 2 strata)
  Real data                       630   6 datasets x 15 (depth, target) pairs x 7 methods
  TOTAL (central)           2,861,034
```

(The `[QUICK]` block was printed too: 17,610 central; it matches Reports 2/3A.)

## 2. Per-phase wall times and evaluation counts

`python reproduce_all.py` (no `--quick`), exit code 0, all nine steps OK. Phase 5a ran at the 3A-validated job count: `reproduce_all.py` passes no `--jobs`, so `run_phase5a.py` used its default `cpu_count() - 1 = 7` (8 physical cores, no SMT), as in the 3A first-chunk measurement.

| # | step | wall | central evaluations | diagnostic / perturbation calls |
|---|---|---:|---:|---:|
| 1 | Phase 0 — analytic unit tests | 2 s | 204 | – |
| 2 | Phase 1 — main benchmark | 1,839 s (30.7 min) | 362,880 | – |
| 3 | Dangerous re-derivation | 1 s | – (12,096 aggregated rows read) | – |
| 4 | Phase 2 — failure detection | 979 s (16.3 min) | 772,200 | + 2 skill-reference calls per cell |
| 5 | Phase 3 — adaptive selection | 5 s | – (analysis) | – |
| 6 | Phase 4 — perturbation diagnostics | 1,797 s (30.0 min) | 224,640 | 1,555,200 |
| 7 | Phase 5a — ensemble ablation | 8,868 s (2.46 h) | 967,680 | 4,406,400 (5 trials) |
| 8 | Phase 5b — sensitivity sweeps | 2,287 s (38.1 min) | 532,800 | – |
| 9 | Real data — recorded-curve re-evaluation | 5 s | 630 | 450 (perturb_iqr, 5 trials x 90 cells) |
| | **total** | **15,783 s (4.38 h)** | **2,861,034** | |

**Perturbation-trial path: the 5-trial path was taken; nothing was reduced.** The decision rule was "reduce Phase-5a `perturb_trials` from 5 to 3 if the first-chunk projection exceeds 16 h at full parallelism". The printout at the first completed block:

```
  [  1/12] block obs=90   sigma=0        80,640 rows  compute  54.0 min  wall  54.0 min
  FIRST CHUNK: obs=90 sigma=0: 1,440 cells, 80,640 rows in 54.0 min compute (2248.0 ms/cell), 54.0 min wall since start
  PROJECTION  : 12 chunks / 7 jobs = 2 round(s) x 54.0 min = 1.80 h evaluation wall time at --jobs 7  (serial equivalent 10.79 h; equal-cost chunks assumed; aggregation and figures extra)
```

1.80 h at `--jobs 7` (even the serial equivalent, 10.79 h, is under 16 h), so `config.PHASE5A["full"]["perturb_trials"]` stayed at `PERTURB_TRIALS = 5` and the run continued untouched. Actual Phase-5a evaluation wall time was **2.41 h** (all 12 blocks), plus ~3 min of concatenation, aggregation and the four figures, for the 8,868 s step total. The blocks were far from equal-cost (anomaly 3):

| block | obs | sigma | compute | finished at (wall since start) |
|---|---|---|---:|---:|
| 1 | 90 | 0 | 54.0 min | 54.0 min |
| 2 | 60 | 0.02 | 83.5 min | 83.6 min |
| 3 | 30 | 0.02 | 85.0 min | 85.0 min |
| 4 | 60 | 0.005 | 1.57 h | 1.57 h |
| 5 | 60 | 0 | 1.61 h | 1.61 h |
| 6 | 30 | 0.005 | 1.63 h | 1.63 h |
| 7 | 30 | 0 | 1.72 h | 1.72 h |
| 8 | 90 | 0.005 | 49.4 min | 1.72 h |
| 9 | 90 | 0.02 | 38.9 min | 2.04 h |
| 10 | 120 | 0 | 56.3 min | 2.35 h |
| 11 | 120 | 0.02 | 46.8 min | 2.39 h |
| 12 | 120 | 0.005 | 50.5 min | 2.41 h |

The seven workers each ran at ~98 % of a core (102 CPU-min per worker after 104 min wall). `phase5a_raw.csv`: 967,680 rows, 263.8 MB; the `shards/` directory was removed cleanly after concatenation (no retry needed).

## 3. Invalid / NaN rates per phase

Method: after each phase, the pooled invalid rate (`valid == 0`, i.e. the estimate is NaN or outside `[MIN_VALID, MAX_VALID]`) over all non-oracle rows of methods **outside the dangerous set** (the full-run artifact of section 4), computed from the per-record file where one exists (`phase1_records.csv`, `phase4_raw.csv`, `phase5a_raw.csv`) and from the seed-weighted `valid_rate` of the aggregated table otherwise (Phase 2, Phase 5b sweep 3). The threshold in the prompt was 1 %.

| phase | rows checked | non-dangerous non-oracle evaluations | invalid | rate | lowest non-dangerous valid rates | verdict |
|---|---:|---:|---:|---:|---|---|
| Phase 1 | 362,880 | 304,560 | 2,716 | **0.892 %** | pade_22 0.944, wynn_eps_3 0.969, log_linear 0.975, wynn_eps_2 0.978 | under threshold |
| Phase 2 (aggregated) | 38,610 | 702,000 | 4,907 | **0.699 %** | log_linear 0.967, pade_22 0.976, weniger_d2 0.987; all others 1.000 | under threshold |
| Phase 3 | – | – | – | – | analysis only | – |
| Phase 4 | 224,640 | 224,640 | 1,242 | **0.553 %** | log_linear 0.970, pade_22 0.972, weniger_d2 0.986, anderson_1 0.999 | under threshold |
| Phase 5a | 967,680 | 812,160 | 12,822 | **1.579 %** | richardson_3 0.761, levin_u1 0.960, log_linear 0.970, wynn_eps_3 0.971, pade_22 0.972, pade_11 0.973 | **threshold crossed; ruled not a flood** (below) |
| Phase 5b sweep 3 (aggregated) | 220 rows per cat_mult | 43,200 cells x 3 cat_mult | – | **1.504 % (g = 0.1) / 1.327 % (g = 0.5)** | pade_22 0.932, log_linear 0.950, wynn_eps_3 0.954, wynn_eps_2 0.962 | **threshold crossed; same phenomenon** |
| Phase 5b sweeps 1-2 | – | – | – | – | selector fire/precision/recall sweeps; no per-method validity column | – |
| Real data | 630 | 630 | 0 | **0.000 %** | NaN prediction 0, NaN error 0, NaN perturb_iqr 0 | under threshold |

Dangerous-set rows, for the record: 38.9 % invalid in Phase 1, 42.9 % in Phase 5a, 37.1 % in Phase 5b sweep 3.

**The Phase-5a crossing, and the decision.** The run was paused for a decision at the Phase-5a check (Phase 5b was allowed to keep computing in the background; nothing had been deleted, committed or pushed). Decomposition of the 12,822 invalid evaluations outside the dangerous set:

* `richardson_3` alone accounts for 32.2 % of them (4,131 of its 17,280 rows; every one a NaN estimate, none out-of-range). Its valid rate is 0.481 at obs 30, 0.583 at obs 60, 0.981 at obs 90 and 0.999 at obs 120; by regime at obs 30 it is 0.02-0.08 on `rational_decay`, `multiphase`, `real_boot_a/b`, `two_exp`. `richardson_3` is a 7-parameter nonlinear `curve_fit` (`L + c1/n^a1 + c2/n^a2 + c3/n^a3`, `maxfev = 3000`, `src/accelerators.py:157`) that does not converge on 30-60-point windows and returns NaN by design. The dangerous set is derived from Phase 1 at obs_idx = 90 only, where the method is 98 % valid, so the derivation cannot see this depth dependence. Phase 5a is the only phase that evaluates all 56 methods at obs 30 and 60.
* Excluding `richardson_3` the rate is 1.093 %; excluding `richardson_3` and sigma = 0 it is 0.515 %. By noise: 2.82 % at sigma = 0, 0.85 % at 0.005, 1.06 % at 0.02. By depth: 2.19 % (obs 30), 1.94 % (60), 1.07 % (90), 1.11 % (120). The sigma = 0 excess is exact-cancellation NaNs (zero denominators, `DENOM_TOL`) in the difference-based Levin (`u1`, `v1`, `u2`, `t2`), Wynn (`eps_2/3`, `rho_3`) and Brezinski families on noise-free plateaus; the same families sit at 1.9 % at sigma = 0 in Phase 1.
* 36 of the 47 non-dangerous non-oracle methods are >= 98 % valid; 98.4 % of all non-dangerous evaluations are valid.

This was reported and put to Kian as a three-way choice (proceed / treat as a flood and stop / proceed but do not push); the answer was **"Not a flood — proceed"**. The Phase-5b sweep-3 figure was found after the ruling; it is the same phenomenon at the same depth as Phase 1 (per-method valid rates correlate 0.944 with Phase 1 at obs 90, sigma in {0, 0.005}, where Phase 1 itself sits at 1.18 % for g = 0.1 — its pooled 0.91 % is pulled down by the sigma = 0.001 level that sweep 3 does not use; Phase 5a at obs 90, g = 0.1 with sweep 3's noise set gives 1.08 %), so it was treated under the same ruling rather than re-asked. Both crossings are listed as anomaly 1.

## 4. Dangerous-artifact method list (full Phase 1) and the diffs

`scripts/derive_dangerous.py` on the full `phase1_aggregated.csv` (12,096 rows): 55 methods scored = 51 accelerators (eligible) + 4 trivial comparators (scored for the record, never in the artifact). Criterion: pooled `S = valid_rate - 2.0 * cat_rate + 0.4 * beats_rate < 0` over the 18 core regimes, pooled over the three strata and the three noise levels, capped cells excluded, oracle excluded; 147 cells per method.

**Dangerous (8):** `geom_avg_diff, linear, neville_2, neville_3, neville_4, pade_21, pade_31, pade_32`

| method | S | valid | cat | beats | cells |
|---|---:|---:|---:|---:|---:|
| neville_4 | -1.0592 | 0.345 | 0.748 | 0.231 | 147 |
| neville_3 | -0.9423 | 0.502 | 0.768 | 0.232 | 147 |
| neville_2 | -0.6766 | 0.673 | 0.721 | 0.233 | 147 |
| linear | -0.6543 | 0.583 | 0.631 | 0.063 | 147 |
| pade_31 | -0.5447 | 0.802 | 0.703 | 0.148 | 147 |
| pade_21 | -0.3393 | 0.673 | 0.546 | 0.199 | 147 |
| pade_32 | -0.1981 | 0.749 | 0.524 | 0.255 | 147 |
| geom_avg_diff | -0.1470 | 0.591 | 0.449 | 0.402 | 147 |
| *nearest non-dangerous:* weniger_d1 | 0.6463 | 0.977 | 0.233 | 0.337 | 147 |
| weniger_d2 | 0.6693 | 0.984 | 0.225 | 0.340 | 147 |
| *(trivial, scored only)* constant_assumed | 0.7218 | 1.000 | 0.209 | 0.352 | 147 |

The margin between the last dangerous method (S = -0.147) and the first safe accelerator (S = +0.646) is wide; no trivial comparator has S < 0 (`window_mean` 0.996, `last_value` 1.000, `window_min` 1.052).

**Diff against the legacy set** (`config.LEGACY_DANGEROUS_METHODS`, the hard-coded set of the rejected design): **`+[]  -[]` — identical.** The derivation prints `vs legacy : +[] -[]`; `scripts/check_dangerous.py`: `declared 8, re-derived 8, MATCH`.

Diff against the quick-run artifact of Report 3A (2 seeds, 4 regimes), for context: `+[geom_avg_diff, pade_32]  -[wynn_eps_2, wynn_eps_3]`. At full scale `wynn_eps_2/3` have S = 1.02/1.04 (valid 0.98/0.96) and `pade_32`/`geom_avg_diff` fall below zero.

Artifact (`results/phase1/dangerous_methods.json`, committed): `schema dangerous_methods/v2`, `pool = accelerators`, `n_pool = 51`, `table` = 51 rows (no trivial), `asymptote_mode hetero`, `assumed_mode zero`, `gap_fractions [0.5, 0.1, 0.02]`, `git_head 0375f24`, `created 2026-09-21T00:06:04`. Consumers confirmed in the log: Phase 5a header "51 accelerators (8 dangerous per artifact)", Phase 5b sweep-3 unranked block = the same 8 methods, Phase-1 UNRANKED block at g = 0.1 (core) = the same 8 methods.

## 5. Results-file inventory (after the run, as committed)

Status is relative to the pre-run commit `0375f24`: `new` = first committed in `30de724`, `modified` = regenerated with different content, `unchanged` = regenerated byte-identical (Phase 0) or legacy files left in place, `ignored (raw)` = per-record files excluded by `.gitignore`.

| file | bytes | status |
|---|---:|---|
| `results/README.md` | 1,133 | unchanged |
| `results/phase0_unit_tests.csv` | 27,100 | unchanged |
| `results/phase0_unit_tests.txt` | 24,215 | unchanged |
| `results/phase1/dangerous_methods.json` | 13,549 | new |
| `results/phase1/figure_01_stability_ranking.png` | 237,483 | modified |
| `results/phase1/figure_02_heatmap.png` | 717,821 | modified |
| `results/phase1/figure_03_horizon_sensitivity.png` | 104,057 | modified |
| `results/phase1/figure_04_method_type.png` | 99,256 | modified |
| `results/phase1/figure_05_regime_recommendations.png` | 125,778 | modified |
| `results/phase1/figure_06_richardson_profile.png` | 89,821 | modified |
| `results/phase1/phase1_aggregated.csv` | 2,253,132 | modified |
| `results/phase1/phase1_capped.csv` | 40,552 | new |
| `results/phase1/phase1_global.csv` | 23,550 | modified |
| `results/phase1/phase1_global_holdout.csv` | 23,566 | new |
| `results/phase1/phase1_heatmap_g0.02.csv` | 9,453 | new |
| `results/phase1/phase1_heatmap_g0.1.csv` | 9,419 | new |
| `results/phase1/phase1_heatmap_g0.5.csv` | 9,529 | new |
| `results/phase1/phase1_horizons.csv` | 7,107 | new |
| `results/phase1/phase1_records.csv` | 91,484,950 | ignored (raw) |
| `results/phase1/phase1_regime_best.csv` | 13,844 | modified |
| `results/phase1/phase1_unranked.csv` | 3,856 | new |
| `results/phase2/figure_p2_01_phase_diagram.png` | 298,816 | modified |
| `results/phase2/figure_p2_02_crossover.png` | 308,750 | modified |
| `results/phase2/figure_p2_03_correlations.png` | 137,839 | modified |
| `results/phase2/figure_p2_04_global_correlations.png` | 56,720 | modified |
| `results/phase2/figure_p2_05_rules.png` | 133,705 | modified |
| `results/phase2/phase2_capped.csv` | 81,969 | new |
| `results/phase2/phase2_correlations.csv` | 5,580 | modified |
| `results/phase2/phase2_correlations_g0.02.csv` | 5,529 | new |
| `results/phase2/phase2_correlations_g0.1.csv` | 5,580 | new |
| `results/phase2/phase2_correlations_g0.5.csv` | 5,839 | new |
| `results/phase2/phase2_features.csv` | 3,907,038 | modified |
| `results/phase2/phase2_phase_diagram_g0.02.csv` | 180,242 | new |
| `results/phase2/phase2_phase_diagram_g0.1.csv` | 178,494 | new |
| `results/phase2/phase2_phase_diagram_g0.5.csv` | 175,847 | new |
| `results/phase2/phase2_rules.csv` | 1,453 | modified |
| `results/phase2/phase2_rules_g0.02.csv` | 1,467 | new |
| `results/phase2/phase2_rules_g0.1.csv` | 1,453 | new |
| `results/phase2/phase2_rules_g0.5.csv` | 1,451 | new |
| `results/phase2/phase2_sweep_aggregated.csv` | 5,883,584 | modified |
| `results/phase3/figure_p3_01_selector_comparison.png` | 99,091 | modified |
| `results/phase3/figure_p3_02_improvement_map.png` | 124,991 | modified |
| `results/phase3/figure_p3_03_classifier_accuracy.png` | 110,347 | modified |
| `results/phase3/figure_p3_04_obs_depth.png` | 133,667 | modified |
| `results/phase3/figure_p3_05_cv_summary.png` | 52,482 | modified |
| `results/phase3/phase3_capped_cells.csv` | 219,934 | new |
| `results/phase3/phase3_cv_results.csv` | 3,536 | modified |
| `results/phase3/phase3_regime_classifier.csv` | 734 | modified |
| `results/phase3/phase3_regime_results.csv` | 15,047 | modified |
| `results/phase3/phase3_selector_comparison.csv` | 774 | modified |
| `results/phase4/figure_p4_01_correlations.png` | 59,472 | modified |
| `results/phase4/figure_p4_02_rejection_rules.png` | 181,100 | modified |
| `results/phase4/figure_p4_03_obs_reliability.png` | 131,615 | modified |
| `results/phase4/figure_p4_04_iqr_vs_error.png` | 840,820 | modified |
| `results/phase4/figure_p4_05_regime_diagnostic.png` | 115,762 | modified |
| `results/phase4/phase4_capped.csv` | 39,665 | new |
| `results/phase4/phase4_cascade_filter.csv` | 389 | modified |
| `results/phase4/phase4_diagnostic_correlations.csv` | 824 | modified |
| `results/phase4/phase4_diagnostic_correlations_holdout.csv` | 804 | new |
| `results/phase4/phase4_ensemble.csv` | 509 | modified |
| `results/phase4/phase4_obs_reliability.csv` | 3,540 | modified |
| `results/phase4/phase4_raw.csv` | 53,981,828 | ignored (raw) |
| `results/phase4/phase4_rejection_rules.csv` | 7,444 | modified |
| `results/phase5a/figure_p5a_01_comparison.png` | 203,577 | modified |
| `results/phase5a/figure_p5a_02_by_sigma.png` | 86,371 | modified |
| `results/phase5a/figure_p5a_03_method_weights.png` | 127,412 | modified |
| `results/phase5a/figure_p5a_04_obs_gap.png` | 93,561 | modified |
| `results/phase5a/phase5a_ablation.csv` | 1,131 | modified |
| `results/phase5a/phase5a_by_sigma.csv` | 11,843 | modified |
| `results/phase5a/phase5a_capped.csv` | 167,548 | new |
| `results/phase5a/phase5a_ensemble.csv` | 4,001 | modified |
| `results/phase5a/phase5a_ensemble_holdout.csv` | 4,196 | new |
| `results/phase5a/phase5a_method_weights.csv` | 5,050 | modified |
| `results/phase5a/phase5a_oracle_gap.csv` | 2,869 | modified |
| `results/phase5a/phase5a_per_regime.csv` | 103,169 | modified |
| `results/phase5a/phase5a_raw.csv` | 263,824,582 | ignored (raw) |
| `results/phase5b/figure_p5b_01_linf.png` | 93,926 | modified |
| `results/phase5b/figure_p5b_02_window.png` | 189,917 | modified |
| `results/phase5b/figure_p5b_03_catmult.png` | 98,367 | modified |
| `results/phase5b/phase5b_sweep1_global.csv` | 3,482 | modified |
| `results/phase5b/phase5b_sweep1_regime.csv` | 49,757 | modified |
| `results/phase5b/phase5b_sweep2_global.csv` | 3,304 | modified |
| `results/phase5b/phase5b_sweep2_regime.csv` | 48,294 | modified |
| `results/phase5b/phase5b_sweep3_champions.csv` | 6,504 | modified |
| `results/phase5b/phase5b_sweep3_concordance.csv` | 284 | modified |
| `results/phase5b/phase5b_sweep3_global.csv` | 26,486 | modified |
| `results/phase5b/phase5b_sweep3_unranked.csv` | 2,695 | new |
| `results/real_data/figure_rd_01_comparison.png` | 79,464 | unchanged |
| `results/real_data/figure_rd_02_regimes.png` | 85,785 | unchanged |
| `results/real_data/figure_rd_03_improvement.png` | 65,569 | unchanged |
| `results/real_data/figure_rd_v2_01_skill.png` | 123,184 | new |
| `results/real_data/real_data_curves.csv` | 60,357 | unchanged |
| `results/real_data/real_data_results.csv` | 4,595 | unchanged |
| `results/real_data/real_data_results_v2.csv` | 127,565 | new |
| `results/real_data/real_data_summary_v2.csv` | 38,567 | new |
| **total on disk** | **428,558,314** | 95 files |
| **total committed (excl. ignored raw)** | **19,266,954** | 92 files |

Summary: 95 files on disk (428.6 MB), of which 92 are committed (19.3 MB) and 3 are the git-ignored raw per-record files (`phase1_records.csv` 91.5 MB, `phase4_raw.csv` 54.0 MB, `phase5a_raw.csv` 263.8 MB; regenerated on every run). 27 new, 57 modified, 8 unchanged.

**Deleted in `30de724`** (the stale v1 files of Report 2, deviation 11; nothing in v2 writes them): `results/phase1/phase1_heatmap_n150.csv`, `phase1_heatmap_n1000.csv`, `phase1_heatmap_n5000.csv`, `phase1_raw_sample.csv`, `results/phase2/phase2_phase_diagram_n5000.csv`, `results/real_data/real_diagnostics.csv` (6 files, 363,101 bytes).

**Kept on purpose** (the legacy 18-cell real-data run): `results/real_data/real_data_results.csv` (18 cells), `real_data_curves.csv` (the recorded curves, also the v2 input), `figure_rd_01_comparison.png`, `figure_rd_02_regimes.png`, `figure_rd_03_improvement.png`. The v2 real-data outputs sit beside them (`real_data_results_v2.csv` 630 rows, `real_data_summary_v2.csv` 90 rows, `figure_rd_v2_01_skill.png`).

Also committed outside `results/`: `REPORT_3B_full_run.txt` (the console log) and, in the follow-up commit, this report.

## 6. Headline sanity lines from the log (orientation only, not analysed here)

* Phase 1, g = 0.1 core: 8 methods below the validity floor (the dangerous 8); `richardson_1` in the rank pool. Phase 2: rank pool 9, `richardson_1` never below the floor (0 unranked cells at g = 0.1); `constant_assumed` S = 0.909 and `constant_oracle` S = 1.396 reported alongside.
* Phase 5a, global mean error at g = 0.1 (core, capped excluded): `oracle_51` 0.00715, `constant_oracle` 0.01349, `oracle_9` 0.01840, `fixed_rational` 0.05313, `phase2_cascade` 0.06762, `diag_ensemble_9` 0.07097, `equal_ensemble_9` 0.07528, `fixed_richardson` 0.08298.
* Real data: 630 rows, `perturb_iqr` finite on 90/90 cells (median 0.00301, range [0.00019, 0.05367]; routed `richardson_1` 54 cells median 0.00168, routed `rational_fit` 36 cells median 0.00573); provenance on every summary row: `phase2_features_rows = 23400`, `git_head = 0375f24`; `nearest_regime` mapped for all 90 cells (none `unknown`).

## 7. Pushed commit and `git ls-remote`

Results commit (results + deletions + run log), pushed with `git push -u origin redesign-v2` (the branch did not exist on the remote before; created):

```
30de724489248c054f57f94bf75b4b64af0a2092   Replace committed results with the redesign-v2 full run (Prompt 3B)
```

```
$ git ls-remote origin redesign-v2
30de724489248c054f57f94bf75b4b64af0a2092	refs/heads/redesign-v2
```

This report (`REPORT_3B.md`) is committed on top of `30de724` and pushed as a second commit; its hash and the post-push `ls-remote` are printed in the console after the report.

## 8. Anomalies and observations

1. **NaN-flood threshold crossed in Phase 5a (1.579 %) and Phase 5b sweep 3 (1.50 % / 1.33 %), ruled not a flood** (section 3). The substantive content: `richardson_3` fails to converge on windows of 30-60 points (valid 0.48 / 0.58 at obs 30 / 60) and is effectively dangerous at shallow depth, but the artifact is derived at obs 90 only, where it is 98 % valid. The Phase-5a ensembles therefore pool a member that is NaN on ~24 % of its rows at the two shallow depths. A per-depth derivation, or a minimum-window rule for the 7-parameter fit, would be a code change and was not made. The sigma = 0 cancellation NaNs in the Levin/Wynn/Brezinski families (2.8 % of non-dangerous evaluations at sigma = 0 in Phase 5a, 1.9 % in Phase 1) are the second component.
2. **Phase 1 sub-rates above 1 %** although the pooled rate is 0.892 %: core-only 1.111 % (held-out 0.234 %), sigma = 0 1.906 % (0.36 % / 0.41 % at 0.001 / 0.005); uncapped 0.941 %, capped 0.297 %; by stratum 0.88-0.91 %. Same mechanisms as above.
3. **The first-chunk projection under-estimated Phase 5a by 34 %** (1.80 h projected, 2.41 h actual). The first block to finish is by construction the cheapest of the seven started together (obs 90, sigma 0: 54 min), while the obs 30/60 blocks took 1.57-1.72 h (about 2 x the ms/cell of the obs 90/120 blocks). The projection's "equal-cost chunks assumed" caveat covered this, and the decision rule was not close to its 16-h limit either way. With 12 blocks on 7 workers the second round ran 5 blocks on 7 workers (3A open question 1); finer chunking would have brought the phase to roughly 1.6 h.
4. **The artifact's `source` field is an absolute local path** (`C:\Users\kianu\...\code\results\phase1\phase1_aggregated.csv`, from `os.path.abspath` in `scripts/derive_dangerous.py:50`); the 3A worktree artifact had the same form. Harmless, but it is now in the committed JSON.
5. **`results/README.md` is out of date on tracking**: it lists `phase2/phase2_features.csv` and `phase2/phase2_sweep_aggregated.csv` among the files "deliberately excluded from version control", but both have been tracked since `900bb0e` and were regenerated and committed here (3.9 MB and 5.9 MB). Not edited (documentation, outside the prompt's scope).
6. **Dangling references to the deleted files**, both benign: `scripts/make_paper_tables.py:150` reads `phase1_heatmap_n{h}.csv` in the v1 code below the v2 refusal stub (exit 2 on v2 results, Report 2); `scripts/analyze_real_diagnostics_legacy.py:96` *writes* `real_diagnostics.csv`, so running the legacy script would recreate the deleted file (it is not part of `reproduce_all.py`).
7. **Phase 0 output is byte-identical** to the committed pre-redesign files (`phase0_unit_tests.csv/.txt` unchanged), as expected since no accelerator implementation changed.
8. **Figure P4-3** (`figure_p4_03_obs_reliability.png`), which the quick runs could not produce (Report 2, deviation 14), was produced at full scale.
9. **Run conditions.** The machine was otherwise idle (the 3A Phase-5a probe had been stopped; no other Python process). The Report-3A worktree at `../wt-report3a` (detached at `8d33e82`) was left untouched.
10. `git` warned that `REPORT_3B_full_run.txt` has LF endings that autocrlf will normalise on the next touch; cosmetic.

Stopped here.
