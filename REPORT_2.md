**Report back**

# Redesign v2, Prompt 2: the redesign wired through the experiment pipeline

Date: 2026-09-19. Branch `redesign-v2` in `code/`. Code commit `b97e691` on top of `672de59` (Report 1). This report and the quick-run log are committed separately on top of `b97e691`. No method implementation was changed. The full pipeline was not run.

## 1. Files changed

Created:

| file | purpose |
|---|---|
| `src/pipeline.py` | Shared helpers used by every phase: `resolve_regimes` (core / held-out), `horizon_meta` (target_g, achieved_g, n_f, capped per cell), `exclude_capped` (the pooled-ranking rule), `capped_block` (the separate capped report), skill helpers, `ACCEL_METHODS` (the 51 accelerators). |
| `src/dangerous.py` | `derive_dangerous` (S < 0 on core regimes, pooled over strata and noise, capped and oracle excluded), `write_artifact` / `load_artifact` / `load_dangerous` for `results/phase1/dangerous_methods.json`. A missing artifact raises with instructions. |
| `scripts/derive_dangerous.py` | The pipeline step between Phase 1 and Phases 2-5: reads `phase1_aggregated.csv`, writes the artifact, prints the scored table and the diff against the legacy set. |
| `tests/test_pipeline_v2.py` | Config-v2 decisions, regime resolution, dangerous derivation and artifact, Phase-1 capped exclusion and median-error ranking, Phase-2 sweep schema, recorded-curve re-evaluation, Phase-0 exclusion. |
| `REPORT_2_quick_run.txt` | The complete console output of `python reproduce_all.py --quick` (progress ticks removed, 1,095 lines). Stored as `.txt` because `.gitignore` excludes `*.log`. |
| `REPORT_2.md` | this file |

Modified:

| file | change |
|---|---|
| `src/config.py` | Config v2: per-phase `full` / `quick` grids (`PHASE1`, `PHASE2`, `PHASE4`, `PHASE5A`, `PHASE5B`, `REAL_DATA`), `HEADLINE_G = 0.1`, `PHASE5B_GAP_FRACTIONS = [0.5, 0.1]`, `EXCLUDE_CAPPED_FROM_POOLED`, `RANK_METRIC = "med_error"`, quick = 2 seeds / 2 regimes per group / 2 noise levels. `DANGEROUS_METHODS` removed; `LEGACY_DANGEROUS_METHODS` kept, labelled, unconsumed; `DANGEROUS_ARTIFACT` added. |
| `src/evaluation.py` | Phase 1: schema columns `is_holdout`, `is_trivial`, `is_oracle`; pooled tables exclude capped cells, sort by median error, rank non-oracle methods with deterministic tie-break; `phase1_capped.csv` block; `regime_best` led by `best_by_skill` with `best_by_stability` secondary; `core_regimes` / `holdout_regimes` parameters. |
| `src/plots.py` | Figure 5 plots the best-by-skill method and its skill; capped cells marked. |
| `src/trajectories.py` | `evaluate_recorded_curves` for the real-data re-evaluation grid (long + summary frames, skill vs best-of-four). |
| `phases/phase2.py` | Pool = 9 methods + `constant_assumed` + `constant_oracle`; targets are per-depth gap-stratified horizons for all three strata; records and aggregates keyed by `target_g` with `n_f`, `achieved_g`, `capped`, `skill`, `L_true`, `L_hat`, flags; rank pool of 10 (oracle excluded); capped cells excluded from the pooled phase-diagram mean, correlations and rules; per-stratum output files; `phase2_capped.csv`. |
| `phases/phase3.py` | Grid keyed by `target_g`; candidates = the 10-method rank pool; capped cells excluded from selector means and CV and listed in `phase3_capped_cells.csv`; enhanced cascade default keys on the cell's `n_f`; CV folds adapt to class counts. |
| `phases/phase4.py` | Three gap strata per (regime, depth); core + held-out regimes evaluated with `is_holdout`; four trivial references evaluated for skill (no diagnostics); pooled analyses core-only with capped excluded; held-out correlations and the capped block written separately; ensemble table carries trivial references and `med_skill`. |
| `phases/phase5a.py` | Same strata / regime treatment; ensemble and oracle pool = the 51 accelerators; the five trivial comparators are fixed reference selectors (oracle labelled) and never pooled; dangerous set from the artifact (refuses to run without it); `med_skill` per selector; held-out and capped tables. |
| `phases/phase5b.py` | Sweeps at g in {0.5, 0.1}; sweep 1 over the five assumed-asymptote modes; pooled rows for core (capped excluded) and held-out separately; sweep 3 ranks 55 methods (51 + 4 non-oracle trivials); dangerous flag from the artifact. |
| `scripts/run_phase1.py` … `run_phase5b.py` | Read the config-v2 grids, expose `n_evaluations`, print v2 summaries (median-error tables, trivial comparators, capped block, best-by-skill). `run_phase3.py` accepts `--quick` / `--full` as labels. |
| `scripts/run_real_data.py` | Default = re-evaluate the recorded curves (no download, no training) on depths {30, 60, 90, 120, 150} x targets {300, 400, 500}; writes `real_data_results_v2.csv`, `real_data_summary_v2.csv`, `figure_rd_v2_01_skill.png`; the 18-cell `real_data_results.csv` and its figures are untouched; the old download-and-train path is kept behind `--retrain`. |
| `scripts/check_dangerous.py` | Compares the artifact with a fresh derivation from the current Phase-1 table. |
| `scripts/make_paper_figures.py` | Reads the dangerous set from the artifact, falling back to the legacy set with a note. |
| `scripts/make_paper_tables.py` | `OUTPUT_CONTRACT_V2` stub (31 fragment names with source files and descriptions), `--contract` flag; refuses v2 results with exit 2 until Prompt 4 implements the fragments. |
| `reproduce_all.py` | v2 orderer: Phase 0 → Phase 1 → dangerous re-derivation → Phases 2, 3, 4, 5a, 5b → real-data re-evaluation; `--quick`; `--plan` prints evaluation counts; per-step status and timing table; stops if Phase 1 or the derivation fails. |
| `tests/test_accelerators.py` | Phase-0 analytic harness runs the 51 accelerators only (`PHASE0_METHODS`); one-line note; `L_true` removed from its config. |
| `tests/test_redesign.py` | Column renames (`is_holdout`, `is_trivial`), median-error ranking, `best_by_skill`. |

## 2. Config v2 summary

```
ASYMPTOTE_MODE hetero | L_TRUE_RANGE (0.005, 0.5) | ASSUMED_L_MODE zero
HORIZON_GAP_FRACTIONS [0.5, 0.1, 0.02] | HEADLINE_G 0.1 | PHASE5B_GAP_FRACTIONS [0.5, 0.1] | HORIZON_N_CAP 50000
EXCLUDE_CAPPED_FROM_POOLED True | RANK_METRIC med_error
DANGEROUS_ARTIFACT results/phase1/dangerous_methods.json | LEGACY_DANGEROUS_METHODS 8 (unconsumed)
QUICK: seeds 2 | core ['single_exp', 'log_slow'] | holdout ['stretched_exp', 'random_knots'] | noise [0.0, 0.005]

PHASE1[full]:  n_seeds 30, noise [0.0, 0.001, 0.005], strata [0.5, 0.1, 0.02], obs_idx 90, window 60, 18 + 6 regimes
PHASE1[quick]: n_seeds 2,  noise [0.0, 0.005],        strata [0.5, 0.1, 0.02], obs_idx 90, window 60, 2 + 2 regimes
PHASE2[full]:  depths [20..140 step 10] (13), noise [0.0, 0.003, 0.005, 0.01, 0.02], strata x3, n_seeds 20, core 18 only
PHASE2[quick]: depths (13), noise [0.0, 0.005], strata x3, n_seeds 2, core 2 only
PHASE4[full]:  depths [30, 60, 90, 120], noise [0.0, 0.005, 0.02], strata x3, n_seeds 20, shifts [-2..2], perturb 5, 18 + 6
PHASE4[quick]: depths [60, 90], noise [0.0, 0.005], strata x3, n_seeds 2, shifts [-1, 0, 1], perturb 3, 2 + 2
PHASE5A[full]: depths [30, 60, 90, 120], noise [0.0, 0.005, 0.02], strata x3, n_seeds 20, perturb 5, 18 + 6
PHASE5A[quick]: depths [60, 90], noise [0.0, 0.005], strata x3, n_seeds 2, perturb 3, 2 + 2
PHASE5B[full]: modes [zero, half, oracle, double, winmin], windows [20, 40, 60, 80, 100], CAT_MULT [2, 5, 10],
               obs_idx 90, noise [0.0, 0.005, 0.02], strata [0.5, 0.1], n_seeds 20, 18 + 6
PHASE5B[quick]: modes (all 5), windows [20, 60, 100], CAT_MULT [2, 10], noise [0.0, 0.005], strata [0.5, 0.1], n_seeds 2, 2 + 2
REAL_DATA:     recorded curves results/real_data/real_data_curves.csv, depths [30, 60, 90, 120, 150],
               targets [300, 400, 500] (depth < target), window 60, mode zero
```

Held-out regimes are evaluation-only in every phase; Phase 2 (selector / cascade training data) and Phase 3 (its consumer) use the core regimes only. Quick mode picks one fast core regime and one that hits the horizon cap, plus one closed-form held-out regime and the per-seed `random_knots` family, so the capped and seed-dependent paths are exercised end to end.

## 3. The `--quick` run

Executed as `python reproduce_all.py --quick` in an isolated git worktree of commit `b97e691` (so the committed `results/` in the main tree stayed untouched). Exit code 0, zero warnings, 177 s wall time. The complete console output is in `REPORT_2_quick_run.txt`; below is the transcript with the progress ticks and per-phase detail tables removed. The dangerous re-derivation runs as step 3 of 9, after Phase 1 and before Phase 2, and its full output is kept.

```
========================================================================
  [1/9]  Phase 0 — analytic unit tests
  $ scripts/run_phase0_tests.py
========================================================================
========================================================================
  PHASE 0 UNIT TEST REPORT
  Sequence Acceleration — Validation on Known Analytic Sequences
========================================================================
  Completed in 2s
========================================================================
  [2/9]  Phase 1 — main benchmark
  $ scripts/run_phase1.py --quick
========================================================================
========================================================================
  PHASE 1 — Main Synthetic Benchmark  [QUICK mode, redesign v2]
========================================================================
  ... (progress and horizon table omitted; see REPORT_2_quick_run.txt) ...
  TOP-20 GLOBAL BY MEDIAN ERROR  (core regimes, g = 0.1, capped cells excluded, L_hat mode = zero)
  ────────────────────────────────────────────────────────────────────────────────────────────
  Method                   Type            MedErr   Skill  Valid    Cat    Stab  Cells  Rank
  ────────────────────────────────────────────────────────────────────────────────────────────
  double_exp_fit           trajectory     0.00083   0.055  1.000  0.000   1.400      2     1
  single_exp_fit           trajectory     0.00083   0.055  1.000  0.000   1.400      2     2
  neville_2                trajectory     0.00154   0.120  0.500  0.500  -0.300      2     3
  constant_oracle          limit          0.00201   0.163  1.000  0.000   1.400      2     -
  richardson_a20           trajectory     0.00840   0.681  1.000  0.000   1.400      2     4
  shanks_3                 limit          0.00895   0.703  1.000  0.000   1.300      2     5
  median_ensemble          trajectory     0.00954   0.888  1.000  0.000   1.300      2     6
  stability_weighted       trajectory     0.01008   0.885  1.000  0.000   1.300      2     7
  geom_avg_diff            limit          0.01095   0.729  0.750  0.250   0.450      2     8
  levin_v2                 limit          0.01113   0.954  1.000  0.000   1.200      2     9
  levin_v1                 limit          0.01118   0.961  1.000  0.000   1.200      2    10
  log_linear               trajectory     0.01155   0.734  1.000  0.000   1.200      2    11
  shanks_4                 limit          0.01223   0.989  1.000  0.000   1.200      2    12
  shanks_2                 limit          0.01228   1.075  1.000  0.000   1.200      2    13
  levin_u2                 limit          0.01287   1.091  1.000  0.000   1.200      2    14
  levin_t2                 limit          0.01288   1.091  1.000  0.000   1.200      2    15
  brezinski_theta1         limit          0.01321   1.109  1.000  0.000   1.200      2    16
  wynn_eps_1               limit          0.01424   1.203  1.000  0.000   1.200      2    17
  levin_t1                 limit          0.01424   1.203  1.000  0.000   1.200      2    18
  best_shanks_wynn         limit          0.01424   1.203  1.000  0.000   1.200      2    19

  TRIVIAL COMPARATORS  (g = 0.1, core, capped excluded)
  ────────────────────────────────────────────────────────────
  constant_oracle          med_err=0.00201  skill=0.163  stab=1.400 (oracle, unranked)
  window_min               med_err=0.01589  skill=1.262  stab=1.100
  last_value               med_err=0.01599  skill=1.269  stab=1.000
  constant_assumed         med_err=0.03962  skill=2.593  stab=1.200
  window_mean              med_err=0.08180  skill=6.620  stab=0.000

  CAPPED CELLS  (excluded from pooled tables; 4 regime x g cells)
  ────────────────────────────────────────────────────────────
  log_slow           g=0.1   n_f= 50000  achieved g = 0.419
  log_slow           g=0.02  n_f= 50000  achieved g = 0.419
  random_knots       g=0.1   n_f= 28361  achieved g = 0.113
  random_knots       g=0.02  n_f= 50000  achieved g = 0.079

  Completed in 18s
========================================================================
  [3/9]  Dangerous re-derivation (Phase 1 -> artifact)
  $ scripts/derive_dangerous.py
========================================================================
========================================================================
  DANGEROUS-METHOD RE-DERIVATION  (Phase 1 -> artifact for Phases 2-5)
========================================================================
  source     : results\phase1\phase1_aggregated.csv  (1344 aggregated rows)
  criterion  : S < 0, core regimes, pooled over strata [0.5, 0.1, 0.02], capped cells excluded, oracle excluded
  methods    : 55 scored
  dangerous  : 8  -> ['linear', 'neville_2', 'neville_3', 'neville_4', 'pade_21', 'pade_31', 'wynn_eps_2', 'wynn_eps_3']
  legacy set : 8 (not consumed; for the record)
  vs legacy  : +['wynn_eps_2', 'wynn_eps_3']  -['geom_avg_diff', 'pade_32']
  artifact   : <repo>/results/phase1/dangerous_methods.json

  method                        S   valid     cat   beats  cells  flag
  ----------------------------------------------------------------------
  linear                  -1.2500   0.750   1.000   0.000      8  DANGEROUS
  pade_21                 -0.4500   0.750   0.625   0.125      8  DANGEROUS
  pade_31                 -0.2750   0.875   0.625   0.250      8  DANGEROUS
  neville_2               -0.2375   0.562   0.500   0.500      8  DANGEROUS
  neville_3               -0.1125   0.688   0.500   0.500      8  DANGEROUS
  neville_4               -0.1125   0.688   0.500   0.500      8  DANGEROUS
  wynn_eps_3              -0.0750   0.625   0.375   0.125      8  DANGEROUS
  wynn_eps_2              -0.0750   0.625   0.375   0.125      8  DANGEROUS
  pade_32                  0.0250   0.875   0.500   0.375      8
  window_mean              0.1250   1.000   0.438   0.000      8
  log_fit                  0.3500   1.000   0.375   0.250      8
  geom_avg_diff            0.4750   0.750   0.250   0.562      8
  brezinski_theta2         0.8250   1.000   0.188   0.500      8
  pade_23                  0.8500   1.000   0.188   0.562      8
  weniger_d1               0.8500   1.000   0.125   0.250      8
  richardson_a05           0.8500   1.000   0.125   0.250      8
  weniger_d2               0.8500   1.000   0.125   0.250      8
  constant_assumed         0.8500   1.000   0.125   0.250      8
  wynn_rho_1               0.9250   1.000   0.062   0.125      8
  wynn_rho_3               0.9500   1.000   0.062   0.188      8
  pade_22                  0.9500   1.000   0.125   0.500      8
  current_value            1.0000   1.000   0.000   0.000      8
  last_value               1.0000   1.000   0.000   0.000      8
  levin_u1                 1.0250   1.000   0.062   0.375      8
  pade_12                  1.0500   1.000   0.062   0.438      8
  pade_13                  1.0500   1.000   0.062   0.438      8
  richardson_3             1.0750   1.000   0.062   0.500      8
  shanks_1                 1.0750   1.000   0.062   0.500      8
  levin_t1                 1.0750   1.000   0.062   0.500      8
  anderson_1               1.0750   1.000   0.062   0.500      8
  wynn_eps_1               1.0750   1.000   0.062   0.500      8
  best_shanks_wynn         1.0750   1.000   0.062   0.500      8
  shanks_2                 1.1000   1.000   0.062   0.562      8
  richardson_1             1.1000   1.000   0.062   0.562      8
  richardson_a10           1.1000   1.000   0.000   0.250      8
  rational_fit             1.1000   1.000   0.062   0.562      8
  richardson_2             1.1000   1.000   0.062   0.562      8
  window_min               1.1000   1.000   0.000   0.250      8
  pade_11                  1.1000   1.000   0.000   0.250      8
  wynn_rho_2               1.1500   1.000   0.000   0.375      8
  brezinski_theta1         1.1500   1.000   0.000   0.375      8
  anderson_2               1.2000   1.000   0.000   0.500      8
  anderson_3               1.2000   1.000   0.000   0.500      8
  levin_v1                 1.2000   1.000   0.000   0.500      8
  levin_u2                 1.2000   1.000   0.000   0.500      8
  shanks_4                 1.2000   1.000   0.000   0.500      8
  levin_v2                 1.2000   1.000   0.000   0.500      8
  levin_t2                 1.2000   1.000   0.000   0.500      8
  log_linear               1.2500   1.000   0.000   0.625      8
  median_ensemble          1.2750   1.000   0.000   0.688      8
  stability_weighted       1.3000   1.000   0.000   0.750      8
  shanks_3                 1.3000   1.000   0.000   0.750      8
  richardson_a20           1.3250   1.000   0.000   0.812      8
  single_exp_fit           1.4000   1.000   0.000   1.000      8
  double_exp_fit           1.4000   1.000   0.000   1.000      8
========================================================================

  Completed in 1s
========================================================================
  [4/9]  Phase 2 — failure detection
  $ scripts/run_phase2.py --quick
========================================================================
========================================================================
  PHASE 2 — Richardson Failure Condition Mapping  [QUICK, redesign v2]
========================================================================
  Completed in 9s
========================================================================
  [5/9]  Phase 3 — adaptive selection
  $ scripts/run_phase3.py --quick
========================================================================
========================================================================
  PHASE 3 — Adaptive Selector Pipeline  [QUICK, redesign v2]
========================================================================
  Completed in 4s
========================================================================
  [6/9]  Phase 4 — perturbation diagnostics
  $ scripts/run_phase4.py --quick
========================================================================
========================================================================
  PHASE 4 — Stability Diagnostic Stress Testing  [QUICK, redesign v2]
========================================================================
  Completed in 9s
========================================================================
  [7/9]  Phase 5a — ensemble ablation
  $ scripts/run_phase5a.py --quick
========================================================================
========================================================================
  PHASE 5A — Full-Pool Ensemble  [QUICK, redesign v2]
========================================================================
  Completed in 109s
========================================================================
  [8/9]  Phase 5b — sensitivity sweeps
  $ scripts/run_phase5b.py --quick
========================================================================
========================================================================
  PHASE 5B — Sensitivity Analysis  [QUICK, redesign v2]
========================================================================
  Completed in 23s
========================================================================
  [9/9]  Real data — recorded-curve re-evaluation
  $ scripts/run_real_data.py --quick
========================================================================
========================================================================
  REAL DATA — RE-EVALUATION OF RECORDED CURVES  (redesign v2)
========================================================================
  REAL-DATA RE-EVALUATION SUMMARY  (recorded curves, no retraining)
================================================================================

  Datasets   : ['adult', 'bank_marketing', 'covertype', 'higgs', 'jannis', 'miniboone']
  Cells      : 90  (depth < target)
  L_hat mode : zero

  Per (depth, target): mean errors and median cascade skill
   depth  target    cascade  richardson   rational    current  best-triv  med skill  beats triv
  ────────────────────────────────────────────────────────────────────────────────────────────
      30     300    0.03444     0.02530    0.02842    0.11247    0.11247      0.305        1.00
      30     400    0.03891     0.03206    0.03512    0.11914    0.11914      0.368        1.00
      30     500    0.04263     0.03778    0.04061    0.12358    0.12358      0.381        1.00
      60     300    0.02503     0.02653    0.01854    0.05657    0.05657      0.496        0.67
      60     400    0.03397     0.03639    0.02473    0.06324    0.06324      0.607        0.67
      60     500    0.04137     0.04457    0.02990    0.06767    0.06767      0.728        0.50
      90     300    0.00516     0.00575    0.00645    0.03489    0.03489      0.209        1.00
      90     400    0.00988     0.01068    0.01161    0.04156    0.04156      0.281        1.00
      90     500    0.01406     0.01501    0.01615    0.04600    0.04600      0.353        1.00
     120     300    0.00695     0.00780    0.00785    0.02308    0.02308      0.262        0.83
     120     400    0.01199     0.01333    0.01331    0.02975    0.02975      0.351        0.67
     120     500    0.01638     0.01814    0.01803    0.03418    0.03418      0.414        0.67
     150     300    0.00432     0.00388    0.00480    0.01609    0.01608      0.274        0.83
     150     400    0.00855     0.00808    0.00945    0.02276    0.02275      0.356        0.67
     150     500    0.01233     0.01173    0.01363    0.02725    0.02725      0.419        0.67

  Method median skill over all cells (skill < 1 beats the best trivial reference):
    cascade            med skill = 0.336   finite = 90/90
    constant_assumed   med skill = 10.650   finite = 90/90
    last_value         med skill = 1.000   finite = 90/90
    rational_fit       med skill = 0.355   finite = 90/90
    richardson_1       med skill = 0.345   finite = 90/90
    window_mean        med skill = 1.876   finite = 90/90
    window_min         med skill = 1.000   finite = 90/90

  Cascade method selection: {'richardson_1': np.int64(54), 'rational_fit': np.int64(36)}
  Best trivial reference:   {'last_value': np.int64(88), 'window_min': np.int64(2)}

========================================================================
  Completed in 3s
========================================================================
  PIPELINE SUMMARY  [QUICK]
========================================================================
   #  step                                             status       time
  ----------------------------------------------------------------------
   1  Phase 0 — analytic unit tests                    OK             2s
   2  Phase 1 — main benchmark                         OK            18s
   3  Dangerous re-derivation (Phase 1 -> artifact)    OK             1s
   4  Phase 2 — failure detection                      OK             9s
   5  Phase 3 — adaptive selection                     OK             4s
   6  Phase 4 — perturbation diagnostics               OK             9s
   7  Phase 5a — ensemble ablation                     OK           109s
   8  Phase 5b — sensitivity sweeps                    OK            23s
   9  Real data — recorded-curve re-evaluation         OK             3s
  ----------------------------------------------------------------------
      total                                                         177s
  All steps completed successfully.  Results are in results/
========================================================================

exit=0
```

Files produced per phase in the quick run: phase1 21 (incl. `dangerous_methods.json`, `phase1_capped.csv`, `phase1_global_holdout.csv`), phase2 20 (per-stratum diagrams, correlations, rules, `phase2_capped.csv`), phase3 10, phase4 13 (incl. `_holdout` correlations and `phase4_capped.csv`), phase5a 13 (incl. `phase5a_ensemble_holdout.csv`, `phase5a_capped.csv`), phase5b 10, real_data 9 (3 new v2 files, the legacy 18-cell files preserved). The dangerous artifact records schema `dangerous_methods/v2`, git head `b97e691`, mode `zero`, strata `[0.5, 0.1, 0.02]`.

Quick-run numbers are 2 seeds on 4 regimes and are not results; they only show that every phase runs and that every schema column is populated.

## 4. Sample rows with the required columns populated

One Phase-1 aggregate row (`phase1_aggregated.csv`; `rational_fit`, `single_exp`, g = 0.1, sigma = 0.005):

```
method rational_fit | family parametric | method_type trajectory | is_trivial 0 | is_oracle 0
regime single_exp | is_holdout 0 | noise 0.005 | target_g 0.1 | n_f 148 | achieved_g 0.09827 | capped 0
L_true 0.037602 | L_hat 0.0
valid_rate 1.0 | cat_rate 0.0 | beats_rate 0.5 | med_error 0.018719 | med_improve 1.0776
med_skill 1.7449 | stability 1.2 | n_seeds 2
```

A second Phase-1 aggregate row on a capped cell (`constant_assumed`, `log_slow`, g = 0.1, sigma = 0), showing the flag that removes it from pooled tables while it stays in the capped block:

```
method constant_assumed | is_trivial 1 | is_oracle 0 | regime log_slow | is_holdout 0 | noise 0.0
target_g 0.1 | n_f 50000 | achieved_g 0.41864 | capped 1 | L_true 0.071912 | L_hat 0.0
valid_rate 1.0 | cat_rate 0.0 | beats_rate 0.5 | med_error 0.145851 | med_skill 1.5192 | stability 1.2
```

One Phase-4 raw row (`phase4_raw.csv`; `richardson_1`, held-out `random_knots`, obs 90, seed 1, g = 0.1, sigma = 0.005). This seed's shape caps at g = 0.1, which is the per-seed behaviour the flag exists for:

```
regime random_knots | is_holdout 1 | obs_idx 90 | noise 0.005 | seed 1 | L_true 0.005571 | L_hat 0.0
method richardson_1 | true_val 0.017140 | estimate 0.018843 | error 0.001703 | valid 1 | catastrophic 0
curr_err 0.084094 | ref_error 0.017140 | skill 0.09935 | shift_iqr 0.010414 | perturb_iqr 0.008098
target_g 0.1 | achieved_g 0.12601 | n_f 50000 | capped 1 | is_trivial 0 | is_oracle 0
```

One Phase-5A raw row (`phase5a_raw.csv`; `single_exp_fit`, held-out `stretched_exp`, obs 60, seed 0, g = 0.02, sigma = 0):

```
regime stretched_exp | is_holdout 1 | obs_idx 60 | noise 0.0 | seed 0 | L_true 0.022268 | L_hat 0.0
method single_exp_fit | true_val 0.026377 | estimate 0.215414 | error 0.189037 | valid 1 | catastrophic 0
curr_err 0.201575 | ref_error 0.026377 | skill 7.1668 | perturb_iqr 0.003050 | is_dangerous 0
casc_slope -0.28688 | casc_r2 0.92335 | target_g 0.02 | achieved_g 0.019974 | n_f 1056 | capped 0
is_trivial 0 | is_oracle 0
```

## 5. Full-run evaluation counts (from `python reproduce_all.py --plan`)

| step | central method evaluations | detail |
|---|---|---|
| Phase 0 | 204 | 51 accelerators x 4 analytic cases (trivials excluded) |
| Phase 1 | 362,880 | 24 regimes x 30 seeds x 3 noise x 3 strata x 56 methods |
| Dangerous derivation | 0 | reads `phase1_aggregated.csv` |
| Phase 2 | 772,200 | 13 depths x 5 noise x 20 seeds x 18 core regimes x 3 strata x 11 methods, plus 140,400 skill-reference calls (2 per cell) |
| Phase 3 | 0 | analysis of Phase 2 output |
| Phase 4 | 224,640 | 4 depths x 3 noise x 20 seeds x 24 regimes x 3 strata x 13 methods, plus 1,555,200 diagnostic calls (9 methods x (5 shifts + 5 perturbations)) |
| Phase 5a | 967,680 | 4 depths x 3 noise x 20 seeds x 24 regimes x 3 strata x 56 methods, plus 4,406,400 perturbation calls (51 methods x 5) |
| Phase 5b | 532,800 | sweep 1 28,800 + sweep 2 28,800 + sweep 3 475,200 (24 regimes, 2 strata, 55 methods in sweep 3) |
| Real data | 630 | 6 datasets x 15 (depth, target) pairs x 7 methods |
| **Total central** | **2,861,034** | plus 6,102,000 diagnostic / reference calls, 8,963,034 method calls in all |

Quick mode plans 17,610 central evaluations (plus 19,872 diagnostic calls), which matches the 177 s run.

Runtime expectation for a full run, scaled from the old per-phase timings: Phase 1 about 30-60 min, Phase 2 about 50 min, Phase 4 about 1 h, Phase 5b about 30 min, and Phase 5a about 12-16 h (its call count is roughly four times the old design's 1.3 M). The full pipeline is therefore an overnight job dominated by Phase 5a. See open question 1.

## 6. Deviations from the prompt and decisions taken

1. **Ensemble and selector pools contain the 51 accelerators only.** The trivial comparators are evaluated everywhere and appear in every table as fixed reference selectors, but they never enter an ensemble, the Phase-2 "best other", the Phase-3 selector oracle or a rank. The constant oracle is additionally excluded from every rank, best-of and pool; it is shown, labelled, in tables. The Phase-2 rank pool is therefore 10 methods (9 + `constant_assumed`), not 9, and Phase 3's candidates are those 10.
2. **Skill in Phase 2** needs `window_mean` and `window_min` in the denominator even though they are not in the Phase-2 pool; they are evaluated per cell (two cheap calls) and not recorded as pool rows.
3. **Phase-3 enhanced cascade** kept its horizon-aware default but keys it on the cell's actual `n_f` (rational_fit when n_f <= 300) since fixed horizons no longer exist; the 300 threshold is unchanged.
4. **Held-out regimes in Phases 4, 5a, 5b** are evaluated with `is_holdout = 1`; pooled analyses (correlations, rejection rules, cascade filter, ensembles, reliability, sweep metrics, sweep-3 rankings) are core-only with capped cells excluded; held-out results go to separate files (`_holdout` correlations, `phase5a_ensemble_holdout.csv`, `regime_set = holdout` rows in 5b). Per-regime tables cover all regimes with the flag.
5. **Phase 4 diagnostics** (shift and perturbation IQR) are computed for the nine-method diagnostic pool only; the four trivial references are evaluated for skill with NaN diagnostics.
6. **Dangerous derivation scores the non-oracle trivial comparators too** (`window_mean` sits at S = 0.125 in the quick run). Since trivials never enter pools, a "dangerous" trivial would only be a label. Open question 2.
7. **Phase 2 runs the correlation and rule analyses for all three strata** (per-stratum files) and also writes the headline stratum under the legacy file names `phase2_correlations.csv` and `phase2_rules.csv` so downstream readers keep working. Figures are produced for the headline stratum.
8. **Quick mode uses two noise levels** [0.0, 0.005] in every phase (the prompt fixed seeds and regimes only) and the reduced shift / perturbation counts of the old quick modes in Phases 4 and 5a.
9. **Real-data grid.** All 15 (depth, target) pairs satisfy depth < target, so every dataset yields 15 cells (90 in total). The nearest-regime mapping uses whichever `phase2_features.csv` is present; in the quick run that file holds two regimes, so the mapping column is not meaningful there.
10. **The quick run was executed in a git worktree**, not in the main checkout, so the committed `results/` of the old design remain untouched pending the full run. Nothing from that worktree is committed except the console log.
11. **Stale v1 result files** that no v2 phase regenerates and that are still committed: `phase1_heatmap_n{150,1000,5000}.csv`, `phase1_raw_sample.csv`, `phase2_phase_diagram_n5000.csv`, and `real_diagnostics.csv` (from `analyze_real_diagnostics.py`, which still targets the stored legacy run). They should be deleted when the full-run results are committed. The legacy real-data files are preserved on purpose.
12. **`make_paper_tables.py`** refuses v2 results (exit 2) and prints the contract; the v1 code below the stub is untouched and still runs on v1 results. `analyze_real_diagnostics.py` is unchanged and still reproduces the stored legacy run.
13. **Median-error ranking has no validity floor.** In the quick run `neville_2` takes rank 3 at g = 0.1 with a valid rate of 0.5 and S = -0.3 (its median error is over the cells where it was valid). Open question 3.
14. **Phase-4 figure P4-3** was not produced in quick mode because `obs_reliability` needs at least five rows per (depth, method) cell; at full scale that condition holds. The old figure file therefore remained in the worktree.
15. **`reproduce_all.py` stops** after a failed Phase 1 or a failed derivation (later phases depend on them) and otherwise continues, reporting every failure in the summary table. `--plan` counts are computed from the same config grids the runners use.
16. **Schema names.** Phase-1 records renamed `holdout` to `is_holdout` to match the prompt's schema; all other phases use the same three flags.

Tests: `python -m pytest tests/ -q` gives `339 passed in 13.26s` (`test_generators` 189, `test_redesign` 138, `test_pipeline_v2` 8, `test_accelerators` 4).

## 7. Open questions for Kian / Claude

1. **Phase 5a cost.** The full grid is about 5.4 M method calls (12-16 h). Options: 3 perturbation trials instead of 5, depths {60, 90, 120}, or running it in parallel by depth. Which, if any, before the full run?
2. **Trivial comparators and the dangerous flag.** Keep scoring them (label only) or restrict the derivation to the 51 accelerators?
3. **Rank eligibility.** Should the median-error ranking require a minimum valid rate (for example 0.9), or report `med_error` over all cells with invalid counted as a failure? As it stands, a rarely-valid method can rank high.
4. **Phase-2 rank pool.** Is the 10-method pool (9 + `constant_assumed`) acceptable for the "Richardson rank" narrative, or should the rank be computed over the original 9 with `constant_assumed` reported alongside?
5. **Held-out treatment in Phases 4 and 5a.** Pooled statistics are core-only with held-out reported separately. Should any headline pooled number include the held-out regimes?
6. **Stale v1 files** (deviation 11): delete in Prompt 3 with the full-run commit, or now?
7. **`analyze_real_diagnostics.py`** still verifies the legacy 18-cell run at the legacy assumed asymptote 0.01. Port it to the v2 grid in Prompt 3, or retire it?
8. **Real-data regime mapping** depends on the Phase-2 feature file present at run time. Should the re-evaluation record which Phase-2 run produced the centroids (commit hash) for provenance?

Stopped here; nothing from Prompt 3 was started.
