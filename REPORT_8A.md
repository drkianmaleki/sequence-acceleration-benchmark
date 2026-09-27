**Report back**

Prompt R8a — evaluation core and phases (redesign v2, pre-rerun). Branch `redesign-v2`. All work committed locally, nothing pushed, the full pipeline not run. This report is left uncommitted for the author's review, as `REPORT_7.md` was before this prompt.

## 1. Commits

- `PRE_R8` = `0b7766efcf057efde20bb87df3984bb97c87cace` (Part A: `Add REPORT_7 (Prompt 7 report)` on top of 842ddb9).
- Final `HEAD` = `d6beb532bcc9dd3e7f674cba264284657633ed27`.

```
d6beb53 Add scripts/dev/compare_quick_runs.py (before/after regression check for code changes)
9deb68c README and docstrings: 52-accelerator roster, validity criterion, descriptive reporting, new real_boot sources
a2e7e21 Table generator follows the descriptive-panel schema
ad0245d real_boot_a/b rebuilt from OpenML electricity (151) and nomao (1486) under pre-specified inclusion criteria, outside the recorded-curve test set
ab04bfc Regime classifier: grouped-by-depth CV (primary), grouped-by-noise check, legacy stratified for comparison; explicit split unit
c6ba693 Phases 2, 3 and 5b: descriptive panels; Richardson error normalised by the last-value error; rules and cascade evaluated by what happened when they fired; no fallback for invalid selections
cfead26 Remove the stability score; exclusion by validity; descriptive-panel columns in the Phase 1 aggregate; drop Phase 5b sweep 3 (outputs remain at 842ddb9)
db2a64b Roster: add pade_33/34/44/45; current_value leaves the accelerator roster (last_value stays as the trivial)
0b7766e Add REPORT_7 (Prompt 7 report)
```

One commit per part, messages as specified. Two deviations, both recorded under Open questions: the Part C commit was amended once (a test schema-lock I had just written was wrong; amended before any later commit so that one-commit-per-part holds), and the Part J commit also carries two small edits found by the Part J audit (a hard-coded evaluation-count FACTS row replaced by a derivation from `reproduce_all.plan`, and the README line for the new dev script).

## 2. Baseline and after

| | pytest | quick run wall time |
|---|---|---|
| Baseline (PRE_R8, Part A) | 350 passed in 93.9 s | 138 s (`..\R8_baseline_quick\results\`) |
| After (HEAD, Part J) | 363 passed in 125.0 s | 144 s (`..\R8_after_quick\results\`) |

Two further quick runs were made as integration checks after Parts E and G (132 s and 149 s; both restored). `git status` is clean after every restore.

## 3. Roster

```
len(ACCEL_METHODS) = 52
len(METHOD_NAMES)  = 57   (52 accelerators + 5 trivial comparators)

sorted(ACCEL_METHODS) =
['anderson_1', 'anderson_2', 'anderson_3', 'best_shanks_wynn', 'brezinski_theta1', 'brezinski_theta2',
 'double_exp_fit', 'geom_avg_diff', 'levin_t1', 'levin_t2', 'levin_u1', 'levin_u2', 'levin_v1', 'levin_v2',
 'linear', 'log_fit', 'log_linear', 'median_ensemble', 'neville_2', 'neville_3', 'neville_4',
 'pade_11', 'pade_12', 'pade_13', 'pade_21', 'pade_22', 'pade_23', 'pade_31', 'pade_32', 'pade_33', 'pade_34',
 'pade_44', 'pade_45', 'rational_fit', 'richardson_1', 'richardson_2', 'richardson_3', 'richardson_a05',
 'richardson_a10', 'richardson_a20', 'shanks_1', 'shanks_2', 'shanks_3', 'shanks_4', 'single_exp_fit',
 'stability_weighted', 'wynn_eps_1', 'wynn_eps_2', 'wynn_eps_3', 'wynn_rho_1', 'wynn_rho_2', 'wynn_rho_3']

PHASE2_POOL = ['last_value', 'richardson_1', 'richardson_a10', 'single_exp_fit', 'rational_fit', 'pade_22',
               'log_linear', 'levin_t2', 'anderson_1']
```

`tests/test_pipeline_v2.py::test_accelerator_roster_is_exactly_the_expected_list` holds this list explicitly (`EXPECTED_ACCELERATORS`).

B3 flag check — `is_trivial` filters inspected: `scripts/make_paper_tables.py` (accelerator-only rankings, counts and the f12 roster; correct to exclude `last_value`), `scripts/analyze_by_ltrue.py` (top-10 accelerators; correct), `scripts/run_phase1.py` (prints the trivial block; includes `last_value`), `phases/phase5b.py` (a column name only). Pools are explicit name lists (`PHASE2_POOL`, `RANK_POOL`, `CANDIDATES`, `PHASE4_METHODS`, `PHASE2_METHODS` in Phase 5a) and all contain `last_value`. One non-`is_trivial` filter did remove it: Phase 5a computed `perturb_iqr` only for `method in POOL` (the accelerator roster), which would have dropped `last_value` from `diag_ensemble_9` where `current_value` used to take part; `perturb_iqr` is now computed for `PERTURB_METHODS` (roster plus Phase-2 pool), and the plan counts perturbation calls from that list. The Phase-5a method-weights table (`phase5a_method_weights.csv`) is the accelerator roster and no longer carries a trivial row (it carried `current_value` only because it was an accelerator by registration).

## 4. `python reproduce_all.py --plan`

```
==============================================================================
  PLANNED EVALUATIONS  [FULL]
==============================================================================
  Phase 0                         208   52 accelerators x 4 analytic cases (trivials excluded)
  Phase 1                     369,360   24 regimes x 30 seeds x 3 noise x 3 strata x 57 methods
  Dangerous derivation              0   reads phase1_aggregated.csv
  Phase 2                     772,200   13 depths x 5 noise x 20 seeds x 18 core regimes x 3 strata x 11 methods (+2 skill-reference calls per cell)
  Phase 3                           0   analysis of Phase 2 output
  Phase 4                     207,360   central evaluations (24 regimes, 12 methods); + 1,555,200 diagnostic calls
  Phase 5a                    984,960   central evaluations (24 regimes, 57 methods); + 4,579,200 perturbation calls
  Phase 5b                    259,200   sweep1a 28,800 + sweep1b 201,600 + sweep2 28,800 (24 regimes, 2 strata)
  Real data                       630   6 datasets x 15 (depth, target) pairs x 7 methods
  --------------------------------------------------------------------------
  TOTAL (central)           2,593,918

==============================================================================
  PLANNED EVALUATIONS  [QUICK]
==============================================================================
  Phase 0                         208   52 accelerators x 4 analytic cases (trivials excluded)
  Phase 1                       2,736   4 regimes x 2 seeds x 2 noise x 3 strata x 57 methods
  Dangerous derivation              0   reads phase1_aggregated.csv
  Phase 2                       3,432   13 depths x 2 noise x 2 seeds x 2 core regimes x 3 strata x 11 methods (+2 skill-reference calls per cell)
  Phase 3                           0   analysis of Phase 2 output
  Phase 4                       1,152   central evaluations (4 regimes, 12 methods); + 5,184 diagnostic calls
  Phase 5a                      5,472   central evaluations (4 regimes, 57 methods); + 15,264 perturbation calls
  Phase 5b                      2,752   sweep1a 320 + sweep1b 2,240 + sweep2 192 (4 regimes, 2 strata)
  Real data                       630   6 datasets x 15 (depth, target) pairs x 7 methods
  --------------------------------------------------------------------------
  TOTAL (central)              16,382
```

## 5. Regression check (`scripts/dev/compare_quick_runs.py ..\R8_baseline_quick\results ..\R8_after_quick\results`)

```
BEFORE: ../R8_baseline_quick/results
AFTER : ../R8_after_quick/results

  phase1_records: 53 shared methods, 2,544 records compared on ['estimate', 'error', 'valid', 'catastrophic', 'n_f', 'capped', 'L_true', 'L_hat']: IDENTICAL
    methods only in AFTER : ['pade_33', 'pade_34', 'pade_44', 'pade_45']
    methods only in BEFORE: ['current_value']
  BEFORE phase1_records: beats_current == win_vs_last on all 2,592 records
  AFTER phase1_records: E_last equals last_value's error on 2,736 of 2,736 records
  phase4_raw.csv: 12 shared methods, 1,152 records compared on ['estimate', 'error']: IDENTICAL
    methods only in AFTER : []
    methods only in BEFORE: ['current_value']
  phase5a_raw.csv: 53 shared methods, 5,088 records compared on ['estimate', 'error']: IDENTICAL
    methods only in AFTER : ['pade_33', 'pade_34', 'pade_44', 'pade_45']
    methods only in BEFORE: ['current_value']
  phase1_aggregated: 53 shared methods, 1,272 records compared on ['valid_rate', 'cat_rate', 'med_error', 'med_skill', 'win_rate_vs_last']: IDENTICAL
    methods only in AFTER : ['pade_33', 'pade_34', 'pade_44', 'pade_45']
    methods only in BEFORE: ['current_value']
  AFTER phase1_aggregated: q25 <= med <= q75 <= p90 on 1,294 of 1,294 rows with valid records; n_valid <= n_total on 1,368 of 1,368 rows; conditional fields NaN on every row without valid records: True

RESULT: every shared method is byte-for-byte identical on every compared column; the roster differs only by the listed methods
```

(53 shared methods = the 48 unchanged accelerators + 5 trivial comparators, `last_value` among them; Phase 4 shares 12 methods because its pool holds `last_value` and three further trivials.)

## 6. Grep audits (run at HEAD, `__pycache__` removed first)

```
$ grep -rn "current_value" src phases scripts tests reproduce_all.py
(no output; exit 1)

$ grep -rni "stabilit" src phases scripts tests reproduce_all.py
src/accelerators.py:1025:def accel_stability_weighted(seq, indices, future_x: float,
src/accelerators.py:1028:    Stability-weighted average over the same pool as the median ensemble.
src/accelerators.py:1032:    retired composite stability score S of the evaluation.
src/accelerators.py:1155:    "stability_weighted": accel_stability_weighted,
src/evaluation.py:117:    'median_ensemble': 'ensemble',   'stability_weighted': 'ensemble',
src/evaluation.py:137:    'median_ensemble', 'stability_weighted',   # pool contains Richardson
phases/phase5b.py:104:                  'log_linear', 'stability_weighted', 'median_ensemble']
scripts/dev/compare_quick_runs.py:148:        gone = [c for c in ("stability", "beats_rate") if c in aa.columns]
scripts/make_paper_tables.py:1334:         "log_linear, richardson_1, ..., stability_weighted, median_ensemble (10; measured on the 96 audit windows)",
tests/test_input_dependence.py:63:# stability_weighted is an ensemble over a pool that contains them).  The
tests/test_input_dependence.py:70:    "stability_weighted", "median_ensemble",
tests/test_pipeline_v2.py:95:    "median_ensemble", "stability_weighted", "best_shanks_wynn",

$ grep -rn "W_CAT\|W_BEATS\|beats_rate\|beats_current\|S < 0\|S<0\|stability_score\|by_stab\|stab_best" src phases scripts tests reproduce_all.py
scripts/dev/compare_quick_runs.py:19:  * asserts in the BEFORE file that beats_current == win_vs_last on every
scripts/dev/compare_quick_runs.py:100:    if "beats_current" in b1.columns and "win_vs_last" in b1.columns:
scripts/dev/compare_quick_runs.py:101:        same = (b1["beats_current"].to_numpy() == b1["win_vs_last"].to_numpy())
scripts/dev/compare_quick_runs.py:103:            print(f"  BEFORE phase1_records: beats_current == win_vs_last on all {len(b1):,} records")
scripts/dev/compare_quick_runs.py:105:            problem(f"BEFORE phase1_records: beats_current != win_vs_last on {int((~same).sum())} records")
scripts/dev/compare_quick_runs.py:107:        problem("BEFORE phase1_records lacks beats_current or win_vs_last")
scripts/dev/compare_quick_runs.py:148:        gone = [c for c in ("stability", "beats_rate") if c in aa.columns]
```

Every hit is either the `stability_weighted` method (its name, its docstring — which now carries the sentence required by ground rule 5 — and its registry / family / L_hat-consumer entries) or the development tool `scripts/dev/compare_quick_runs.py`, which must name `beats_current` to perform the assertion Part J.4 asks for on the BEFORE file and names the two retired columns to assert their absence in the AFTER aggregate. Outside `scripts/dev` the second grep returns nothing.

```
$ grep -rn "stability\|beats_rate\|richardson_stability\|best_other\|r1_losing\|losing_margin\|mean_stability\|precision\|recall\|spearman_vs_margin" src phases scripts tests reproduce_all.py
src/accelerators.py:745:    identical to levin_t1 / levin_t2 to machine precision
src/accelerators.py:1025:def accel_stability_weighted(seq, indices, future_x: float,
src/accelerators.py:1032:    retired composite stability score S of the evaluation.
src/accelerators.py:1155:    "stability_weighted": accel_stability_weighted,
src/evaluation.py:117:    'median_ensemble': 'ensemble',   'stability_weighted': 'ensemble',
src/evaluation.py:137:    'median_ensemble', 'stability_weighted',   # pool contains Richardson
phases/phase2.py:659:    """Return index of element in lst nearest to val (handles float precision)."""
phases/phase4.py:289:      precision = P(catastrophic | diagnostic > threshold)
phases/phase4.py:290:      recall    = P(diagnostic > threshold | catastrophic)
phases/phase4.py:310:                precision = true_pos / n_fire
phases/phase4.py:311:                recall    = (true_pos / n_bad) if n_bad > 0 else float('nan')
phases/phase4.py:325:                    'precision':      round(precision, 3),
phases/phase4.py:326:                    'recall':         round(float(recall), 3)
phases/phase4.py:327:                                      if math.isfinite(recall) else float('nan'),
phases/phase4.py:333:            'precision', 'recall', 'mean_err_saved']
phases/phase4.py:517:    Spearman r(diagnostic, error) and precision of high-diagnostic = catastrophic."""
phases/phase4.py:603:    """Precision vs recall for both diagnostics, side-by-side panels."""
phases/phase4.py:614:            sub = sub_d[sub_d['method'] == method].dropna(subset=['precision', 'recall'])
phases/phase4.py:617:            ax.plot(sub['recall'], sub['precision'],
phases/phase4.py:623:                            (row['recall'], row['precision']),
phases/phase5b.py:104:                  'log_linear', 'stability_weighted', 'median_ensemble']
scripts/dev/compare_quick_runs.py:148:        gone = [c for c in ("stability", "beats_rate") if c in aa.columns]
scripts/make_paper_tables.py:1325:         "so the corrected weniger_d1/d2 == levin_t1/t2 to machine precision on the 96 audit windows ...
scripts/make_paper_tables.py:1334:         "log_linear, richardson_1, ..., stability_weighted, median_ensemble (10; measured on the 96 audit windows)",
scripts/run_phase4.py:107:    print('\n  Best rejection rules (precision >= 0.30):')
scripts/run_phase4.py:111:        good = (df_rules[df_rules['precision'] >= 0.30]
scripts/run_phase4.py:112:                .sort_values('precision', ascending=False)
scripts/run_phase4.py:116:                  f"{row['threshold']:>8.3f} {row['precision']:>7.3f} {row['recall']:>7.3f}")
tests/test_accelerators.py:346:    weight (n0+j+1)^{k-1} of Levin for k = 1, 2.  Checked to machine precision
tests/test_input_dependence.py:63:# stability_weighted is an ensemble over a pool that contains them).  The
tests/test_input_dependence.py:70:    "stability_weighted", "median_ensemble",
tests/test_pipeline_v2.py:95:    "median_ensemble", "stability_weighted", "best_shanks_wynn",
tests/test_redesign.py:99:    fx = pd.read_csv(FIXTURE, float_precision="round_trip")
```

Data-flow audit classification: `stability_weighted` (method); "machine precision" / "float precision" / `float_precision="round_trip"` (unrelated numerical contexts: `src/accelerators.py:745`, `phases/phase2.py:659`, `scripts/make_paper_tables.py:1325`, `tests/test_accelerators.py:346`, `tests/test_redesign.py:99`); the compare script (see above); and Phase 4's rejection-rule table (`phases/phase4.py` Q2 `test_rejection_rules`, its figure P4-2 and the `scripts/run_phase4.py` print of it). The Phase-4 precision/recall are the detection metrics of a *diagnostic-threshold rule against the catastrophe flag* (error above `CAT_MULT` times the last-value error, or invalid); they never involved the stability score and Phase 4 was not in this prompt's scope. Flagged under Open questions in case the author wants them replaced by a rule panel as well. No `richardson_stability`, `best_other`, `r1_losing`, `losing_margin`, `mean_stability` or `spearman_vs_margin` anywhere.

```
$ grep -rn "49\|\b48\b\|\b54\b" src phases scripts tests reproduce_all.py README.md   (tests/fixtures excluded)
phases/phase5a.py:532:    'capped_diag_safe':   '#3949ab',
scripts/make_paper_tables.py:1355:    fact(sec, "evaluation counts (central)", "Phase 0 196; Phase 1 349,920; ...   <- fixed in the Part J commit (see below)
scripts/make_paper_tables.py:1378:        f.write("- **richardson_3 validity by depth**: ... (0.48 / 0.58 / 0.98 / 1.00 ...   <- fixed in the Part J commit
reproduce_all.py:212:    print(f"  {'#':>2}  {'step':<48} {'status':<8} {'time':>8}")
reproduce_all.py:215:        print(f"  {i:>2}  {label:<48} {'OK' if ok else 'FAILED':<8} {elapsed:>7.0f}s")
reproduce_all.py:217:    print(f"  {'':>2}  {'total':<48} {'':<8} {time.time() - t_start:>7.0f}s")
```

Restricted to roster-count uses this returns nothing: the `phase5a.py` hit is a colour code, the `reproduce_all.py` hits are print-format widths. The two `make_paper_tables.py` hits were hard-coded evaluation counts and previous-run numbers in a FACTS row and the FACTS header; both were replaced in the Part J commit (the evaluation-count fact is now built from `reproduce_all.plan("full")`, the header no longer quotes numbers), so the grep is clean at HEAD apart from the colour and the widths.

## 7. `real_boot` sources (Part F)

Both candidates passed every pre-specified criterion; `src/generators.py` was switched to them and the two result files were committed in ad0245d before any `git clean`.

```
  sources   : {'electricity': 151, 'nomao': 1486}
  test set  : {'covertype': 1596, 'higgs': 23512, 'adult': 1590, 'jannis': 41168, 'miniboone': 41150, 'bank_marketing': 1461}  (disjoint by construction)
  training  : src.datasets.train_xgboost, 500 rounds, seed 42
  criteria  : rows >= 20,000; gap(90)/gap(0) >= 0.05; argmin round >= 150

  electricity
    loss at round 1 / 90 / 500: 0.672324 / 0.348262 / 0.244944
    argmin round (1-based): 500   min loss 0.244944
    envelope gap ratio gap(90)/gap(0): 0.2381
    criterion rows           value 45312        threshold 20000    PASS
    criterion gap_ratio_90   value 0.238106     threshold 0.05     PASS
    criterion argmin_round   value 500          threshold 150      PASS
  nomao
    loss at round 1 / 90 / 500: 0.560152 / 0.105999 / 0.069994
    argmin round (1-based): 495   min loss 0.069931
    envelope gap ratio gap(90)/gap(0): 0.0730
    criterion rows           value 34465        threshold 20000    PASS
    criterion gap_ratio_90   value 0.0730078    threshold 0.05     PASS
    criterion argmin_round   value 495          threshold 150      PASS
  ALL CRITERIA PASSED: YES
```

Resulting gap profiles: `real_boot_a` (electricity) gap(0) = 0.700, gap(90) = 0.1667, gap(499) = 0; `real_boot_b` (nomao) gap(0) = 0.700, gap(90) = 0.0511, gap(499) = 0. The CSV written into `results/real_data/` is byte-identical to a first probe run into a scratch directory (deterministic training). `results/real_data/real_boot_sources_provenance.json`:

```json
{
  "purpose": "generator inputs of the held-out real_boot_a / real_boot_b regimes; never scored; disjoint from the recorded-curve test set (src.datasets.DATASET_IDS)",
  "sources": {"electricity": 151, "nomao": 1486},
  "test_set": {"covertype": 1596, "higgs": 23512, "adult": 1590, "jannis": 41168, "miniboone": 41150, "bank_marketing": 1461},
  "training": {"function": "src.datasets.train_xgboost", "n_rounds": 500, "seed": 42, "val_fraction": 0.2,
               "learning_rate": 0.05, "max_depth": 6, "subsample": 0.8, "colsample_bytree": 0.8},
  "versions": {"xgboost": "3.3.0", "sklearn": "1.9.0", "openml": "0.15.1", "numpy": "2.5.1", "pandas": "3.0.3"},
  "criteria": {"min_rows": 20000, "min_gap_ratio_at_obs_depth": 0.05, "obs_depth": 90, "min_argmin_round": 150,
               "gap_ratio_definition": "gap(obs_depth) / gap(0) of the smoothing-spline lower-envelope profile of src.generators.envelope_profile (the real_boot construction)"},
  "git_head": "ab04bfc",
  "created": "2026-09-27T14:16:08",
  "all_passed": true,
  "datasets": {
    "electricity": {"openml_id": 151, "openml_version": 1, "openml_name": "electricity", "n_rows": 45312, "n_features": 8, "n_classes": 2,
                    "loss_round_1": 0.6723241493045053, "loss_round_obs": 0.3482620105727242, "loss_round_last": 0.2449443401276191,
                    "min_loss": 0.2449443401276191, "argmin_round": 500, "gap_ratio_obs": 0.23810558116512387,
                    "criteria": {"rows": {"value": 45312, "threshold": 20000, "passed": true},
                                 "gap_ratio_90": {"value": 0.23810558116512387, "threshold": 0.05, "passed": true},
                                 "argmin_round": {"value": 500, "threshold": 150, "passed": true}},
                    "passed_all": true},
    "nomao": {"openml_id": 1486, "openml_version": 1, "openml_name": "nomao", "n_rows": 34465, "n_features": 118, "n_classes": 2,
              "loss_round_1": 0.5601515983715469, "loss_round_obs": 0.10599868264476649, "loss_round_last": 0.06999417693558856,
              "min_loss": 0.06993148478542166, "argmin_round": 495, "gap_ratio_obs": 0.07300777546376783,
              "criteria": {"rows": {"value": 34465, "threshold": 20000, "passed": true},
                           "gap_ratio_90": {"value": 0.07300777546376783, "threshold": 0.05, "passed": true},
                           "argmin_round": {"value": 495, "threshold": 150, "passed": true}},
              "passed_all": true}
  }
}
```

(`git_head` is the commit before the Part F commit, as with every artifact that is committed together with the code that produced it.)

## 8. Phase 2 denominator counts, Phase 3 selector validity (after-snapshot, quick grid)

`results/phase2/phase2_denominator_counts.csv`:

```
target_g,n_records_zero_denominator,n_cells_affected,n_records_total,n_cells_total
0.02,0,0,52,26
0.1,0,0,52,26
0.5,0,0,104,52
```

`results/phase3/phase3_selector_comparison.csv` at g = 0.1 (quick grid: 26 uncapped cells × 2 seeds; the 26 `log_slow` cells are capped at this stratum and excluded):

| selector | valid_rate | n_valid / n_total | med_error |
|---|---|---|---|
| fixed_last | 1.0000 | 52/52 | 0.027616 |
| fixed_richardson | 1.0000 | 52/52 | 0.023053 |
| fixed_rational | 1.0000 | 52/52 | 0.021518 |
| fixed_single_exp | 1.0000 | 52/52 | 0.000030 |
| phase2_cascade | 1.0000 | 52/52 | 0.023053 |
| enhanced_cascade | 1.0000 | 52/52 | 0.021518 |
| oracle | 1.0000 | 52/52 | 0.000030 |

Validity warning: **fired**, at g = 0.5 (not at 0.1 or 0.02). Console line of the after run:

```
WARNING: validity rate differs between selectors by 0.0096 at g = 0.5 (fixed_last 1.0000, fixed_richardson 1.0000, fixed_rational 1.0000, fixed_single_exp 1.0000, phase2_cascade 1.0000, enhanced_cascade 1.0000, oracle 0.9904); the error statistics are conditional on validity -- read them together with the validity rates
```

The cause is the designed behaviour: the hindsight oracle picks the candidate with the lowest cell-median error over its *valid* records, and on one cell that candidate had one invalid seed, which the oracle inherits (no fallback). Validity spread per stratum: g = 0.5: 0.0096, g = 0.1: 0.0, g = 0.02: 0.0. The comparison file was written regardless.

## 9. Regime classifier (after-snapshot, quick grid: 2 regimes × 13 depths × 2 noise levels = 52 rows)

| protocol | overall accuracy | folds |
|---|---|---|
| grouped_by_depth (primary) | 0.942 | 13 folds, one per obs_idx |
| grouped_by_noise (check) | 0.827 | only 2 distinct noise values: 2 folds (noted in the `note` column) |
| stratified_5fold_legacy (comparison only) | 0.962 | 5 stratified folds |

`split_unit` strings, one per protocol, repeated on every row of `phase3_regime_classifier.csv`:

- grouped_by_depth: `one row is the seed-averaged feature vector of one (regime, obs_idx, noise) cell (seeds never straddle a fold boundary); GroupKFold with groups = obs_idx: every fold holds out all cells at one observation depth and trains on the cells at every other depth, so the same regime at adjacent depths remains in training and the protocol tests transfer across depth`
- grouped_by_noise: `one row is the seed-averaged feature vector of one (regime, obs_idx, noise) cell (seeds never straddle a fold boundary); GroupKFold with groups = noise: every fold holds out all cells at one noise level and trains on the cells at every other noise level`
- stratified_5fold_legacy: `one row is the seed-averaged feature vector of one (regime, obs_idx, noise) cell (seeds never straddle a fold boundary); random stratified 5-fold over cells; cells of the same regime at adjacent depths may fall on both sides`

The scipy 1-NN fallback is gone; a missing scikit-learn raises an `ImportError` with an explicit message.

## 10. Padé sanity lines (after-snapshot, `phase1_global.csv`, core, g = 0.1; quick grid = 2 core regimes × 2 noise levels, `log_slow` capped)

```
pade_33: valid_rate 1.0000  med_error 0.020023  cat_rate 0.0000  rank_eligible 1  n_cells 2
pade_34: valid_rate 1.0000  med_error 0.025285  cat_rate 0.2500  rank_eligible 1  n_cells 2
pade_44: valid_rate 1.0000  med_error 0.020087  cat_rate 0.0000  rank_eligible 1  n_cells 2
pade_45: valid_rate 1.0000  med_error 0.014589  cat_rate 0.0000  rank_eligible 1  n_cells 2
```

Excluded set derived from the quick Phase 1 (schema v3, `check_dangerous.py --phase1-dir` MATCH): `geom_avg_diff, linear, neville_2, neville_3, neville_4, pade_21, pade_31, pade_32, wynn_eps_2, wynn_eps_3` (quick-grid numbers; meaningless for the paper, listed only to show the criterion running end to end).

Part J.6: `make_paper_tables.py --results ..\R8_after_quick\results --out ..\R8_after_quick\paper_fragments --facts ..\R8_after_quick\FACTS.md` wrote 22 fragments and 214 facts (f10b skipped: the quick grid has no `perturb_iqr` rows at the headline stratum, a pre-existing quick-mode limitation); `make_paper_figures.py --results ... --out ...` wrote both figures; `check_dangerous.py --phase1-dir ..\R8_after_quick\results\phase1` returned MATCH and confirmed schema v3, the criterion text and the floor.

## 11. Files changed and output-column changes

Files (48 changed, `git diff --stat 0b7766e d6beb53`):

- `.gitignore` — `results/phase2/phase2_records.csv` added next to the other raw files.
- `README.md` — title and venue; roster sentences with derived counts (52 printed by the command); exclusion paragraph = validity criterion + legacy-name note; new *Descriptive reporting* and *Regime classifier* paragraphs; `real_boot` sources and criteria; Phase 5b without sweep 3 (outputs at 842ddb9); results/raw-file paragraph; scripts table; tests paragraph (no exemption; new test files; 363 tests); repository tree (`panels.py`, `make_real_boot_sources.py`, `dev/compare_quick_runs.py`); Report 3B references → 5B / 8A; citation block.
- `reproduce_all.py` — plan: Phase-5b sweep 3 removed, real-data counts from `DATASET_IDS` and `REAL_DATA_METHODS`, Phase-5a perturbation calls from `PERTURB_METHODS`; docstring without hard-coded counts and without S.
- `src/accelerators.py` — `accel_pade_33/34/44/45` (one-line wrappers) registered after `pade_32`; `accel_current_value` and its registry entries deleted; `stability_weighted` docstring sentence (weight = 1/(shift IQR + 1e-6), unrelated to the retired score); `best_shanks_wynn` docstring wording; module docstring with derived counts.
- `src/trivial.py` — `last_value` docstrings (no alias); `SKILL_EPS` public.
- `src/evaluation.py` — `FAMILY` / `USES_FUTURE_X` for the four Padé variants, `current_value` removed; `stability_score` deleted; `W_*` out of `build_cfg`; record: `E_last` added, `beats_current` removed; aggregate through `error_panel` (panel columns added, `beats_rate`/`stability` removed); pooled tables and unranked block without `beats_rate`/`stability`; `regime_best`: `best_by_stability` block replaced by `best_by_error` under the validity floor; heatmaps = median error (+ valid-rate companion); docstring.
- `src/config.py` — `W_CAT`/`W_BEATS` block deleted; exclusion comment = validity criterion + legacy-name note; `catmult_values` removed from `PHASE5B`; section header wording.
- `src/diagnostics.py` — `stability_score` deleted; module/section docstrings ("perturbation diagnostics").
- `src/dangerous.py` — `derive_dangerous` by pooled validity (`valid_rate`, `cat_rate`, `med_error` tabulated; flag = eligible and `valid_rate < RANK_MIN_VALID`; sorted by `valid_rate`); schema `dangerous_methods/v3` with the criterion text, `rank_min_valid` and `legacy_name`; `load_artifact` accepts only v3; module docstring with the legacy-name note.
- `src/pipeline.py` — `TRIVIAL_NON_ORACLE` defined before `PHASE2_POOL`; pool starts with `last_value`; assertion `m in ACCEL_METHODS or m in TRIVIAL_NON_ORACLE`; comments with derived counts.
- `src/panels.py` (new) — `error_panel`, `panel_of`, `zero_denominator_flags`, `threshold_rule`, `rule_panel`, `PANEL_COLS`, `RULE_COLS`.
- `src/plots.py` — six Phase-1 figures redrawn on median error with the validity floor; no S anywhere; `figure_01_error_ranking.png`.
- `src/generators.py` — `envelope_profile(n, y)` factored out of `real_boot_profile`; `_REAL_CURVES_PATH` → `real_boot_sources.csv`; `_REAL_BOOT_SOURCES = {real_boot_a: electricity, real_boot_b: nomao}`; docstrings.
- `src/datasets.py` — `load_datasets(ids=DATASET_IDS, with_meta=False)`; `REAL_BOOT_SOURCE_IDS` with disjointness assertions; docstring.
- `src/trajectories.py` — `REAL_DATA_METHODS` (the 7 reported methods, asserted against the per-cell prediction dict); cascade docstring wording.
- `phases/phase1.py` — `stability_score` re-export dropped.
- `phases/phase2.py` — rewritten as described in Part D (records file, panel aggregate, Richardson targets, denominator counts, two-target correlations, `evaluate_rules` via `rule_panel`, figures redrawn: P2-1 = map of `R_R_med` over depth × noise, my choice as allowed).
- `phases/phase3.py` — rewritten: per-record selector panels (no fallback), `fixed_last`, oracle by lowest cell-median error, validity warning, `cross_validate` for the two cascades, three-protocol classifier with `split_unit` and `note`, figures error-based with validity alongside.
- `phases/phase4.py` — `last_value` in the colour table and as the perturbation-filter fallback (both places); docstring title.
- `phases/phase5a.py` — `current_value` selector removed; small-pool selector names derived from `len(PHASE2_POOL)` (`oracle_9`, `equal_ensemble_9`, `diag_ensemble_9`, `weighting_gain_9` unchanged as strings); `PERTURB_METHODS`; docstring.
- `phases/phase5b.py` — sweep 3 and figure P5B-3 removed; `_stability`, `W_*` and `CAT_MULT` cfg keys removed; `_cascade_metrics` replaced by `rule_panel` on per-window cells (`_cascade_cell_record` stores both methods' error/valid/catastrophic and `E_last`); figures P5B-1/2 plot fire rate, lower-error fraction and median relative change; the artifact is loaded as the ordering guard only.
- `scripts/run_phase1.py` — prints without `stability`; `best_by_error` shown; docstring.
- `scripts/run_phase2.py` — rewritten summary (Richardson targets, denominator counts, two-target correlations, rule panels).
- `scripts/run_phase3.py` — rewritten summary (panel columns, warning outcome, LORO, three protocols); requires `phase2_records.csv`.
- `scripts/run_phase4.py` — title wording.
- `scripts/run_phase5a.py` — perturbation-call count from `PERTURB_METHODS`.
- `scripts/run_phase5b.py` — sweep 3 removed from plan, arguments and console; sweep 1/2 tables print the rule panel.
- `scripts/run_real_data.py` — `current_err` documented as the last-value error; counts from `REAL_DATA_METHODS` / `DATASET_IDS`.
- `scripts/derive_dangerous.py`, `scripts/check_dangerous.py` — validity criterion printed and verified; `check_dangerous.py --phase1-dir` for snapshots; schema/criterion/floor checks.
- `scripts/make_paper_tables.py` — roster labels (`fixed_last`, small-pool names); excluded-set section on the v3 artifact (pooled valid rate of each excluded method; lowest valid rate among the non-excluded); parameters row without S; f05 on the rule panel; f09a on the panel columns with the caption "conditional on validity; validity rate alongside"; `\Stab` macro dropped; evaluation-count fact derived from `reproduce_all.plan`.
- `scripts/make_paper_figures.py` — `--results/--out`; fig1 on `target_g == HEADLINE_G`, excluded set from the snapshot's artifact, legend "excluded (valid rate below 0.9)".
- `scripts/make_real_boot_sources.py` (new), `scripts/dev/compare_quick_runs.py` (new).
- `results/phase5b/phase5b_sweep3_{champions,concordance,global,unranked}.csv`, `results/phase5b/figure_p5b_03_catmult.png` — deleted (available at 842ddb9); `results/real_data/real_boot_sources.csv`, `results/real_data/real_boot_sources_provenance.json` — added. Every other committed results file is byte-identical to 842ddb9.
- Tests: `tests/test_accelerators.py` (baseline test reformulated; `W_*` keys gone), `tests/test_generators.py` (source disjointness; envelope profiles), `tests/test_input_dependence.py` (no exemption), `tests/test_panels.py` (new), `tests/test_phase3_classifier.py` (new), `tests/test_pipeline_v2.py` (derived roster count; explicit 52-name roster; validity-criterion artifact test with 0.85 excluded / 0.95 kept / 0.9 kept / trivial never excluded / v2 refused; Phase-1 schema locks; Phase-2 records/targets/rules and Phase-3 selector test), `tests/test_redesign.py` (`trivial_last_value` identity; derived counts; `best_by_error`; source-curve test).

Output columns added / renamed / removed:

- `phase1_records.csv`: + `E_last`; − `beats_current`.
- `phase1_aggregated.csv`: + `n_total`, `n_valid`, `mean_error`, `sd_error`, `q25_error`, `q75_error`, `p90_error`; − `beats_rate`, `stability`.
- `phase1_global.csv`, `phase1_global_holdout.csv`: − `beats_rate`, `stability`. `phase1_unranked.csv`: − `stability`.
- `phase1_regime_best.csv`: − `best_by_stability`, `stab_best_stability`, `stab_best_med_error`; + `best_by_error`, `err_best_med_error`, `err_best_valid_rate`, `err_best_cat_rate`, `n_at_floor`.
- `phase1_heatmap_g{g}.csv`: values are now the method × regime median error (were S); new companion `phase1_heatmap_valid_g{g}.csv` (valid rate). `figure_01_stability_ranking.png` → `figure_01_error_ranking.png`.
- `dangerous_methods.json`: schema `v3`; − `W_CAT`, `W_BEATS`; + `rank_min_valid`, `legacy_name`; table − `beats_rate`, `stability`; + `med_error`.
- `phase2_records.csv` (new, git-ignored): `method, regime, obs_idx, noise, seed, target_g, estimate, error, valid, catastrophic, E_last, capped, L_true, L_hat, n_f, achieved_g`.
- `phase2_sweep_aggregated.csv`: + the seven panel columns above; − `beats_rate`, `stability`.
- `phase2_phase_diagram_g{g}.csv`: removed (files). `phase2_richardson_targets_g{g}.csv` (new): `regime, obs_idx, noise, target_g, n_f, achieved_g, capped, R_R_med, log_med_error_R, med_error_R, n_valid_R, n_total, n_zero_denominator, zero_denominator_cell`. `phase2_denominator_counts.csv` (new): `target_g, n_records_zero_denominator, n_cells_affected, n_records_total, n_cells_total`.
- `phase2_correlations*.csv`: − `spearman_vs_margin`, `p_vs_margin`, `spearman_vs_losing`, `p_vs_losing`, `n_obs`; + `spearman_vs_RR`, `p_vs_RR`, `spearman_vs_log_err`, `p_vs_log_err`, `n_cells`, `n_dropped_nan`.
- `phase2_rules*.csv`: − `n_fire`, `n_total`, `precision`, `recall`, `mean_gain`; + `RULE_COLS` = `n_cells_total, n_cells_fired, fire_rate, r1_<panel>, alt_<panel>, lower_error_frac_records, lower_error_frac_cells, median_rel_change, nf_r1_<panel>, nf_alt_<panel>` (`<panel>` = the eleven `PANEL_COLS`); rows in `CANDIDATE_RULES` order, no sorting by a score.
- `phase3_selector_comparison.csv`, `phase3_regime_results.csv`: − `mean_stability`, `n`; + the eleven `PANEL_COLS`, `n_cells` (`n_capped_excluded` kept); selector `fixed_current` → `fixed_last`.
- `phase3_cv_results.csv`: − `mean_stability`, `n`; + `target_g`, `PANEL_COLS`, `n_cells`; rows for the two cascades only.
- `phase3_capped_cells.csv`: − `achieved_stability`; + `chosen_med_error`.
- `phase3_regime_classifier.csv`: + leading `protocol`, `split_unit`, `note`; the per-regime and `__OVERALL__` rows repeat per protocol.
- `phase5b_sweep1_global.csv`, `phase5b_sweep2_global.csv`: − `precision`, `recall`, `mean_gain`, `n`; + `RULE_COLS` (`fire_rate` kept as part of them). `phase5b_sweep1_regime.csv`, `phase5b_sweep2_regime.csv`: − `precision`, `mean_gain`, `n`; + `RULE_COLS`.
- `phase5b_sweep3_*.csv`, `figure_p5b_03_catmult.png`: removed.
- `phase5a_ensemble*.csv`, `phase5a_by_sigma.csv`, `phase5a_per_regime.csv`: the `current_value` selector row is gone (`last_value` row unchanged); `phase5a_raw.csv`: `perturb_iqr` now finite for `last_value`.
- `phase4_raw.csv` and the Phase-4 tables: `last_value` rows in place of `current_value` rows (`current_value` is no longer a method); `phase4_cascade_filter.csv` falls back to `last_value`.
- `real_boot_sources.csv` (new): `round` 0–499, `electricity`, `nomao`; `real_boot_sources_provenance.json` (new). Real-data tables: unchanged (`current_err` documented).

## 12. Open questions

1. **Intermediate state at the Part C commit.** As the prompt's Part C text anticipates ("Part D rewrites the analysis"), at cfead26 `phases/phase2.py`'s phase-diagram / correlation / rule functions and `phases/phase3.py` still referenced the removed `stability` column (one Phase-2 test failed at that commit) and the "stabilit" grep returned those names; both are clean from c6ba693 on. The Part C commit was amended once before Part D to correct a column-order expectation in a schema-lock test I had added (`aggregate_skill_vs` interleaves the per-trivial median and win-rate columns).
2. **Part J commit carries two extra edits** found by the J.5 audit: `scripts/make_paper_tables.py` hard-coded the previous run's evaluation counts in a FACTS row (ground rule 2) — now derived from `reproduce_all.plan("full")` — and the FACTS header quoted previous-run validity numbers (removed); plus the README line for the dev script. An extra commit would have broken one-commit-per-part, so they ride with J.
3. **Phase 5a `perturb_iqr` for `last_value`** (B3): computed so that `last_value` takes part in `diag_ensemble_9` as `current_value` did. Because the perturbation RNG stream is shared across methods in evaluation order, `perturb_iqr` values of the other methods are not comparable with the baseline anyway (the four new Padé methods and the removed `current_value` already shift the stream); `estimate`/`error` are unaffected, as the compare script shows. The plan's perturbation-call count rose accordingly (4,579,200 in the full grid).
4. **Phase 4 precision/recall** (Q2 rejection rules against the catastrophe flag, figure P4-2, `run_phase4.py` print) remain: they are detection metrics of a diagnostic threshold rule and never involved S; Phase 4 was outside this prompt. If the author wants Phase 4's rules reported as rule panels too, that is a separate change.
5. **`load_artifact` accepts only v3**, so the committed `results/phase1/dangerous_methods.json` (v2) is refused by the new code: Phases 5a/5b, `check_dangerous.py`, `make_paper_tables.py` and `make_paper_figures.py` cannot run against the committed `results/` until the full run regenerates it. Expected and documented; all four were verified against the after-snapshot instead.
6. **`results/README.md`** still describes sweep 3, the stability heatmap and the three raw files; I left it untouched because of the byte-identical rule for `results/`. It needs the corresponding edits with the R9 full run (or permission to edit it now). Likewise the committed `FACTS.md` / `paper_fragments/` are from the previous run and are regenerated in R9; the `make_paper_tables.py` "pipeline provenance: full run" fact is a hard-coded description of the 2026-09-21 run and must be updated after R9.
7. **`figure_01_stability_ranking.png`** stays in the committed `results/phase1/` (byte-identical rule) although the code now writes `figure_01_error_ranking.png`; the stale file disappears only when the author removes it or the tree is regenerated.
8. **Definitions resolved on my side** (please check): (a) the denominator counts are taken over the *uncapped* cells (the cells that enter the correlations), the per-cell targets file carries every cell with its `capped` flag; (b) the pooled `ALL` correlation row uses the same `MIN_OBS = 15` as the per-regime rows (the old code used 10 for the pooled row); (c) in Phase 5b the cascade fires per window, so the rule-panel "cell" is one (regime, noise, seed) window with one record per method — `lower_error_frac_records` and `lower_error_frac_cells` coincide there and the docstrings say so; (d) when no candidate has a valid record on a cell, the oracle's pick is immaterial (every choice is invalid) and the first candidate is recorded; (e) `sd_error` is the sample standard deviation (NaN below two valid records), quantiles use numpy's default linear interpolation; (f) rates in the written tables are rounded to four decimals as before (the panel function itself rounds nothing); (g) `best_by_error` in `phase1_regime_best.csv` applies the validity floor (a regime with no method at the floor gets an empty name and `n_at_floor = 0`); (h) the Phase-1 heatmap files now hold median error with a valid-rate companion file rather than being dropped; (i) figure P2-1 was redrawn as the descriptive `R_R_med` map (the option the prompt offered); (j) `cross_validate` keeps only the two cascades and, because their thresholds are fixed, its rows coincide with the cascades' rows of `phase3_regime_results.csv` — kept as specified, but it is redundant information.
9. **`envelope_profile` refactor before the verdicts.** `real_boot_profile`'s construction was factored into `envelope_profile(n, y)` in `src/generators.py` before the criteria were evaluated, so the script measures candidates with exactly the generator's construction; the switch of `_REAL_BOOT_SOURCES` / `_REAL_CURVES_PATH` waited for both verdicts. The refactor changes no number (the generator tests pass unchanged on the `adult`/`higgs` profiles before the switch).
10. **Git-ignored raw files.** The quick runs overwrite `phase1_records.csv`, `phase4_raw.csv` and `phase5a_raw.csv` of the previous full run (git cannot restore them). I backed them up to `..\R8_fullrun_raw_backup\results\` before Part A's run and copied them back after Part J.7, so the working tree again holds the full-run raw files (sizes 115.7 / 71.5 / 326.9 MB). There is no full-run `phase2_records.csv` yet (new file; the after-snapshot has the quick one).
11. **Stray directory `C:\c\Users\kianu\AppData\Local\Temp\claude\...`** (2.5 MB): two ad-hoc smoke runs wrote to a POSIX-style path that Python resolved relative to the drive root. It contains only my scratch output; its removal was blocked by the tool policy, so please delete it.
12. **Quick-mode observations, not defects:** the validity warning fires at g = 0.5 (Section 8) by design; `f10b` is skipped on quick results because the quick grid has no `perturb_iqr` rows at the headline stratum (pre-existing).
13. **Not changed, possibly worth a look:** `scripts/analyze_by_ltrue.py` (Prompt 6 analysis) still reads the full-run `phase1_records.csv` and asserts 21 classical variants (true), but its `f13` fragment and FACTS section belong to the previous run; `scripts/analyze_real_diagnostics_legacy.py` is untouched legacy.

## Verification snapshot folders (outside git)

- `..\R8_baseline_quick\results\` — quick run at PRE_R8 (with raw files).
- `..\R8_after_quick\results\` (+ `paper_fragments\`, `FACTS.md`, `paper_figures\`) — quick run at HEAD (with raw files, including `phase2_records.csv`).
- `..\R8_after_quick_g\` — interim quick run after Part G (used to verify the table generator before Part H); `..\R8_baseline_quick_run.txt`, `..\R8_after_quick_run.txt`, `..\R8_check_quick_run.txt`, `..\R8_check_quick_run_g.txt` — console logs.
- `..\R8_fullrun_raw_backup\results\` — the previous full run's git-ignored raw files.
