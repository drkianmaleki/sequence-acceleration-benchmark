**Report back**

# Redesign v2, Prompt 5A: Weniger fix, input-dependence guard, fixed-reference skill, sweep 1b, real-data pre/post-minimum strata, validity by depth

Date: 2026-09-21. Branch `redesign-v2` in `code/`. Code commit `ddca12a` on top of `070f32d` (Report 4); this report and the quick-run log are committed on top. **Not pushed** (the prompt did not ask for a push; see deviation 1). The full pipeline was **not** run; the committed `results/` (commit `30de724`, code `0375f24`) therefore predate every change below and no longer match the code (deviation 2). Verification: `pytest tests/` (348 passed) and `python reproduce_all.py --quick` in a detached worktree of `ddca12a` (`../wt-5a`; log `REPORT_5A_quick_run.txt`, 1,318 lines, exit 0).

## 1. Files changed (commit `ddca12a`: 39 files, +972 / -159)

| file | change |
|---|---|
| `src/accelerators.py` | Item 1. `_weniger_delta`: remainder estimate `w_j = s_{j+1} - s_j` (`np.diff` of the last `order + 2` window values), Pochhammer weight `(n0 + j + 1)_{k-1}` via `scipy.special.poch` (new import `sp_poch`), the `abs(w_j) < tol -> NaN` guard kept, `D < tol -> NaN` kept. Docstring states the formula and, in one paragraph, the old behaviour (`w_n = s_n` cancelled the sequence out of the numerator; both orders returned 0 for every input). |
| `tests/test_accelerators.py` | `test_weniger_recovers_geometric_limit[weniger_d1/d2]`: limit of `0.3 + 0.5 * 0.9**n` recovered to <= 1e-12 (`import pytest` added). |
| `tests/test_input_dependence.py` | **new**, item 2 (section 3). |
| `src/trivial.py` | Item 3. `REFERENCE_TAGS`, the column-name tuples, `skill_vs_table` (per record), `skill_vs_from_arrays` (selector arrays), `aggregate_skill_vs` (DataFrame slice); module docstring rewritten: `skill` = hindsight best-of-four (strict), `skill_vs_*` / `win_vs_*` = fixed reference. |
| `src/evaluation.py` | Phase 1: the eight per-record columns; `med_skill_vs_*` / `win_rate_vs_*` per (method, regime, noise, g) cell and in the pooled core / held-out tables (median of per-cell medians, mean of per-cell win rates); docstring. |
| `phases/phase2.py` | Records and `phase2_sweep_aggregated.csv` carry the columns (the three extra trivials were already evaluated on every cell as skill references). |
| `phases/phase4.py` | Records carry the columns; `phase4_ensemble.csv` gets `med_skill_vs_*` / `win_rate_vs_*` per selector (the four trivial selectors are the references). |
| `phases/phase5a.py` | Records carry the columns; every selector summary (`phase5a_ensemble`, `_ensemble_holdout`, `_by_sigma`, `_per_regime`) gets them (`_selector_summary` reads the trivial selectors' error columns); `phase5a_method_weights.csv` gets them; **item 6**: `phase5a_validity_by_depth.csv` (method x obs_idx x noise: `valid_rate`, `n`, flags). |
| `src/trajectories.py` | Item 3: `real_data_results_v2.csv` rows carry `skill_vs_*` / `win_vs_*`, the summary carries `cascade_skill_vs_*` / `cascade_win_vs_*`. **Item 5**: `curve_minimum_table` (argmin round, minimum, value at round 500, relative rise), `post_min_target = target_round > argmin_round` on every row of both files (+ `argmin_round`, `rise_from_min`), `real_data_strata` (every summary statistic and the perturbation-diagnostic AUC three ways), `_auc_fail_vs_succ`. |
| `scripts/run_real_data.py` | Writes `real_data_curve_minima_v2.csv` and `real_data_strata_v2.csv`; prints the minima table, the three-way summary and the failure crosstab; docstring. |
| `phases/phase5b.py` | **Item 4**. `LHAT_CONSUMERS` (10), `SWEEP1_METHODS` (+ `constant_assumed`), `CASCADE_CLAMP_NOTE`; sweep 1a (cascade) rows kept and labelled with a `note` column; `sweep1_consumers` (sweep 1b) evaluates the 11 methods (+ the three other trivials as references) under the five modes on the same RNG streams as sweep 1a, per (method, mode, g, regime set) with capped cells excluded: `valid_rate`, `cat_rate`, `med_error`, `med_skill`, `med_skill_vs_*`, `win_rate_vs_*`, `n` (`phase5b_sweep1_consumers.csv`; per noise level in `phase5b_sweep1_consumers_noise.csv`); module docstring. |
| `scripts/run_phase5b.py` | Evaluation counts (1a + 1b), the sweep-1b printout, the clamp label line. |
| `reproduce_all.py` | `--plan` counts sweep 1b (Phase 5b full: 734,400 = 28,800 + 201,600 + 28,800 + 475,200). |
| `scripts/make_paper_tables.py` | "hindsight best-of-four (strict)" wording; three-way real-data analysis in `f07` (post-minimum cells marked `+`, pre/post footer rows) and `f08` (AUC rows for pre- and post-minimum targets), with `post_min_target` computed from `real_data_curves.csv` when the summary predates the column; facts for the curve minima and the three strata; when the new files / columns exist: fragment `f05b_sweep1_consumers_{core,holdout}.tex`, the validity-by-depth fact from the committed aggregate, fixed-reference win-rate facts, sweep-1b facts. |
| `FACTS.md`, `paper_fragments/*` | Regenerated on the committed (pre-5A) results with `--no-raw`: wording, the f07 / f08 three-way rows and 9 real-data facts; headers say `code 070f32d-dirty`. |
| `README.md`, `results/README.md` | Two kinds of skill; the input-dependence guard and the Weniger correction; the new Phase-5a / 5b / real-data files. |
| `REPORT_5A_quick_run.txt`, `REPORT_5A.md` | this run's log and this report (second commit). |

Not changed: `src/config.py`, the generators, Phases 1/3 logic, the dangerous derivation, the committed `results/`.

## 2. Weniger fix

Root cause as stated by the review, confirmed on the branch code before the fix: with `w_j = s_j` the numerator `sum_j (-1)^j C(k,j) beta_j s_j / w_j = sum_j (-1)^j C(k,j) beta_j` is the k-th difference of a degree-(k-1) polynomial in j, identically 0 for k = 1, 2, so `weniger_d1/d2` returned `0 / D = 0` for every input (the input-dependence test below shows it on a random walk too: 0 before, 8.80 / 12.00 after).

Geometric-limit recovery after the fix (`s_n = 0.3 + 0.5 * 0.9**n`, n = 31..90, `future_x` 1000, `L_hat` 0):

| method | estimate | abs. error |
|---|---|---|
| `weniger_d1` | 0.300000000000001 | 1.22e-15 |
| `weniger_d2` | 0.300000000000001 | 1.44e-15 |

(`levin_t1` / `levin_t2` on the same sequence: 1.22e-15 / 1.44e-15 -- see open question 1.)

Phase-0 analytic harness after the fix (quick-run worktree, `results/phase0_unit_tests.csv`; identical to a direct run in the main checkout into the scratchpad):

| case | weniger_d1 | weniger_d2 |
|---|---|---|
| GEOM (r = 0.92) | exact, PASS | exact (improve 3.1e11x), PASS |
| HARMONIC (threshold 5x) | 1.98x, fail | 2.95x, fail |
| LOGSLOW (threshold 2x) | 1.18x, fail | 1.28x, fail |
| ALTGEOM | exact, PASS | exact, PASS |
| **passes** | **2 of 4** (was 1 of 4: LOGSLOW only, because 0 happened to be 17.7x closer to the limit 0.01 than the current value) | **2 of 4** (was 1 of 4) |

Overall Phase 0: 63 of 204 passes (was 61). The committed `results/phase0_unit_tests.*` still show the pre-fix rows (estimate `-0.000000` on all four cases).

## 3. Input-dependence test: BEFORE and AFTER

`tests/test_input_dependence.py`: 96 windows (8 regimes `single_exp, two_exp, power_law, rational_decay, mixed_pow_rat, multiphase, osc_exp, log_slow` x 6 seeds x sigma in {0, 0.005}; n_obs 90, window 60, L_hat 0; generated with Phase 1's generator and RNG seeding), all 51 accelerators, `future_x` 500. (a) reaction = the output changes (|delta| > 1e-12, or a finite value turns NaN) under `seq * (1 + 0.01 U(-1, 1))` (fixed seed per window), rate over the windows where the base output is finite, threshold >= 90 %; (b) equality = the base output is within 1e-12 of the last value, window mean, window min or L_hat, rate over all 96 windows, threshold <= 10 %; `current_value` exempt. Three tests + one that checks `phases.phase5b.LHAT_CONSUMERS` against the measured list.

**BEFORE** (same test file run in a worktree of `070f32d`, the pre-fix code; `pytest -s`):

```
input-dependence over 96 windows (8 regimes x 6 seeds x sigma [0.0, 0.005]; n_obs 90, window 60, L_hat 0.0)
  method               finite  reaction  equality  flag
  weniger_d1               96     0.000     1.000  REACTION < 90%, EQUALS TRIVIAL > 10%
  weniger_d2               96     0.052     0.948  REACTION < 90%, EQUALS TRIVIAL > 10%
  current_value            96     1.000     1.000  exempt
  ... 49 of 51 accelerators react on >= 99.9 % of finite windows; 48 never equal a trivial
  flagged: reaction ['weniger_d1', 'weniger_d2']; equality ['weniger_d1', 'weniger_d2']
FAILED tests/test_input_dependence.py::test_every_accelerator_reacts_to_its_input
FAILED tests/test_input_dependence.py::test_no_accelerator_is_a_trivial_predictor
2 failed, 1 passed in 85.46s
```

`weniger_d2`'s 5 % reaction / 95 % equality are the floating-point residue of `0 / D` when D is tiny (the same 5 records that were not exactly 0 in Report 4). No other method is flagged; every other accelerator reacts on >= 99.9 % of its finite windows and never equals a trivial.

**AFTER** (`ddca12a`):

```
input-dependence over 96 windows (8 regimes x 6 seeds x sigma [0.0, 0.005]; n_obs 90, window 60, L_hat 0.0)
  method               finite  reaction  equality  flag
  current_value            96     1.000     1.000  exempt
  ... 51 of 51 accelerators react on >= 99.9 % of finite windows; 50 never equal a trivial
  flagged: reaction []; equality []
  L_hat consumers (10; output differs between L_hat = 0 and 0.5 * min(window)): log_linear (96), richardson_2 (96), richardson_3 (96), single_exp_fit (96), log_fit (95), double_exp_fit (94), richardson_1 (92), rational_fit (91), stability_weighted (88), median_ensemble (10)
3 passed in 108.26s
```

(`weniger_d1` / `weniger_d2` after the fix: reaction 1.000 / 1.000, equality 0.000 / 0.000, finite on 96 / 96 windows.)

**The printed L_hat-consumer list** (the number is the count of the 96 windows on which the output differs between L_hat = 0 and L_hat = 0.5 * min(window)):

> log_linear (96), richardson_2 (96), richardson_3 (96), single_exp_fit (96), log_fit (95), double_exp_fit (94), richardson_1 (92), rational_fit (91), stability_weighted (88), median_ensemble (10)

The review expected nine; the measured list has ten: `median_ensemble` is also a consumer (its pool contains the nine, so its median moves on the 10 windows where a consumer's value crosses it). `EXPECTED_LHAT_CONSUMERS` in the test and `phases.phase5b.LHAT_CONSUMERS` hold the measured ten (deviation 3).

## 4. Fixed-reference skill: one Phase-1 aggregate row (quick run)

`results/phase1/phase1_aggregated.csv` of the worktree, `rational_fit`, `single_exp`, sigma 0.005, g = 0.1 (2 seeds):

```
method rational_fit | regime single_exp | noise 0.005 | target_g 0.1 | n_f 148 | achieved_g 0.098274 | capped 0
valid_rate 1.0 | cat_rate 0.0 | beats_rate 0.5 | med_error 0.018719 | med_improve 1.077581
med_skill 1.744883                       <- hindsight best-of-four (strict), unchanged in value
med_skill_vs_assumed 1.5153 | win_rate_vs_assumed 0.5
med_skill_vs_last    1.5209 | win_rate_vs_last    0.5
med_skill_vs_wmean   0.2282 | win_rate_vs_wmean   1.0
med_skill_vs_wmin    1.5283 | win_rate_vs_wmin    0.5
```

The same eight columns appear in `phase1_global.csv` / `phase1_global_holdout.csv` (pooled over cells), `phase2_sweep_aggregated.csv`, `phase4_ensemble.csv`, every Phase-5a selector table and `phase5a_method_weights.csv`, and as `cascade_skill_vs_*` / `cascade_win_vs_*` in `real_data_summary_v2.csv` (per-row `skill_vs_*` / `win_vs_*` in `real_data_results_v2.csv`). Phase 5a, quick run, g = 0.1 core: `oracle_51` win rate vs assumed / last / wmin = 1.000 / 1.000 / 1.000; `fixed_rational` 0.500 / 0.625 / 0.625; `constant_assumed` 0.000 / 0.500 / 0.500 (`med_skill_vs_assumed` = 1.0000 by construction).

## 5. Sweep 1b: one consumer row (quick run)

`results/phase5b/phase5b_sweep1_consumers.csv` (220 rows = 11 methods x 5 modes x 2 strata x 2 regime sets), `log_linear`, mode `winmin`, g = 0.1, core:

```
assumed_mode winmin | method log_linear | target_g 0.1 | regime_set core | n 4 | n_capped_excluded 4
valid_rate 1.0 | cat_rate 0.0 | med_error 0.012913 | med_skill 1.349363
med_skill_vs_assumed 1.3494 | win_rate_vs_assumed 0.00
med_skill_vs_last    0.8237 | win_rate_vs_last    0.75
med_skill_vs_wmean   0.1573 | win_rate_vs_wmean   1.00
med_skill_vs_wmin    0.8237 | win_rate_vs_wmin    0.75
```

The runner's sweep-1b block (g = 0.1, core, pooled over noise; `median error / median skill / win rate vs constant_assumed`):

```
  method                                 zero                   half                 oracle                 double                 winmin
  richardson_1               0.0156/1.51/0.50       0.0156/2.42/0.50       0.0156/7.73/0.00       0.0156/2.46/0.50       0.0156/1.17/0.50
  richardson_2               0.0181/1.57/0.50       0.0181/2.27/0.50       0.0181/9.00/0.00       0.0181/2.30/0.50       0.0181/1.90/0.00
  richardson_3               0.0192/1.60/0.50       0.0192/2.19/0.50       0.0192/9.55/0.00       0.0192/2.22/0.50       0.0192/2.01/0.00
  single_exp_fit             0.0000/0.00/1.00       0.0000/0.00/1.00       0.0000/0.01/0.75       0.0000/0.00/1.00       0.0000/0.00/1.00
  double_exp_fit             0.0000/0.00/1.00       0.0000/0.00/1.00       0.0000/0.01/0.75       0.0000/0.00/1.00       0.0000/0.00/1.00
  rational_fit               0.0186/1.73/0.50       0.0186/2.69/0.50       0.0186/9.24/0.00       0.0186/2.73/0.50       0.0186/1.58/0.25
  log_fit                    0.0621/5.29/0.50       0.0621/7.56/0.00      0.0621/30.83/0.00       0.0621/7.67/0.50       0.0621/6.52/0.00
  log_linear                 0.0113/0.65/1.00       0.0068/0.64/1.00       0.0059/2.95/0.00       0.0106/1.81/0.00       0.0129/1.35/0.00
  stability_weighted         0.0075/0.54/0.75       0.0075/0.60/0.75       0.0075/3.75/0.00       0.0075/0.61/0.75       0.0075/1.10/0.50
  median_ensemble            0.0042/0.32/0.75       0.0042/0.38/0.75       0.0042/2.10/0.00       0.0042/0.39/0.75       0.0042/0.59/0.75
  constant_assumed           0.0396/2.38/0.00       0.0208/1.47/0.00       0.0020/1.00/0.00       0.0356/2.27/0.00       0.0096/1.00/0.00
```

Already visible on the quick grid (open question 2): apart from `log_linear` and `constant_assumed`, the consumers' median error is the same to four decimals under every mode -- the Richardson and curve fits use L_hat only as a `curve_fit` starting value -- while their skill and win columns swing because the reference `constant_assumed` itself changes with the mode (under `oracle` it returns L_true and nearly everything loses to it). The sweep-1a cascade rows are kept in `phase5b_sweep1_global/_regime.csv` with the column `note = "cascade features clamp L0 = max(0, min(L_hat, 0.5*min(window))); non-zero modes coincide when L_hat >= 0.5*min(window)"`; the clamp was not lifted.

## 6. Real data: curve minima and the pre/post-minimum split

Per dataset (`real_data_curve_minima_v2.csv`, from the committed `real_data_curves.csv`; the worktree run reproduces it exactly):

| dataset | rounds | argmin round | minimum | value at round 500 | rise from minimum |
|---|---:|---:|---:|---:|---:|
| covertype | 500 | 500 | 0.34916 | 0.34916 | +0.00 % |
| higgs | 500 | 500 | 0.53009 | 0.53009 | +0.00 % |
| adult | 500 | **337** | 0.27172 | 0.27298 | **+0.46 %** |
| jannis | 500 | 500 | 0.68955 | 0.68955 | +0.00 % |
| miniboone | 500 | 500 | 0.13712 | 0.13712 | +0.00 % |
| bank_marketing | 500 | **257** | 0.19853 | 0.20116 | **+1.32 %** |

Failure crosstab on the 90 cells (`post_min_target = target_round > argmin_round`; failure = cascade skill >= 1, which on these curves is the same event as improvement < 0 because `last_value` is the best trivial on 88 of 90 cells); computed from the committed summary and reproduced by the worktree run:

| | fail | ok | all |
|---|---:|---:|---:|
| pre-minimum targets | **2** | 63 | 65 |
| post-minimum targets | **15** | 10 | 25 |
| all | 17 | 73 | 90 |

As the review expected: 2/65 vs 15/25. The three-way summary (`real_data_strata_v2.csv`, printed by the runner):

| stratum | cells | fail | rate | median skill | median improvement | win vs assumed | win vs last | perturbation AUC | p | ordering |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| all | 90 | 17 | 0.189 | 0.336 | +0.664 | 1.000 | 0.811 | 0.317 | 0.019 | failing < succeeding |
| pre-minimum | 65 | 2 | 0.031 | 0.268 | +0.732 | 1.000 | 0.969 | 0.198 | 0.175 | failing < succeeding |
| post-minimum | 25 | 15 | 0.600 | 1.345 | -0.345 | 1.000 | 0.400 | 0.433 | 0.598 | failing < succeeding |

So the Report-4 "inverse ordering" of the perturbation diagnostic (AUC 0.317, p = 0.019 pooled) is largely the pre/post composition: within post-minimum targets the AUC is 0.433 (p = 0.60), i.e. the diagnostic does not separate the failing cells there either, and within pre-minimum targets there are only two failures. The paper's real-data story is a pre/post-minimum story, not a diagnostic story. `f07` marks the 25 post-minimum cells with `+` and adds pre/post footer rows; `f08` carries the two stratum rows; `FACTS.md` has the minima and the three strata.

## 7. pytest

`python -m pytest tests/ -q` at `ddca12a`: **348 passed in 108.8 s** (342 before + `test_weniger_recovers_geometric_limit` x 2 + the 4 input-dependence tests; `test_input_dependence.py` accounts for ~100 s of the wall time because it evaluates 51 methods x 96 windows x 3 calls).

## 8. `python reproduce_all.py --quick` (worktree `../wt-5a` at `ddca12a`)

```
========================================================================
  PIPELINE SUMMARY  [QUICK]
========================================================================
   #  step                                             status       time
  ----------------------------------------------------------------------
   1  Phase 0 — analytic unit tests                    OK             2s
   2  Phase 1 — main benchmark                         OK            21s
   3  Dangerous re-derivation (Phase 1 -> artifact)    OK             1s
   4  Phase 2 — failure detection                      OK             9s
   5  Phase 3 — adaptive selection                     OK             3s
   6  Phase 4 — perturbation diagnostics               OK             8s
   7  Phase 5a — ensemble ablation                     OK            40s
   8  Phase 5b — sensitivity sweeps                    OK            72s
   9  Real data — recorded-curve re-evaluation         OK             6s
  ----------------------------------------------------------------------
      total                                                         162s
  All steps completed successfully.  Results are in results/
========================================================================
```

Phase 5b went from 28 s (Report 3A quick) to 72 s with sweep 1b (2,240 extra calls on the quick grid; 201,600 on the full grid, roughly 10-20 min at full scale). New files present in the worktree: `phase5a_validity_by_depth.csv` (224 rows), `phase5b_sweep1_consumers.csv` (220) and `_noise.csv`, `real_data_curve_minima_v2.csv` (6), `real_data_strata_v2.csv` (3). `scripts/make_paper_tables.py --no-raw` on the worktree results: 22 fragments (the two `f05b` files appear) and 188 facts, including the validity-by-depth fact from the committed aggregate and the win-rate facts. Quick-run dangerous artifact: 8 methods (same as Report 3A's quick set); `check_dangerous.py` MATCH.

## 9. Deviations and open questions

1. **Committed but not pushed.** A worktree checks out a commit, so the code was committed (`ddca12a`; this report is a second commit). The prompt did not ask for a push; `origin/redesign-v2` is still at `070f32d`.
2. **The committed `results/` are stale.** They come from the pre-5A code (`0375f24`); the Weniger fix changes every phase's numbers (Phase 1 records for `weniger_d1/d2`, the Phase 2/3/4 pool that contains `weniger_d2`, sweep 3, the artifact's table), and the new columns / files exist only in the worktree's quick output. Per the results policy the whole set needs a full re-run (~4.5 h + sweep 1b) before any number is quoted; `FACTS.md` and the fragments in this commit are still derived from the stale results (their headers say so).
3. **Ten L_hat consumers, not nine**: `median_ensemble` reacts on 10 of 96 windows (section 3). It is included in `LHAT_CONSUMERS` and in sweep 1b. If the review meant the direct consumers only, drop it from both.
4. **Reaction criterion**: a finite estimate that turns NaN under the perturbation counts as a reaction. Without that rule the six dangerous methods (`neville_2/3/4`, `pade_21/32`, `geom_avg_diff`) fail (a) at 0.00-0.90 because the perturbed window pushes them out of the validity range -- the opposite of insensitivity.
5. **Win-rate denominator**: `win_rate_vs_*` is the mean of the 0/1 win indicator over all records of the slice (an invalid method never wins), so it is a deployable statement; `med_skill_vs_*` is the median over records where the method is valid. Both are documented in `src/trivial.py`.
6. **Real-data stratification in the paper generator** uses the curves file directly, so `f07` / `f08` / the facts are already three-way on the committed results; the runner's `real_data_strata_v2.csv` will duplicate them after the full run.

Open questions:

1. **`weniger_d1/d2` are now numerically identical to `levin_t1/t2`.** With the d-type remainder and Pochhammer weights, `(x)_{k-1} = x^{k-1}` for k <= 2, which is exactly the Levin `t` weight; on the 96 test windows the pairs agree to machine precision on 96/96 windows (max |diff| 0). The roster therefore has two duplicate pairs. Options: raise the Weniger orders to 3 and 4 (where the Pochhammer and power weights differ), keep one pair, or keep both and say so in the paper. Not changed here.
2. **Sweep 1b shows that L_hat barely enters the accelerators** (quick grid): the Richardson and curve fits use it only as a starting value, so their errors are mode-invariant to four decimals; `log_linear` (offset) and `constant_assumed` are the only methods whose predictions move. If that holds on the full grid, the sensitivity claim becomes "the assumed asymptote matters only through the trivial comparator and `log_linear`", and the interesting quantity is the win rate against `constant_assumed` under the `winmin` / `half` modes rather than the method errors.
3. **Sweep 1a clamp**: kept as instructed; the `note` column documents it. Whether the paper should show sweep 1a at all is a paper decision.
4. **Phase-0 thresholds**: after the fix the Weniger pair passes 2 of 4 (exact on the two geometric cases, 1.98x / 2.95x on HARMONIC against a 5x threshold). The harness's thresholds were set for the v1 roster; nothing was retuned.
5. **The `post_min_target` split explains the real-data failures** (15 of 17 failures are post-minimum), which raises the design question of whether post-minimum targets belong in the real-data grid at all, or whether the deployment recommendation should be conditioned on a "minimum not yet passed" test that can be evaluated without the future. Not addressed here.

Stopped here.
