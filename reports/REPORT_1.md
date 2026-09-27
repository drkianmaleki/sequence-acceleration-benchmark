**Report back**

# Redesign v2, Prompt 1: benchmark infrastructure

Date: 2026-09-19. Repository: `code/` (the git repo; the Dropbox folder above it is not a repository).

## 1. Branch and commit state

| item | value |
|---|---|
| branch | `redesign-v2`, created from `main` at `d5dc041` |
| code commit | `c1b2cc5` "Redesign v2 (Prompt 1): hidden heterogeneous asymptotes, assumed-L modes, trivial comparators, gap-stratified horizons, held-out regimes, regression tests" |
| this report | committed separately on top of `c1b2cc5` (hash printed in the terminal) |
| working tree | clean after each commit; nothing under `results/` was modified or regenerated |
| remote | not pushed |

Baseline before any change: 78 tests passing on `main`. After: 331 passing.

## 2. Files created or modified

Created:

| file | purpose |
|---|---|
| `src/asymptote.py` | `assumed_asymptote(L_true, window, mode)` computes L_hat for the five `ASSUMED_L_MODES`; only "oracle" returns L_true. |
| `src/trivial.py` | The five trivial comparators, `ORACLE_METHODS`, `SKILL_REFERENCE_METHODS`, `skill_score`, `best_reference_error`. |
| `src/horizons.py` | `horizon_for_gap(regime, n_obs, g, seed)` forward search on the noiseless gap with the 50,000 cap; `horizon_table`, `format_horizon_table`. |
| `tests/test_redesign.py` | 138 tests: determinism, gap preservation, horizon inversion, the flaw regression, trivial methods, holdout contracts, oracle isolation, evaluation-loop end-to-end. |
| `tests/fixtures/legacy_means_d5dc041.csv` / `.npz` | Noiseless means of the 18 regimes captured from the unmodified code (commit d5dc041) on a 308-point grid; ground truth for the legacy-equality test. |
| `REPORT_1.md` | this file |

Modified:

| file | change |
|---|---|
| `src/generators.py` | Rewritten: every regime is a gap function; `mean(n, L*) = L* + gap(n)`; `true_asymptote(regime, seed, mode)`; `regime_functions`; six held-out regimes; `TRUTH`/`GENERATORS` now require `L_star`. |
| `src/config.py` | Added `ASYMPTOTE_MODE`, `L_TRUE_RANGE`, `LEGACY_L_INF`, `ASSUMED_L_MODE(S)`, `HORIZON_GAP_FRACTIONS`, `HORIZON_N_CAP`, `HEADLINE_G`. Removed `L_INF`. |
| `src/evaluation.py` | Main loop rewritten for gap-stratified horizons, hidden L_true, L_hat, trivial comparators, skill, holdout flag, oracle-excluded ranks; new output files. |
| `src/accelerators.py` | `cfg.get("L_inf", 0.01)` (7 sites) replaced by `_assumed_L(cfg)` which raises `KeyError` when absent; trivial methods registered (56 methods); `last_value` is the `current_value` function object. |
| `src/diagnostics.py` | Same removal of the hidden 0.01 default in `extract_features`. |
| `src/trajectories.py` | `process_curves(..., assumed_mode=...)` computes L_hat per window; `_DEFAULT_CFG` no longer carries an asymptote; `apply_accelerator` requires `cfg['L_inf']`. |
| `src/plots.py` | Horizon key is `target_g`; oracle rows dropped from ranking figures; `trivial` family colour. |
| `phases/phase2.py`, `phase4.py`, `phase5a.py` | Per-seed `regime_functions`; every `_cfg(...)` and feature call receives L_hat; L_true/L_hat recorded (phase-2 features, phase-4 records). |
| `phases/phase5b.py` | Same wiring; sweep 1 now sweeps `ASSUMED_L_MODES` instead of numeric L_inf values; its figure is categorical. |
| `scripts/run_phase1.py` | Gap strata instead of fixed horizons; `--assumed-mode`, `--no-holdout`, `--out-dir`; prints horizon table, trivial comparators, oracle-unranked top-20. |
| `scripts/run_phase5b.py` | `assumed_modes` replaces `l_inf_values` in quick/full configs and summaries. |
| `scripts/run_real_data.py` | `ASSUMED_MODE = config.ASSUMED_L_MODE` replaces `L_INF = 0.01`. |
| `scripts/analyze_real_diagnostics.py` | Comment only: it reproduces the stored pre-redesign run and keeps the explicit legacy 0.01. |
| `tests/test_generators.py` | Updated to the new signatures; "converges to shared asymptote" replaced by "converges to its own L_true"; holdout regimes included. |
| `tests/test_accelerators.py` | Phase-0 harness config gains `L_true` so the oracle comparator is valid on the analytic cases. |
| `.gitignore`, `results/README.md` | `results/phase1/phase1_records.csv` (per-seed records) excluded from version control and documented. |

Not touched: `README.md` (still describes the old design), committed `results/`, `reproduce_all.py`, `scripts/make_paper_tables.py`, `DANGEROUS_METHODS` (comment added that it is stale until the Prompt-2 run).

## 3. Design decisions, ambiguities resolved, deviations from the prompt

1. **REJECTION_CONTEXT does not exist.** No file by that name exists anywhere in the project or the archive clones. I used the (a)/(b)/(c) paragraph in the prompt as the rejection context and read `documents/submission/07-24-2026/revision_notes/FEEDBACK_ANALYSIS.md` (the pre-submission expert review) for background.
2. **Legacy equality is to floating-point re-association, not bit-identical.** `L* + (a + b + c)` differs from the old `((L* + a) + b) + c` by at most 1.11e-16 (six multi-term regimes; the rest are bit-exact). The test tolerance is 1e-15. The fixture was captured from the unmodified code before any edit.
3. **Registries require the asymptote.** `TRUTH[r](n, L_star, seed=None)` and `GENERATORS[r](n, rng, sigma, L_star, seed=None)`. The old three-argument call raises `TypeError` so no caller can silently keep a shared constant. `regime_functions(regime, seed)` returns `(gen, truth, L_true)` closures with the old call shapes for the phase loops.
4. **L_hat lives in a new module** `src/asymptote.py` rather than `config.py`, which is constants-only. The prompt named `config.py` for the mode; the constants are there, the function is next door.
5. **The hidden `0.01` default inside methods was removed.** Accelerators, diagnostics and trajectories used `cfg.get("L_inf", 0.01)`, which was a third route by which the true constant reached methods. It now raises `KeyError`. A test covers it.
6. **Oracle plumbing.** `constant_oracle` reads `cfg['L_true']`, which only the evaluation harness sets. A test runs all 55 other methods with and without that key and asserts identical outputs.
7. **Phase-0 harness consequence.** `tests/test_accelerators.py` now supplies `L_true = 0.01` so the oracle is valid there. Because the analytic cases have limit 0.01, `constant_assumed` and `constant_oracle` trivially pass those cases. `results/phase0_unit_tests.*` were not regenerated.
8. **Skill edge cases.** Denominator below 1e-12 gives skill 1.0 if the method is also exact and +inf otherwise; NaN when the method is invalid or no reference is valid. Medians keep +inf.
9. **Horizon "within 1%".** True for the slow regimes (per-step decay below 1% beyond n = 90). For exponential regimes integer quantisation sets the residual (single_exp at g = 0.5 achieves 0.4868), so the test checks the exact first-crossing property for all 19 monotone regimes and the 1% criterion for the 9 slow ones. Oscillatory regimes use the first crossing. Staircase is a step function: g = 0.5 achieves 0.278; g = 0.1 and 0.02 land at n = 320 where the gap is exactly 0.
10. **Cap handling.** Capped cells carry `n_f = 50000`, `capped = 1` and the achieved fraction. At n_obs = 90, seed 0: 6 of 72 cells cap (table below). `random_knots` caps at g = 0.02 for 14 of 30 seeds because its shape is per seed.
11. **random_knots is seeded by the sequence seed** (a family of curves, one per seed): knots drawn without replacement from {20..200} and sorted, four exponents in U[0.3, 0.9] for the four segments, segment constants chosen for continuity with gap(0) = 0.7. Its horizon therefore depends on the seed; per-record `n_f` is per seed and the printed table uses seed 0.
12. **real_boot construction.** Smoothing spline of the recorded loss against log(n+1) (lambda 0.01: keeps the steep start, removes round-to-round noise), lower envelope by running minimum (the adult curve overfits after round ~331), rescaled to gap(0) = 0.7 with the envelope's final value as floor, gap identically 0 beyond the 500 recorded rounds. L_true is therefore the exact limit, reached at round ~331 (adult) and 499 (higgs). Loaded lazily from `results/real_data/real_data_curves.csv`.
13. **logistic_tail** clips its exponent at 700 to avoid overflow warnings; values are unchanged.
14. **Clip-at-zero in the intrinsic-noise regimes was kept.** With L_true as low as 0.005, the Laplace noise of `heavy_noise_plat` (scale 0.004) is clipped for roughly 14% of tail draws and the Gaussian of `noisy_plateau` for roughly 5%, which biases observations upward relative to TRUTH. The prompt says gap shapes must not change; I also left the noise models alone. Open question 2.
15. **Sequence generation in the main loop only covers the observed prefix** (n up to obs_idx). Methods see the window and the target comes from the noiseless truth, so nothing else was ever used. Window realisations are identical to before for the Gaussian generators.
16. **Phases 2, 4, 5a, 5b received only the L_true / L_hat wiring.** They still evaluate at fixed indices (150, 1000, 5000) and aggregate by `future_idx`. Converting them to gap strata is a separate redesign of their aggregations and figures and is left for Prompt 2. Phase-5b sweep 1 was necessarily redesigned (it swept numeric L_inf against a known true 0.01); it now sweeps the five modes. `scripts/make_paper_tables.py` still reads the old `L_inf_assumed` column from the committed CSVs and will need updating when results are regenerated.
17. **Real-data path.** `process_curves` takes `assumed_mode`; modes that need L_true raise on real curves. `analyze_real_diagnostics.py` keeps its explicit 0.01 because it asserts fidelity against the stored pre-redesign results.
18. **New Phase-1 outputs.** `phase1_records.csv` (every per-seed record; gitignored, about 5 MB in quick mode, roughly 90 MB for a full run), `phase1_global_holdout.csv`, `phase1_horizons.csv`, heatmaps keyed by g. `phase1_raw_sample.csv` is gone (superseded by the records file). The global ranking pools the 18 core regimes; oracle rows are present with `rank = NaN`; `regime_best` excludes the oracle and adds `best_by_skill`, `best_skill`, `oracle_med_error`. In aggregated tables `n_f` is the median over seeds and `capped` means any seed capped (only matters for random_knots).
19. **`HEADLINE_G = 0.1`** is used for console summaries and figures only; it is not a paper decision (open question 1).
20. **Phase-2 feature CSV gains `L_true` and `L_hat` columns.** Phase 3 selects its feature columns explicitly, so this is safe (verified).
21. **`--out-dir` on `run_phase1.py`** was added so the smoke run did not overwrite committed results.
22. **Old `FUTURE_IDX_*` constants remain** for the legacy phase runners and the flaw regression test, with a comment not to use them for headline claims.

## 4. Test summary

Full suite on the final tree:

```
python -m pytest tests/ -q
331 passed in 10.05s
```

Breakdown: `test_accelerators.py` 4, `test_generators.py` 189 (parametrised over 24 regimes and 4 asymptote values), `test_redesign.py` 138.

Measured values printed by the regression tests (`pytest tests/test_redesign.py -s -k "referee_flaw or legacy_mode_reproduces"`):

```
LEGACY GAP CHECK: max |old mean - (0.01 + gap)| = 1.110e-16
FLAW REGRESSION: median |L_true - target(n_f=5000)| over 18 legacy regimes = 0.001030  (referee: 0.00103)
```

The flaw test asserts the median within 1% of 0.00103 and that at least 9 of 18 regimes sit within 1.1e-3 of the constant. A companion test asserts that under the redesign (hetero L_true, mode "zero") the `constant_assumed` error at n_f = 5000 has median above 0.03 (measured: the whole target level, about the median L_true of 0.05).

Additional verification, not part of the suite:

- Smoke run of phases 2, 4, 5a and 5b (sweeps 1, 2, 3 and the sweep-1 figure) on tiny grids into the scratchpad: all ran, 28 s, L_true/L_hat columns present where added.
- `scripts/run_phase1.py --quick --out-dir <scratchpad>` (24 regimes, 5 seeds, sigma 0, three strata, 20,160 method evaluations): ran end to end, six figures, nine CSVs, zero warnings. Preview at g = 0.1 with L_hat = 0: `constant_oracle` median error 0.0115 and skill 0.20 (shown, unranked); `constant_assumed` median error 0.0669, skill 1.0; rank 1 `single_exp_fit` (skill 1.17), rank 2 `rational_fit` (skill 0.60), rank 3 `richardson_a10` (skill 1.39). Five seeds at sigma 0 are not a result, only evidence that the pipeline runs.
- Real-curve pipeline on the recorded six curves: modes "zero" and "winmin" produce 18 finite predictions each; "oracle" raises as intended.

## 5. Horizon table, n_obs = 90

Cell = `n_f (achieved g)`. Capped cells are bold with the achieved fraction. Seed 0 for `random_knots`; its per-seed spread is given below the table.

| regime | g = 0.5 | g = 0.1 | g = 0.02 |
|---|---|---|---|
| single_exp | 108 (g = 0.4868) | 148 (g = 0.0983) | 188 (g = 0.0198) |
| two_exp | 129 (g = 0.4956) | 218 (g = 0.0999) | 308 (g = 0.0198) |
| three_exp | 176 (g = 0.4988) | 377 (g = 0.0999) | 579 (g = 0.0198) |
| four_exp | 231 (g = 0.4981) | 631 (g = 0.0998) | 1033 (g = 0.0200) |
| power_law | 244 (g = 0.4999) | 2441 (g = 0.1000) | 24329 (g = 0.0200) |
| rational_decay | 205 (g = 0.5000) | 1125 (g = 0.1000) | 5725 (g = 0.0200) |
| mixed_pow_rat | 254 (g = 0.4990) | 2260 (g = 0.1000) | 27113 (g = 0.0200) |
| multiphase | 504 (g = 0.4998) | 3852 (g = 0.1000) | 20594 (g = 0.0200) |
| osc_exp | 107 (g = 0.4783) | 165 (g = 0.0957) | 222 (g = 0.0197) |
| damped_osc_pow | 260 (g = 0.4994) | 3014 (g = 0.1000) | 39688 (g = 0.0200) |
| log_slow | 8594 (g = 0.5000) | **50000 CAP** (achieved g = 0.419) | **50000 CAP** (achieved g = 0.419) |
| delayed_plateau | 160 (g = 0.4966) | 321 (g = 0.0993) | 482 (g = 0.0198) |
| staircase | 170 (g = 0.2778) | 320 (g = 0.0000) | 320 (g = 0.0000) |
| noisy_plateau | 104 (g = 0.4966) | 137 (g = 0.0954) | 169 (g = 0.0193) |
| slow_osc_power | 415 (g = 0.4984) | 3483 (g = 0.1000) | 39003 (g = 0.0200) |
| log_oscillatory | 533 (g = 0.4984) | **50000 CAP** (achieved g = 0.299) | **50000 CAP** (achieved g = 0.299) |
| heavy_noise_plat | 97 (g = 0.4966) | 114 (g = 0.0907) | 130 (g = 0.0183) |
| broken_power_law | 514 (g = 0.4999) | 28776 (g = 0.1000) | **50000 CAP** (achieved g = 0.080) |
| stretched_exp | 193 (g = 0.4983) | 579 (g = 0.0998) | 1172 (g = 0.0200) |
| logistic_tail | 105 (g = 0.4775) | 135 (g = 0.0961) | 164 (g = 0.0194) |
| inv_sqrt_log | 245 (g = 0.4998) | 2929 (g = 0.1000) | 41315 (g = 0.0200) |
| random_knots | 400 (g = 0.4994) | 6723 (g = 0.1000) | **50000 CAP** (achieved g = 0.032) |
| real_boot_a | 125 (g = 0.4971) | 209 (g = 0.0986) | 273 (g = 0.0200) |
| real_boot_b | 151 (g = 0.4968) | 335 (g = 0.0992) | 444 (g = 0.0200) |

Capped: 6 of 72 cells. `random_knots` over 30 seeds: g = 0.5 has n_f in [203, 708], none capped; g = 0.1 has n_f in [1335, 50000], 1 capped; g = 0.02 has median n_f 45170, 14 of 30 capped.

## 6. Sample L_true draws

| regime | seed | L_true |
|---|---|---|
| single_exp | 0 | 0.06736 |
| single_exp | 1 | 0.00785 |
| power_law | 29 | 0.00776 |
| log_slow | 0 | 0.13535 |
| staircase | 12 | 0.08085 |
| heavy_noise_plat | 5 | 0.01489 |
| broken_power_law | 17 | 0.00897 |
| stretched_exp | 0 | 0.02227 |
| random_knots | 3 | 0.19248 |
| real_boot_b | 29 | 0.01251 |

Over all 24 regimes and 30 seeds: 720 distinct values, min 0.0050, median 0.0510, max 0.4975.

## 7. Open questions for Kian / Claude

1. **Headline stratum.** Which g carries the paper's headline (0.1 is the console default here)? And are capped cells reported as their own row, or excluded from pooled rankings?
2. **Clip-at-zero bias at small L_true** in `noisy_plateau` and `heavy_noise_plat` (decision 14). Keep the observation model, or scale the intrinsic noise with L_true?
3. **random_knots as a per-seed family** (decision 11). Keep, or freeze one shape per regime? Its g = 0.02 stratum caps for 14 of 30 seeds.
4. **Phases 2, 4, 5a, 5b still use fixed-index horizons.** Convert to gap strata in Prompt 2? `DANGEROUS_METHODS` must be re-derived after the new Phase-1 run, and `make_paper_tables.py` must be updated for the new column names.
5. **Phase-0 harness** now lists the trivial comparators as passing the analytic cases (their limit is 0.01). Exclude trivial methods from Phase 0, or keep them as a visible reminder?
6. **Default mode.** "zero" is the deployment-honest default as instructed. "winmin" is the other deployable mode; should the main run report both?
7. **Skill denominator** is the best of four trivial predictors per cell, a small per-cell selection. Acceptable, or should skill be against a single fixed reference as well?
8. **Stability-score saturation.** At sigma 0 most methods tie at S = 1.4, so the per-regime "best" column is order-dependent (pre-existing; FEEDBACK_ANALYSIS 2.6). `best_by_skill` is provided; should it become the primary per-regime recommendation?
9. **Noise scale versus L_true.** Absolute sigma up to 0.02 now exceeds the smallest asymptotes (0.005). Intended?
10. **Per-seed records file** (~90 MB for a full run) is gitignored. Commit a compressed copy, or keep regenerable?
11. **Documentation.** `README.md` and `reproduce_all.py` still describe the old design; rewrite after the Prompt-2 run?

Stopped here; nothing from Prompt 2 was started.
