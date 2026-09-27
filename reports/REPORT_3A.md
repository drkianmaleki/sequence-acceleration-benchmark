**Report back**

# Redesign v2, Prompt 3A: the Report-2 review decisions applied

Date: 2026-09-19/20. Branch `redesign-v2` in `code/`. Code commit `8d33e82` on top of `b43449a` (Report 2); the verification below ran in a detached git worktree of `8d33e82`. One five-line follow-up (shard-directory cleanup retry, section 8) and this report are committed on top. No method implementation was changed. The full pipeline was not run; the full Phase-5a grid was started only long enough to obtain the first-chunk timing (section 3) and then stopped.

## 1. Files changed

| file | change |
|---|---|
| `phases/phase5a.py` | Decision 1. The `(obs_idx, sigma)` loop body is now `evaluate_block(...)`, a self-contained function (the two RNG streams are created per `(obs_idx, sigma, seed)` as before; the generated sequence length is fixed by `n_max = max(obs_idx_list)` so the noise draws do not depend on which blocks share a process). `run_phase5a(jobs, keep_shards)` builds the block list in serial order, runs it in-process (`jobs=1`) or in a `spawn` process pool, writes one shard per block to `<out_dir>/shards/`, prints the first completed block's timing and the projected evaluation wall time at the chosen job count, then concatenates the block frames in serial order into `phase5a_raw.csv` (shards deleted unless `keep_shards`). `run_all` passes the two arguments through. |
| `scripts/run_phase5a.py` | `--jobs N` (default `cpu_count() - 1`, here 7; `1` = serial) and `--keep-shards`; header line with the job count and block count. |
| `src/config.py` | `RANK_MIN_VALID = 0.9` (decision 3). |
| `src/pipeline.py` | `assign_ranks(df, metric, group_cols, ascending)`: rank-eligible = not oracle, finite metric, `valid_rate >= RANK_MIN_VALID`; ranks within groups with the method name as tie-break; adds `rank` (NaN when ineligible) and `rank_eligible`. `unranked_block(df, cols)`: the below-floor rows (oracle excluded), `valid_rate` always shown. `git_head()` for provenance fields. |
| `src/evaluation.py` | Phase 1 `_pool` uses `assign_ranks`; global tables carry `rank_eligible`; new `phase1_unranked.csv` (core + holdout) and `results['unranked']`. |
| `scripts/run_phase1.py` | TOP-20 block lists rank-eligible rows (oracle shown with `-`); new UNRANKED block (below the floor, with `valid_rate`). |
| `phases/phase2.py` | Decision 4: `RANK_POOL = PHASE2_BASE_METHODS` (9); `UNRANKED_COMPARATORS = [constant_assumed, constant_oracle]`. Decision 3: per-cell rank pool restricted to methods with `valid_rate >= RANK_MIN_VALID`; new phase-diagram columns `richardson_valid_rate`, `richardson_below_floor`, `n_rank_pool`, `n_rank_eligible`; `richardson_rank` / `richardson_err_rank` are NaN when Richardson is below the floor. Reported alongside, unranked: `constant_assumed_{stability,med_error,valid_rate}`, `constant_oracle_{stability,med_error}`. Figure P2-1 legend updated (blank = below floor). |
| `scripts/run_phase2.py` | Header prints the 9-method rank pool and the floor; the per-regime table gains a "Below floor" column (NaN-safe min rank); new UNRANKED CELLS block; one line with the alongside comparators. |
| `phases/phase3.py` | Docstring only: `CANDIDATES = RANK_POOL` is now the 9 methods (see deviation 2). |
| `phases/phase5b.py` | Sweep-3 global table ranked by stability through `assign_ranks` (per `cat_mult` x `target_g`; `rank`, `rank_eligible`); new `phase5b_sweep3_unranked.csv`; the Kendall-tau concordance is computed over rank-eligible methods only. |
| `scripts/run_phase5b.py` | Sweep-3 printout: top-5 ranked and the unranked block at the headline stratum. |
| `src/dangerous.py` | Decision 2: `derive_dangerous` scores every non-oracle method but `eligible = method in ACCEL_METHODS`; `dangerous = eligible and S < 0`; the table carries `eligible`. `write_artifact` drops non-eligible rows from the artifact's `table`, refuses a non-accelerator in `dangerous` (`ValueError`), and records `pool = "accelerators"`, `n_pool = 51`. |
| `scripts/derive_dangerous.py` | Prints "55 scored = 51 accelerators (eligible) + 4 trivial comparators (scored for the record, never in the artifact)"; trivial rows are flagged `trivial (scored only; not eligible, not in artifact)`. |
| `src/trajectories.py` | Decision 5: `perturb_seed(dataset, depth)` (crc32 of `"dataset:depth"`, as in the legacy script) and `perturb_iqr_real(...)` (`PERTURB_TRIALS` = 5 evaluations on windows x `(1 + 0.02 U(-1,1))`, IQR of the finite estimates, NaN below three). `evaluate_recorded_curves` computes it for the routed method at every `(dataset, depth, target)` cell, writes it to every row of the cell in `df_long` and to `df_summary`, and adds `phase2_features_path`, `phase2_features_rows`, `git_head` to every summary row. |
| `scripts/run_real_data.py` | Summary prints the perturb_iqr statistics (per routed method) and the provenance block; docstring. |
| `scripts/analyze_real_diagnostics.py` → `scripts/analyze_real_diagnostics_legacy.py` | Decision 6: `git mv`; docstring note that it verifies the stored pre-redesign 18-cell run at the legacy assumed asymptote 0.01 and is not part of `reproduce_all.py`. References updated in `README.md` (the only ones outside historical reports). |
| `README.md` | The legacy script's new name and role; the v2 `perturb_iqr` column. |
| `.gitignore` | `results/phase5a/shards/`. |
| `tests/test_pipeline_v2.py` | Dangerous test rewritten with real method names and a trivial with S < 0 (scored, never dangerous, never written; `write_artifact` refuses it). Phase-2 test: 9-method pool, alongside columns, floor semantics, `phases.phase3.CANDIDATES == RANK_POOL`. Real-data test: `perturb_iqr` present, one value per cell, bit-exact re-derivation from `perturb_iqr_real` with the crc32 seed, provenance columns (0 rows / `unknown` mapping when the feature file is absent; the file's row count when present). New: `assign_ranks` / `unranked_block` unit test; Phase-1 floor test (`phase1_unranked.csv`); Phase-5a serial-vs-`jobs=2` byte-identity test including the shard concatenation and shard cleanup. |
| `tests/test_redesign.py` | The Phase-1 rank assertion now states the floor rule (`rank_eligible` equals the three-condition mask; ineligible rows unranked). |
| `REPORT_3A_quick_run.txt` | Complete console log of `python reproduce_all.py --quick` in the worktree (1,259 lines). |
| `REPORT_3A.md` | this file |

## 2. Serial-vs-parallel identity check (decision 1)

Quick grid (`config.PHASE5A["quick"]`: 2 depths x 2 noise = 4 blocks of 24 cells, 5,376 rows), run four ways from the same `results/phase1/dangerous_methods.json`:

| run | command | rows | SHA-256 of `phase5a_raw.csv` |
|---|---|---|---|
| new code, serial | `run_phase5a.py --quick --jobs 1 --keep-shards` | 5,376 | `574c35b31ed37f9cb517…` |
| new code, 2 jobs | `run_phase5a.py --quick --jobs 2` | 5,376 | `574c35b31ed37f9cb517…` |
| new code, 4 workers (inside `reproduce_all.py --quick`, default `--jobs 7` capped at 4 blocks) | | 5,376 | `574c35b31ed37f9cb517…` |
| pre-change serial loop, commit `b43449a` (worktree) | `run_phase5a.py --quick` | 5,376 | `574c35b31ed37f9cb517…` |

`cmp` of the unsorted files: identical. `diff <(sort serial) <(sort jobs2)`: no differences. The textual concatenation of the four serial shards (`phase5a_raw_part000_obs60_sigma0.csv` … `part003_obs90_sigma0.005.csv`, header once) equals the raw file byte for byte. All eight aggregated Phase-5a tables (`phase5a_ensemble`, `_ensemble_holdout`, `_by_sigma`, `_per_regime`, `_ablation`, `_method_weights`, `_oracle_gap`, `_capped`) are byte-identical between the serial and the 2-job run. The unit test `test_phase5a_block_parallel_byte_identical` repeats the serial-vs-2-jobs comparison on a 448-row grid in the suite.

Why it holds by construction: every RNG stream is seeded per `(obs_idx, sigma, seed)`, so a block never consumes draws from another block; the only cross-block coupling in the old loop was the generated sequence length `max(obs_idx_list) + 1`, which is now passed to every block as `n_max`; the raw table is `pd.concat` of the block frames in the serial block order, and pandas formats floats element-wise.

## 3. First-chunk timing and projected Phase-5a wall time

The printout is produced by `run_phase5a` when the first block completes (quick run, inside `reproduce_all.py --quick`, 4 blocks on 4 workers, measured while a second Phase-5a process was running):

```
  [  1/4] block obs=90   sigma=0.005     1,344 rows  compute    30.4 s  wall    31.9 s

  FIRST CHUNK: obs=90 sigma=0.005: 24 cells, 1,344 rows in 30.4 s compute (1267.4 ms/cell), 31.9 s wall since start
  PROJECTION  : 4 chunks / 4 jobs = 1 round(s) x 31.9 s = 31.9 s evaluation wall time at --jobs 4  (serial equivalent 2.0 min; equal-cost chunks assumed; aggregation and figures extra)
```

(the four blocks finished at 31.9 / 33.7 / 39.9 / 43.5 s wall; the whole step, with aggregation and figures, took 48 s.)

**Full grid at the default job count.** `python scripts/run_phase5a.py --full --jobs 7` was started in the worktree (`config.PHASE5A["full"]`: 4 depths x 3 noise = 12 blocks of 1,440 cells = 20 seeds x 24 regimes x 3 strata; 967,680 central + 4,406,400 perturbation calls) on the 8-core machine (Intel Core Ultra 7 258V, 8 physical cores, no SMT), alone, and stopped by a watcher one minute after the first chunk reported:

```
  [  1/12] block obs=90   sigma=0        80,640 rows  compute  57.3 min  wall  57.4 min

  FIRST CHUNK: obs=90 sigma=0: 1,440 cells, 80,640 rows in 57.3 min compute (2388.8 ms/cell), 57.4 min wall since start
  PROJECTION  : 12 chunks / 7 jobs = 2 round(s) x 57.4 min = 1.91 h evaluation wall time at --jobs 7  (serial equivalent 11.47 h; equal-cost chunks assumed; aggregation and figures extra)
```

So the first full-grid chunk took **57.4 min** wall (2.39 s per cell; the seven workers each ran at ~100 % of a core: 53.6 CPU-min per worker after 55 min), and the projected Phase-5a evaluation wall time at `--jobs 7` is **1.91 h** (2 rounds of 57.4 min), against a serial equivalent of 11.47 h, which is inside Report 2's 12-16 h estimate. Two caveats: the projection assumes equal-cost chunks, and the six chunks started together with this one (obs 30 and 60, all three noise levels) had not finished when the run was stopped 65 s later, so some blocks are somewhat slower and a realistic figure is 2-2.5 h; and the second round runs only 5 of 12 chunks on 7 workers (open question 1). Aggregation and the four figures come on top (48 s of the quick step; expected a few minutes on the 967,680-row table). The first chunk's shard (`phase5a_raw_part006_obs90_sigma0.csv`, 22 MB) was written before the kill; nothing from this run was kept.

## 4. pytest

`python -m pytest tests/ -q` in the worktree of `8d33e82`: **342 passed in 34.10 s** (`test_generators` 189, `test_redesign` 138, `test_pipeline_v2` 11, `test_accelerators` 4; three tests new, four modified). After the post-verification cleanup change (section 8) the suite was re-run in the main checkout: **342 passed in 25.79 s**.

## 5. `python reproduce_all.py --quick`

Executed in the worktree (`8d33e82`), exit code 0; complete log in `REPORT_3A_quick_run.txt`. Phase 5a ran with the default `--jobs 7`, capped to the 4 blocks of the quick grid (48 s against 109 s in Report 2, while a second Phase-5a process was competing for the CPU; see section 3).

```
========================================================================
  PIPELINE SUMMARY  [QUICK]
========================================================================
   #  step                                             status       time
  ----------------------------------------------------------------------
   1  Phase 0 — analytic unit tests                    OK             2s
   2  Phase 1 — main benchmark                         OK            25s
   3  Dangerous re-derivation (Phase 1 -> artifact)    OK             1s
   4  Phase 2 — failure detection                      OK            10s
   5  Phase 3 — adaptive selection                     OK             4s
   6  Phase 4 — perturbation diagnostics               OK             9s
   7  Phase 5a — ensemble ablation                     OK            48s
   8  Phase 5b — sensitivity sweeps                    OK            28s
   9  Real data — recorded-curve re-evaluation         OK             6s
  ----------------------------------------------------------------------
      total                                                         133s
  All steps completed successfully.  Results are in results/
========================================================================
```

New rules visible in the log: the Phase-1 UNRANKED block (6 methods at g = 0.1, core), the derivation's "55 scored = 51 accelerators (eligible) + 4 trivial comparators (… never in the artifact)", the Phase-2 header "rank pool : 9", the Phase-5a FIRST CHUNK / PROJECTION lines, the Phase-5b sweep-3 unranked block (6 methods), and the real-data perturb_iqr and provenance lines. `scripts/check_dangerous.py` on the worktree output: MATCH (8 declared, 8 re-derived).

## 6. Confirmation items for decisions 2-6

**Decision 2, artifact method list length.** `results/phase1/dangerous_methods.json` (worktree): `pool = "accelerators"`, `n_pool = 51`, `table` = **51 rows** (no `is_trivial = 1` row), `dangerous_methods` = 8 (`linear, neville_2, neville_3, neville_4, pade_21, pade_31, wynn_eps_2, wynn_eps_3`; unchanged from Report 2 because no trivial had S < 0 there either). The derivation printout still scores and prints the four trivials, e.g.

```
  window_mean              0.1250   1.000   0.438   0.000      8  trivial (scored only; not eligible, not in artifact)
  constant_assumed         0.8500   1.000   0.125   0.250      8  trivial (scored only; not eligible, not in artifact)
```

**Decision 3, an unranked-block example** (Phase 1, quick run, g = 0.1, core; `phase1_unranked.csv` has 44 rows over both regime sets and three strata):

```
  UNRANKED — BELOW VALIDITY FLOOR  (g = 0.1, core; valid_rate < 0.9; shown, never ranked; 6 methods)
  Method                   Type          Valid    Cat    MedErr   Skill    Stab  Cells
  geom_avg_diff            limit         0.750  0.250   0.01095   0.729   0.450      2
  neville_3                trajectory    0.750  0.500  24.65084 1640.334  -0.050      2
  neville_4                trajectory    0.750  0.500 208.97676 21194.835  -0.050      2
  neville_2                trajectory    0.500  0.500   0.00154   0.120  -0.300      2
  wynn_eps_2               limit         0.500  0.500   0.02018   1.707  -0.500      2
  wynn_eps_3               limit         0.500  0.500   0.02002   1.695  -0.500      2
```

`neville_2` held rank 3 in Report 2 (valid rate 0.5, median error over the cells where it was valid); it is now unranked and `richardson_a20` moves from rank 4 to rank 3. At g = 0.1 the core table has 56 rows: 49 ranked, 6 below the floor, 1 oracle. The same six methods form the Phase-5b sweep-3 unranked block (g = 0.1, CAT_MULT = 2), and the sweep-3 concordance now compares 49 (g = 0.1) and 45 (g = 0.5) ranked methods instead of 55.

**Decision 4, Phase-2 rank pool size.** `RANK_POOL` = 9 (`current_value, richardson_1, richardson_a10, single_exp_fit, rational_fit, pade_22, log_linear, weniger_d2, anderson_1`); `phase2_phase_diagram_g0.1.csv` has `n_rank_pool = 9` in all 52 cells and `n_rank_eligible = 9` in 51 of them (one cell, `log_slow`, obs 90, sigma 0.005, has 8: a competitor fell below the floor). No cell had `richardson_1` below the floor. Reported alongside, unranked, on every cell: `constant_assumed_stability` (mean 1.004 at g = 0.1), `constant_assumed_med_error`, `constant_assumed_valid_rate`, `constant_oracle_stability` (mean 1.392), `constant_oracle_med_error`. Runner line: `rank pool : 9 (the original 9; constant_assumed and constant_oracle reported alongside, unranked)`.

**Decision 5, one real-data v2 row with perturb_iqr** (`real_data_results_v2.csv`, 630 rows, `perturb_iqr` finite on 630/630):

```
dataset adult | obs_depth 90 | target_round 500 | method cascade | selected_method richardson_1
is_cascade 1 | is_trivial 0 | prediction 0.267256 | true_val 0.272978 | error 0.005723
current_err 0.013033 | ref_error 0.013033 | skill 0.439105 | perturb_iqr 0.008713
L_hat 0.0 | assumed_mode zero | nearest_regime log_slow
```

Over the 90 cells: median perturb_iqr 0.00301, range [0.00019, 0.05367]; routed `richardson_1` (54 cells) median 0.00168, routed `rational_fit` (36 cells) median 0.00573.

**Decision 5, provenance metadata** (`real_data_summary_v2.csv`, 90 rows, on every row): `phase2_features_path = results\phase2\phase2_features.csv`, `phase2_features_rows = 104` (the quick run's two-regime feature file), `git_head = 8d33e82`. When the feature file is absent the columns read `''`, `0`, `<hash>` and `nearest_regime` is `unknown` (unit-tested).

**Decision 6.** `scripts/analyze_real_diagnostics_legacy.py` exists (git rename, history preserved), imports cleanly, and its docstring opens with the legacy note; `scripts/analyze_real_diagnostics.py` no longer exists; `README.md` points to the new name. `REPORT_1.md` / `REPORT_2.md` keep the old name as historical record.

## 7. Deviations from the prompt and decisions taken

1. **Commits.** The prompt did not ask for a commit, but a git worktree checks out a commit, so the code was committed (`8d33e82`) before the verification; the report, the quick-run log and the cleanup follow-up are a second commit on top. Nothing from the worktree's `results/` is committed; the main checkout's committed `results/` (the old design's) is untouched, as in Report 2. The worktree is kept at `../wt-report3a` (detached at `8d33e82`, with the quick-run outputs and the dangerous artifact, for inspection; `git worktree remove --force ../wt-report3a` drops it); the `b43449a` worktree used for the old-loop comparison was removed.
2. **Phase-3 candidates follow the Phase-2 rank pool.** `phases.phase3.CANDIDATES = list(RANK_POOL)`, so the selector candidates (and the Phase-3 "oracle" selector) are now the 9 methods; `constant_assumed` receives exactly the oracle's treatment there too. Report 2 had 10 candidates. Open question 2.
3. **Validity floor in Phase 2 is per cell.** The prompt names pooled tables; Phase 2 emits per-cell ranks, so the floor was applied inside each `(regime, obs_idx, noise)` cell: below-floor methods take no rank slot, and a cell where `richardson_1` itself is below the floor gets `richardson_rank = NaN`, `richardson_below_floor = 1` (listed by the runner; blank in figure P2-1; excluded from the mean-rank map). The quick run produced no such cell. Open question 3.
4. **Sweep-3 concordance over ranked methods only.** With the floor, the Kendall tau between CAT_MULT settings is computed over rank-eligible methods common to both settings (45-49 in the quick run, was 55); the champions are unaffected.
5. **perturb_iqr seeding is keyed on (dataset, depth), not target**, exactly as the legacy script keys `"dataset:obs"`: all target rounds of one window share the same perturbed windows and differ only through `future_x`. The value is a cell-level quantity of the routed method; it is repeated on every row of the cell in `real_data_results_v2.csv` (the same convention as `selected_method`) and also written to the summary.
6. **Phase 5a chunk size is (depth x noise) as asked**: 12 equal blocks on the full grid, so `--jobs 7` needs 2 rounds with 5 of 7 workers idle in the second. Open question 1.
7. **Shard cleanup on Windows.** In the quick run the shard files were deleted but the empty `shards/` directory survived `shutil.rmtree` (a file-sync client held the folder). After the verification a five-line retry loop was added to `run_phase5a` (section 8); the directory is git-ignored either way.
8. **Timing conditions.** The quick-run first-chunk figures (30-42 s per 24-cell block) were measured while the pre-change serial Phase-5a run of section 2 was executing on the same machine, so they are pessimistic; the full-grid timing of section 3 ran alone.
9. **`reproduce_all.py` was not changed**: it does not pass `--jobs`, so the pipeline uses Phase 5a's default (`cpu_count() - 1`). Open question 4.

## 8. Post-verification change

`phases/phase5a.py`, end of `run_phase5a`: `shutil.rmtree(shard_dir, ignore_errors=True)` is retried up to five times with a 0.5 s pause until the directory is gone. No other code differs from `8d33e82`. Re-tested with `tests/test_pipeline_v2.py::test_phase5a_block_parallel_byte_identical` (shard cleanup asserted) and the full suite (section 4).

## 9. Open questions for Kian / Claude

1. **Phase 5a chunk granularity.** 12 blocks / 7 jobs leaves the second round 5/7 idle. Chunking additionally by seed group (blocks of `(obs_idx, sigma, seed-range)`, still RNG-independent because streams are seeded per seed) would give 24-60 equal chunks and cut the projected wall time by roughly 1.6x. Do it now, or keep the (depth x noise) blocks as specified?
2. **Phase-3 candidate pool.** Is 9 (constant_assumed excluded like the oracle) the intended consequence of decision 4, or should Phase 3 keep constant_assumed as a candidate (10)?
3. **Phase-2 floor semantics.** Keep the per-cell rule (Richardson unranked where it fails >10 % of seeds), or exempt Richardson itself so the failure map never has blank cells?
4. **`reproduce_all.py --jobs`.** Expose a pass-through for Phase 5a, or leave the default (`cpu_count() - 1`)?
5. **Stale v1 files** and the committed `results/` still await the full run (Report 2, open question 6).

Stopped here; nothing beyond the six decisions and the verification was started.
