**Report back**

# Redesign v2, Prompt 5B: roster correction, pre-flight gate, full run at the 49-method roster, tables and facts

Date: 2026-09-21. Branch `redesign-v2` in `code/`. Commits on top of `68d4bc1` (Report 5A): `e5a802b` (part 1, roster), `74c02f1` (part 3, generator), `928092a` (results of the full run), `91b221a` (fragments + FACTS), `73e2ea5` + `ae0f688` (a fact-label fix and the regeneration at that clean HEAD); this report is one more commit on top. Everything through `ae0f688` is pushed (section 8). The three binding review decisions were applied as stated: (1) the Weniger pair is retired and `levin_t2` replaces `weniger_d2` in the pool; (2) sweep 1b's headline reference is `last_value`, `constant_assumed`'s own error per mode is reported as the value of knowing the floor, the vs-assumed columns stay in the CSVs; (3) the post-minimum cells stay in the real-data grid and everything is reported three ways.

## 1. Roster change and pre-flight gate

**Roster.** `weniger_d1` / `weniger_d2` removed from the registry (`src/accelerators.py`): 49 accelerators in 12 families, 54 registered methods with the five trivial comparators. The corrected `_weniger_delta` and its two wrappers are kept under `RETIRED_METHODS`, with the geometric-limit test and the new identity test `tests/test_accelerators.py::test_weniger_equals_levin_t[1|2]`: on the 96 audit windows of `test_input_dependence.py`, `_weniger_delta(k) == _levin_transform(k, 't')` to 1e-12 relative (NaN on the same windows) for k = 1, 2 -- the identity is on permanent record. Reason, as the review stated it: this codebase's Levin `t` already uses the forward-difference remainder `w_n = Delta s_n` (Weniger's d~-type), and the Pochhammer weight `(n0+j+1)_(k-1)` equals the power weight `(n0+j+1)^(k-1)` for k <= 2.

**Pool.** The 9-method Phase 2/3/4 pool is now defined once, `src.pipeline.PHASE2_POOL = [current_value, richardson_1, richardson_a10, single_exp_fit, rational_fit, pade_22, log_linear, levin_t2, anderson_1]`; `phases/phase2.py` (`PHASE2_BASE_METHODS`, the three candidate rules that named `weniger_d2`), `phases/phase3.py` (the enhanced cascade's staircase branch), `phases/phase4.py` (`PHASE4_METHODS`), `phases/phase5a.py` (`PHASE2_METHODS`, the `_9` selectors) and `phases/phase5b.py` (`CASCADE_METHODS`) read it.

**Levin docstrings** (`_levin_transform`): `'t'` = forward-difference remainder `w_n = Delta s_n` (d~-type in Weniger's notation, not Levin's backward-difference t), `'u'` = `(n+1) Delta s_n`, `'v'` = the ratio-of-differences form; names unchanged, no numerical change (the sweep-3 / Phase-1 numbers of every Levin variant are bit-identical to the previous run, section 4).

**Hard-coded counts removed.** `src.pipeline.N_ACCEL = len(ACCEL_METHODS)`; Phase-5a selector names are built from the pool size (`oracle_49`, `equal_ensemble_49`, `diag_ensemble_49`, `capped_diag_49`, ablation keys `threshold_vs_equal_49`, `diag_vs_equal_49`); the artifact's `n_pool` was already `len(_ELIGIBLE)` (now 49); `src/dangerous.py`, `scripts/run_phase1.py`, `reproduce_all.py --plan`, the tests (`len(ACCEL_METHODS)`, `len(METHOD_NAMES) == len(ACCEL_METHODS) + 5`), `scripts/make_paper_tables.py` (`N_ACC`) and both READMEs derive or state 49 / 54 / 12 families. `grep -rn "\b51\b"` over the Python tree finds only colour hex codes.

**Pre-flight gate (both passed before the full run started).**

| check | result |
|---|---|
| `python -m pytest tests/ -q` at `e5a802b` | **350 passed in 105.6 s** (348 + the two identity tests) |
| `python reproduce_all.py --quick` in worktree `../wt-5b` at `e5a802b` | **exit 0, 157 s**: Phase 0 3 s, Phase 1 19 s, derivation 1 s, Phase 2 10 s, Phase 3 3 s, Phase 4 8 s, Phase 5a 40 s, Phase 5b 67 s, real data 6 s; log `Methods : 54 (49 accelerators + 5 trivial comparators)`, `Pool : 49 accelerators`, `oracle_49` in the Phase-5a output, no `weniger` / `oracle_51` string anywhere in the log; quick artifact `n_pool 49` |

## 2. Full run: per-phase wall times

`python reproduce_all.py` in the main checkout at `74c02f1` (pipeline code = `e5a802b`; the only change in between is the generator), 2026-09-21 11:38:20 to 17:00:02, exit 0, log `REPORT_5B_full_run.txt` (1,386 lines). Phase 5a at the default `--jobs 7`, 5 perturbation trials.

| # | step | wall | previous run (Report 3B) | central evaluations (`--plan`) |
|---|---|---:|---:|---:|
| 1 | Phase 0 | 2 s | 2 s | 196 (49 x 4) |
| 2 | Phase 1 | 1,729 s (28.8 min) | 1,839 s | 349,920 (24 x 30 x 3 x 3 x 54) |
| 3 | Dangerous re-derivation | 1 s | 1 s | -- |
| 4 | Phase 2 | 876 s (14.6 min) | 979 s | 772,200 |
| 5 | Phase 3 | 4 s | 5 s | -- |
| 6 | Phase 4 | 2,079 s (34.7 min) | 1,797 s | 224,640 (+ 1,555,200 diagnostic calls) |
| 7 | Phase 5a | 9,223 s (2.56 h) | 8,868 s | 933,120 (+ 4,233,600 perturbation calls) |
| 8 | Phase 5b | 5,375 s (1.49 h) | 2,287 s | 717,120 (sweep 1a 28,800 + **sweep 1b 201,600** + sweep 2 28,800 + sweep 3 457,920) |
| 9 | Real data | 15 s | 5 s | 630 |
| | **total** | **19,302 s (5.36 h)** | 15,783 s | 2,997,826 |

Phase 5a: first chunk (obs 90, sigma 0) 57.1 min / 77,760 rows / 2,378 ms per cell; projection 1.90 h at `--jobs 7` (serial equivalent 11.42 h); actual 2.49 h evaluation wall (block times 57 min to 1.80 h, the obs 30/60 blocks slowest, as in Report 3B) plus ~3 min aggregation. Phase 5b's extra 51 min is sweep 1b (7,200 windows x 2 strata x 14 method calls, the two ensembles being the expensive members). Phase 4 is 4.7 min slower than last time because `levin_t2` now receives the shift / perturbation diagnostics that the constant-output `weniger_d2` used to short-circuit.

## 3. NaN rule

Pooled invalid rate over non-dangerous, non-oracle records (dangerous set = this run's artifact), the two known patterns removed for the residual:

```
dangerous set (8): ['geom_avg_diff', 'linear', 'neville_2', 'neville_3', 'neville_4', 'pade_21', 'pade_31', 'pade_32']
phase1   rows=  349,920 non-dang/non-oracle=  291,600 invalid=  2,542 rate=0.872% | after removing the known patterns: rate=0.460% (n=246,240)
         by noise: 0: 1.84%, 0.001: 0.37%, 0.005: 0.41%
         lowest valid (non-dangerous, full data): pade_22=0.944, wynn_eps_3=0.969, log_linear=0.975, wynn_eps_2=0.978, wynn_rho_3=0.981, richardson_3=0.981
         residual methods < 0.95 valid after removing known patterns: ['pade_22']
         VERDICT: ok (under threshold)
phase2   agg rows=38,610 rate=0.725% | after removing known patterns 0.621%; lowest: log_linear=0.967, pade_22=0.976, levin_t2=0.997, anderson_1=0.999, current_value=1.000
         VERDICT: ok (under threshold)
phase4   rows=  224,640 non-dang/non-oracle=  224,640 invalid=  1,341 rate=0.597% | after removing the known patterns: rate=0.490% (n=213,120)
         by noise: 0: 0.63%, 0.005: 0.40%, 0.02: 0.75%
         by depth: 30: 0.32%, 60: 0.40%, 90: 0.98%, 120: 0.69%
         richardson_3 valid by depth: 
         lowest valid (non-dangerous, full data): log_linear=0.970, pade_22=0.972, levin_t2=0.980, anderson_1=0.999, last_value=1.000, current_value=1.000
         residual methods < 0.95 valid after removing known patterns: none
         VERDICT: ok (under threshold)
phase5a  rows=  933,120 non-dang/non-oracle=  777,600 invalid= 12,267 rate=1.578% | after removing the known patterns: rate=0.488% (n=648,000)
         by noise: 0: 2.79%, 0.005: 0.88%, 0.02: 1.06%
         by depth: 30: 2.21%, 60: 1.96%, 90: 1.04%, 120: 1.10%
         richardson_3 valid by depth: 30: 0.481, 60: 0.583, 90: 0.981, 120: 0.999
         lowest valid (non-dangerous, full data): richardson_3=0.761, levin_u1=0.960, log_linear=0.970, wynn_eps_3=0.971, pade_22=0.972, pade_11=0.973
         residual methods < 0.95 valid after removing known patterns: none
         VERDICT: THRESHOLD CROSSED -> known patterns only (richardson_3 at obs 30/60; sigma = 0 cancellation in the classical families) -> continue
phase5b  sweep-3 rate=1.353% (pooled over noise incl. sigma = 0; no per-record file) lowest: pade_22=0.932, log_linear=0.950, wynn_eps_3=0.954, wynn_eps_2=0.962, wynn_rho_2=0.974, wynn_rho_3=0.974
         VERDICT: THRESHOLD CROSSED (obs 90; same methods as Phase 1; sigma = 0 share not separable in the aggregate)
real_data rows=630 NaN prediction=0 NaN error=0 VERDICT: ok
```

| phase | rate outside the dangerous set | residual after removing the known patterns | verdict |
|---|---:|---:|---|
| Phase 1 | 0.872 % | 0.460 % | under threshold |
| Phase 2 (aggregated) | 0.725 % | 0.621 % | under threshold |
| Phase 4 | 0.597 % | 0.490 % | under threshold |
| Phase 5a | **1.578 %** | 0.488 %, no residual method below 0.95 valid | crossed; known patterns only (richardson_3 at obs 30/60: valid 0.481 / 0.583; sigma = 0: 2.79 % vs 0.88 % / 1.06 %) -> continued |
| Phase 5b sweep 3 (aggregated, obs 90) | **1.353 %** | not separable in the aggregate (no per-record file) | crossed; the same methods as Phase 1 at the same depth (pade_22 0.932, log_linear 0.950, wynn_eps_3 0.954, wynn_eps_2 0.962 -- the sigma = 0 cancellation family plus the two Phase-1 low-validity fits) -> continued |
| Real data | 0 NaN in 630 rows | -- | ok |

Nothing outside the two known patterns appeared; the run was not stopped.

## 4. Dangerous artifact and the diff against the previous run

`results/phase1/dangerous_methods.json`: **8 dangerous = `geom_avg_diff, linear, neville_2, neville_3, neville_4, pade_21, pade_31, pade_32`**; `n_pool 49`, 49-row table, `source results/phase1/phase1_aggregated.csv`, `git_head 74c02f1`; `scripts/check_dangerous.py`: MATCH.

Diff against the previous run's artifact (`30de724`, 51-method roster): `+[] -[]` on the dangerous list; the 49 common methods have **bit-identical** S / valid / cat / beats values (the Weniger pair never entered another method's evaluation, and the RNG streams are per (regime, seed, sigma)); the only table difference is the two Weniger rows that no longer exist (`weniger_d1` S = +0.646 and `weniger_d2` S = +0.669 in the old table). Against the legacy hard-coded set: identical, as in every run of the redesign.

## 5. Fragment f01 (trivial baseline, the paper's first result), verbatim

```latex
\begin{tabular}{lrrrrrrrrrrrr}
\toprule
Stratum / row & \multicolumn{6}{c}{core (18 regimes)} & \multicolumn{6}{c}{held-out (6 regimes)} \\
\cmidrule(lr){2-7}\cmidrule(lr){8-13}
 & med.\ err & win vs $\hat L$ & skill vs $\hat L$ & win vs last & skill vs last & strict skill & med.\ err & win vs $\hat L$ & skill vs $\hat L$ & win vs last & skill vs last & strict skill \\
\midrule
\multicolumn{13}{l}{\textit{$g = 0.5$}} \\
predict $\hat L$ (\meth{constant\_assumed}) & 0.1094 & -- & -- & 0.040 & 1.973 & 2.313 & 0.0853 & -- & -- & 0.000 & 2.227 & 2.261 \\
\meth{last\_value} & 0.0690 & 0.961 & 0.507 & -- & -- & 1.021 & 0.0356 & 1.000 & 0.456 & -- & -- & 1.000 \\
\meth{window\_mean} & 0.1066 & 0.611 & 0.838 & 0.197 & 1.706 & 1.792 & 0.0815 & 0.556 & 0.873 & 0.000 & 2.273 & 2.438 \\
\meth{window\_min} & 0.0484 & 0.947 & 0.438 & 0.391 & 1.000 & 1.000 & 0.0340 & 1.000 & 0.451 & 0.356 & 1.000 & 1.000 \\
\meth{constant\_oracle} \textit{(reference, not deployable)} & 0.0579 & 1.000 & 0.494 & 0.728 & 0.995 & 1.000 & 0.0356 & 1.000 & 0.448 & 0.702 & 0.993 & 0.997 \\
\addlinespace[2pt]
best accelerator, core rank 1: \meth{log\_linear} & 0.0111 & 0.904 & 0.092 & 0.825 & 0.334 & 0.362 & 0.0077 & 0.972 & 0.102 & 0.824 & 0.279 & 0.283 \\
best accelerator, held-out rank 1: \meth{rational\_fit} & 0.0159 & 0.952 & 0.145 & 0.849 & 0.310 & 0.495 & 0.0048 & 0.900 & 0.045 & 0.826 & 0.148 & 0.160 \\
median accelerator (49) & 0.0497 & 0.892 & 0.454 & 0.540 & 1.000 & 1.044 & 0.0297 & 0.939 & 0.372 & 0.513 & 0.999 & 1.028 \\
accelerators with win rate $> 0.5$ / strict skill $< 1$ & & 43/49 & & 26/49 & & 14/49 & & 44/49 & & 27/49 & & 17/49 \\
\midrule
\multicolumn{13}{l}{\textit{$g = 0.1$} (headline)} \\
predict $\hat L$ (\meth{constant\_assumed}) & 0.0620 & -- & -- & 0.524 & 0.682 & 1.000 & 0.0561 & -- & -- & 0.524 & 0.627 & 1.000 \\
\meth{last\_value} & 0.1030 & 0.476 & 1.488 & -- & -- & 1.570 & 0.0767 & 0.475 & 1.616 & -- & -- & 1.632 \\
\meth{window\_mean} & 0.1463 & 0.381 & 2.071 & 0.158 & 1.427 & 2.733 & 0.1382 & 0.333 & 2.773 & 0.000 & 1.803 & 2.773 \\
\meth{window\_min} & 0.0681 & 0.503 & 1.304 & 0.348 & 1.000 & 1.493 & 0.0766 & 0.489 & 1.616 & 0.331 & 1.000 & 1.616 \\
\meth{constant\_oracle} \textit{(reference, not deployable)} & 0.0077 & 1.000 & 0.151 & 1.000 & 0.111 & 0.164 & 0.0084 & 1.000 & 0.173 & 1.000 & 0.111 & 0.174 \\
\addlinespace[2pt]
best accelerator, core rank 1: \meth{log\_linear} & 0.0202 & 0.809 & 0.299 & 0.756 & 0.332 & 0.601 & 0.0253 & 0.840 & 0.273 & 0.862 & 0.370 & 0.558 \\
best accelerator, held-out rank 1: \meth{rational\_fit} & 0.0296 & 0.787 & 0.328 & 0.877 & 0.299 & 0.670 & 0.0074 & 0.824 & 0.084 & 0.853 & 0.150 & 0.232 \\
median accelerator (49) & 0.0489 & 0.543 & 0.948 & 0.571 & 1.000 & 1.538 & 0.0357 & 0.607 & 0.584 & 0.600 & 0.999 & 1.337 \\
accelerators with win rate $> 0.5$ / strict skill $< 1$ & & 35/49 & & 28/49 & & 9/49 & & 35/49 & & 31/49 & & 14/49 \\
\midrule
\multicolumn{13}{l}{\textit{$g = 0.02$}} \\
predict $\hat L$ (\meth{constant\_assumed}) & 0.0541 & -- & -- & 0.544 & 0.558 & 1.000 & 0.0545 & -- & -- & 0.569 & 0.499 & 1.000 \\
\meth{last\_value} & 0.0970 & 0.456 & 1.871 & -- & -- & 1.873 & 0.0834 & 0.431 & 2.042 & -- & -- & 2.042 \\
\meth{window\_mean} & 0.1749 & 0.353 & 2.682 & 0.102 & 1.398 & 2.782 & 0.1450 & 0.313 & 3.590 & 0.000 & 1.738 & 3.590 \\
\meth{window\_min} & 0.0970 & 0.440 & 1.751 & 0.304 & 1.000 & 1.962 & 0.0834 & 0.449 & 2.042 & 0.331 & 1.000 & 2.042 \\
\meth{constant\_oracle} \textit{(reference, not deployable)} & 0.0011 & 1.000 & 0.031 & 1.000 & 0.020 & 0.031 & 0.0017 & 1.000 & 0.041 & 1.000 & 0.020 & 0.041 \\
\addlinespace[2pt]
best accelerator, core rank 1: \meth{richardson\_2} & 0.0218 & 0.777 & 0.319 & 0.847 & 0.394 & 0.883 & 0.0117 & 0.769 & 0.315 & 0.849 & 0.185 & 0.537 \\
best accelerator, held-out rank 1: \meth{rational\_fit} & 0.0228 & 0.761 & 0.398 & 0.843 & 0.278 & 0.605 & 0.0089 & 0.793 & 0.120 & 0.900 & 0.240 & 0.250 \\
median accelerator (49) & 0.0514 & 0.518 & 1.101 & 0.564 & 1.000 & 1.718 & 0.0386 & 0.553 & 0.630 & 0.593 & 0.999 & 1.329 \\
accelerators with win rate $> 0.5$ / strict skill $< 1$ & & 33/49 & & 27/49 & & 8/49 & & 31/49 & & 31/49 & & 12/49 \\
\bottomrule
\end{tabular}
```

## 6. Fragment f04 (classical no-op, 21 variants), verbatim

```latex
\begin{tabular}{llrrrrrrrrr}
\toprule
Family & Method & \multicolumn{3}{c}{$g = 0.5$} & \multicolumn{3}{c}{$g = 0.1$} & \multicolumn{3}{c}{$g = 0.02$} \\
\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}
 & & MI & win vs last & med.\ skill & MI & win vs last & med.\ skill & MI & win vs last & med.\ skill \\
\midrule
shanks & \meth{shanks\_1} & \textbf{1.001} & 0.604 & \textbf{1.043} & \textbf{1.000} & 0.613 & \textbf{1.538} & \textbf{1.000} & 0.607 & \textbf{1.720} \\
shanks & \meth{shanks\_2} & \textbf{0.965} & 0.331 & \textbf{1.091} & \textbf{0.980} & 0.353 & \textbf{1.672} & \textbf{0.980} & 0.347 & \textbf{1.928} \\
shanks & \meth{shanks\_3} & \textbf{0.975} & 0.395 & \textbf{1.093} & \textbf{0.982} & \textbf{0.401} & \textbf{1.516} & \textbf{0.984} & 0.396 & \textbf{1.585} \\
shanks & \meth{shanks\_4} & \textbf{0.971} & 0.385 & \textbf{1.098} & \textbf{0.973} & 0.370 & \textbf{1.581} & \textbf{0.974} & 0.362 & \textbf{1.686} \\
\addlinespace[2pt]
wynn\_eps & \meth{wynn\_eps\_1} & \textbf{1.001} & 0.604 & \textbf{1.040} & \textbf{1.000} & 0.613 & \textbf{1.523} & \textbf{1.001} & 0.607 & \textbf{1.654} \\
wynn\_eps & \meth{wynn\_eps\_2} & \textbf{0.982} & \textbf{0.488} & \textbf{1.078} & \textbf{0.989} & \textbf{0.471} & \textbf{1.521} & \textbf{0.990} & \textbf{0.466} & \textbf{1.595} \\
wynn\_eps & \meth{wynn\_eps\_3} & \textbf{0.987} & \textbf{0.498} & \textbf{1.080} & \textbf{0.991} & \textbf{0.488} & \textbf{1.518} & \textbf{0.990} & \textbf{0.482} & \textbf{1.606} \\
\addlinespace[2pt]
wynn\_rho & \meth{wynn\_rho\_1} & \textbf{0.995} & \textbf{0.421} & \textbf{1.060} & \textbf{0.997} & \textbf{0.438} & \textbf{1.528} & \textbf{0.997} & \textbf{0.432} & \textbf{1.657} \\
wynn\_rho & \meth{wynn\_rho\_2} & \textbf{1.000} & \textbf{0.465} & \textbf{1.078} & \textbf{1.000} & \textbf{0.454} & \textbf{1.651} & \textbf{1.000} & \textbf{0.464} & \textbf{1.871} \\
wynn\_rho & \meth{wynn\_rho\_3} & \textbf{0.987} & \textbf{0.436} & \textbf{1.092} & \textbf{0.995} & \textbf{0.494} & \textbf{1.546} & \textbf{0.995} & \textbf{0.491} & \textbf{1.653} \\
\addlinespace[2pt]
levin & \meth{levin\_t1} & \textbf{1.001} & 0.604 & \textbf{1.040} & \textbf{1.000} & 0.613 & \textbf{1.523} & \textbf{1.001} & 0.607 & \textbf{1.654} \\
levin & \meth{levin\_t2} & \textbf{0.988} & \textbf{0.474} & \textbf{1.051} & \textbf{0.993} & \textbf{0.503} & \textbf{1.520} & \textbf{0.992} & \textbf{0.498} & \textbf{1.602} \\
levin & \meth{levin\_u1} & \textbf{1.000} & \textbf{0.414} & \textbf{1.079} & \textbf{1.000} & \textbf{0.503} & \textbf{1.603} & \textbf{1.000} & \textbf{0.512} & \textbf{1.924} \\
levin & \meth{levin\_u2} & \textbf{0.984} & 0.399 & \textbf{1.068} & \textbf{0.992} & \textbf{0.462} & \textbf{1.592} & \textbf{0.991} & \textbf{0.453} & \textbf{1.868} \\
levin & \meth{levin\_v1} & \textbf{0.965} & 0.277 & \textbf{1.096} & \textbf{0.980} & 0.390 & \textbf{1.571} & \textbf{0.981} & 0.382 & \textbf{1.766} \\
levin & \meth{levin\_v2} & \textbf{0.973} & \textbf{0.418} & \textbf{1.094} & \textbf{0.981} & \textbf{0.420} & \textbf{1.680} & \textbf{0.981} & \textbf{0.416} & \textbf{1.898} \\
\addlinespace[2pt]
brezinski & \meth{brezinski\_theta1} & \textbf{1.048} & \textbf{0.546} & \textbf{1.018} & \textbf{1.037} & \textbf{0.592} & \textbf{1.518} & \textbf{1.033} & \textbf{0.584} & \textbf{1.900} \\
brezinski & \meth{brezinski\_theta2} & \textbf{0.907} & 0.328 & \textbf{1.140} & \textbf{0.941} & 0.362 & \textbf{1.806} & \textbf{0.944} & 0.353 & \textbf{2.067} \\
\addlinespace[2pt]
anderson & \meth{anderson\_1} & \textbf{1.001} & 0.604 & \textbf{1.043} & \textbf{1.000} & 0.613 & \textbf{1.573} & \textbf{1.000} & 0.607 & \textbf{1.921} \\
anderson & \meth{anderson\_2} & \textbf{1.000} & \textbf{0.568} & \textbf{1.038} & \textbf{1.000} & \textbf{0.571} & \textbf{1.530} & \textbf{1.000} & \textbf{0.564} & \textbf{1.904} \\
anderson & \meth{anderson\_3} & \textbf{0.999} & \textbf{0.540} & \textbf{1.044} & \textbf{0.999} & \textbf{0.540} & \textbf{1.559} & \textbf{0.999} & \textbf{0.534} & \textbf{1.895} \\
\midrule
\multicolumn{2}{l}{in the $\pm 10\%$ band (MI in $[0.9, 1.1]$)} & 21/21 &  &  & 21/21 &  &  & 21/21 &  &  \\
\multicolumn{2}{l}{win rate vs last in $[0.4, 0.6]$ (coin flip)} &  & 11/21 &  &  & 13/21 &  &  & 12/21 &  \\
\multicolumn{2}{l}{median skill $\geq 0.9$} &  &  & 21/21 &  &  & 21/21 &  &  & 21/21 \\
\bottomrule
\end{tabular}
```

With the degenerate Weniger pair gone, **all 21 classical variants are inside the +/-10 % MI band at every stratum** (Report 4 had 21 of 23: the two outliers were `weniger_d1/d2`); in win-rate terms 11 / 13 / 12 of 21 sit within a coin flip of the last value at g = 0.5 / 0.1 / 0.02 (the rest are mostly *below* 0.4, i.e. worse than the last value more often than not), and every one has strict median skill >= 0.9.

## 7. Fragment f05b (sweep 1b), verbatim -- core

```latex
\begin{tabular}{llrrrrrrrrrr}
\toprule
Method & row & \multicolumn{5}{c}{$g = 0.5$} & \multicolumn{5}{c}{$g = 0.1$} \\
\cmidrule(lr){3-7}\cmidrule(lr){8-12}
 & & zero & half & oracle & double & winmin & zero & half & oracle & double & winmin \\
\midrule
\meth{richardson\_1} & med.\ err & 0.0173 & 0.0173 & 0.0173 & 0.0173 & 0.0173 & 0.0309 & 0.0309 & 0.0309 & 0.0309 & 0.0309 \\
 & win vs last & 0.774 & 0.774 & 0.774 & 0.774 & 0.774 & 0.799 & 0.799 & 0.799 & 0.800 & 0.799 \\
\addlinespace[1pt]
\meth{richardson\_2} & med.\ err & 0.0158 & 0.0156 & 0.0160 & 0.0159 & 0.0162 & 0.0263 & 0.0263 & 0.0263 & 0.0263 & 0.0265 \\
 & win vs last & 0.778 & 0.778 & 0.778 & 0.779 & 0.776 & 0.778 & 0.778 & 0.778 & 0.779 & 0.775 \\
\addlinespace[1pt]
\meth{richardson\_3} & med.\ err & 0.0164 & 0.0161 & 0.0161 & 0.0162 & 0.0161 & 0.0224 & 0.0230 & 0.0225 & 0.0225 & 0.0229 \\
 & win vs last & 0.766 & 0.769 & 0.771 & 0.772 & 0.778 & 0.754 & 0.758 & 0.758 & 0.765 & 0.766 \\
\addlinespace[1pt]
\meth{single\_exp\_fit} & med.\ err & 0.0214 & 0.0214 & 0.0214 & 0.0216 & 0.0214 & 0.0378 & 0.0378 & 0.0378 & 0.0384 & 0.0378 \\
 & win vs last & 0.863 & 0.863 & 0.863 & 0.863 & 0.863 & 0.920 & 0.919 & 0.920 & 0.912 & 0.920 \\
\addlinespace[1pt]
\meth{double\_exp\_fit} & med.\ err & 0.0176 & 0.0173 & 0.0161 & 0.0176 & 0.0199 & 0.0265 & 0.0263 & 0.0246 & 0.0263 & 0.0319 \\
 & win vs last & 0.850 & 0.850 & 0.846 & 0.852 & 0.859 & 0.880 & 0.888 & 0.871 & 0.890 & 0.884 \\
\addlinespace[1pt]
\meth{rational\_fit} & med.\ err & 0.0167 & 0.0167 & 0.0167 & 0.0167 & 0.0167 & 0.0290 & 0.0290 & 0.0290 & 0.0290 & 0.0290 \\
 & win vs last & 0.835 & 0.835 & 0.835 & 0.835 & 0.835 & 0.851 & 0.851 & 0.851 & 0.851 & 0.851 \\
\addlinespace[1pt]
\meth{log\_fit} & med.\ err & 0.0385 & 0.0385 & 0.0385 & 0.0385 & 0.0385 & 0.0684 & 0.0684 & 0.0684 & 0.0684 & 0.0684 \\
 & win vs last & 0.618 & 0.618 & 0.618 & 0.618 & 0.618 & 0.675 & 0.675 & 0.675 & 0.675 & 0.675 \\
\addlinespace[1pt]
\meth{log\_linear} & med.\ err & 0.0145 & 0.0138 & 0.0163 & 0.0253 & 0.0420 & 0.0240 & 0.0215 & 0.0162 & 0.0500 & 0.0752 \\
 & win vs last & 0.800 & 0.807 & 0.772 & 0.498 & 0.756 & 0.747 & 0.763 & 0.760 & 0.479 & 0.827 \\
\addlinespace[1pt]
\meth{stability\_weighted} & med.\ err & 0.0454 & 0.0454 & 0.0454 & 0.0454 & 0.0454 & 0.0454 & 0.0454 & 0.0454 & 0.0454 & 0.0454 \\
 & win vs last & 0.608 & 0.608 & 0.608 & 0.608 & 0.608 & 0.658 & 0.658 & 0.658 & 0.658 & 0.658 \\
\addlinespace[1pt]
\meth{median\_ensemble} & med.\ err & 0.0387 & 0.0387 & 0.0387 & 0.0387 & 0.0387 & 0.0457 & 0.0457 & 0.0457 & 0.0457 & 0.0457 \\
 & win vs last & 0.678 & 0.678 & 0.678 & 0.678 & 0.678 & 0.678 & 0.678 & 0.678 & 0.678 & 0.678 \\
\addlinespace[1pt]
predict $\hat L$ (\meth{constant\_assumed}) -- the value of knowing the floor & med.\ err & 0.1206 & 0.0972 & 0.0579 & 0.0616 & 0.0323 & 0.0593 & 0.0322 & 0.0077 & 0.0348 & 0.0507 \\
 & win vs last & 0.084 & 0.133 & 0.714 & 0.582 & 0.785 & 0.544 & 0.682 & 0.995 & 0.580 & 0.844 \\
\addlinespace[1pt]
\bottomrule
\end{tabular}
```

## 7b. Fragment f05b, verbatim -- held-out

```latex
\begin{tabular}{llrrrrrrrrrr}
\toprule
Method & row & \multicolumn{5}{c}{$g = 0.5$} & \multicolumn{5}{c}{$g = 0.1$} \\
\cmidrule(lr){3-7}\cmidrule(lr){8-12}
 & & zero & half & oracle & double & winmin & zero & half & oracle & double & winmin \\
\midrule
\meth{richardson\_1} & med.\ err & 0.0104 & 0.0104 & 0.0104 & 0.0104 & 0.0104 & 0.0149 & 0.0149 & 0.0149 & 0.0149 & 0.0149 \\
 & win vs last & 0.786 & 0.786 & 0.786 & 0.786 & 0.786 & 0.793 & 0.793 & 0.793 & 0.793 & 0.793 \\
\addlinespace[1pt]
\meth{richardson\_2} & med.\ err & 0.0090 & 0.0089 & 0.0090 & 0.0081 & 0.0083 & 0.0156 & 0.0148 & 0.0173 & 0.0167 & 0.0173 \\
 & win vs last & 0.772 & 0.775 & 0.781 & 0.781 & 0.769 & 0.787 & 0.790 & 0.801 & 0.798 & 0.784 \\
\addlinespace[1pt]
\meth{richardson\_3} & med.\ err & 0.0091 & 0.0093 & 0.0094 & 0.0086 & 0.0091 & 0.0142 & 0.0164 & 0.0161 & 0.0155 & 0.0156 \\
 & win vs last & 0.750 & 0.761 & 0.750 & 0.753 & 0.756 & 0.781 & 0.787 & 0.779 & 0.781 & 0.784 \\
\addlinespace[1pt]
\meth{single\_exp\_fit} & med.\ err & 0.0146 & 0.0146 & 0.0146 & 0.0149 & 0.0146 & 0.0277 & 0.0277 & 0.0277 & 0.0280 & 0.0277 \\
 & win vs last & 0.878 & 0.878 & 0.878 & 0.875 & 0.878 & 0.910 & 0.910 & 0.910 & 0.908 & 0.910 \\
\addlinespace[1pt]
\meth{double\_exp\_fit} & med.\ err & 0.0130 & 0.0126 & 0.0113 & 0.0127 & 0.0146 & 0.0255 & 0.0249 & 0.0240 & 0.0244 & 0.0251 \\
 & win vs last & 0.875 & 0.872 & 0.878 & 0.875 & 0.875 & 0.880 & 0.882 & 0.880 & 0.896 & 0.891 \\
\addlinespace[1pt]
\meth{rational\_fit} & med.\ err & 0.0064 & 0.0064 & 0.0064 & 0.0064 & 0.0064 & 0.0114 & 0.0114 & 0.0114 & 0.0114 & 0.0114 \\
 & win vs last & 0.792 & 0.792 & 0.792 & 0.792 & 0.792 & 0.818 & 0.818 & 0.818 & 0.818 & 0.818 \\
\addlinespace[1pt]
\meth{log\_fit} & med.\ err & 0.0399 & 0.0399 & 0.0399 & 0.0399 & 0.0399 & 0.0504 & 0.0504 & 0.0504 & 0.0504 & 0.0504 \\
 & win vs last & 0.483 & 0.483 & 0.483 & 0.483 & 0.483 & 0.597 & 0.597 & 0.597 & 0.597 & 0.597 \\
\addlinespace[1pt]
\meth{log\_linear} & med.\ err & 0.0100 & 0.0085 & 0.0096 & 0.0255 & 0.0261 & 0.0266 & 0.0189 & 0.0142 & 0.0414 & 0.0504 \\
 & win vs last & 0.772 & 0.775 & 0.753 & 0.369 & 0.772 & 0.801 & 0.843 & 0.922 & 0.479 & 0.838 \\
\addlinespace[1pt]
\meth{stability\_weighted} & med.\ err & 0.0285 & 0.0285 & 0.0285 & 0.0285 & 0.0285 & 0.0305 & 0.0305 & 0.0305 & 0.0305 & 0.0305 \\
 & win vs last & 0.667 & 0.667 & 0.667 & 0.667 & 0.667 & 0.753 & 0.753 & 0.753 & 0.753 & 0.753 \\
\addlinespace[1pt]
\meth{median\_ensemble} & med.\ err & 0.0203 & 0.0203 & 0.0203 & 0.0203 & 0.0203 & 0.0262 & 0.0262 & 0.0262 & 0.0262 & 0.0262 \\
 & win vs last & 0.742 & 0.742 & 0.742 & 0.742 & 0.742 & 0.779 & 0.779 & 0.779 & 0.779 & 0.779 \\
\addlinespace[1pt]
predict $\hat L$ (\meth{constant\_assumed}) -- the value of knowing the floor & med.\ err & 0.0981 & 0.0712 & 0.0423 & 0.0397 & 0.0207 & 0.0624 & 0.0363 & 0.0084 & 0.0432 & 0.0441 \\
 & win vs last & 0.039 & 0.094 & 0.739 & 0.556 & 0.847 & 0.507 & 0.639 & 0.994 & 0.566 & 0.938 \\
\addlinespace[1pt]
\bottomrule
\end{tabular}
```

L_hat-invariance (from `FACTS.md`, g = 0.1 core; max minus min of the median error over the five modes): `richardson_1` 0.00000 (0.0 %), `rational_fit` 0.00000, `log_fit` 0.00000, `median_ensemble` 0.00000, `stability_weighted` 0.00001, `richardson_2` 0.00024 (0.9 %), `single_exp_fit` 0.00064 (1.7 %), `richardson_3` 0.00060 (2.7 %), `double_exp_fit` 0.00732 (29.7 %), `log_linear` 0.05894 (362.8 %); `constant_assumed` 0.05156 (667 %: zero 0.0593, half 0.0322, oracle 0.0077, double 0.0348, winmin 0.0507 -- the value of knowing the floor). Win rates vs `last_value` move by < 0.01 for every consumer except `log_linear` (0.479 under `double`, 0.827 under `winmin`) and `double_exp_fit`.

## 8. Real data, three ways (all / pre-minimum / post-minimum)

Curve minima (`real_data_curve_minima_v2.csv`):

| dataset | argmin round | minimum | value at round 500 | rise from minimum |
|---|---:|---:|---:|---:|
| covertype | 500 | 0.34916 | 0.34916 | +0.00 % |
| higgs | 500 | 0.53009 | 0.53009 | +0.00 % |
| adult | 337 | 0.27172 | 0.27298 | +0.46 % |
| jannis | 500 | 0.68955 | 0.68955 | +0.00 % |
| miniboone | 500 | 0.13712 | 0.13712 | +0.00 % |
| bank_marketing | 257 | 0.19853 | 0.20116 | +1.32 % |

The three-way summary (`real_data_strata_v2.csv`, verbatim values):

| stratum | cells | fail (skill >= 1) | rate | median cascade err | median skill | median improvement | win vs L_hat | win vs last | perturbation AUC | p | ordering |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| all | 90 | 17 | 0.189 | 0.01032 | 0.336 | +0.664 | 1.000 | 0.811 | 0.317 | 0.019 | failing < succeeding |
| pre_min | 65 | 2 | 0.031 | 0.01009 | 0.268 | +0.732 | 1.000 | 0.969 | 0.198 | 0.175 | failing < succeeding |
| post_min | 25 | 15 | 0.600 | 0.01055 | 1.345 | -0.345 | 1.000 | 0.400 | 0.433 | 0.598 | failing < succeeding |

The f07 footer rows (per target round 300 / 400 / 500; cells / skill >= 1 / median skill):

```latex
\multicolumn{2}{l}{median skill / cells with skill $\geq 1$ (of 30)} &  & & 0.26 / 4 &  & & 0.34 / 6 &  & & 0.41 / 7 \\
\multicolumn{2}{l}{pre-minimum targets: cells / skill $\geq 1$ / median skill} &  & & 25 / 1 / 0.23 &  & & 20 / 0 / 0.28 &  & & 20 / 1 / 0.32 \\
\multicolumn{2}{l}{post-minimum targets: cells / skill $\geq 1$ / median skill} &  & & 5 / 3 / 1.13 &  & & 10 / 6 / 1.23 &  & & 10 / 6 / 1.89 \\
```

Fragment f08 (perturbation diagnostic), verbatim:

```latex
\begin{tabular}{llrrrrrrl}
\toprule
Cells & failure definition & $n_{\mathrm{fail}}$ & $n_{\mathrm{succ}}$ & med.\ IQR$_{\mathrm{fail}}$ & med.\ IQR$_{\mathrm{succ}}$ & AUC & $p$ & ordering \\
\midrule
all 90 & skill $\geq 1$ & 17 & 73 & 0.0013 & 0.0054 & 0.317 & 0.019 & failing $<$ succeeding \\
all 90 & improvement $< 0$ & 17 & 73 & 0.0013 & 0.0054 & 0.317 & 0.019 & failing $<$ succeeding \\
all 90 & either & 17 & 73 & 0.0013 & 0.0054 & 0.317 & 0.019 & failing $<$ succeeding \\
\addlinespace[2pt]
pre-minimum targets (65) & either & 2 & 63 & 0.0010 & 0.0056 & 0.198 & 0.175 & failing $<$ succeeding \\
post-minimum targets (25) & either & 15 & 10 & 0.0013 & 0.0028 & 0.433 & 0.598 & failing $<$ succeeding \\
\addlinespace[2pt]
routed \meth{richardson\_1} (54) & either & 7 & 47 & 0.0008 & 0.0035 & 0.219 & 0.015 & failing $<$ succeeding \\
routed \meth{rational\_fit} (36) & either & 10 & 26 & 0.0023 & 0.0083 & 0.300 & 0.069 & failing $<$ succeeding \\
\bottomrule
\end{tabular}
```

Same numbers as Report 5A (the real-data step depends on nothing that changed): 2 of 65 pre-minimum cells fail vs 15 of 25 post-minimum; the pooled inverse ordering of the diagnostic (AUC 0.317, p = 0.019) is a composition effect -- within the post-minimum stratum the AUC is 0.433 (p = 0.60).

## 9. Ten headline facts with provenance (from `FACTS.md`, 209 facts)

| # | fact | value | file | filter | formula |
|---|---|---|---|---|---|
| 1 | g = 0.1 core: predict-L_hat / last_value / oracle median error | 0.0620 / 0.1030 / 0.0077 | `results/phase1/phase1_global.csv` | `target_g == 0.1`, core, capped excluded | `med_error` rows of `constant_assumed`, `last_value`, `constant_oracle` |
| 2 | g = 0.1 core: best accelerator | `log_linear`: med. err 0.0202; win rate vs L_hat 0.809, vs last 0.756; median skill vs L_hat 0.299, vs last 0.332; strict 0.601 | `phase1_global.csv` | `target_g == 0.1`, rank-eligible accelerators | argmin `med_error`; `win_rate_vs_*` = mean over cells of the per-cell win rate |
| 3 | g = 0.1 core: accelerators with win rate > 0.5 vs L_hat / vs last; with strict pooled skill < 1 | 35 / 28 / 9 of 49 (41 rank-eligible) | `phase1_global.csv` | `target_g == 0.1`, `is_trivial == 0` | count(`win_rate_vs_assumed` > 0.5); count(`win_rate_vs_last` > 0.5); count(`med_skill` < 1) |
| 4 | g = 0.1 held-out: best accelerator and the same counts | `rational_fit` 0.0074, win vs L_hat 0.824 / vs last 0.853, strict 0.232; 35 / 31 / 14 of 49 (42 eligible) | `phase1_global_holdout.csv` | `target_g == 0.1`, held-out | as above |
| 5 | classical no-op, g = 0.1 core | 21 of 21 in the MI band; 13 of 21 with win rate vs last in [0.4, 0.6]; 21 of 21 with strict skill >= 0.9 | `phase1_global.csv` | family in the six classical families | count(0.9 <= `med_improve` <= 1.1); count(0.4 <= `win_rate_vs_last` <= 0.6); count(`med_skill` >= 0.9) |
| 6 | dangerous set | `geom_avg_diff, linear, neville_2, neville_3, neville_4, pade_21, pade_31, pade_32`; identical to the legacy set and to the previous run | `results/phase1/dangerous_methods.json` | `dangerous == 1` | `S = valid - 2 cat + 0.4 beats < 0`, core, obs 90, capped excluded |
| 7 | richardson_3 validity by depth (committed aggregate) | obs 30: 0.481, 60: 0.583, 90: 0.981, 120: 0.999; the only accelerator below the 0.9 floor at some depth that the obs-90 artifact does not flag | `results/phase5a/phase5a_validity_by_depth.csv` | `method == richardson_3` | n-weighted `valid_rate` over noise per `obs_idx` |
| 8 | sigma = 0 cancellation NaNs | the 21 classical variants: 3.11 % invalid at sigma = 0 vs 0.12 % / 0.07 % at 0.001 / 0.005 (Phase 1); 4.18 % vs 0.16 % / 0.45 % over four depths (Phase 5a) | `phase1_aggregated.csv`; `phase5a_raw.csv` | family in the six classical families | 100 (1 - mean `valid_rate`) per noise |
| 9 | L_hat invariance (sweep 1b, g = 0.1 core) | max change of median error across the five modes: 0.0 % for richardson_1 / rational_fit / log_fit / median_ensemble, <= 2.7 % for richardson_2/3 and single_exp_fit, 29.7 % for double_exp_fit, 362.8 % for log_linear; constant_assumed 667 % (0.0593 -> 0.0077 from zero to oracle) | `results/phase5b/phase5b_sweep1_consumers.csv` | `regime_set == core`, `target_g == 0.1` | max - min over `assumed_mode` of `med_error`, relative to the min |
| 10 | real data, pre / post minimum | 65 cells / 2 fail / median skill 0.268 / median improvement +0.732 vs 25 / 15 / 1.345 / -0.345; adult argmin 337, bank_marketing 257 | `results/real_data/real_data_summary_v2.csv`; `real_data_curve_minima_v2.csv` | `post_min_target` 0 / 1 | count(`cascade_skill` >= 1); medians |
| + | Phase 5a, g = 0.1 core | `oracle_49` mean err 0.007298 (was `oracle_51` 0.007153), `oracle_9` 0.023986 (was 0.018396), `constant_oracle` 0.013494, `fixed_rational` 0.053127 (median skill 1.0000; 1,895 / 1,885 records beat / lose to the hindsight best-of-four; win rate vs last 0.884), `diag_ensemble_9` 0.094583 (was 0.070965) | `phase5a_ensemble.csv`; `phase5a_raw.csv` | `target_g == 0.1` | mean error over records |
| + | top-10 win rates (vs assumed / last / wmean / wmin), g = 0.1 core | 1. log_linear 0.81/0.76/0.79/0.79; 2. richardson_3 0.74/0.77/0.78/0.82; 3. richardson_2 0.75/0.81/0.82/0.84; 4. double_exp_fit 0.77/0.91/0.90/0.86; 5. rational_fit 0.79/0.88/0.84/0.84; 6. richardson_1 0.74/0.81/0.84/0.83; 7. pade_23 0.76/0.70/0.72/0.70; 8. pade_12 0.77/0.68/0.72/0.68; 9. pade_22 0.65/0.66/0.67/0.64; 10. single_exp_fit 0.67/0.96/0.89/0.92 | `phase1_global.csv` | top 10 by `med_error` | `win_rate_vs_<ref>` |

Fragment f12 (validity by depth), for the record:

```latex
\begin{tabular}{lrrrrr}
\toprule
Method & $\nobs = 30$ & $\nobs = 60$ & $\nobs = 90$ & $\nobs = 120$ & min (depth, $\sigma$) \\
\midrule
$\dagger$\meth{neville\_4} & \textbf{0.473} & \textbf{0.336} & \textbf{0.323} & \textbf{0.328} & 0.034 \\
$\dagger$\meth{linear} & \textbf{0.352} & \textbf{0.363} & \textbf{0.555} & \textbf{0.621} & 0.349 \\
$\dagger$\meth{neville\_3} & \textbf{0.648} & \textbf{0.490} & \textbf{0.403} & \textbf{0.413} & 0.117 \\
\meth{richardson\_3} & \textbf{0.481} & \textbf{0.583} & 0.981 & 0.999 & 0.348 \\
$\dagger$\meth{pade\_21} & \textbf{0.631} & \textbf{0.547} & \textbf{0.617} & \textbf{0.663} & 0.244 \\
$\dagger$\meth{pade\_31} & \textbf{0.586} & \textbf{0.594} & \textbf{0.751} & \textbf{0.772} & 0.407 \\
$\dagger$\meth{geom\_avg\_diff} & \textbf{0.587} & \textbf{0.597} & \textbf{0.605} & \textbf{0.618} & 0.471 \\
$\dagger$\meth{neville\_2} & \textbf{0.708} & \textbf{0.682} & \textbf{0.632} & \textbf{0.674} & 0.481 \\
$\dagger$\meth{pade\_32} & \textbf{0.668} & \textbf{0.731} & \textbf{0.642} & \textbf{0.666} & 0.499 \\
\midrule
\multicolumn{6}{l}{9 of 49 accelerators fall below $\rhoV = 0.9$ at some depth; the dangerous artifact (obs 90 only) flags 8 of them} \\
\bottomrule
\end{tabular}
```

## 10. Pushed commits and `git ls-remote`

```
e5a802b  Redesign v2 (Prompt 5B, part 1): retire the Weniger pair, levin_t2 in the pool, derive every roster count
74c02f1  Paper tables (Prompt 5B, part 3): f01 fixed-reference layout, f04 on 21 classical variants ..., f12 validity by depth
928092a  Replace committed results with the redesign-v2 full run at the 49-method roster (Prompt 5B)
91b221a  Regenerate paper fragments and FACTS.md at the 49-method full run (Prompt 5B)
73e2ea5  make_paper_tables.py: derive the classical-variant count in the sigma = 0 fact labels
ae0f688  Regenerate paper fragments and FACTS.md at the clean HEAD (label fix)
```

Fragment headers read `generated by scripts/make_paper_tables.py at code 73e2ea5, results as of commit 928092a` (clean HEAD, no `-dirty`); all 23 fragments compile against the paper's macros (23 pages, 0 errors).

```
$ git ls-remote origin redesign-v2
ae0f688dd1f1ba173cdd72e01844b9d18e1bbdd6	refs/heads/redesign-v2
```

(`HEAD` at the time of writing this report: `ae0f688`; the report is committed on top and pushed, its hash is printed in the console after the report.)

## 11. Anomalies and open questions

1. **Sweep 1b confirms that L_hat barely enters the accelerators on the full grid** (section 7): `richardson_1`, `rational_fit`, `log_fit` and both ensembles are mode-invariant to four decimals; `richardson_2/3` and `single_exp_fit` move by < 3 %; only `double_exp_fit` (30 %) and `log_linear` (363 %) respond, and `constant_assumed` is the method that the assumed asymptote actually changes (7.7x between `zero` and `oracle`). The paper's assumed-asymptote sensitivity statement is therefore about the trivial comparator and `log_linear`, not about the accelerators; the vs-assumed skill columns move with the reference, which is why decision 2 made `last_value` the headline reference.
2. **`log_linear` under the `double` mode** drops to a 0.48 win rate vs last (core, g = 0.1) with median error 0.0500 vs 0.0240 under `zero`: an over-estimated asymptote breaks the log transform. The rank-1 core method is the L_hat-sensitive one.
3. **Phase 5b sweep 3 at 1.353 %** crosses the threshold with the Phase-1 method set at obs 90; the sweep writes no per-record file, so the sigma = 0 share cannot be separated after the fact. If the rule is to be applied strictly in future runs, sweep 3 needs a per-record or per-noise validity output (a small change to `phases/phase5b.py::sweep3_catmult`).
4. **`richardson_3` is still the one accelerator that f12 flags and the artifact does not** (below 0.9 at obs 30 and 60, 0.98 at obs 90). The Prompt-5A open question stands: a per-depth derivation or a minimum-window rule for the 7-parameter fit.
5. **The pool swap moved the Phase-5a ensemble numbers substantially -- the retired `weniger_d2` had been an asymptote anchor.** At g = 0.1 core, mean error: `diag_ensemble_9` 0.0710 -> 0.0946, `equal_ensemble_9` 0.0753 -> 0.0967, `oracle_9` 0.0184 -> 0.0240, `diag_ensemble_49` / `capped_diag_49` 0.0952 -> 0.1075 (from the 51-versions), `oracle_49` 0.00715 -> 0.00730; the fixed defaults, the cascade, the trivial selectors and Phase 3 are unchanged to 1e-4. Mechanism: the previous run's `weniger_d2` returned the constant 0 = L_hat on every record, so inside the 9-method pool it was a `constant_assumed` member that pulled every ensemble toward the asymptote and was the per-record best member whenever the true value was near 0; `levin_t2` is a genuine, noisier accelerator. Consequences in the ablation (`phase5a_ablation.csv`, g = 0.1): `diag_vs_equal_49` 0.0129 -> 0.0018, `weighting_gain_9` 0.0043 -> 0.0021, `pool_expansion_gain` -0.033 -> -0.013. Every ensemble conclusion in the paper must be re-read from this run: the diagnostic-weighting benefit is now ~0.002 and `fixed_rational` (0.0531) beats every ensemble by a wider margin.
6. **The Phase-4 pooled diagnostic correlation rose from 0.225 to 0.385** (`ALL`, perturb_IQR vs error, core) once the constant-output member left the pool: `weniger_d2`'s perturbation IQR was 0 on every record while its error was not, which dragged the pooled Spearman r down; `levin_t2` now contributes r = 0.446 (shift_IQR undefined for it, NaN, as for `anderson_1`). Per-method correlations of the other eight methods are unchanged. `f10a` and the diagnostics narrative should be re-read from this run.
7. **Phase 5b now takes 1.5 h** (sweep 1b 51 min); `reproduce_all.py` total 5.4 h. Sweep 1b's cost is the two ensembles (each call evaluates an 8-method pool); restricting sweep 1b to the eight direct consumers would halve it. Not changed.
8. **Both quick-run worktrees are still present** (`../wt-5b` at `e5a802b`, `../wt-report3a` at `8d33e82`); `../wt-5a` was removed. `git worktree remove --force` drops them.
9. **The previous run's `phase5b_sweep1_global.csv` gained a `note` column** and the sweep-1a numbers are unchanged; f05 (sweep 1a) is still generated and still labelled as the clamped, near-no-op part.
10. **f01's median-accelerator "skill vs last" is 1.000 at every stratum**: the median over the 49 accelerators of `med_skill_vs_last` sits exactly at the identity because the 21 classical variants return (almost) the last value on the median cell; this is the no-op result seen from the other side, not a formatting artefact.

Stopped here.
