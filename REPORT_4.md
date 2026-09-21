**Report back**

# Redesign v2, Prompt 4: paper tables for the v2 schema, FACTS.md, docs and hygiene

Date: 2026-09-21. Branch `redesign-v2` in `code/`, on top of the full-run results commit `30de724` (Report 3B). Deliverables commit `16372ee`, pushed (section 7); this report is committed on top. No method implementation, generator, phase or pipeline code was changed; the results were not re-run. `python -m pytest tests/ -q`: 342 passed in 34.4 s after the changes.

## 1. What was done

| item | where |
|---|---|
| `scripts/make_paper_tables.py` rewritten for the v2 schema (1,100 lines; the v1 code and the Prompt-2 refusal stub are gone). Reads the per-stratum files explicitly (`phase1_global.csv` / `phase1_global_holdout.csv` keyed by `target_g`, `phase2_*_g{g}.csv`, `phase5b_sweep*_global.csv`, `real_data_*_v2.csv`, the `*_capped.csv` blocks); never reads `phase2_correlations.csv` / `phase2_rules.csv` (the legacy-named headline copies). Writes 20 booktabs `tabular` fragments to `paper_fragments/` with a provenance comment header (sources, filter, code and results commit), an index `paper_fragments/README.md`, and `FACTS.md`. Options `--results`, `--out`, `--facts`, `--no-raw`. | `scripts/make_paper_tables.py` |
| 20 fragments (section 2), all compiled once with `pdflatex` against the paper's macro definitions (`\meth`, `\diag`, `\Stab`, `\rhoC`, `\rhoV`, `\TE`, `\LE`, `\nobs`, `\nf` from `paper_final.tex`): 20 pages, 0 errors, 0 overfull boxes. | `paper_fragments/` |
| `FACTS.md`: 172 facts in 19 sections, each row = fact, value, file, filter, formula; the five named facts have their own entries (section 5). Two facts (richardson_3 validity by depth; the record-level skill split of `fixed_rational`) come from the git-ignored `phase5a_raw.csv` and are labelled so; with `--no-raw`, or when the file is absent, they print "not recomputed" with the recorded values. | `FACTS.md` |
| `README.md` rewritten for the v2 design: hidden heterogeneous asymptotes, the L_hat modes, trivial comparators and skill, gap strata with capping, core / held-out regimes, validity floor, the dangerous-artifact step, provenance; reproduction commands with the measured wall times; table generation; per-phase grid table; tests; full file map; where the numbers are; legacy material; data availability. The v1 "headline findings" paragraph is gone (nothing about the v2 findings is asserted in the README beyond pointers to `FACTS.md`). | `README.md` |
| `results/README.md`: the stale note ("five large intermediate files ... excluded", naming the two Phase-2 CSVs) replaced by the correct one (three raw per-record files excluded; the two Phase-2 CSVs are committed) plus a layout table of the v2 result files. | `results/README.md` |
| `scripts/derive_dangerous.py`: the artifact's `source` is now the repo-relative POSIX path (`os.path.relpath(..., _ROOT)`); docstring updated. Artifact regenerated and committed (section 6). | `scripts/derive_dangerous.py`, `results/phase1/dangerous_methods.json` |
| `reproduce_all.py` docstring: the v2 ordering step by step (Phase 0, Phase 1, derivation, Phases 2-5b, real data), the Phase-5a worker-process note, measured wall times of the full run, `--plan`, and the two follow-up commands (`check_dangerous.py`, `make_paper_tables.py`). Code unchanged. | `reproduce_all.py` |

## 2. Fragment inventory (`paper_fragments/`, 20 files + `README.md`)

Prompt item -> fragment. Items that split by stratum or regime set have one file per part; every file is one `tabular`.

| fragment | lines / bytes | content | sources |
|---|---|---|---|
| `f01_trivial_baseline.tex` | 49 / 4,215 | Trivial comparators vs accelerators: the four deployable trivials, the oracle constant as a labelled reference, the rank-1 accelerator of each regime set, the median accelerator, and how many of the 51 accelerators beat the best deployable trivial (count by pooled median error / count with pooled median skill < 1) | results/phase1/phase1_global.csv; results/phase1/phase1_global_holdout.csv |
| `f02_ranking_g0.5.tex` | 66 / 7,055 | Main ranking at g = 0.5: rank pool = accelerators above the validity floor, core and held-out side by side, unranked block appended | results/phase1/phase1_global.csv; results/phase1/phase1_global_holdout.csv |
| `f02_ranking_g0.1.tex` | 66 / 7,061 | Main ranking at g = 0.1: rank pool = accelerators above the validity floor, core and held-out side by side, unranked block appended | results/phase1/phase1_global.csv; results/phase1/phase1_global_holdout.csv |
| `f02_ranking_g0.02.tex` | 66 / 7,057 | Main ranking at g = 0.02: rank pool = accelerators above the validity floor, core and held-out side by side, unranked block appended | results/phase1/phase1_global.csv; results/phase1/phase1_global_holdout.csv |
| `f03_skill_summary.tex` | 30 / 1,991 | Skill summary: fraction of (method x regime x noise) cells whose median skill is below 1 (the method beat the best-of-four deployable trivial on the median seed), per family, per stratum, core vs held-out | results/phase1/phase1_aggregated.csv |
| `f04_classical_noop.tex` | 44 / 4,264 | Classical no-op table by stratum: median improvement factor vs the +/-10 % band, and median skill | results/phase1/phase1_global.csv |
| `f05_sweep1_modes.tex` | 52 / 3,823 | Assumed-asymptote mode sweep: Phase-2 cascade metrics under each L_hat mode, core and held-out, per stratum | results/phase5b/phase5b_sweep1_global.csv |
| `f06_generalisation.tex` | 66 / 5,888 | Held-out vs core generalisation: per-method rank shift per stratum, with median errors at the headline stratum; rational_fit and log_linear in bold | results/phase1/phase1_global.csv; results/phase1/phase1_global_holdout.csv |
| `f07_real_data_v2.tex` | 49 / 3,633 | Real-data re-evaluation over the (depth x target) grid: routed method, cascade error and skill per cell | results/real_data/real_data_summary_v2.csv |
| `f07b_real_data_legacy18.tex` | 35 / 2,520 | Legacy 18-cell real-data table (pre-redesign design), preserved unchanged | results/real_data/real_data_results.csv |
| `f08_real_perturb_diagnostic.tex` | 17 / 2,114 | Real-data perturbation diagnostic: does the routed method's perturb_iqr separate failing cells from succeeding ones? | results/real_data/real_data_summary_v2.csv |
| `f09a_selectors_phase3.tex` | 18 / 1,307 | Phase 3 selectors: mean achieved stability per stratum and under leave-one-regime-out cross-validation | results/phase3/phase3_selector_comparison.csv; results/phase3/phase3_cv_results.csv |
| `f09b_ensemble_phase5a_core.tex` | 37 / 2,957 | Phase 5a selectors and ensembles, core regimes: median error and median skill per stratum (mean error at the headline stratum), oracle_51 vs constant_oracle vs the fixed defaults | results/phase5a/phase5a_ensemble.csv |
| `f09c_ensemble_phase5a_holdout.tex` | 37 / 2,973 | Phase 5a selectors and ensembles, held-out regimes: median error and median skill per stratum (mean error at the headline stratum), oracle_51 vs constant_oracle vs the fixed defaults | results/phase5a/phase5a_ensemble_holdout.csv |
| `f09d_ablation_phase5a.tex` | 19 / 1,190 | Phase 5a ablation: paired mean error differences between selectors per stratum | results/phase5a/phase5a_ablation.csv |
| `f10a_diagnostics_correlations.tex` | 24 / 1,708 | Phase 4 diagnostics: correlation of perturb_IQR and shift_IQR with error, core pooled and held-out | results/phase4/phase4_diagnostic_correlations.csv; results/phase4/phase4_diagnostic_correlations_holdout.csv |
| `f10b_diagnostics_depth.tex` | 19 / 1,055 | Phase 4: perturb_IQR reliability by observation depth at the headline stratum | results/phase4/phase4_obs_reliability.csv |
| `f10c_diagnostics_ensemble.tex` | 19 / 1,102 | Phase 4 selectors with the trivial references: error and skill at the headline stratum | results/phase4/phase4_ensemble.csv |
| `f10d_diagnostics_filter.tex` | 17 / 819 | Phase 4: the perturb_IQR screen applied to the Phase-2 cascade | results/phase4/phase4_cascade_filter.csv |
| `f11_capped_block.tex` | 22 / 2,173 | Capped block: every capped (regime x stratum) cell of the main run with the achieved gap fraction, and the capped-cell counts of the depth grids | results/phase1/phase1_capped.csv; results/phase1/phase1_horizons.csv; results/phase2/phase2_capped.csv; results/phase4/phase4_capped.csv; results/phase5a/phase5a_capped.csv; results/phase3/phase3_capped_cells.csv |

Mapping to the prompt: (1) `f01`; (2) `f02_ranking_g{0.5,0.1,0.02}`; (3) `f03`; (4) `f04`; (5) `f05`; (6) `f06`; (7) `f07` + `f07b`; (8) `f08`; (9) `f09a-d`; (10) `f10a-d`; (11) `f11`.

Conventions common to all fragments: ranks are among the 51 accelerators only (trivial comparators are never ranked; the oracle never enters a count); the rank pool is the accelerators with `rank_eligible == 1` (pooled valid rate >= 0.9); every pooled number excludes capped cells; a dagger marks a method in the dangerous artifact; error = |estimate - true value at n_f|; skill = error / best-of-four deployable trivial error on the same cell.

## 3. Fragment (1), the paper's first result, verbatim

```latex
% AUTO-GENERATED -- do not hand-edit.  generated by scripts/make_paper_tables.py at code 43eb1be-dirty, results as of commit 30de724
% Trivial comparators vs accelerators: the four deployable trivials, the oracle constant as a labelled reference, the rank-1 accelerator of each regime set, the median accelerator, and how many of the 51 accelerators beat the best deployable trivial (count by pooled median error / count with pooled median skill < 1)
% source : results/phase1/phase1_global.csv (core regimes, all three strata, capped cells excluded)
% source : results/phase1/phase1_global_holdout.csv (held-out regimes)
% filter : per stratum; oracle excluded from every count; skill = err / best-of-four deployable trivial per cell
% note   : 'accelerators beating best deployable trivial': first number = med_error below the best single deployable trivial's med_error; second = pooled med_skill < 1 (beats the per-cell best-of-four on the median cell)
\begin{tabular}{lrrrrrrrr}
\toprule
Stratum / row & \multicolumn{4}{c}{core (18 regimes)} & \multicolumn{4}{c}{held-out (6 regimes)} \\
\cmidrule(lr){2-5}\cmidrule(lr){6-9}
 & med.\ err & med.\ skill & $\rhoV$ & $\rhoC$ & med.\ err & med.\ skill & $\rhoV$ & $\rhoC$ \\
\midrule
\multicolumn{9}{l}{\textit{$g = 0.5$}} \\
\meth{window\_min} & 0.0484 & 1.000 & 1.000 & 0.040 & 0.0340 & 1.000 & 1.000 & 0.000 \\
\meth{last\_value} & 0.0690 & 1.021 & 1.000 & 0.000 & 0.0356 & 1.000 & 1.000 & 0.000 \\
\meth{window\_mean} & 0.1066 & 1.792 & 1.000 & 0.065 & 0.0815 & 2.438 & 1.000 & 0.141 \\
\meth{constant\_assumed} & 0.1094 & 2.313 & 1.000 & 0.251 & 0.0853 & 2.261 & 1.000 & 0.320 \\
\meth{constant\_oracle} \textit{(reference, not deployable)} & 0.0579 & 1.000 & 1.000 & 0.001 & 0.0356 & 0.997 & 1.000 & 0.006 \\
\addlinespace[2pt]
best accelerator, core rank 1: \meth{log\_linear} & 0.0111 & 0.362 & 0.982 & 0.091 & 0.0077 & 0.283 & 1.000 & 0.004 \\
best accelerator, held-out rank 1: \meth{rational\_fit} & 0.0159 & 0.495 & 1.000 & 0.012 & 0.0048 & 0.160 & 1.000 & 0.000 \\
median accelerator (51) & 0.0507 & 1.047 & 0.982 & 0.048 & 0.0306 & 1.028 & 1.000 & 0.009 \\
accelerators beating best deployable trivial & 17/51 & 14/51 & & & 33/51 & 17/51 & & \\
\midrule
\multicolumn{9}{l}{\textit{$g = 0.1$} (headline)} \\
\meth{constant\_assumed} & 0.0620 & 1.000 & 1.000 & 0.183 & 0.0561 & 1.000 & 1.000 & 0.136 \\
\meth{window\_min} & 0.0681 & 1.493 & 1.000 & 0.044 & 0.0766 & 1.616 & 1.000 & 0.000 \\
\meth{last\_value} & 0.1030 & 1.570 & 1.000 & 0.000 & 0.0767 & 1.632 & 1.000 & 0.000 \\
\meth{window\_mean} & 0.1463 & 2.733 & 1.000 & 0.015 & 0.1382 & 2.773 & 1.000 & 0.000 \\
\meth{constant\_oracle} \textit{(reference, not deployable)} & 0.0077 & 0.164 & 1.000 & 0.000 & 0.0084 & 0.174 & 1.000 & 0.000 \\
\addlinespace[2pt]
best accelerator, core rank 1: \meth{log\_linear} & 0.0202 & 0.601 & 0.953 & 0.112 & 0.0253 & 0.558 & 1.000 & 0.000 \\
best accelerator, held-out rank 1: \meth{rational\_fit} & 0.0296 & 0.670 & 1.000 & 0.009 & 0.0074 & 0.232 & 1.000 & 0.002 \\
median accelerator (51) & 0.0502 & 1.530 & 0.979 & 0.043 & 0.0359 & 1.307 & 1.000 & 0.002 \\
accelerators beating best deployable trivial & 35/51 & 9/51 & & & 33/51 & 14/51 & & \\
\midrule
\multicolumn{9}{l}{\textit{$g = 0.02$}} \\
\meth{constant\_assumed} & 0.0541 & 1.000 & 1.000 & 0.187 & 0.0545 & 1.000 & 1.000 & 0.127 \\
\meth{last\_value} & 0.0970 & 1.873 & 1.000 & 0.000 & 0.0834 & 2.042 & 1.000 & 0.000 \\
\meth{window\_min} & 0.0970 & 1.962 & 1.000 & 0.048 & 0.0834 & 2.042 & 1.000 & 0.000 \\
\meth{window\_mean} & 0.1749 & 2.782 & 1.000 & 0.014 & 0.1450 & 3.590 & 1.000 & 0.000 \\
\meth{constant\_oracle} \textit{(reference, not deployable)} & 0.0011 & 0.031 & 1.000 & 0.000 & 0.0017 & 0.041 & 1.000 & 0.000 \\
\addlinespace[2pt]
best accelerator, core rank 1: \meth{richardson\_2} & 0.0218 & 0.883 & 0.992 & 0.028 & 0.0117 & 0.537 & 0.989 & 0.013 \\
best accelerator, held-out rank 1: \meth{rational\_fit} & 0.0228 & 0.605 & 1.000 & 0.012 & 0.0089 & 0.250 & 1.000 & 0.002 \\
median accelerator (51) & 0.0515 & 1.686 & 0.979 & 0.042 & 0.0386 & 1.314 & 1.000 & 0.004 \\
accelerators beating best deployable trivial & 33/51 & 8/51 & & & 34/51 & 12/51 & & \\
\bottomrule
\end{tabular}
```

Reading it: at the headline stratum (g = 0.1, core) the best deployable trivial is `constant_assumed` (L_hat = 0) at median error 0.0620 against the oracle constant's 0.0077; the rank-1 accelerator `log_linear` reaches 0.0202 (median skill 0.601); 35 of 51 accelerators have a lower pooled median error than the best single deployable trivial, but only 9 of 51 beat the per-cell best-of-four on their median cell (pooled median skill < 1), and the median accelerator has median skill 1.53. On the held-out regimes `rational_fit` is rank 1 (0.0074, skill 0.232) and 14 of 51 have median skill < 1. At g = 0.5 `window_min` beats even the oracle constant (0.0484 vs 0.0579): the target has not approached the asymptote yet.

## 4. Skill summary (fragment 3), verbatim

```latex
% AUTO-GENERATED -- do not hand-edit.  generated by scripts/make_paper_tables.py at code 43eb1be-dirty, results as of commit 30de724
% Skill summary: fraction of (method x regime x noise) cells whose median skill is below 1 (the method beat the best-of-four deployable trivial on the median seed), per family, per stratum, core vs held-out
% source : results/phase1/phase1_aggregated.csv (per (method, regime, noise, g) cell: median skill over 30 seeds)
% filter : capped == 0, is_oracle == 0; a cell counts when its median-seed skill is < 1 (NaN median skill counts as not < 1)
% note   : K = methods in the family; T = method type (\TE trajectory extrapolator / \LE limit estimator)
\begin{tabular}{lrlrrrrrr}
\toprule
Family & $K$ & T & \multicolumn{2}{c}{$g = 0.5$} & \multicolumn{2}{c}{$g = 0.1$} & \multicolumn{2}{c}{$g = 0.02$} \\
\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}
 & & & core & held-out & core & held-out & core & held-out \\
\midrule
richardson & 6 & \TE & 67\% & 75\% & 49\% & 63\% & 49\% & 66\% \\
parametric & 4 & \TE & 72\% & 78\% & 54\% & 62\% & 47\% & 53\% \\
pade & 8 & \TE & 46\% & 64\% & 32\% & 46\% & 27\% & 36\% \\
neville & 3 & \TE & 24\% & 33\% & 19\% & 31\% & 19\% & 31\% \\
baseline & 4 & mixed & 30\% & 35\% & 21\% & 30\% & 21\% & 27\% \\
ensemble & 3 & mixed & 26\% & 35\% & 20\% & 36\% & 21\% & 33\% \\
shanks & 4 & \LE & 20\% & 29\% & 18\% & 33\% & 18\% & 30\% \\
wynn\_eps & 3 & \LE & 23\% & 30\% & 19\% & 33\% & 18\% & 31\% \\
wynn\_rho & 3 & \LE & 19\% & 17\% & 14\% & 11\% & 13\% & 11\% \\
levin & 6 & \LE & 15\% & 22\% & 17\% & 26\% & 17\% & 24\% \\
brezinski & 2 & \LE & 28\% & 39\% & 19\% & 33\% & 18\% & 30\% \\
weniger & 2 & \LE & 0\% & 0\% & 0\% & 0\% & 0\% & 0\% \\
anderson & 3 & \LE & 23\% & 31\% & 4\% & 20\% & 4\% & 20\% \\
\midrule
trivial \textit{(4 deployable)} & 4 & \LE & 0\% & 0\% & 0\% & 0\% & 0\% & 0\% \\
\midrule
\textbf{all accelerators} & 51 & & 34\% & 43\% & 25\% & 36\% & 24\% & 33\% \\
\bottomrule
\end{tabular}
```

The `trivial (4 deployable)` row is 0 % by construction (a deployable trivial's skill is >= 1, = 1 when it is the best of the four); it is a sanity row. `weniger` at 0 % everywhere is explained in section 8, anomaly 1.

## 5. The answer fragment (8) gives, verbatim

Header note of `f08_real_perturb_diagnostic.tex` (also the last bullet of the "Named facts" block of `FACTS.md`):

> On the 90-cell grid the perturb_iqr of the routed method weakly separates failing cells (skill >= 1 or negative improvement; n = 17) from succeeding ones (n = 73) -- but in the INVERSE direction (failing cells have the LOWER IQR): AUC = 0.317 with a higher IQR read as 'failure', two-sided Mann-Whitney p = 0.019; ordering: failing < succeeding (median IQR 0.0013 vs 0.0054). With skill >= 1 alone: AUC = 0.317, p = 0.019, failing < succeeding (n_fail = 17). With negative improvement alone: AUC = 0.317, p = 0.019, failing < succeeding (n_fail = 17). Routed richardson_1 only: AUC = 0.219, p = 0.015, failing < succeeding (n_fail = 7 of 54). Routed rational_fit only: AUC = 0.300, p = 0.069, failing < succeeding (n_fail = 10 of 36). The two failure definitions select the same cells.

The tabular:

```latex
\begin{tabular}{llrrrrrrl}
\toprule
Cells & failure definition & $n_{\mathrm{fail}}$ & $n_{\mathrm{succ}}$ & med.\ IQR$_{\mathrm{fail}}$ & med.\ IQR$_{\mathrm{succ}}$ & AUC & $p$ & ordering \\
\midrule
all 90 & skill $\geq 1$ & 17 & 73 & 0.0013 & 0.0054 & 0.317 & 0.019 & failing $<$ succeeding \\
all 90 & improvement $< 0$ & 17 & 73 & 0.0013 & 0.0054 & 0.317 & 0.019 & failing $<$ succeeding \\
all 90 & either & 17 & 73 & 0.0013 & 0.0054 & 0.317 & 0.019 & failing $<$ succeeding \\
\addlinespace[2pt]
routed \meth{richardson\_1} (54) & either & 7 & 47 & 0.0008 & 0.0035 & 0.219 & 0.015 & failing $<$ succeeding \\
routed \meth{rational\_fit} (36) & either & 10 & 26 & 0.0023 & 0.0083 & 0.300 & 0.069 & failing $<$ succeeding \\
\bottomrule
\end{tabular}
```

So: the diagnostic does carry signal on the recorded curves, but with the sign reversed relative to the synthetic Phase-4 finding (where perturb_IQR correlates positively with error): the 17 failing cells are the ones whose routed prediction is *stable* under window perturbation. The 17 failing cells are 9 in `bank_marketing`, 7 in `adult`, 1 in `jannis`, at depths 60 / 120 / 150 (none at 30 or 90); the worst is `bank_marketing` depth 150 -> round 500 (`rational_fit`, skill 51.7). The two failure definitions coincide because `last_value` is the best trivial on 88 of the 90 cells, so skill >= 1 is the same event as improvement <= 0.

## 6. Ten headline numbers from FACTS.md, with provenance

| # | fact | value | file | filter | formula |
|---|---|---|---|---|---|
| 1 | g = 0.1 core: best deployable trivial / oracle constant / best accelerator | `constant_assumed` 0.0620 / `constant_oracle` 0.0077 / `log_linear` 0.0202 (median skill 0.601) | `results/phase1/phase1_global.csv` | `target_g == 0.1`, `regime_set == core`, capped excluded; best accelerator over `is_trivial == 0`, `rank_eligible == 1` | argmin `med_error` over the four deployable trivials; the oracle row; argmin `med_error` over rank-eligible accelerators |
| 2 | g = 0.1 core: accelerators beating the best deployable trivial | 35 of 51 by pooled median error (43 rank-eligible); 9 of 51 with pooled median skill < 1 | `results/phase1/phase1_global.csv` | `target_g == 0.1`, `is_trivial == 0` | count(`med_error` < min trivial `med_error`); count(`med_skill` < 1) |
| 3 | g = 0.1 held-out: best accelerator and the same counts | `rational_fit` 0.0074 (skill 0.232); 33 of 51 by median error, 14 of 51 by skill | `results/phase1/phase1_global_holdout.csv` | `target_g == 0.1`, `regime_set == holdout` | as above |
| 4 | cells with median skill < 1, all accelerators, g = 0.1 | core 25.2 % (617 / 2,448 cells); held-out 36.5 % (279 / 765) | `results/phase1/phase1_aggregated.csv` | `is_trivial == 0`, `target_g == 0.1`, `capped == 0`, `is_holdout` 0 / 1 | mean over (method, regime, noise) cells of [`med_skill` < 1] |
| 5 | classical no-op band | 21 of 23 classical variants with MI in [0.9, 1.1] at every stratum; 23 of 23 with median skill >= 0.9 | `results/phase1/phase1_global.csv` | core; family in the seven classical families | count(0.9 <= `med_improve` <= 1.1); count(`med_skill` >= 0.9) |
| 6 | dangerous set vs legacy | identical: `geom_avg_diff, linear, neville_2, neville_3, neville_4, pade_21, pade_31, pade_32`; `+[] -[]`; nearest non-dangerous `weniger_d1` S = +0.646 | `results/phase1/dangerous_methods.json` vs `config.LEGACY_DANGEROUS_METHODS` | `dangerous == 1`; `table` | set difference; min S over `dangerous == 0` |
| 7 | richardson_3 validity by depth; scope of the derivation | valid rate obs 30: 0.481, 60: 0.583, 90: 0.981, 120: 0.999; the derivation reads Phase 1 only (obs_idx = 90, noise {0, 0.001, 0.005}, 30 seeds, core regimes) | `results/phase5a/phase5a_raw.csv` (git-ignored; 2026-09-21 run) ; `src/config.py PHASE1`, `scripts/derive_dangerous.py` | `method == richardson_3` | mean(`valid`) per `obs_idx` |
| 8 | sigma = 0 cancellation NaNs | the 23 classical (difference-based) variants: invalid 3.14 % at sigma = 0 vs 0.12 % / 0.10 % at 0.001 / 0.005; all non-dangerous accelerators: 2.08 % vs 0.39 % / 0.45 % | `results/phase1/phase1_aggregated.csv` | family in the seven classical families; `method` not dangerous, `is_trivial == 0` | 100 (1 - mean `valid_rate`) per noise level |
| 9 | generalisation | Spearman rho(core rank, held-out rank) = 0.885 / 0.818 / 0.896 at g = 0.5 / 0.1 / 0.02; `rational_fit` rank 5 -> 1 and `log_linear` 1 -> 10 at g = 0.1 | `phase1_global.csv`; `phase1_global_holdout.csv` | rank-eligible in both sets | `spearmanr` of the two rank vectors |
| 10 | Phase 5a at g = 0.1, core | `oracle_51` mean err 0.007153 (median skill 0.085) vs `constant_oracle` 0.013494 (0.207) vs `fixed_rational` 0.053127 (median skill 1.0000: 1,895 of 3,780 records beat the best-of-four trivial, 1,885 lose) | `results/phase5a/phase5a_ensemble.csv`; `phase5a_raw.csv` for the split | `selector in (oracle_51, constant_oracle, fixed_rational)`, `target_g == 0.1` | mean / median error over records; count(`skill` < 1), count(`skill` > 1) |
| + | real data (v2) | median cascade skill 0.336 over 90 cells; 17 of 90 with skill >= 1; perturbation diagnostic AUC 0.317, p = 0.019, inverse ordering | `results/real_data/real_data_summary_v2.csv` | all rows | median(`cascade_skill`); count(`cascade_skill` >= 1); Mann-Whitney U / (n_fail n_succ) |
| + | capped cells (Phase 1, obs 90) | 7 (regime, g) cells: `log_slow` g 0.1 / 0.02 (achieved 0.419), `log_oscillatory` 0.1 / 0.02 (0.299), `broken_power_law` 0.02 (0.080), `random_knots` 0.1 (median n_f 3,850) and 0.02 (45,170); depth grids: Phase 2 72, Phase 4 29, Phase 5a 29 (regime, depth, g) cells | `phase1_capped.csv`, `phase1_horizons.csv`, `phase{2,4,5a}_capped.csv` | `capped == 1` | distinct (regime, obs_idx, target_g) |

## 7. Regenerated artifact and the pushed commit

`python scripts/derive_dangerous.py` after the `relpath` change, on the committed `phase1_aggregated.csv` (12,096 rows):

| field | before (`30de724`) | after (`16372ee`) |
|---|---|---|
| `source` | `C:\Users\kianu\Dropbox\Projects\Ongoing\Accelerator_project\code\results\phase1\phase1_aggregated.csv` | `results/phase1/phase1_aggregated.csv` |
| `dangerous_methods` | 8 | **unchanged**: `geom_avg_diff, linear, neville_2, neville_3, neville_4, pade_21, pade_31, pade_32` |
| `table` | 51 rows | **byte-identical** (51 rows, every S / valid / cat / beats value equal) |
| `git_head` / `created` | `0375f24` / 2026-09-21T00:06:04 | `43eb1be` / 2026-09-21T10:20:22 |
| `pool` / `n_pool` / `criterion` / weights / modes | | unchanged |

`python scripts/check_dangerous.py`: `declared in artifact 8, re-derived from Phase 1 table 8, MATCH`. The `git_head` field now records the commit the derivation ran from (`43eb1be`, code identical to `0375f24` for everything the derivation touches) rather than the commit of the Phase-1 run; the Phase-1 run itself is dated by `REPORT_3B_full_run.txt` and the results commit `30de724`.

Deliverables commit (generator, 20 fragments + index, `FACTS.md`, `README.md`, `results/README.md`, `reproduce_all.py`, `scripts/derive_dangerous.py`, the artifact; 22 added, 6 modified), pushed:

```
16372ee81d195acfb09304d55f625b91626837b5   Paper tables for redesign v2 (Prompt 4): make_paper_tables.py, fragments, FACTS.md, docs
```

```
$ git ls-remote origin redesign-v2
16372ee81d195acfb09304d55f625b91626837b5	refs/heads/redesign-v2
```

This report is committed on top as a second commit and pushed; its hash and the post-push `ls-remote` are printed in the console after the report.

## 8. Anomalies and open questions

1. **`weniger_d1` and `weniger_d2` are the `constant_assumed` predictor.** With `w_n = s_n` the Weniger numerator in `src/accelerators.py::_weniger_delta` is `sum_j (-1)^j C(k, j) beta_j`, which is identically 0 for k = 1 and k = 2, so the transform returns 0 = L_hat whenever its denominator is non-zero. Verified on the raw Phase-1 records: `weniger_d1` returns |estimate| < 1e-12 on all 6,378 valid records and its error equals `constant_assumed`'s on every one; `weniger_d2` is ~0 on 6,387 of 6,408 (the rest are floating-point residue up to 0.5); on the committed aggregate the median error equals `constant_assumed`'s on 204 of 213 / 216 cells (`FACTS.md`, *method implementation notes*). Consequences for the paper: the "weniger" family row of the skill summary (0 %) and the two Weniger rows of the no-op table are properties of a degenerate implementation, not of the Weniger delta transform; `weniger_d2` is one of the 9 methods of the Phase 2/3/4 pool and one of the Phase-3 selector candidates, so it enters those results as a second copy of `constant_assumed`; and `weniger_d1` is the nearest non-dangerous accelerator (S = +0.646). **Open question**: fix the remainder estimate (a re-run of every phase; the 5b sweep-3 champions and the artifact would have to be re-derived) or document it as a known no-op and drop the two rows from the paper's method roster? Not changed here.
2. **Sweep 1 (assumed-asymptote modes) is a near no-op by construction.** The feature extractor clamps `L0 = max(0, min(L_hat, 0.5 * min(window)))` (`phases/phase5b.py::_cascade_features`, `src/trajectories.py::extract_features`), so `half`, `oracle`, `double` and `winmin` coincide whenever `L_hat >= 0.5 * min(window)`, which is the normal case near convergence. In `f05` the four non-zero modes agree to 8e-6 in mean gain at g = 0.1 (max spread across all five modes 1.5e-4; `FACTS.md`, *sweep 1*). The sweep therefore tests only L0 in {0, ~0.5 min(window)}, and the two Phase-2 rule methods use `L_hat` only as a curve-fit starting value. **Open question**: is this the intended robustness statement ("the cascade is insensitive to L_hat because the features barely use it") or should the clamp be lifted for the sweep?
3. **`fixed_rational` median skill prints as exactly 1.0000 in Phase 4 and 5a.** Not a clip: `rational_fit` beats the best-of-four trivial on 1,895 of 3,780 core g = 0.1 records and loses on 1,885, so the median falls on the boundary (fact 10). Worth a sentence in the paper, since a reader will suspect a bug.
4. **The two real-data failure definitions coincide** (skill >= 1 and improvement < 0 select the same 17 cells) because `last_value` is the best trivial reference on 88 of 90 cells (`window_min` on the other 2). The fragment reports all three definitions anyway.
5. **Fragment (1)'s "beating" count is two numbers**, not one: by pooled median error against the best single deployable trivial (35/51 at g = 0.1 core) and by pooled median skill < 1 against the per-cell best-of-four (9/51). The second is the stricter, per-cell statement; the paper should say which it uses. Both are in `FACTS.md`.
6. **Paper title.** The README keeps the submitted title including the subtitle "Why Sequence Acceleration Fails and Simple Curve Fits Suffice" (marked "title as submitted") because I have no authority to retitle the paper; under the redesign the subtitle is a claim to be re-verified. Open question for Kian.
7. **Provenance hash in fragment headers.** The generator records HEAD at run time plus `-dirty` when the tree has uncommitted changes; the committed fragments therefore say `code 43eb1be-dirty, results as of commit 30de724` (the generator's own changes landed in `16372ee`). Documented in `paper_fragments/README.md`; regenerating at a clean HEAD changes only that header line.
8. **`richardson_3` is rank 2 at g = 0.1 (core) and rank 3 on the held-out regimes** on the obs-90 main run while being 48-58 % valid at obs 30-60 (fact 7). The ranking table cannot show this depth dependence; the paper's deployment recommendation should not rest on `richardson_3` without a minimum-window condition.
9. **`f10a`: shift_IQR is undefined for `current_value`, `weniger_d2` and `anderson_1`** in the stored Phase-4 table (NaN Spearman r); for the two accelerators this is again the constant-output degeneracy (`weniger_d2`) or an identical-under-shift estimate. Shown as `--`.
10. **Fragment sizes.** `f02` (57 lines, 13 columns) and `f06` (57 lines, 12 columns) are appendix-width landscape tables; `f05` has 30 data rows. Trimming is a paper-layout decision; the generator keeps every row so nothing is hand-selected.
11. **Legacy 18-cell table (`f07b`)** recomputes the old "reduction in mean error" aggregate by depth from `real_data_results.csv` (all 18 cells: +58.7 %; 3 of 18 cells worse than the current value) purely for the record; it is a v1 quantity and must not be mixed with the v2 grid.
12. **Nothing else was touched**: no results re-run, no phase code changed, the 3A worktree `../wt-report3a` still exists, the `*.log` ignore rule still applies (the run logs are `.txt`).

Stopped here; Claude (chat) takes over for verification and the paper.
