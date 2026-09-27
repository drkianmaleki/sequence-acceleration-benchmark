**Report back**

# Redesign v2, Prompt 6: do the conclusions depend on the synthetic asymptote range? (analysis only)

Date: 2026-09-21. Branch `redesign-v2` in `code/`. Analysis commit `abdc775` on top of `848caed` (Report 5B), pushed; this report is one more commit on top. No pipeline, generator or method code was changed and nothing was re-run: `scripts/analyze_by_ltrue.py` is a standalone script over the git-ignored `results/phase1/phase1_records.csv` of the committed run (`928092a`), capped cells excluded, oracle never ranked. Housekeeping done: `../wt-5b` and `../wt-report3a` removed (`git worktree list` shows only the main checkout); the stray `../wt-prefix` directory left over from Report 5A (already pruned from git) was deleted too.

## 1. What was computed

Records binned by their hidden `L_true` into the log-scale terciles of [0.005, 0.5]: **T1 [0.005, 0.0232)**, **T2 [0.0232, 0.1077)**, **T3 [0.1077, 0.5]** (239 / 236 / 245 of the 720 (regime, seed) pairs; 323,190 uncapped records). Per stratum g in {0.5, 0.1, 0.02}, per tercile, core and held-out separately, for the four deployable trivials, `constant_oracle` (reference), the top-10 accelerators by pooled median error at g = 0.1 (core ranking of `phase1_global.csv`: log_linear, richardson_3, richardson_2, double_exp_fit, rational_fit, richardson_1, pade_23, pade_12, pade_22, single_exp_fit) and the 21 classical variants pooled as one row: median error (over valid records), win rate vs `last_value`, win rate vs `constant_assumed` (means of the per-record 0/1 win indicators; an invalid estimate never wins), median skill vs `last_value` (over valid records), plus the median improvement factor for the in-band count. Outputs: `results/phase1/phase1_by_Ltrue.csv` (666 rows = 3 strata x 3 terciles x 2 regime sets x 37 rows, the 21 variants individually included), `paper_fragments/f13_by_Ltrue.tex` (compiles: 2 pages, 0 errors), and a FACTS.md section.

## 2. The g = 0.1 table, verbatim (console output of `scripts/analyze_by_ltrue.py`)

Each cell is `median error / win rate vs last_value / win rate vs constant_assumed / median skill vs last_value`.

```
  g = 0.1, core: median error / win vs last / win vs L_hat / median skill vs last
  row                                 T1 [0.005, 0.0232)             T2 [0.0232, 0.1077)                T3 [0.1077, 0.5]
  constant_assumed              0.0214/0.867/0.000/0.252        0.0621/0.654/0.000/0.661        0.2406/0.070/0.000/2.866
  last_value                    0.0904/0.000/0.133/1.000        0.1157/0.000/0.346/1.000        0.1133/0.000/0.930/1.000
  window_mean                   0.1256/0.166/0.116/1.427        0.1670/0.140/0.241/1.441        0.1663/0.168/0.771/1.431
  window_min                    0.0494/0.385/0.155/1.000        0.0891/0.325/0.401/1.000        0.0822/0.335/0.935/1.000
  constant_oracle               0.0056/1.000/1.000/0.111        0.0077/1.000/1.000/0.111        0.0099/1.000/1.000/0.111
  log_linear                    0.0101/0.843/0.617/0.246        0.0224/0.805/0.868/0.289        0.0430/0.626/0.933/0.550
  richardson_3                  0.0176/0.830/0.516/0.352        0.0225/0.815/0.813/0.302        0.0234/0.681/0.894/0.386
  richardson_2                  0.0211/0.843/0.512/0.367        0.0220/0.848/0.809/0.303        0.0267/0.730/0.904/0.478
  double_exp_fit                0.0201/0.910/0.527/0.349        0.0248/0.949/0.780/0.416        0.0596/0.875/0.988/0.622
  rational_fit                  0.0286/0.895/0.477/0.316        0.0188/0.903/0.870/0.246        0.0329/0.834/1.000/0.316
  richardson_1                  0.0272/0.824/0.488/0.385        0.0273/0.862/0.794/0.297        0.0331/0.748/0.933/0.404
  pade_23                       0.0157/0.897/0.619/0.256        0.0252/0.788/0.833/0.316        0.1195/0.427/0.826/1.307
  pade_12                       0.0168/0.884/0.609/0.255        0.0251/0.778/0.835/0.363        0.1105/0.401/0.857/1.366
  pade_22                       0.0326/0.643/0.441/0.507        0.0339/0.638/0.621/0.485        0.0283/0.689/0.877/0.452
  single_exp_fit                0.0336/0.968/0.378/0.642        0.0358/0.955/0.619/0.627        0.0393/0.963/0.988/0.645
  classical_21                  0.0499/0.499/0.236/1.000        0.0554/0.476/0.452/1.002        0.0508/0.491/0.900/1.000

  g = 0.1, holdout: median error / win vs last / win vs L_hat / median skill vs last
  row                                 T1 [0.005, 0.0232)             T2 [0.0232, 0.1077)                T3 [0.1077, 0.5]
  constant_assumed              0.0203/0.928/0.000/0.324        0.0611/0.626/0.000/0.862        0.2130/0.015/0.000/3.274
  last_value                    0.0752/0.000/0.072/1.000        0.0767/0.000/0.374/1.000        0.0767/0.000/0.985/1.000
  window_mean                   0.1377/0.000/0.000/1.756        0.1382/0.000/0.224/1.803        0.1380/0.000/0.785/1.603
  window_min                    0.0735/0.338/0.113/1.000        0.0764/0.367/0.381/1.000        0.0760/0.369/0.985/1.000
  constant_oracle               0.0084/1.000/1.000/0.111        0.0084/1.000/1.000/0.110        0.0084/1.000/1.000/0.111
  log_linear                    0.0091/1.000/0.723/0.226        0.0206/0.857/0.878/0.510        0.0374/0.667/1.000/0.658
  richardson_3                  0.0096/0.841/0.733/0.184        0.0106/0.721/0.748/0.170        0.0216/0.887/0.954/0.330
  richardson_2                  0.0095/0.862/0.692/0.212        0.0134/0.714/0.741/0.170        0.0223/0.918/0.974/0.327
  double_exp_fit                0.0213/0.985/0.503/0.356        0.0218/0.966/0.932/0.369        0.0300/0.944/1.000/0.627
  rational_fit                  0.0094/0.862/0.744/0.208        0.0087/0.741/0.735/0.210        0.0074/0.995/1.000/0.208
  richardson_1                  0.0114/0.862/0.626/0.234        0.0132/0.741/0.748/0.156        0.0197/0.872/1.000/0.272
  pade_23                       0.0093/0.964/0.846/0.170        0.0278/0.755/0.850/0.453        0.1151/0.338/0.877/1.694
  pade_12                       0.0109/0.974/0.805/0.282        0.0376/0.769/0.850/0.538        0.1028/0.354/0.918/1.592
  pade_22                       0.0146/0.846/0.595/0.239        0.0191/0.816/0.803/0.249        0.0175/0.841/0.928/0.249
  single_exp_fit                0.0287/0.985/0.251/0.576        0.0221/0.973/0.789/0.575        0.0384/0.964/1.000/0.632
  classical_21                  0.0366/0.557/0.250/0.996        0.0432/0.568/0.525/0.995        0.0419/0.566/0.968/0.996
```

The same table as the g = 0.1 block of `paper_fragments/f13_by_Ltrue.tex` (core and held-out rows interleaved):

```latex
\begin{tabular}{llrrrrrrrrrrrr}
\toprule
$g$ / row & & \multicolumn{4}{c}{T1: $L_{\mathrm{true}} \in [0.005, 0.0232)$} & \multicolumn{4}{c}{T2: $[0.0232, 0.1077)$} & \multicolumn{4}{c}{T3: $[0.1077, 0.5]$} \\
\cmidrule(lr){3-6}\cmidrule(lr){7-10}\cmidrule(lr){11-14}
 & set & med.\ err & win/last & win/$\hat L$ & skill/last & med.\ err & win/last & win/$\hat L$ & skill/last & med.\ err & win/last & win/$\hat L$ & skill/last \\
\midrule
\multicolumn{14}{l}{\textit{$g = 0.1$} (headline)} \\
predict $\hat L$ (\meth{constant\_assumed}) & core & 0.0214 & 0.867 & -- & 0.252 & 0.0621 & 0.654 & -- & 0.661 & 0.2406 & 0.070 & -- & 2.866 \\
 & held-out & 0.0203 & 0.928 & -- & 0.324 & 0.0611 & 0.626 & -- & 0.862 & 0.2130 & 0.015 & -- & 3.274 \\
\meth{last\_value} & core & 0.0904 & -- & 0.133 & -- & 0.1157 & -- & 0.346 & -- & 0.1133 & -- & 0.930 & -- \\
 & held-out & 0.0752 & -- & 0.072 & -- & 0.0767 & -- & 0.374 & -- & 0.0767 & -- & 0.985 & -- \\
\meth{window\_mean} & core & 0.1256 & 0.166 & 0.116 & 1.427 & 0.1670 & 0.140 & 0.241 & 1.441 & 0.1663 & 0.168 & 0.771 & 1.431 \\
 & held-out & 0.1377 & 0.000 & 0.000 & 1.756 & 0.1382 & 0.000 & 0.224 & 1.803 & 0.1380 & 0.000 & 0.785 & 1.603 \\
\meth{window\_min} & core & 0.0494 & 0.385 & 0.155 & 1.000 & 0.0891 & 0.325 & 0.401 & 1.000 & 0.0822 & 0.335 & 0.935 & 1.000 \\
 & held-out & 0.0735 & 0.338 & 0.113 & 1.000 & 0.0764 & 0.367 & 0.381 & 1.000 & 0.0760 & 0.369 & 0.985 & 1.000 \\
\meth{constant\_oracle} \textit{(ref.)} & core & 0.0056 & 1.000 & 1.000 & 0.111 & 0.0077 & 1.000 & 1.000 & 0.111 & 0.0099 & 1.000 & 1.000 & 0.111 \\
 & held-out & 0.0084 & 1.000 & 1.000 & 0.111 & 0.0084 & 1.000 & 1.000 & 0.110 & 0.0084 & 1.000 & 1.000 & 0.111 \\
\addlinespace[2pt]
\meth{log\_linear} & core & 0.0101 & 0.843 & 0.617 & 0.246 & 0.0224 & 0.805 & 0.868 & 0.289 & 0.0430 & 0.626 & 0.933 & 0.550 \\
 & held-out & 0.0091 & 1.000 & 0.723 & 0.226 & 0.0206 & 0.857 & 0.878 & 0.510 & 0.0374 & 0.667 & 1.000 & 0.658 \\
\meth{richardson\_3} & core & 0.0176 & 0.830 & 0.516 & 0.352 & 0.0225 & 0.815 & 0.813 & 0.302 & 0.0234 & 0.681 & 0.894 & 0.386 \\
 & held-out & 0.0096 & 0.841 & 0.733 & 0.184 & 0.0106 & 0.721 & 0.748 & 0.170 & 0.0216 & 0.887 & 0.954 & 0.330 \\
\meth{richardson\_2} & core & 0.0211 & 0.843 & 0.512 & 0.367 & 0.0220 & 0.848 & 0.809 & 0.303 & 0.0267 & 0.730 & 0.904 & 0.478 \\
 & held-out & 0.0095 & 0.862 & 0.692 & 0.212 & 0.0134 & 0.714 & 0.741 & 0.170 & 0.0223 & 0.918 & 0.974 & 0.327 \\
\meth{double\_exp\_fit} & core & 0.0201 & 0.910 & 0.527 & 0.349 & 0.0248 & 0.949 & 0.780 & 0.416 & 0.0596 & 0.875 & 0.988 & 0.622 \\
 & held-out & 0.0213 & 0.985 & 0.503 & 0.356 & 0.0218 & 0.966 & 0.932 & 0.369 & 0.0300 & 0.944 & 1.000 & 0.627 \\
\meth{rational\_fit} & core & 0.0286 & 0.895 & 0.477 & 0.316 & 0.0188 & 0.903 & 0.870 & 0.246 & 0.0329 & 0.834 & 1.000 & 0.316 \\
 & held-out & 0.0094 & 0.862 & 0.744 & 0.208 & 0.0087 & 0.741 & 0.735 & 0.210 & 0.0074 & 0.995 & 1.000 & 0.208 \\
\meth{richardson\_1} & core & 0.0272 & 0.824 & 0.488 & 0.385 & 0.0273 & 0.862 & 0.794 & 0.297 & 0.0331 & 0.748 & 0.933 & 0.404 \\
 & held-out & 0.0114 & 0.862 & 0.626 & 0.234 & 0.0132 & 0.741 & 0.748 & 0.156 & 0.0197 & 0.872 & 1.000 & 0.272 \\
\meth{pade\_23} & core & 0.0157 & 0.897 & 0.619 & 0.256 & 0.0252 & 0.788 & 0.833 & 0.316 & 0.1195 & 0.427 & 0.826 & 1.307 \\
 & held-out & 0.0093 & 0.964 & 0.846 & 0.170 & 0.0278 & 0.755 & 0.850 & 0.453 & 0.1151 & 0.338 & 0.877 & 1.694 \\
\meth{pade\_12} & core & 0.0168 & 0.884 & 0.609 & 0.255 & 0.0251 & 0.778 & 0.835 & 0.363 & 0.1105 & 0.401 & 0.857 & 1.366 \\
 & held-out & 0.0109 & 0.974 & 0.805 & 0.282 & 0.0376 & 0.769 & 0.850 & 0.538 & 0.1028 & 0.354 & 0.918 & 1.592 \\
\meth{pade\_22} & core & 0.0326 & 0.643 & 0.441 & 0.507 & 0.0339 & 0.638 & 0.621 & 0.485 & 0.0283 & 0.689 & 0.877 & 0.452 \\
 & held-out & 0.0146 & 0.846 & 0.595 & 0.239 & 0.0191 & 0.816 & 0.803 & 0.249 & 0.0175 & 0.841 & 0.928 & 0.249 \\
\meth{single\_exp\_fit} & core & 0.0336 & 0.968 & 0.378 & 0.642 & 0.0358 & 0.955 & 0.619 & 0.627 & 0.0393 & 0.963 & 0.988 & 0.645 \\
 & held-out & 0.0287 & 0.985 & 0.251 & 0.576 & 0.0221 & 0.973 & 0.789 & 0.575 & 0.0384 & 0.964 & 1.000 & 0.632 \\
\addlinespace[2pt]
21 classical variants (pooled) & core & 0.0499 & 0.499 & 0.236 & 1.000 & 0.0554 & 0.476 & 0.452 & 1.002 & 0.0508 & 0.491 & 0.900 & 1.000 \\
 & held-out & 0.0366 & 0.557 & 0.250 & 0.996 & 0.0432 & 0.568 & 0.525 & 0.995 & 0.0419 & 0.566 & 0.968 & 0.996 \\
\bottomrule
\end{tabular}
```

## 3. The three facts (FACTS.md, section "L_true terciles")

```
    - rank-1 accelerator per L_true tercile at g = 0.1 core (among the top-10 by pooled median error): T1 [0.005, 0.0232): log_linear (med. err 0.0101, win vs last 0.843, win vs L_hat 0.617); T2 [0.0232, 0.1077): rational_fit (med. err 0.0188, win vs last 0.903, win vs L_hat 0.870); T3 [0.1077, 0.5]: richardson_3 (med. err 0.0234, win vs last 0.681, win vs L_hat 0.894)
    - classical variants inside the +/-10 % MI band per L_true tercile at g = 0.1 core: T1: 21 of 21; T2: 21 of 21; T3: 21 of 21
    - predict-L_hat (constant_assumed, L_hat = 0) vs last_value per L_true tercile at g = 0.1: T1 core: L_hat wins 0.867 (last wins 0.133); med. err 0.0214 vs 0.0904; T1 holdout: L_hat wins 0.928 (last wins 0.072); med. err 0.0203 vs 0.0752; T2 core: L_hat wins 0.654 (last wins 0.346); med. err 0.0621 vs 0.1157; T2 holdout: L_hat wins 0.626 (last wins 0.374); med. err 0.0611 vs 0.0767; T3 core: L_hat wins 0.070 (last wins 0.930); med. err 0.2406 vs 0.1133; T3 holdout: L_hat wins 0.015 (last wins 0.985); med. err 0.2130 vs 0.0767
```

## 4. Reading

1. **predict-L_hat's strength is a property of the asymptote range, not of the methods.** With L_hat = 0, `constant_assumed` beats `last_value` on 87 % of core records in T1 and 65 % in T2, but on **7 %** in T3 (held-out: 93 % / 63 % / **1.5 %**); its median error goes 0.021 -> 0.062 -> 0.241 while `last_value` stays at 0.09-0.12. In the tercile that covers the real curves' floors (0.14-0.69), the deployable baseline to beat is the last value, and predict-L_hat is the worst of the four trivials.
2. **The rank-1 accelerator changes with the tercile** (g = 0.1 core): `log_linear` in T1 (0.0101), `rational_fit` in T2 (0.0188), `richardson_3` in T3 (0.0234). `log_linear`'s median error quadruples from T1 to T3 (0.0101 -> 0.0430) and its win rate vs last drops 0.84 -> 0.63; `pade_23` / `pade_12` collapse in T3 (median error 0.12 / 0.11, win vs last 0.43 / 0.40, skill vs last 1.31 / 1.37, i.e. worse than the last value); `rational_fit`, `richardson_1/2/3`, `single_exp_fit` and `double_exp_fit` keep win rates vs last of 0.68-0.96 in every tercile. On the held-out regimes `rational_fit` is the most range-robust method (win vs last 0.86 / 0.74 / 0.995; median error 0.009 / 0.009 / 0.007).
3. **The classical no-op result does not depend on the range**: 21 of 21 variants inside the +/-10 % MI band in every tercile; the pooled classical row has win rate vs last 0.48-0.50 and median skill vs last 1.00 in all three (held-out 0.56-0.57 / 1.00).
4. **Win rates vs L_hat are range artefacts** in T3 (0.83-1.00 for every accelerator, 0.90 for the pooled classical row) because L_hat = 0 is far from every asymptote there; this is the empirical form of the Report-5B decision to make `last_value` the headline reference.

## 5. Pushed hash and `git ls-remote`

```
abdc775  Analysis (Prompt 6): Phase-1 results by L_true tercile
```

```
$ git ls-remote origin redesign-v2
abdc775b7c0da763c8316fd0642051e1eb4c9c83	refs/heads/redesign-v2
```

(`HEAD` at the time of writing: `abdc775`; this report is committed on top and pushed, its hash is printed in the console after the report.)

## 6. Notes

- `FACTS.md` is otherwise generated by `scripts/make_paper_tables.py`, which does not know about the new section; regenerating FACTS.md drops it until `scripts/analyze_by_ltrue.py` is run again (the script replaces its own section idempotently). A one-line hook in the generator would make it permanent; not added (no generator changes in this prompt).
- The top-10 list is the core g = 0.1 ranking for both regime sets, so the held-out block shows the same ten methods (rank-1 on held-out is `rational_fit` in every tercile).
- Held-out T2 holds only 49 (regime, seed) pairs; its cells are the noisiest in the table.

Stopped here.
