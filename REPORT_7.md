**Report back**

# Redesign v2, Prompt 7: FACTS.md evaluation counts for the 49-method run

Date: 2026-09-24. Branch `redesign-v2` in `code/`. One commit on top of `dd03b66` (Report 6), pushed.

## 1. What was wrong

`FACTS.md`, section "pipeline provenance", still carried two rows from the superseded 51-method run (Report 3B):

- "evaluation counts (central)" listed Phase 1 362,880; Phase 5a 967,680 (+4,406,400 perturbation calls); Phase 5b 532,800.
- "full run" named `REPORT_3B_full_run.txt; REPORT_3B.md` with the Report-3B timings (2026-09-20 23:35 to 2026-09-21 03:58, 15,783 s).

Both rows are emitted verbatim by `scripts/make_paper_tables.py` (lines 1318-1321), so a regeneration of `FACTS.md` would have restored the stale text. Both files were changed together.

## 2. Corrected rows (verbatim, `FACTS.md` lines 283-284)

```
| full run | 2026-09-21 11:38 to 17:00, 19,302 s; Phase 1 1,729 s, Phase 2 876 s, Phase 4 2,079 s, Phase 5a 9,223 s (--jobs 7, 5 perturbation trials), Phase 5b 5,375 s | REPORT_5B_full_run.txt; REPORT_5B.md | - | - |
| evaluation counts (central) | Phase 0 196; Phase 1 349,920; Phase 2 772,200 (+2 skill-reference calls per cell); Phase 4 224,640 (+1,555,200 diagnostic calls); Phase 5a 933,120 (+4,233,600 perturbation calls); Phase 5b 717,120 (sweep 1a 28,800 + 1b 201,600 + 2 28,800 + 3 457,920); real data 630; TOTAL 2,997,826 | python reproduce_all.py --plan | - | - |
```

Sources checked:

- `python reproduce_all.py --plan` at HEAD `dd03b66` (before the commit), FULL block: Phase 0 196; Phase 1 349,920; Phase 2 772,200 (+2 skill-reference calls per cell); Phase 4 224,640 (+1,555,200 diagnostic calls); Phase 5a 933,120 (+4,233,600 perturbation calls); Phase 5b 717,120 (sweep1a 28,800 + sweep1b 201,600 + sweep2 28,800 + sweep3 457,920); Real data 630; TOTAL (central) 2,997,826. Identical to the numbers in the prompt.
- Full-run timings: `REPORT_5B.md` section 2 and the step table at the end of `REPORT_5B_full_run.txt` (START 2026-09-21 11:38:20; Phase 1 1729s, Phase 2 876s, Phase 4 2079s, Phase 5a 9223s, Phase 5b 5375s, total 19302s). The two agree.
- "results commit" row: left at `928092a`, which is the commit that replaced `results/` with the 49-method full run (Report 5B, section 10). Note for the next regeneration: the script computes this row as `git log -1 -- results`, which now returns `abdc775` (Prompt 6 wrote its tercile analysis under `results/`), so the regenerated row will read `abdc775`, not `928092a`.

## 3. Grep results

Before the edit:

```
$ grep -n -E "362,880|967,680|532,800|2,861,034" FACTS.md README.md
FACTS.md:284:| evaluation counts (central) | Phase 1 362,880; Phase 2 772,200; Phase 4 224,640 (+1,555,200 diagnostic calls); Phase 5a 967,680 (+4,406,400 perturbation calls); Phase 5b 532,800; real data 630 | python reproduce_all.py --plan | - | - |
```

After the edit:

```
$ grep -n -E "362,880|967,680|532,800|2,861,034" FACTS.md README.md
(no output; grep exit status 1)
```

`README.md` never contained any of the four numbers. It does still cite the earlier run in prose that was outside this prompt's scope and was not touched: line 49 ("4.4 h on 8 cores"), line 54 (Report-3B wall times, "Phase 5b 38 min"), line 56 ("commit `30de724`, log in `REPORT_3B_full_run.txt`"), and line 142 ("`REPORT_3B.md`: the full-run report"). The corresponding 49-method values are 5.4 h, Phase 5b 1.5 h, commit `928092a`, log `REPORT_5B_full_run.txt`.

## 4. Commit and push

```
$ git diff --stat   (before commit)
 FACTS.md                     | 4 ++--
 scripts/make_paper_tables.py | 6 +++---
 2 files changed, 5 insertions(+), 5 deletions(-)

$ python -m py_compile scripts/make_paper_tables.py
py_compile OK

$ git log -1 --format='%H %s'
842ddb917ada04e72a39a95d1d245f6b3901a9d4 FACTS: evaluation counts for the 49-method run

$ git push origin redesign-v2
To https://github.com/drkianmaleki/sequence-acceleration-benchmark.git
   dd03b66..842ddb9  redesign-v2 -> redesign-v2
error: update_ref failed for ref 'refs/remotes/origin/redesign-v2': couldn't set 'refs/remotes/origin/redesign-v2'

$ git ls-remote origin redesign-v2
842ddb917ada04e72a39a95d1d245f6b3901a9d4	refs/heads/redesign-v2
```

The push itself succeeded (the remote ref is at `842ddb9`); only the local tracking ref `refs/remotes/origin/redesign-v2` failed to update, most likely a Dropbox file lock on the ref file. A subsequent `git fetch origin redesign-v2` set it (`dd03b66..842ddb9  redesign-v2 -> origin/redesign-v2`); `git status -sb` now reports `## redesign-v2...origin/redesign-v2` with no ahead/behind.

Pushed hash: `842ddb917ada04e72a39a95d1d245f6b3901a9d4`.

This report file is written to the working tree and not committed.
