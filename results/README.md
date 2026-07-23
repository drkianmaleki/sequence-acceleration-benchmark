# results/

This directory holds generated output. It is populated by running the
benchmark; nothing here is written by hand.

```
python reproduce_all.py            # everything
python scripts/run_phase1.py --full   # one phase at a time
```

Each phase writes to its own subdirectory (`phase1/`, `phase2/`, ... ,
`real_data/`), creating it if needed.

## Provenance

Results are only meaningful alongside the commit that produced them. Any
change to `src/generators.py`, `src/accelerators.py`, `src/evaluation.py`,
or the validity window in `src/config.py` invalidates every stored table,
so regenerate the whole set rather than mixing output from different
commits.

After re-running Phase 1, check that the derived dangerous-method set is
still correct:

```
python scripts/check_dangerous.py
```

Four large intermediate files are deliberately excluded from version
control (see `.gitignore`): `phase2/phase2_features.csv`,
`phase2/phase2_sweep_aggregated.csv`, `phase4/phase4_raw.csv`, and
`phase5a/phase5a_raw.csv`. They are regenerated on each run.
