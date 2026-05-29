# sequence-acceleration-benchmark

This project is in progress. 

Benchmark for sequence acceleration methods applied to machine-learning training curves.

## Structure

| Path | Purpose |
|------|---------|
| `src/` | Core library: accelerators, evaluation, datasets, plotting |
| `phases/` | Phase-level experiment logic (2–5b) |
| `scripts/` | Entry points — run one per phase |
| `data/` | Downloaded datasets and model artefacts (gitignored) |
| `tests/` | Unit tests |
| `results/` | Output figures and tables (gitignored) |

## Quick start

```bash
python scripts/run_phase0_tests.py   # sanity checks
python scripts/run_real_data.py      # real-data experiment (OpenML + XGBoost)
```
