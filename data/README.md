# data/

This directory is populated at runtime and is excluded from version control.

## Contents

When you run `scripts/run_real_data.py`, the script downloads six tabular
classification datasets from the OpenML CC-18 benchmark suite via the `openml`
Python package. Downloaded files are stored here automatically.

## Dataset registry

| Name            | OpenML id | Instances | Features | Classes |
|-----------------|-----------|-----------|----------|---------|
| covertype       | 1596      | 581,012   | 54       | 7       |
| higgs           | 23512     | 98,050    | 28       | 2       |
| adult           | 1590      | 48,842    | 14       | 2       |
| jannis          | 41168     | 83,733    | 54       | 4       |
| miniboone       | 41150     | 130,064   | 50       | 2       |
| bank_marketing  | 1461      | 45,211    | 16       | 2       |

All datasets are publicly available at <https://www.openml.org> under the
Creative Commons licence associated with each dataset on the OpenML platform.

## Reproducibility note

The synthetic benchmark (Phases 0–5) does not use any files in this directory.
All synthetic sequences are generated deterministically from the regime
definitions and random seeds in `src/generators.py` and `src/config.py`.
Only the real-data transfer check (Section 9 of the paper) requires a download.
