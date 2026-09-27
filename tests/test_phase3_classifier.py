"""
tests/test_phase3_classifier.py
===============================
The regime classifier's three cross-validation protocols (phases/phase3.py):
the grouped splits hold out whole groups, every row is tested exactly once,
a perfectly separable table is classified perfectly, and the output table
carries every protocol label with its split-unit sentence.
"""

import itertools
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from phases.phase2 import FEATURE_COLS  # noqa: E402
from phases.phase3 import (CLASSIFIER_PROTOCOLS, PRIMARY_PROTOCOL, classifier_folds,  # noqa: E402
                           classifier_table, regime_classifier)

REGIMES = [f"regime_{k}" for k in range(6)]
DEPTHS = [20, 40, 60, 80, 100]
NOISES = [0.0, 0.005, 0.02]
SEEDS = range(3)


def _feature_table(separable: bool, rng: np.random.RandomState) -> pd.DataFrame:
    """A synthetic phase2_features frame: regimes x depths x noises x seeds."""
    rows = []
    for k, regime in enumerate(REGIMES):
        for d, nz, s in itertools.product(DEPTHS, NOISES, SEEDS):
            base = {"regime": regime, "obs_idx": d, "noise": nz, "seed": s, "L_true": 0.1, "L_hat": 0.0}
            for j, f in enumerate(FEATURE_COLS):
                if separable:
                    # each regime occupies its own band in every feature; depth and
                    # noise move the value within the band only
                    base[f] = 10.0 * k + 0.02 * d / 100 + 0.1 * nz + 0.01 * rng.randn()
                else:
                    base[f] = rng.randn()
            rows.append(base)
    return pd.DataFrame(rows)


def _check_grouped(protocol: str, group: str):
    table = classifier_table(_feature_table(True, np.random.RandomState(0)))
    folds, note = classifier_folds(table, protocol)
    values = table[group].values
    assert len(folds) == table[group].nunique()
    tested = np.zeros(len(table), dtype=int)
    for train_idx, test_idx in folds:
        assert not set(values[train_idx]) & set(values[test_idx])   # no group on both sides
        assert len(set(values[test_idx])) == 1                        # one group per fold
        tested[test_idx] += 1
    assert (tested == 1).all()                                        # every row tested once
    assert "one per" in note


def test_grouped_by_depth_folds_hold_out_whole_depths():
    _check_grouped("grouped_by_depth", "obs_idx")


def test_grouped_by_noise_folds_hold_out_whole_noise_levels():
    _check_grouped("grouped_by_noise", "noise")


def test_rows_are_seed_averaged_cells():
    df = _feature_table(False, np.random.RandomState(1))
    table = classifier_table(df)
    assert len(table) == len(REGIMES) * len(DEPTHS) * len(NOISES)
    assert "seed" not in table.columns
    one = df[(df.regime == REGIMES[0]) & (df.obs_idx == DEPTHS[0]) & (df.noise == NOISES[0])]
    row = table[(table.regime == REGIMES[0]) & (table.obs_idx == DEPTHS[0]) & (table.noise == NOISES[0])].iloc[0]
    assert row[FEATURE_COLS[0]] == float(one[FEATURE_COLS[0]].mean())


def test_separable_table_is_classified_perfectly_and_csv_carries_the_protocols(tmp_path):
    df = _feature_table(True, np.random.RandomState(2))
    out = regime_classifier(df, str(tmp_path))
    assert list(out.columns) == ["protocol", "regime", "n_samples", "n_correct", "accuracy",
                                 "top_confusion", "split_unit", "note"]
    assert set(out.protocol) == set(CLASSIFIER_PROTOCOLS) == {"grouped_by_depth", "grouped_by_noise",
                                                              "stratified_5fold_legacy"}
    assert PRIMARY_PROTOCOL == "grouped_by_depth"
    overall = out[out.regime == "__OVERALL__"].set_index("protocol")
    assert len(overall) == 3
    for p in ("grouped_by_depth", "grouped_by_noise"):
        assert overall.loc[p, "accuracy"] == 1.0
        assert overall.loc[p, "n_samples"] == len(REGIMES) * len(DEPTHS) * len(NOISES)
    for p, spec in CLASSIFIER_PROTOCOLS.items():
        rows = out[out.protocol == p]
        assert (rows.split_unit == spec["split_unit"]).all()
        assert set(rows.regime) == set(REGIMES) | {"__OVERALL__"}
    assert "holds out all cells at one observation depth" in overall.loc["grouped_by_depth", "split_unit"]
    assert "holds out all cells at one noise level" in overall.loc["grouped_by_noise", "split_unit"]
    assert "may fall on both sides" in overall.loc["stratified_5fold_legacy", "split_unit"]
    csv = pd.read_csv(tmp_path / "phase3_regime_classifier.csv")
    assert list(csv.columns)[0] == "protocol" and set(csv.protocol) == set(CLASSIFIER_PROTOCOLS)
    assert csv.split_unit.notna().all() and (csv.split_unit.str.len() > 50).all()


def test_few_groups_fall_back_and_say_so(tmp_path):
    df = _feature_table(True, np.random.RandomState(3))
    two_noise = df[df.noise.isin(NOISES[:2])]
    out = regime_classifier(two_noise, str(tmp_path))
    note = out[(out.protocol == "grouped_by_noise")].note.iloc[0]
    assert "only 2 distinct noise values: 2 folds" == note
    assert out[(out.protocol == "grouped_by_noise") & (out.regime == "__OVERALL__")].accuracy.iloc[0] == 1.0
    one_noise = df[df.noise == NOISES[0]]
    out1 = regime_classifier(one_noise, str(tmp_path))
    ov = out1[(out1.protocol == "grouped_by_noise") & (out1.regime == "__OVERALL__")].iloc[0]
    assert "cannot hold out a group" in ov.note and ov.n_samples == 0 and np.isnan(ov.accuracy)
    assert out1[(out1.protocol == "grouped_by_depth") & (out1.regime == "__OVERALL__")].accuracy.iloc[0] == 1.0
