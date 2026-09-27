"""
panels.py
=========
The descriptive error panel and the rule panel: the summaries every phase
reports for a set of records, implemented once.

A *record* is one method on one (regime, obs_idx, noise, seed, target_g).
It carries the error |prediction - true value at the target round|, ``valid``
(a usable finite estimate), ``catastrophic`` (invalid, or error above
config.CAT_MULT times the last-value error) and E_last, the error of
repeating the last observed value (the ``last_value`` trivial comparator).
A *cell* is one (regime, obs_idx, noise, target_g); its records are the seeds.
Capped cells are excluded from every summary by the caller.

Descriptive panel (error_panel) of a set of records:

    n_total, n_valid, valid_rate, cat_rate
        over all records (valid_rate = n_valid / n_total; cat_rate = mean of
        the catastrophic flag, so an invalid record is catastrophic);
    mean_error, sd_error, med_error, q25_error, q75_error, p90_error
        over the VALID records only -- every table that prints them labels
        them "conditional on validity" and shows the validity rate alongside
        (sd_error is the sample standard deviation, NaN below two records;
        quantiles use numpy's default linear interpolation);
    win_rate_vs_last
        fraction of ALL records where the method is valid and its error is
        below E_last; an invalid record never wins.

An empty valid set leaves the conditional fields NaN with n_valid = 0.  No
composite score is formed anywhere: the panel is reported as it is.

Rule panel (rule_panel): a rule is "feature op threshold -> routed method"
(or any per-cell firing condition, such as the Phase-2 cascade).  Over the
eligible cells (uncapped, feature finite): n_cells_total, n_cells_fired,
fire_rate; then, over the RECORDS of the fired cells, the descriptive panel
of richardson_1 (prefix r1_) and of the routed method (prefix alt_);
lower_error_frac_records = fraction of fired-cell records where both are
valid and the routed method's error is below richardson_1's;
lower_error_frac_cells = fraction of fired cells where the routed method's
cell-median error is below richardson_1's; median_rel_change = median over
fired cells of (alt_med - r1_med) / r1_med; and the same two panels over the
records of the not-fired cells (prefix nf_).  No detection score of any kind
is formed from these: the panel reports what happened.

zero_denominator_flags marks the records where the last-value normalised
error E / E_last used the documented E_last <= SKILL_EPS branch of
src.trivial.skill_score (ratio 1.0 when E <= SKILL_EPS, +inf otherwise).
"""

from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd

from src.trivial import SKILL_EPS

PANEL_COLS = (
    "n_total", "n_valid", "valid_rate", "cat_rate",
    "mean_error", "sd_error", "med_error", "q25_error", "q75_error", "p90_error",
    "win_rate_vs_last",
)
CONDITIONAL_COLS = ("mean_error", "sd_error", "med_error", "q25_error", "q75_error", "p90_error")
RECORD_COLS = ("error", "valid", "catastrophic", "E_last")
RULE_COLS = (
    "n_cells_total", "n_cells_fired", "fire_rate",
    *[f"r1_{c}" for c in PANEL_COLS], *[f"alt_{c}" for c in PANEL_COLS],
    "lower_error_frac_records", "lower_error_frac_cells", "median_rel_change",
    *[f"nf_r1_{c}" for c in PANEL_COLS], *[f"nf_alt_{c}" for c in PANEL_COLS],
)


def error_panel(errors, valid, catastrophic, last_errors) -> Dict[str, float]:
    """
    The descriptive panel of a set of records (definition in the module docstring).

    Parameters
    ----------
    errors       : per-record error |prediction - truth| (NaN where invalid)
    valid        : per-record validity flag (0/1 or bool)
    catastrophic : per-record catastrophe flag (0/1 or bool)
    last_errors  : per-record E_last, the error of the last observed value

    Returns a dict with the keys in PANEL_COLS.  Rates are plain fractions;
    the caller decides on rounding for its own table.
    """
    e = np.asarray(errors, dtype=float).ravel()
    v = np.asarray(valid).ravel().astype(bool)
    c = np.asarray(catastrophic).ravel().astype(bool)
    el = np.asarray(last_errors, dtype=float).ravel()
    n_total = int(e.size)
    if not (v.size == c.size == el.size == n_total):
        raise ValueError("error_panel: errors, valid, catastrophic and last_errors "
                         "must have the same length")
    n_valid = int(v.sum())
    ev = e[v]
    nan = float("nan")
    out: Dict[str, float] = {
        "n_total": n_total,
        "n_valid": n_valid,
        "valid_rate": (n_valid / n_total) if n_total else nan,
        "cat_rate": float(c.mean()) if n_total else nan,
    }
    if n_valid:
        q25, q50, q75, p90 = np.percentile(ev, [25, 50, 75, 90])
        out.update({
            "mean_error": float(ev.mean()),
            "sd_error": float(ev.std(ddof=1)) if n_valid >= 2 else nan,
            "med_error": float(q50),
            "q25_error": float(q25),
            "q75_error": float(q75),
            "p90_error": float(p90),
        })
    else:
        out.update({k: nan for k in CONDITIONAL_COLS})
    with np.errstate(invalid="ignore"):
        wins = v & (e < el)          # NaN on either side compares False: never a win
    out["win_rate_vs_last"] = float(wins.mean()) if n_total else nan
    return out


def panel_of(records: pd.DataFrame) -> Dict[str, float]:
    """error_panel over the rows of a record frame (columns RECORD_COLS)."""
    return error_panel(records["error"], records["valid"], records["catastrophic"], records["E_last"])


def zero_denominator_flags(e_method, e_last) -> np.ndarray:
    """
    Boolean mask of the records where the normalised error E_method / E_last
    took the E_last <= SKILL_EPS branch of src.trivial.skill_score: both errors
    finite (an invalid record yields NaN, not the branch) and E_last at or
    below SKILL_EPS.  The ratio there is 1.0 when E_method <= SKILL_EPS and
    +inf otherwise; this is the documented rule, not a regulariser, and the
    pipeline counts how often it fired.
    """
    em = np.asarray(e_method, dtype=float).ravel()
    el = np.asarray(e_last, dtype=float).ravel()
    if em.size != el.size:
        raise ValueError("zero_denominator_flags: arrays must have the same length")
    return np.isfinite(em) & np.isfinite(el) & (el <= SKILL_EPS)


def threshold_rule(cells: pd.DataFrame, feature: str, op: str,
                   threshold: float) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    The eligible cells of a "feature op threshold" rule (those whose feature
    is finite) and the boolean firing mask over them.  ``op`` is '<' or '>'.
    """
    if op not in ("<", ">"):
        raise ValueError(f"threshold_rule: unknown operator {op!r}")
    f = pd.to_numeric(cells[feature], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(f)
    elig = cells.loc[finite].reset_index(drop=True)
    fv = f[finite]
    fired = (fv < threshold) if op == "<" else (fv > threshold)
    return elig, fired


def _cell_median_error(records: pd.DataFrame, cell_keys: Sequence[str], name: str) -> pd.DataFrame:
    ok = records[records["valid"].astype(bool)]
    med = ok.groupby(list(cell_keys))["error"].median().rename(name).reset_index()
    return med


def rule_panel(cells: pd.DataFrame, fired, records_r1: pd.DataFrame,
               records_alt: pd.DataFrame, cell_keys: Sequence[str],
               record_keys: Sequence[str] = None) -> Dict[str, float]:
    """
    The rule panel (definition in the module docstring).

    Parameters
    ----------
    cells       : one row per ELIGIBLE cell (uncapped, feature finite) with the
                  ``cell_keys`` columns; the caller has already applied the
                  capped-exclusion rule and the feature-finite rule
    fired       : boolean array aligned with ``cells``: did the rule fire there
    records_r1  : richardson_1's records (cell_keys + RECORD_COLS, one row per
                  record); records of cells absent from ``cells`` are ignored
    records_alt : the routed method's records, same layout
    cell_keys   : the columns that identify a cell
    record_keys : the columns that identify a record (default: cell_keys plus
                  'seed' unless 'seed' is already a cell key); used to pair a
                  richardson_1 record with the routed method's record

    Returns a dict with the keys in RULE_COLS.  Rates are plain fractions.
    """
    cell_keys = list(cell_keys)
    if record_keys is None:
        record_keys = cell_keys + (["seed"] if "seed" not in cell_keys else [])
    record_keys = list(record_keys)
    fired = np.asarray(fired).ravel().astype(bool)
    if fired.size != len(cells):
        raise ValueError("rule_panel: 'fired' must be aligned with 'cells'")
    flags = cells[cell_keys].copy().reset_index(drop=True)
    flags["_fired"] = fired
    n_cells_total = int(len(flags))
    n_cells_fired = int(fired.sum())
    nan = float("nan")
    out: Dict[str, float] = {
        "n_cells_total": n_cells_total,
        "n_cells_fired": n_cells_fired,
        "fire_rate": (n_cells_fired / n_cells_total) if n_cells_total else nan,
    }

    def _join(records: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in dict.fromkeys(record_keys + list(RECORD_COLS)) if c in records.columns]
        return records[cols].merge(flags, on=cell_keys, how="inner")

    r1 = _join(records_r1)
    alt = _join(records_alt)
    for prefix, mask_val in (("", True), ("nf_", False)):
        for tag, frame in (("r1_", r1), ("alt_", alt)):
            sub = frame[frame["_fired"] == mask_val]
            out.update({f"{prefix}{tag}{k}": v for k, v in panel_of(sub).items()})

    # record-wise comparison on the fired cells: pair the two methods' records
    pair = (r1[r1["_fired"]][record_keys + ["error", "valid"]]
            .merge(alt[alt["_fired"]][record_keys + ["error", "valid"]],
                   on=record_keys, how="inner", suffixes=("_r1", "_alt")))
    if len(pair):
        both = pair["valid_r1"].astype(bool) & pair["valid_alt"].astype(bool)
        with np.errstate(invalid="ignore"):
            lower = both & (pair["error_alt"].to_numpy(float) < pair["error_r1"].to_numpy(float))
        out["lower_error_frac_records"] = float(lower.mean())
    else:
        out["lower_error_frac_records"] = nan

    # cell-wise comparison on the fired cells: cell-median errors over valid records
    fired_cells = flags[flags["_fired"]][cell_keys]
    if len(fired_cells):
        med = (fired_cells
               .merge(_cell_median_error(r1, cell_keys, "r1_med"), on=cell_keys, how="left")
               .merge(_cell_median_error(alt, cell_keys, "alt_med"), on=cell_keys, how="left"))
        r1m = med["r1_med"].to_numpy(dtype=float)
        altm = med["alt_med"].to_numpy(dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            lower_cells = np.isfinite(r1m) & np.isfinite(altm) & (altm < r1m)
            rel = (altm - r1m) / r1m
        out["lower_error_frac_cells"] = float(lower_cells.mean())
        rel = rel[~np.isnan(rel)]
        out["median_rel_change"] = float(np.median(rel)) if rel.size else nan
    else:
        out["lower_error_frac_cells"] = nan
        out["median_rel_change"] = nan
    return {k: out[k] for k in RULE_COLS}
