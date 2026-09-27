"""
panels.py
=========
The descriptive error panel: the one summary every phase reports for a set
of records (one method on many cells, or one selector's choices).

A *record* is one method on one (regime, obs_idx, noise, seed, target_g).
It carries the error |prediction - true value at the target round|, ``valid``
(a usable finite estimate), ``catastrophic`` (invalid, or error above
config.CAT_MULT times the last-value error) and E_last, the error of
repeating the last observed value (the ``last_value`` trivial comparator).

The panel of a set of records:

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
error_panel() is the single implementation used by Phases 1, 2, 3 and 5b.
"""

from typing import Dict

import numpy as np

PANEL_COLS = (
    "n_total", "n_valid", "valid_rate", "cat_rate",
    "mean_error", "sd_error", "med_error", "q25_error", "q75_error", "p90_error",
    "win_rate_vs_last",
)
CONDITIONAL_COLS = ("mean_error", "sd_error", "med_error", "q25_error", "q75_error", "p90_error")


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
