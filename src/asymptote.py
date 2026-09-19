"""
asymptote.py
============
The *assumed* asymptote L_hat: what a method is told about the limit.

Redesign v2 separates two quantities that the rejected design conflated:

    L_true   the hidden asymptote of the (regime, seed) sequence
             (src.generators.true_asymptote); methods never see it.
    L_hat    the value supplied to accelerators, feature extractors and
             cascade inputs under cfg['L_inf'].

L_hat is computed here from config.ASSUMED_L_MODE.  Only the "oracle" mode
returns L_true, and every output produced in that mode is labelled oracle.
"""

from typing import Optional, Sequence

import numpy as np

import src.config as CFG_MOD


def resolve_mode(mode: Optional[str] = None) -> str:
    """Return the effective assumed-asymptote mode, validating it."""
    mode = CFG_MOD.ASSUMED_L_MODE if mode is None else mode
    if mode not in CFG_MOD.ASSUMED_L_MODES:
        raise ValueError(
            f"unknown ASSUMED_L_MODE {mode!r}; "
            f"expected one of {CFG_MOD.ASSUMED_L_MODES}")
    return mode


def is_oracle_mode(mode: Optional[str] = None) -> bool:
    return resolve_mode(mode) == "oracle"


def needs_true_asymptote(mode: Optional[str] = None) -> bool:
    """True for modes that cannot be evaluated without L_true (not deployable)."""
    return resolve_mode(mode) in ("half", "oracle", "double")


def assumed_asymptote(L_true: Optional[float],
                      window: Sequence[float],
                      mode: Optional[str] = None) -> float:
    """
    Compute L_hat, the asymptote handed to methods.

    Parameters
    ----------
    L_true : hidden true asymptote of the sequence, or None when it is not
             known (real curves).  Only consulted by half / oracle / double.
    window : the observation window (used by "winmin").
    mode   : one of config.ASSUMED_L_MODES; defaults to config.ASSUMED_L_MODE.
    """
    mode = resolve_mode(mode)

    if mode == "zero":
        return 0.0

    if mode == "winmin":
        w = np.asarray(window, dtype=float)
        if w.size == 0:
            raise ValueError("winmin mode needs a non-empty window")
        return float(max(0.0, 0.9 * float(np.min(w))))

    if L_true is None:
        raise ValueError(
            f"ASSUMED_L_MODE {mode!r} requires L_true, which is unavailable "
            f"(real curves).  Use 'zero' or 'winmin'.")

    if mode == "half":
        return 0.5 * float(L_true)
    if mode == "oracle":
        return float(L_true)
    if mode == "double":
        return 2.0 * float(L_true)

    raise AssertionError("unreachable")   # resolve_mode() validated the mode
