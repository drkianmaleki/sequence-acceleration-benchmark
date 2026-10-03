#!/usr/bin/env python3
"""
derive_selection.py  (redesign v2, R9f Part A)
==============================================
Choosing a method by trial, tested on the synthetic families of Phase 1:
a method is chosen because it had the lowest error on one run of a problem
(the pilot); how does that chosen method do on another run of the same
problem (the final run), compared with always using the default
(selection_defs.DEFAULT_METHOD) and with repeating the last observed value?

The analysis rereads the git-ignored per-record file of the full run,
results/phase1/phase1_records.csv, and evaluates no method: every number is
a choice among, or a statistic of, records already computed.  Like
scripts/derive_raw_facts.py it turns the records into committed aggregates:

    results/phase1/phase1_selection_cells.csv        one row per (set, regime, noise, target_g, pool, design)
    results/phase1/phase1_selection_global.csv       one row per (set, target_g, noise class, pool, design)
    results/phase1/phase1_selection_provenance.json  script, commit, tree state, timing, versions, the records
                                                     file's row count and SHA-256, the pools, the designs,
                                                     cells and trials per design, the Phase-1 agreement check

    python scripts/derive_selection.py [--results results] [--extract PATH]

Population.  The cells are the (regime, noise, target_g) cells of the pooled
Phase 1 tables, all three strata, core and held-out: those with capped == 0
in phase1_aggregated.csv (a cell with any capped seed is excluded, as
there).  Within a cell the runs are the seeds.  A method's error on a run
is its ``error`` when ``valid == 1``; an invalid record has no error.

Pools (scripts/selection_defs.py, derived): lead = the leading methods
(LEAD), named = the default and the conservative alternative, classical =
the classical variants, all = every accelerator.

Designs, for a cell, a pool and the errors of the pool's members on the
cell's seeds:
  one_pilot    every ordered pair of distinct seeds (pilot, final); the chosen
               method is the pool member with the lowest error on the pilot
               seed among those valid there (ties: registry order of
               src.pipeline.ACCEL_METHODS); its record on the final seed is
               taken as it is -- an invalid record stays invalid, no fallback;
  many_pilots  every seed in turn is the final run and the other seeds are the
               pilots; the chosen method is the pool member with the lowest
               median error over the pilot seeds (median over its valid pilot
               records) among the members valid on at least RANK_MIN_VALID of
               the pilot seeds (ties: registry order).
A trial on which no pool member qualifies (no member valid on the pilot; no
member above the validity floor) has no chosen method: its chosen record
counts as invalid (never a win) and it is counted in n_no_choice.

Per-cell statistics, over the trials of the cell: n_seeds, n_trials,
n_no_choice; chosen_valid_rate = share of trials whose chosen record is
valid; win_vs_default = share of trials where the chosen record is valid
and its error is strictly below the default's on the same final seed (an
invalid chosen record never wins; a valid chosen record wins against an
invalid default); default_chosen = share of trials where the chosen method
is the default itself (never a strict win); win_vs_last = the same against
last_value; med_error_chosen = median error of the chosen records over the
trials where it is valid; med_error_default, med_error_last = the medians
of the default's and of the last value's errors over the final seeds of
the cell's trials (over the trials where that method's record is valid);
med_ratio_vs_default = median over the trials where both are valid and the
default's error exceeds 1e-12 of chosen error / default error; top_chosen,
top_chosen_share (the method chosen most often and its share of the trials
with a choice; ties: registry order), n_distinct_chosen.

Pooled table, one row per (set, target_g, noise class, pool, design), noise
classes as in fragment f21 plus 'all': families, cells; the mean over the
cells of chosen_valid_rate, win_vs_default, default_chosen, win_vs_last;
the median over the cells of med_error_chosen, med_error_default,
med_error_last, med_ratio_vs_default (cells without a value skipped);
families_win_default_k / _n and families_win_last_k / _n = the number of
families whose mean over their cells in the class of win_vs_default (of
win_vs_last) exceeds 0.5, and the number of families.

Internal check (every set and stratum, class 'all'): med_error_default
equals med_error of the default in phase1_global.csv /
phase1_global_holdout.csv and med_error_last that of last_value, to six
decimals, and the families / cells equal n_regimes / n_cells there.  In
both designs every seed is the final seed of the same number of trials, so
the cell medians over the trials' final seeds are the cell medians over
seeds, and the pooled medians over cells must reproduce the Phase 1
pooled table; a disagreement means the cell set or the aggregation differs
from Phase 1, and the script then writes nothing and exits 2.

Floats are written at full precision (round-trip repr); rows in a fixed
order; UTF-8, LF.  Pipeline step before the tables step (reproduce_all.py),
a member of GENERATOR_STEPS: it evaluates no method, so it is not in the
path set of the code fingerprint.  --extract PATH additionally writes the
headline-stratum records of the analysis's cells (regime, is_holdout,
noise, seed, method, valid, error; every accelerator and last_value) to
PATH (gzip when the name ends in .gz), for an independent check.
"""

import argparse
import datetime as _dt
import hashlib
import json
import os
import subprocess
import sys
import time
import warnings

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import src.config as C                                    # noqa: E402
from src.pipeline import ACCEL_METHODS, resolve_regimes   # noqa: E402
from scripts.selection_defs import (DEFAULT_METHOD, DESIGNS, NOISE_CLASS_ALL, POOL_NAMES,   # noqa: E402
                                    candidate_pools, lead_rule, leading_methods, noise_class,
                                    noise_class_order, rank_accelerators, registry_index)
from scripts.analyze_by_ltrue import load_manifest        # noqa: E402  (the run manifest: library versions)
from reproduce_all import _versions                       # noqa: E402  (the library versions of this environment)

SCRIPT = "scripts/derive_selection.py"
LAST = "last_value"
STRATA = list(C.HORIZON_GAP_FRACTIONS)          # [0.5, 0.1, 0.02]
HEADLINE_G = float(C.HEADLINE_G)
RANK_MIN_VALID = float(C.RANK_MIN_VALID)
RATIO_EPS = 1e-12                               # the default's error must exceed this for a ratio
CHECK_TOL = 1e-6                                # "to six decimals"
RECORDS = os.path.join("phase1", "phase1_records.csv")
RECORD_COLS = ["regime", "is_holdout", "noise", "seed", "target_g", "capped", "method", "valid", "error"]
CELL_KEYS = ["is_holdout", "regime", "noise", "target_g"]
SET_NAME = {0: "core", 1: "holdout"}
OUT_CELLS = os.path.join("phase1", "phase1_selection_cells.csv")
OUT_GLOBAL = os.path.join("phase1", "phase1_selection_global.csv")
OUT_PROV = os.path.join("phase1", "phase1_selection_provenance.json")
CELL_COLUMNS = ["regime_set", "target_g", "regime", "noise", "noise_class", "pool", "design",
                "n_seeds", "n_trials", "n_no_choice", "chosen_valid_rate", "win_vs_default", "default_chosen", "win_vs_last",
                "med_error_chosen", "med_error_default", "med_error_last", "med_ratio_vs_default",
                "top_chosen", "top_chosen_share", "n_distinct_chosen"]
GLOBAL_COLUMNS = ["regime_set", "target_g", "noise_class", "pool", "design", "families", "cells",
                  "chosen_valid_rate", "win_vs_default", "default_chosen", "win_vs_last",
                  "med_error_chosen", "med_error_default", "med_error_last", "med_ratio_vs_default",
                  "families_win_default_k", "families_win_default_n", "families_win_last_k", "families_win_last_n"]
DESIGN_TEXT = {
    "one_pilot": "every ordered pair of distinct seeds (pilot, final); chosen = the pool member with the lowest error on the "
                 "pilot seed among those valid there (ties: registry order); its record on the final seed is taken as it is "
                 "(an invalid record stays invalid, no fallback)",
    "many_pilots": "every seed in turn is the final run and the other seeds are the pilots; chosen = the pool member with the "
                   "lowest median error over its valid pilot records among the members valid on at least RANK_MIN_VALID of "
                   f"the pilot seeds ({RANK_MIN_VALID:g}; ties: registry order)",
}
NOTE = ("This analysis was added after the full run recorded in run_manifest.json and evaluates no method: it rereads the "
        "per-record file of that run and chooses among, and summarises, records already computed.")


# ── the population ────────────────────────────────────────────────────────────
def uncapped_cells(agg):
    """The (is_holdout, regime, noise, target_g) cells with capped == 0 in phase1_aggregated.csv,
    in the fixed order of the output (set, stratum, canonical regime order, noise)."""
    flags = agg.groupby(CELL_KEYS).capped.agg(["min", "max"])
    bad = flags[flags["min"] != flags["max"]]
    assert bad.empty, f"the capped flag differs between methods of a cell: {bad.index.tolist()[:5]}"
    cells = [k for k, v in flags["max"].items() if int(v) == 0]
    order = {r: i for i, r in enumerate(resolve_regimes())}
    g_order = {g: i for i, g in enumerate(STRATA)}
    cells.sort(key=lambda k: (int(k[0]), g_order.get(float(k[3]), len(STRATA)), order.get(k[1], len(order)), k[1], float(k[2])))
    return cells


def error_matrix(cell_records, methods):
    """errors[i, j] of method i on seed j (seeds ascending): the record's error when
    valid == 1, NaN otherwise (an invalid or missing record has no error)."""
    seeds = sorted(cell_records.seed.unique())
    err = cell_records.pivot(index="method", columns="seed", values="error").reindex(index=methods, columns=seeds)
    val = cell_records.pivot(index="method", columns="seed", values="valid").reindex(index=methods, columns=seeds)
    E = np.array(err.to_numpy(dtype=float), dtype=float)          # a writable copy (pandas may hand back a read-only view)
    E[~(val.to_numpy(dtype=float) == 1)] = np.nan
    return E, seeds


# ── the choices ───────────────────────────────────────────────────────────────
def choose_one_pilot(E_pool):
    """Per pilot seed (column): the row of the member with the lowest error among the
    members valid there (the first row at a tie = registry order when the rows are in
    registry order); -1 when no member is valid on that seed."""
    n_s = E_pool.shape[1]
    chosen = np.full(n_s, -1, dtype=int)
    for p in range(n_s):
        col = E_pool[:, p]
        if np.isfinite(col).any():
            chosen[p] = int(np.nanargmin(col))
    return chosen


def choose_many_pilots(E_pool, min_valid):
    """Per final seed (column): the row of the member with the lowest median error over
    the other seeds (median of its valid records there) among the members valid on at
    least min_valid of those seeds (the first row at a tie); -1 when none qualifies or
    there is no other seed."""
    n_m, n_s = E_pool.shape
    chosen = np.full(n_s, -1, dtype=int)
    if n_s < 2:
        return chosen
    for f in range(n_s):
        P = np.delete(E_pool, f, axis=1)
        n_valid = np.isfinite(P).sum(axis=1)
        eligible = (n_valid / P.shape[1]) >= min_valid
        if not eligible.any():
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)        # all-NaN rows are ineligible anyway
            med = np.nanmedian(P, axis=1)
        chosen[f] = int(np.argmin(np.where(eligible, med, np.inf)))
    return chosen


def trials(E_pool, design, min_valid=RANK_MIN_VALID):
    """The trials of a cell and pool: a list of (final seed column, chosen row or -1)."""
    n_s = E_pool.shape[1]
    if design == "one_pilot":
        ch = choose_one_pilot(E_pool)
        return [(f, int(ch[p])) for p in range(n_s) for f in range(n_s) if f != p]
    if design == "many_pilots":
        ch = choose_many_pilots(E_pool, min_valid)
        return [(f, int(ch[f])) for f in range(n_s)] if n_s >= 2 else []
    raise ValueError(design)


# ── per-cell statistics ───────────────────────────────────────────────────────
def _median(v):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else float("nan")


def cell_stats(E_pool, members, e_default, e_last, trial_list):
    """The per-cell statistics of one (cell, pool, design); members = the pool's methods in
    the row order of E_pool; e_default / e_last = the default's and the last value's errors
    per seed (NaN when invalid)."""
    n_s = E_pool.shape[1]
    n_t = len(trial_list)
    if n_t == 0:
        return dict(n_seeds=n_s, n_trials=0, n_no_choice=0, chosen_valid_rate=float("nan"), win_vs_default=float("nan"),
                    default_chosen=float("nan"), win_vs_last=float("nan"), med_error_chosen=float("nan"),
                    med_error_default=float("nan"), med_error_last=float("nan"), med_ratio_vs_default=float("nan"),
                    top_chosen="", top_chosen_share=float("nan"), n_distinct_chosen=0)
    finals = np.array([f for f, _ in trial_list], dtype=int)
    rows = np.array([c for _, c in trial_list], dtype=int)
    has = rows >= 0
    chosen_err = np.full(n_t, np.nan)
    chosen_err[has] = E_pool[rows[has], finals[has]]
    valid = np.isfinite(chosen_err)
    d, l = np.asarray(e_default, dtype=float)[finals], np.asarray(e_last, dtype=float)[finals]
    with np.errstate(invalid="ignore"):
        win_d = valid & (~np.isfinite(d) | (chosen_err < d))
        win_l = valid & (~np.isfinite(l) | (chosen_err < l))
        both = valid & np.isfinite(d) & (d > RATIO_EPS)
        ratio = chosen_err[both] / d[both]
    names = [members[c] for c in rows[has]]
    is_default = np.zeros(n_t, dtype=bool)
    is_default[has] = np.array([m == DEFAULT_METHOD for m in names], dtype=bool) if names else False
    counts = {}
    for m in names:
        counts[m] = counts.get(m, 0) + 1
    if counts:
        top = sorted(counts, key=lambda m: (-counts[m], registry_index(m)))[0]
        top_share = counts[top] / len(names)
    else:
        top, top_share = "", float("nan")
    return dict(n_seeds=n_s, n_trials=n_t, n_no_choice=int((~has).sum()),
                chosen_valid_rate=float(valid.mean()), win_vs_default=float(win_d.mean()),
                default_chosen=float(is_default.mean()), win_vs_last=float(win_l.mean()),
                med_error_chosen=_median(chosen_err[valid]), med_error_default=_median(d), med_error_last=_median(l),
                med_ratio_vs_default=_median(ratio) if both.any() else float("nan"),
                top_chosen=top, top_chosen_share=float(top_share), n_distinct_chosen=len(counts))


# ── the two tables ────────────────────────────────────────────────────────────
def derive_cells(records, agg, pools, designs=DESIGNS, min_valid=RANK_MIN_VALID):
    """The per-cell table (CELL_COLUMNS) from the records and the aggregated table."""
    methods = list(ACCEL_METHODS) + [LAST]
    missing = sorted(set(m for p in pools.values() for m in p) - set(records.method.unique()))
    assert not missing, f"pool members absent from the records: {missing}"
    assert LAST in set(records.method.unique()) and DEFAULT_METHOD in set(records.method.unique())
    idx = {m: i for i, m in enumerate(methods)}
    groups = {k: v for k, v in records.groupby(CELL_KEYS, sort=False)}
    rows = []
    for key in uncapped_cells(agg):
        hold, regime, noise, g = int(key[0]), key[1], float(key[2]), float(key[3])
        sub = groups.get((hold, regime, noise, g))
        if sub is None:
            sub = groups.get(key)
        assert sub is not None, f"cell {key} of phase1_aggregated.csv has no records"
        E, seeds = error_matrix(sub, methods)
        e_default, e_last = E[idx[DEFAULT_METHOD]], E[idx[LAST]]
        for pool in POOL_NAMES:
            members = sorted(pools[pool], key=registry_index)
            E_pool = E[[idx[m] for m in members]]
            for design in designs:
                st = cell_stats(E_pool, members, e_default, e_last, trials(E_pool, design, min_valid))
                rows.append(dict(regime_set=SET_NAME[hold], target_g=g, regime=regime, noise=noise,
                                 noise_class=noise_class(regime, noise), pool=pool, design=design, **st))
    return pd.DataFrame(rows, columns=CELL_COLUMNS)


def pooled(cells):
    """The pooled table (GLOBAL_COLUMNS) from the per-cell table: one row per (set, target_g,
    noise class including 'all', pool, design); means over cells of the rates, medians over
    cells of the errors and the ratio, and the families whose mean win rate exceeds 0.5."""
    rows = []
    sets = [s for s in SET_NAME.values() if s in set(cells.regime_set)]
    strata = [g for g in STRATA if g in set(cells.target_g)] + sorted(set(cells.target_g) - set(STRATA))
    for rs in sets:
        for g in strata:
            S = cells[(cells.regime_set == rs) & (cells.target_g == g)]
            if S.empty:
                continue
            classes = sorted(S.noise_class.unique(), key=noise_class_order) + [NOISE_CLASS_ALL]
            for cls in classes:
                Sc = S if cls == NOISE_CLASS_ALL else S[S.noise_class == cls]
                for pool in POOL_NAMES:
                    for design in DESIGNS:
                        sub = Sc[(Sc.pool == pool) & (Sc.design == design)]
                        if sub.empty:
                            continue
                        fam = sub.groupby("regime")[["win_vs_default", "win_vs_last"]].mean()
                        rows.append(dict(
                            regime_set=rs, target_g=g, noise_class=cls, pool=pool, design=design,
                            families=int(sub.regime.nunique()), cells=int(len(sub)),
                            chosen_valid_rate=float(sub.chosen_valid_rate.mean()), win_vs_default=float(sub.win_vs_default.mean()),
                            default_chosen=float(sub.default_chosen.mean()), win_vs_last=float(sub.win_vs_last.mean()),
                            med_error_chosen=float(sub.med_error_chosen.median()), med_error_default=float(sub.med_error_default.median()),
                            med_error_last=float(sub.med_error_last.median()), med_ratio_vs_default=float(sub.med_ratio_vs_default.median()),
                            families_win_default_k=int((fam.win_vs_default > 0.5).sum()), families_win_default_n=int(len(fam)),
                            families_win_last_k=int((fam.win_vs_last > 0.5).sum()), families_win_last_n=int(len(fam))))
    return pd.DataFrame(rows, columns=GLOBAL_COLUMNS)


def check_against_phase1(glob, G_core, G_hold, tol=CHECK_TOL):
    """The internal check: at every set and stratum the 'all' rows reproduce the pooled Phase 1
    table (med_error of the default and of last_value to six decimals; n_regimes, n_cells).
    Returns the evidence (one dict per set and stratum); raises AssertionError on a disagreement."""
    evidence, problems = [], []
    for rs, G in (("core", G_core), ("holdout", G_hold)):
        for g in STRATA:
            ref = G[G.target_g == g]
            sub = glob[(glob.regime_set == rs) & (glob.target_g == g) & (glob.noise_class == NOISE_CLASS_ALL)]
            if ref.empty or sub.empty:
                continue
            ref_d = float(ref[ref.method == DEFAULT_METHOD].med_error.iloc[0])
            ref_l = float(ref[ref.method == LAST].med_error.iloc[0])
            n_fam, n_cells = int(ref.n_regimes.max()), int(ref.n_cells.max())
            ok = True
            for _, r in sub.iterrows():
                if not (abs(float(r.med_error_default) - ref_d) <= tol and abs(float(r.med_error_last) - ref_l) <= tol
                        and int(r.families) == n_fam and int(r.cells) == n_cells):
                    ok = False
                    problems.append(f"{rs} g={g:g} pool={r.pool} design={r.design}: med_error_default {r.med_error_default!r} vs "
                                    f"{ref_d!r}; med_error_last {r.med_error_last!r} vs {ref_l!r}; families {r.families} vs {n_fam}; "
                                    f"cells {r.cells} vs {n_cells}")
            d_max = float((sub.med_error_default - ref_d).abs().max())
            l_max = float((sub.med_error_last - ref_l).abs().max())
            evidence.append(dict(regime_set=rs, target_g=g, rows=int(len(sub)), families=n_fam, cells=n_cells,
                                 med_error_default_phase1=ref_d, med_error_last_phase1=ref_l,
                                 max_abs_diff_default=d_max, max_abs_diff_last=l_max, agree=ok))
    assert not problems, "the selection tables disagree with the pooled Phase 1 tables:\n  " + "\n  ".join(problems)
    return evidence


# ── provenance and I/O ───────────────────────────────────────────────────────
def _git(*args):
    try:
        return subprocess.check_output(["git"] + list(args), cwd=_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def load_pools(results):
    """The four pools from the committed pooled Phase 1 tables (LEAD at the headline stratum)."""
    G_core = pd.read_csv(os.path.join(results, "phase1", "phase1_global.csv"))
    G_hold = pd.read_csv(os.path.join(results, "phase1", "phase1_global_holdout.csv"))
    rank_core, rank_hold = rank_accelerators(G_core, STRATA), rank_accelerators(G_hold, STRATA)
    lead = leading_methods(G_core, rank_core, rank_hold, HEADLINE_G)
    return candidate_pools(lead), G_core, G_hold


def write_extract(records, agg, path, g=HEADLINE_G):
    """The headline-stratum records of the analysis's cells (every accelerator and last_value)."""
    keys = {(int(k[0]), k[1], float(k[2])) for k in uncapped_cells(agg) if float(k[3]) == g}
    sub = records[(records.target_g == g) & records.method.isin(list(ACCEL_METHODS) + [LAST])]
    keep = [(int(h), r, float(n)) in keys for h, r, n in zip(sub.is_holdout, sub.regime, sub.noise)]
    out = sub[keep][["regime", "is_holdout", "noise", "seed", "method", "valid", "error"]]
    out = out.sort_values(["is_holdout", "regime", "noise", "seed", "method"], kind="stable")
    out.to_csv(path, index=False, encoding="utf-8", lineterminator="\n",
               compression="gzip" if path.endswith(".gz") else None)
    return len(out)


def main(argv=None):
    p = argparse.ArgumentParser(description="Choosing a method by trial on the Phase 1 records -> results/phase1/phase1_selection_*.csv")
    p.add_argument("--results", default=os.path.join(_ROOT, "results"))
    p.add_argument("--extract", default=None, metavar="PATH",
                   help="also write the headline-stratum records of the analysis's cells to PATH (gzip when it ends in .gz)")
    args = p.parse_args(argv)
    results = args.results
    rec_path = os.path.join(results, RECORDS)
    if not os.path.exists(rec_path):
        print(f"ERROR: {os.path.relpath(rec_path, _ROOT)} absent (git-ignored; written by the run); nothing written")
        return 1
    t0 = time.time()
    started = _dt.datetime.now().isoformat(timespec="seconds")
    print(f"derive_selection.py: results = {os.path.relpath(results, _ROOT)}")
    pools, G_core, G_hold = load_pools(results)
    for name in POOL_NAMES:
        print(f"  pool {name:<9} ({len(pools[name]):>2}): {', '.join(pools[name])}")
    records = pd.read_csv(rec_path, usecols=RECORD_COLS)
    agg = pd.read_csv(os.path.join(results, "phase1", "phase1_aggregated.csv"),
                      usecols=CELL_KEYS + ["method", "capped"])
    n_rows, sha = len(records), sha256_file(rec_path)
    print(f"  records: {n_rows:,} rows, sha256 {sha}")
    cells = derive_cells(records, agg, pools)
    glob = pooled(cells)
    try:
        evidence = check_against_phase1(glob, G_core, G_hold)
    except AssertionError as exc:
        print(f"  INTERNAL CHECK FAILED -- nothing written\n  {exc}")
        return 2
    for e in evidence:
        print(f"  check {e['regime_set']:<7} g={e['target_g']:<5g} {e['families']} families, {e['cells']} cells: "
              f"|d default| {e['max_abs_diff_default']:.2e}, |d last| {e['max_abs_diff_last']:.2e}: {'agree' if e['agree'] else 'DISAGREE'}")
    finished = _dt.datetime.now().isoformat(timespec="seconds")
    seconds = round(time.time() - t0, 1)
    one_pool = cells[cells.pool == POOL_NAMES[0]]
    M = load_manifest(results)
    versions = _versions()
    prov = {
        "script": SCRIPT,
        "git_head": {"full": _git("rev-parse", "HEAD"), "short": _git("rev-parse", "--short", "HEAD")},
        "tracked_tree_clean": _git("status", "--porcelain", "--untracked-files=no") == "",
        "started": started, "finished": finished, "seconds": seconds,
        "versions": versions,
        "versions_equal_run_manifest": (M is not None and M.get("versions") == versions),
        "records_file": RECORDS.replace(os.sep, "/"), "records_rows": int(n_rows), "records_sha256": sha,
        "population": "the (regime, noise, target_g) cells with capped == 0 in phase1_aggregated.csv, all three strata, core and held-out; "
                      "runs = seeds; a method's error on a run is its error when valid == 1",
        "pools": {name: list(pools[name]) for name in POOL_NAMES},
        "lead_rule": lead_rule(HEADLINE_G),
        "default_method": DEFAULT_METHOD,
        "rank_min_valid": RANK_MIN_VALID,
        "designs": dict(DESIGN_TEXT),
        "cells": {d: int((one_pool.design == d).sum()) for d in DESIGNS},
        "trials": {d: int(one_pool[one_pool.design == d].n_trials.sum()) for d in DESIGNS},
        "cells_by_set_and_stratum": [dict(regime_set=e["regime_set"], target_g=e["target_g"], families=e["families"], cells=e["cells"])
                                     for e in evidence],
        "phase1_agreement": evidence,
        "rows": {"cells": int(len(cells)), "global": int(len(glob))},
        "note": NOTE,
    }
    write_csv(cells, os.path.join(results, OUT_CELLS))
    write_csv(glob, os.path.join(results, OUT_GLOBAL))
    with open(os.path.join(results, OUT_PROV), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(prov, fh, indent=2)
        fh.write("\n")
    print(f"  wrote {OUT_CELLS.replace(os.sep, '/')} ({len(cells)} rows), {OUT_GLOBAL.replace(os.sep, '/')} ({len(glob)} rows), "
          f"{OUT_PROV.replace(os.sep, '/')}; {seconds} s")
    print(f"  cells per design: {prov['cells']}; trials per design: {prov['trials']}")
    if args.extract:
        n = write_extract(records, agg, args.extract)
        print(f"  wrote extract {args.extract} ({n:,} rows; g = {HEADLINE_G:g}; {sha256_file(args.extract)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
