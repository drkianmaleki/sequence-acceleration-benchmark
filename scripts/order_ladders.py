"""
order_ladders.py
================
Phase 0b — order ladders on the Phase-1 grid (redesign v2).

For every accelerator family with an order parameter, the *ladder* is the set
of variants obtained by changing only that parameter in the family's own
constructor (LADDERS below).  Every variant, plus the four deployable trivial
comparators and constant_oracle, is evaluated on Phase 1's grid at
g = config.HEADLINE_G with exactly Phase 1's windows, noise streams, horizons
and configuration, so that a ladder variant that coincides with a roster
method reproduces that method's Phase-1 records exactly.  The results are
descriptive panels (src.panels.error_panel) per (family, variant, regime set
in {core, holdout}) over the records of uncapped cells, noise levels pooled,
plus per-noise rows.  No ranking, no score.

The family constructors are the private helpers of src/accelerators.py
(_accel_shanks, _wynn_epsilon, _wynn_rho, _levin_transform, _brezinski_theta,
_anderson, _neville, _pade_fit, _fit_richardson, _richardson_fixed,
_parametric_fit); they are imported here deliberately, because a ladder must
be built from the very code path of the registry entries.  Every ladder order
that is a roster method uses the registered function itself
(src.accelerators.METHODS[name]); the other orders call the constructor with
the changed parameter.  Three ladder-only models that no constructor offers
(the 4-term Richardson fit, the triple exponential and the two-term rational
fit) are written in this script, in the style of their family.

The evaluation loop mirrors src.evaluation.run_phase1 line by line (the same
RandomState(seed * 137 + int(sigma * 1e6) % 9973) per (regime, sigma, seed),
window slicing, assumed-asymptote mode, horizon, build_cfg, validity and
catastrophe rules and E_last) rather than re-using run_phase1, which is
built around the registry and writes the Phase-1 tables; the agreement check
at the end (order_ladders_agreement.txt) proves that every roster-marked
variant reproduces phase1_records.csv exactly (estimate, error, valid,
catastrophic, n_f, capped; NaN == NaN) and the script exits non-zero on any
mismatch.

    python scripts/order_ladders.py --quick|--full [--results results] [--jobs N]

Outputs (<results>/phase0b/)
----------------------------
    order_ladders_records.csv    one row per (variant, regime, noise, seed) at g = HEADLINE_G (git-ignored)
    order_ladders_panels.csv     family, variant, order_label, is_roster, roster_name, regime_set,
                                 noise (a value or 'pooled'), the PANEL_COLS, n_cells, n_cells_capped_excluded
    order_ladders_agreement.txt  the agreement check: variants checked, records compared, verdict
"""

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np                              # noqa: E402
import pandas as pd                             # noqa: E402
from scipy.optimize import curve_fit            # noqa: E402

import src.config as CFG_MOD                    # noqa: E402
from src.accelerators import (                  # noqa: E402  (private constructors, deliberately)
    METHODS, _accel_shanks, _anderson, _assumed_L, _brezinski_theta, _fit_richardson,
    _levin_transform, _neville, _pade_fit, _parametric_fit, _richardson_fixed, _safe,
    _to_arrays, _wynn_epsilon, _wynn_rho)
from src.asymptote import assumed_asymptote, resolve_mode   # noqa: E402
from src.evaluation import build_cfg, is_valid               # noqa: E402
from src.generators import HOLDOUT, regime_functions        # noqa: E402
from src.horizons import horizon_for_gap                    # noqa: E402
from src.panels import PANEL_COLS, error_panel              # noqa: E402
from src.pipeline import resolve_regimes                    # noqa: E402
from src.trivial import ORACLE_METHODS, SKILL_REFERENCE_METHODS  # noqa: E402

RATE_COLS = ('valid_rate', 'cat_rate', 'win_rate_vs_last')
AGREEMENT_COLS = ['estimate', 'error', 'valid', 'catastrophic', 'n_f', 'capped']
RECORD_KEYS = ['regime', 'noise', 'seed']


# =============================================================================
# 1.  VARIANTS
# =============================================================================

@dataclass(frozen=True)
class Variant:
    family: str
    order_label: str
    fn: Callable
    roster_name: str = ''          # '' for a ladder-only order

    @property
    def is_roster(self) -> int:
        return int(bool(self.roster_name))

    @property
    def key(self) -> str:
        return f'{self.family}:{self.order_label}'


def _roster_or(name_by_order: Dict, order, make: Callable, family: str, label: str) -> Variant:
    """A roster order uses the registered function itself; any other order the constructor."""
    roster = name_by_order.get(order, '')
    fn = METHODS[roster] if roster else make(order)
    return Variant(family, label, fn, roster)


# ── constructors with the order parameter exposed (the registry wrappers' code path) ──
def _make_shanks(k):
    def fn(seq, indices, future_x, cfg):
        return _safe(_accel_shanks(seq, k, cfg.get("denom_tol", 1e-14)), cfg)
    return fn


def _make_wynn_eps(k):
    def fn(seq, indices, future_x, cfg):
        return _safe(_wynn_epsilon(seq, k, cfg.get("denom_tol", 1e-14)), cfg)
    return fn


def _make_wynn_rho(k):
    def fn(seq, indices, future_x, cfg):
        return _safe(_wynn_rho(seq, indices, k, cfg.get("denom_tol", 1e-14)), cfg)
    return fn


def _make_levin(variant):
    def make(k):
        def fn(seq, indices, future_x, cfg):
            return _safe(_levin_transform(seq, indices, k, variant, cfg.get("denom_tol", 1e-14)), cfg)
        return fn
    return make


def _make_brezinski(k):
    def fn(seq, indices, future_x, cfg):
        return _safe(_brezinski_theta(seq, k, cfg.get("denom_tol", 1e-14)), cfg)
    return fn


def _make_anderson(m):
    def fn(seq, indices, future_x, cfg):
        return _safe(_anderson(seq, indices, future_x, m, cfg.get("ridge", 1e-8)), cfg)
    return fn


def _make_neville(d):
    def fn(seq, indices, future_x, cfg):
        return _safe(_neville(seq, indices, future_x, d), cfg)
    return fn


def _make_pade(pq):
    p, q = pq

    def fn(seq, indices, future_x, cfg):
        return _pade_fit(seq, indices, future_x, p, q, cfg)
    return fn


def _make_richardson_free(n_terms):
    if n_terms == 4:
        return richardson_4term

    def fn(seq, indices, future_x, cfg):
        y, x = _to_arrays(seq, indices)
        return _fit_richardson(x, y, n_terms, future_x, cfg)
    return fn


def _make_richardson_fixed(alpha):
    def fn(seq, indices, future_x, cfg):
        return _richardson_fixed(seq, indices, future_x, cfg, alpha=alpha)
    return fn


# ── ladder-only models, written in the style of their family ──────────────────
def richardson_4term(seq, indices, future_x: float, cfg: dict) -> float:
    """Richardson 4-term: L + c1/n^a1 + c2/n^a2 + c3/n^a3 + c4/n^a4 by nonlinear LS
    (the _fit_richardson pattern extended by one term; ladder only)."""
    y, x = _to_arrays(seq, indices)
    L0 = _assumed_L(cfg)
    x_safe = np.maximum(x, 1.0)

    def model(n, L, c1, a1, c2, a2, c3, a3, c4, a4):
        return L + c1 / n**a1 + c2 / n**a2 + c3 / n**a3 + c4 / n**a4
    p0     = [L0, 0.3, 0.5, 0.2, 1.0, 0.1, 2.0, 0.05, 3.0]
    bounds = ([0.0, -5.0, 0.05, -5.0, 0.05, -5.0, 0.05, -5.0, 0.05],
              [2.0,  5.0, 4.00,  5.0, 4.00,  5.0, 4.00,  5.0, 4.00])
    if len(x_safe) < 13:
        return np.nan
    try:
        popt, _ = curve_fit(model, x_safe, y, p0=p0, bounds=bounds, maxfev=3000)
        return _safe(float(model(max(future_x, 1.0), *popt)), cfg)
    except Exception:
        return np.nan


def triple_exp_fit(seq, indices, future_x: float, cfg: dict) -> float:
    """Fit L + A1*exp(-l1*n) + A2*exp(-l2*n) + A3*exp(-l3*n) by nonlinear LS (ladder only)."""
    L0 = _assumed_L(cfg)

    def model(n, L, A1, l1, A2, l2, A3, l3):
        return L + A1 * np.exp(-l1 * n) + A2 * np.exp(-l2 * n) + A3 * np.exp(-l3 * n)
    return _parametric_fit(seq, indices, future_x, cfg, model,
                           p0=[L0, 0.4, 0.02, 0.2, 0.15, 0.1, 0.5],
                           bounds=([0, 0, 1e-4, 0, 1e-4, 0, 1e-4], [1, 2, 5, 2, 5, 2, 5]))


def rational_two_term_fit(seq, indices, future_x: float, cfg: dict) -> float:
    """Fit L + A/(1 + B*n) + C/(1 + D*n) by nonlinear LS (ladder only)."""
    L0 = _assumed_L(cfg)

    def model(n, L, A, B, C, D):
        return L + A / (1.0 + B * n) + C / (1.0 + D * n)
    return _parametric_fit(seq, indices, future_x, cfg, model,
                           p0=[L0, 0.5, 0.05, 0.25, 0.5],
                           bounds=([0, 0, 1e-5, 0, 1e-5], [1, 2, 10, 2, 10]))


# ── the ladders: (family, orders, roster orders, constructor, label) ─────────
PADE_ORDERS = [(p, q) for p in range(1, 6) for q in range(1, 6)] + [(5, 6), (7, 8), (9, 10)]
PADE_ROSTER = {(1, 1): 'pade_11', (1, 2): 'pade_12', (1, 3): 'pade_13', (2, 1): 'pade_21',
               (2, 2): 'pade_22', (2, 3): 'pade_23', (3, 1): 'pade_31', (3, 2): 'pade_32',
               (3, 3): 'pade_33', (3, 4): 'pade_34', (4, 4): 'pade_44', (4, 5): 'pade_45'}
PARAMETRIC_MODELS = [
    # (order_label, roster name or None, ladder-only callable)
    ('single-exp',        'single_exp_fit', None),
    ('double-exp',        'double_exp_fit', None),
    ('triple-exp',        None,             triple_exp_fit),
    ('rational one-term', 'rational_fit',   None),
    ('rational two-term', None,             rational_two_term_fit),
    ('log',               'log_fit',        None),
]

LADDERS = [
    # family, orders, roster {order: name}, constructor(order) -> fn, label(order)
    ('shanks',          range(1, 11), {k: f'shanks_{k}' for k in range(1, 5)},           _make_shanks,      lambda k: f'k={k}'),
    ('wynn_eps',        range(1, 9),  {k: f'wynn_eps_{k}' for k in range(1, 4)},         _make_wynn_eps,    lambda k: f'k={k}'),
    ('wynn_rho',        range(1, 8),  {k: f'wynn_rho_{k}' for k in range(1, 4)},         _make_wynn_rho,    lambda k: f'k={k}'),
    ('levin_t',         range(1, 9),  {1: 'levin_t1', 2: 'levin_t2'},                     _make_levin('t'),  lambda k: f'k={k}'),
    ('levin_u',         range(1, 9),  {1: 'levin_u1', 2: 'levin_u2'},                     _make_levin('u'),  lambda k: f'k={k}'),
    ('levin_v',         range(1, 9),  {1: 'levin_v1', 2: 'levin_v2'},                     _make_levin('v'),  lambda k: f'k={k}'),
    ('brezinski_theta', range(1, 7),  {1: 'brezinski_theta1', 2: 'brezinski_theta2'},     _make_brezinski,   lambda k: f'k={k}'),
    ('anderson',        range(1, 7),  {m: f'anderson_{m}' for m in range(1, 4)},         _make_anderson,    lambda m: f'm={m}'),
    ('neville',         range(1, 9),  {d: f'neville_{d}' for d in range(2, 5)},          _make_neville,     lambda d: f'd={d}'),
    ('pade',            PADE_ORDERS,  PADE_ROSTER,                                        _make_pade,        lambda pq: f'[{pq[0]},{pq[1]}]'),
    ('richardson_free', range(1, 5),  {n: f'richardson_{n}' for n in range(1, 4)},       _make_richardson_free, lambda n: f'{n} term' + ('s' if n > 1 else '')),
    ('richardson_fixed', [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
                         {0.5: 'richardson_a05', 1.0: 'richardson_a10', 2.0: 'richardson_a20'}, _make_richardson_fixed, lambda a: f'alpha={a:g}'),
]
LADDER_FAMILIES = [name for name, *_ in LADDERS] + ['parametric']
COMPARATORS = list(SKILL_REFERENCE_METHODS) + sorted(ORACLE_METHODS)   # the four deployable trivials + constant_oracle


def build_ladders() -> List[Variant]:
    """Every ladder variant in definition order (the parametric ladder last)."""
    out: List[Variant] = []
    for family, orders, roster, make, label in LADDERS:
        for order in orders:
            out.append(_roster_or(roster, order, make, family, label(order)))
    for label, roster, fn in PARAMETRIC_MODELS:
        out.append(Variant('parametric', label, METHODS[roster] if roster else fn, roster or ''))
    return out


def comparator_variants() -> List[Variant]:
    """The deployable trivials and constant_oracle, evaluated alongside (family 'trivial')."""
    return [Variant('trivial', m, METHODS[m], m) for m in COMPARATORS]


def all_variants() -> List[Variant]:
    return build_ladders() + comparator_variants()


def ladder_sizes() -> Dict[str, int]:
    sizes: Dict[str, int] = {}
    for v in build_ladders():
        sizes[v.family] = sizes.get(v.family, 0) + 1
    return sizes


# =============================================================================
# 2.  EVALUATION (mirrors src.evaluation.run_phase1 line by line)
# =============================================================================

def evaluate_regime(regime: str, n_seeds: int, noise_levels: List[float], obs_idx: int,
                    window_len: int, g: float, assumed_mode: str, asymptote_mode: str) -> pd.DataFrame:
    """
    Every variant on every (noise, seed) window of one regime at stratum g,
    exactly as run_phase1 builds the window, the horizon and the record.
    """
    variants = all_variants()
    last_key = 'trivial:last_value'
    n_arr   = np.arange(obs_idx + 1, dtype=float)
    w_start = max(0, obs_idx - window_len + 1)
    idx_win = list(range(w_start, obs_idx + 1))
    holdout = int(regime in HOLDOUT)
    records = []
    for sigma in noise_levels:
        for seed in range(n_seeds):
            gen, truth, L_true = regime_functions(regime, seed, asymptote_mode)
            rng      = np.random.RandomState(seed * 137 + int(sigma * 1e6) % 9973)
            seq_full = gen(n_arr, rng, sigma)
            seq_win  = list(seq_full[w_start : obs_idx + 1])
            curr_val = float(seq_full[obs_idx])
            L_hat    = assumed_asymptote(L_true, seq_win, assumed_mode)

            hz       = horizon_for_gap(regime, obs_idx, g, seed=seed)
            n_f      = hz.n_f
            true_val = float(truth(n_f))
            curr_err = abs(curr_val - true_val)
            cfg      = build_cfg(n_f, L_hat, L_true)

            ests: Dict[str, float] = {}
            errs: Dict[str, float] = {}
            for v in variants:
                try:
                    est = float(v.fn(seq_win, idx_win, float(n_f), cfg))
                except Exception:
                    est = float('nan')
                ests[v.key] = est
                errs[v.key] = abs(est - true_val) if is_valid(est, cfg) else float('nan')
            E_last = errs[last_key]

            for v in variants:
                est, err = ests[v.key], errs[v.key]
                valid = math.isfinite(err)
                cat   = (not valid) or (curr_err > 1e-12 and err > CFG_MOD.CAT_MULT * curr_err)
                records.append({
                    'family':       v.family,
                    'variant':      v.key,
                    'order_label':  v.order_label,
                    'is_roster':    v.is_roster,
                    'roster_name':  v.roster_name,
                    'regime':       regime,
                    'is_holdout':   holdout,
                    'noise':        sigma,
                    'seed':         seed,
                    'L_true':       L_true,
                    'L_hat':        L_hat,
                    'target_g':     g,
                    'achieved_g':   hz.achieved_g,
                    'n_f':          int(n_f),
                    'capped':       int(hz.capped),
                    'true_val':     true_val,
                    'curr_val':     curr_val,
                    'estimate':     est if valid else float('nan'),
                    'error':        err if valid else float('nan'),
                    'E_last':       E_last,
                    'valid':        int(valid),
                    'catastrophic': int(cat),
                })
    return pd.DataFrame(records)


def _regime_task(args):
    regime, kw = args
    t0 = time.perf_counter()
    df = evaluate_regime(regime, **kw)
    return regime, df, time.perf_counter() - t0


def evaluate_grid(regimes: List[str], n_seeds: int, noise_levels: List[float], obs_idx: int,
                  window_len: int, g: float, assumed_mode: Optional[str] = None,
                  asymptote_mode: Optional[str] = None, jobs: int = 1, verbose: bool = True) -> pd.DataFrame:
    """All regimes, serially or in a process pool; records concatenated in regime order."""
    assumed_mode   = resolve_mode(assumed_mode)
    asymptote_mode = CFG_MOD.ASYMPTOTE_MODE if asymptote_mode is None else asymptote_mode
    kw = dict(n_seeds=n_seeds, noise_levels=[float(s) for s in noise_levels], obs_idx=int(obs_idx),
              window_len=int(window_len), g=float(g), assumed_mode=assumed_mode, asymptote_mode=asymptote_mode)
    tasks = [(r, kw) for r in regimes]
    jobs = max(1, min(int(jobs), len(tasks)))
    frames: Dict[str, pd.DataFrame] = {}
    t_start = time.perf_counter()

    def _done(regime, df, t):
        frames[regime] = df
        if verbose:
            print(f'  [{len(frames):>2}/{len(tasks)}] {regime:<18} {len(df):>7,} records  '
                  f'{t:6.1f} s  (wall {time.perf_counter() - t_start:6.1f} s)', flush=True)

    if jobs == 1:
        for task in tasks:
            _done(*_regime_task(task))
    else:
        import multiprocessing as mp
        with mp.get_context('spawn').Pool(processes=jobs) as pool:
            for regime, df, t in pool.imap_unordered(_regime_task, tasks):
                _done(regime, df, t)
    return pd.concat([frames[r] for r in regimes], ignore_index=True)


# =============================================================================
# 3.  PANELS
# =============================================================================

def build_panels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per (family, variant, regime_set, noise): the descriptive panel over the
    records of the uncapped cells (noise pooled, and per noise level), the
    number of contributing cells and the number of capped cells excluded.
    A cell is one (regime, noise) at the single stratum.
    """
    rows = []
    meta_cols = ['family', 'variant', 'order_label', 'is_roster', 'roster_name']
    for meta, grp in df.groupby(meta_cols, sort=False):
        for regime_set, flag in (('core', 0), ('holdout', 1)):
            rs = grp[grp['is_holdout'] == flag]
            if rs.empty:
                continue
            unc = rs[rs['capped'] == 0]
            cap = rs[rs['capped'] == 1]
            slices = [('pooled', unc, cap)] + [(f'{s:g}', unc[unc['noise'] == s], cap[cap['noise'] == s])
                                               for s in sorted(rs['noise'].unique())]
            for noise, u, c in slices:
                panel = error_panel(u['error'], u['valid'], u['catastrophic'], u['E_last'])
                row = dict(zip(meta_cols, meta))
                row.update({'regime_set': regime_set, 'noise': noise})
                row.update({k: (round(v, 4) if k in RATE_COLS and math.isfinite(v) else v) for k, v in panel.items()})
                row['n_cells'] = int(u[['regime', 'noise']].drop_duplicates().shape[0])
                row['n_cells_capped_excluded'] = int(c[['regime', 'noise']].drop_duplicates().shape[0])
                rows.append(row)
    cols = meta_cols + ['regime_set', 'noise', *PANEL_COLS, 'n_cells', 'n_cells_capped_excluded']
    return pd.DataFrame(rows, columns=cols)


# =============================================================================
# 4.  AGREEMENT CHECK AGAINST PHASE 1
# =============================================================================

def agreement_check(df: pd.DataFrame, phase1_records: str, g: float) -> Tuple[str, bool]:
    """
    For every roster-marked variant, its records must equal the rows of
    phase1_records.csv with the same (regime, noise, seed) at target_g = g on
    AGREEMENT_COLS (exact equality, NaN == NaN), and the two key sets must
    coincide.  The reference is parsed with float_precision='round_trip' so
    that the CSV text written by Phase 1 (Python's shortest round-trip repr)
    gives back the very floats Phase 1 computed.  Returns (report text, ok).
    """
    lines = [f'Order-ladder agreement check against {phase1_records} at g = {g:g}',
             f'columns compared exactly (NaN == NaN): {AGREEMENT_COLS}']
    if not os.path.exists(phase1_records):
        lines.append('VERDICT: MISMATCH -- phase1_records.csv not found (run Phase 1 first)')
        return '\n'.join(lines) + '\n', False
    p1 = pd.read_csv(phase1_records, usecols=RECORD_KEYS + ['target_g', 'method'] + AGREEMENT_COLS,
                     float_precision='round_trip')
    p1 = p1[p1['target_g'] == g]
    roster = df[df['is_roster'] == 1]
    variants = roster[['variant', 'roster_name']].drop_duplicates()
    n_checked = n_records = 0
    problems: List[str] = []
    for variant, name in variants.itertuples(index=False):
        mine = roster[roster['variant'] == variant][RECORD_KEYS + AGREEMENT_COLS]
        ref = p1[p1['method'] == name][RECORD_KEYS + AGREEMENT_COLS]
        km = set(map(tuple, mine[RECORD_KEYS].to_numpy()))
        kr = set(map(tuple, ref[RECORD_KEYS].to_numpy()))
        if km != kr:
            problems.append(f'{variant} -> {name}: key sets differ ({len(km - kr)} only in ladder, {len(kr - km)} only in Phase 1)')
        merged = mine.merge(ref, on=RECORD_KEYS, suffixes=('_ladder', '_p1'))
        n_checked += 1
        n_records += len(merged)
        for c in AGREEMENT_COLS:
            a = merged[c + '_ladder'].to_numpy(dtype=float)
            b = merged[c + '_p1'].to_numpy(dtype=float)
            bad = ~((a == b) | (np.isnan(a) & np.isnan(b)))
            if bad.any():
                ex = merged.loc[bad, RECORD_KEYS + [c + '_ladder', c + '_p1']].head(3)
                problems.append(f'{variant} -> {name}: {int(bad.sum())} of {len(merged)} records differ in {c}; e.g.\n'
                                + ex.to_string(index=False))
    ok = not problems
    lines.append(f'variants checked : {n_checked} (every ladder variant that is a roster method, and the comparators)')
    lines.append(f'records compared : {n_records:,}')
    lines.append(f'mismatches       : {len(problems)}')
    lines += problems
    lines.append('VERDICT: ' + ('AGREE -- every roster-marked variant reproduces its Phase-1 records exactly'
                                if ok else 'MISMATCH -- see the lines above'))
    return '\n'.join(lines) + '\n', ok


# =============================================================================
# 5.  CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description='Phase 0b — order ladders on the Phase-1 grid')
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--quick', action='store_true', help='config.PHASE1["quick"] grid')
    mode.add_argument('--full', action='store_true', help='config.PHASE1["full"] grid')
    p.add_argument('--results', default=os.path.join('results'),
                   help='results tree: reads <results>/phase1/phase1_records.csv, writes <results>/phase0b/')
    p.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 2) - 1),
                   help='worker processes over regimes (default cpu_count() - 1; 1 = serial)')
    return p.parse_args()


def main() -> int:
    args = parse_args()
    mode = 'quick' if args.quick else 'full'
    cfg = dict(CFG_MOD.PHASE1[mode])
    g = float(CFG_MOD.HEADLINE_G)
    regimes = resolve_regimes(cfg['core_regimes'], cfg['holdout_regimes'], include_holdout=True)
    out_dir = os.path.join(args.results, 'phase0b')
    os.makedirs(out_dir, exist_ok=True)
    variants = all_variants()
    n_ladder = len(build_ladders())
    sizes = ladder_sizes()

    print('=' * 72)
    print(f'  PHASE 0b — Order ladders on the Phase-1 grid  [{mode.upper()}, redesign v2]')
    print('=' * 72)
    print(f'  ladders     : ' + ', '.join(f'{k} {v}' for k, v in sizes.items()) + f'  ({n_ladder} variants)')
    print(f'  roster hits : {sum(v.is_roster for v in build_ladders())} ladder variants are roster methods; '
          f'+ {len(COMPARATORS)} comparators {COMPARATORS}')
    print(f'  grid        : {len(regimes)} regimes x {cfg["n_seeds"]} seeds x {len(cfg["noise_levels"])} noise, '
          f'obs_idx {cfg["obs_idx"]}, window {cfg["window_len"]}, g = {g:g} only')
    print(f'  evaluations : {len(variants) * len(regimes) * cfg["n_seeds"] * len(cfg["noise_levels"]):,}')
    print(f'  jobs        : {args.jobs}    output: {out_dir}/')
    print('=' * 72 + '\n')

    t0 = time.perf_counter()
    df = evaluate_grid(regimes, cfg['n_seeds'], cfg['noise_levels'], cfg['obs_idx'], cfg['window_len'], g,
                       jobs=args.jobs, verbose=True)
    t_eval = time.perf_counter() - t0
    p = os.path.join(out_dir, 'order_ladders_records.csv')
    df.to_csv(p, index=False)
    print(f'\n  Saved: {p}  ({len(df):,} rows; git-ignored)  evaluation {t_eval:.1f} s')

    panels = build_panels(df)
    p = os.path.join(out_dir, 'order_ladders_panels.csv')
    panels.to_csv(p, index=False)
    print(f'  Saved: {p}  ({len(panels)} rows)')

    text, ok = agreement_check(df, os.path.join(args.results, 'phase1', 'phase1_records.csv'), g)
    p = os.path.join(out_dir, 'order_ladders_agreement.txt')
    with open(p, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(text)
    print(f'  Saved: {p}\n')
    print(text)

    # console summary: core, noise pooled, per family
    core = panels[(panels['regime_set'] == 'core') & (panels['noise'] == 'pooled')]
    print(f"  {'family':<16} {'variant':<20} {'roster':<16} {'valid':>6} {'med err':>10} {'q25':>9} {'q75':>9} {'win/last':>9}  (core, noise pooled, conditional on validity)")
    print('  ' + '-' * 112)
    for _, r in core.iterrows():
        me = f"{r['med_error']:.5f}" if math.isfinite(r['med_error']) else 'n/a'
        q1 = f"{r['q25_error']:.5f}" if math.isfinite(r['q25_error']) else 'n/a'
        q3 = f"{r['q75_error']:.5f}" if math.isfinite(r['q75_error']) else 'n/a'
        print(f"  {r['family']:<16} {r['order_label']:<20} {r['roster_name']:<16} {r['valid_rate']:>6.3f} "
              f"{me:>10} {q1:>9} {q3:>9} {r['win_rate_vs_last']:>9.3f}")
    print('=' * 72)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
