"""
accelerators.py
===============
All 51 sequence-acceleration methods organised by family.

Each accelerator has the unified signature:
    method(seq, indices, future_x, cfg) -> float

where
    seq      : list or 1-D array of observed values (the window)
    indices  : corresponding integer indices (same length as seq)
    future_x : int, the far-future index to predict
    cfg      : dict with at least keys 'ridge', 'L_inf', 'denom_tol'

Return value is the scalar prediction, or np.nan on failure.

Mathematical references
-----------------------
[SH55]  Shanks, D. (1955). Non-linear transformation of divergent and
        slowly convergent sequences. J. Math. Phys., 34(1-4), 1-42.
[WY56]  Wynn, P. (1956). On a device for computing the e_m(S_n)
        transformation. Math. Tables Aids Comput., 10, 91-96.
[RI11]  Richardson, L. F. (1911). The approximate arithmetical solution
        by finite differences of physical problems. Phil. Trans. R.
        Soc. London A, 210, 307-357.
[LE73]  Levin, D. (1973). Development of non-linear transformations for
        improving convergence of sequences. Int. J. Comput. Math.,
        3(1-4), 371-388.
[WE89]  Weniger, E. J. (1989). Nonlinear sequence transformations for
        the acceleration of convergence. Comput. Phys. Rep., 10, 189-371.
[BR71]  Brezinski, C. (1971). Acceleration de suites a convergence
        logarithmique. C. R. Acad. Sci. Paris, 273, 727-730.
[BRZ91] Brezinski, C. & Redivo-Zaglia, M. (1991). Extrapolation Methods.
        Elsevier, Amsterdam.
[NE66]  Neville, E. H. (1934). Iterative interpolation. J. Indian Math.
        Soc., 20, 87-120.
[AN65]  Anderson, D. G. (1965). Iterative procedures for nonlinear
        integral equations. J. ACM, 12(4), 547-560.
"""

import numpy as np
from typing import List
from scipy.optimize import curve_fit
from scipy.special import comb as sp_comb

# ── Internal helpers ───────────────────────────────────────────────────────────

def _safe(v: float, cfg: dict) -> float:
    """Return v if finite and within validity bounds, else np.nan."""
    if not np.isfinite(v):
        return np.nan
    if v < cfg.get("min_valid", -0.5) or v > cfg.get("max_valid", 500.0):
        return np.nan
    return float(v)


def _to_arrays(seq, indices):
    return np.asarray(seq, dtype=float), np.asarray(indices, dtype=float)


def _normalise_x(x_raw: np.ndarray, future_x: float):
    """Map x_raw to [0,1] and scale future_x accordingly."""
    x0, x1 = x_raw[0], x_raw[-1]
    span = x1 - x0
    if abs(span) < 1e-10:
        return None, None, None
    return (x_raw - x0) / span, (future_x - x0) / span, span


# =============================================================================
# FAMILY 1 — Baselines
# =============================================================================

def accel_current_value(seq, indices, future_x: float, cfg: dict) -> float:
    """Last observed value — the hard baseline every method must beat."""
    return float(seq[-1])


def accel_linear(seq, indices, future_x: float, cfg: dict) -> float:
    """Linear extrapolation: fit s = a + b*n and evaluate at future_x."""
    y, x = _to_arrays(seq, indices)
    if len(x) < 2:
        return np.nan
    try:
        p = np.polyfit(x, y, 1)
        return _safe(float(np.polyval(p, future_x)), cfg)
    except Exception:
        return np.nan


def accel_log_linear(seq, indices, future_x: float, cfg: dict) -> float:
    """
    Log-linear extrapolation.
    Fits log(s - L_est) vs log(n) and returns L + exp(fit) at future_x.
    """
    y, x = _to_arrays(seq, indices)
    L_est = cfg.get("L_inf", 0.01)
    shifted = y - L_est
    if np.any(shifted <= 0):
        return np.nan
    try:
        log_y = np.log(shifted)
        log_x = np.log(np.maximum(x, 1.0))
        p = np.polyfit(log_x, log_y, 1)
        log_pred = np.polyval(p, np.log(max(future_x, 1.0)))
        return _safe(L_est + float(np.exp(log_pred)), cfg)
    except Exception:
        return np.nan


def accel_geom_avg_diff(seq, indices, future_x: float, cfg: dict) -> float:
    """
    Geometric-average-of-differences baseline.
    Estimates the common ratio r of recent differences and extrapolates
    the remaining geometric series sum to future_x.
    """
    y, x = _to_arrays(seq, indices)
    if len(y) < 4:
        return np.nan
    diffs = np.diff(y[-10:])
    neg = diffs[diffs < 0]
    if len(neg) < 2:
        return np.nan
    ratios = neg[1:] / neg[:-1]
    r = float(np.exp(np.mean(np.log(np.abs(ratios)))))
    if not (0.0 < r < 1.0):
        return np.nan
    d_last = diffs[-1]
    n_steps = int(future_x - x[-1])
    if n_steps <= 0:
        return float(y[-1])
    remainder = d_last * r * (1 - r ** n_steps) / (1 - r)
    return _safe(float(y[-1]) + float(remainder), cfg)


# =============================================================================
# FAMILY 2 — Richardson / Parametric Power-Law  [RI11]
# =============================================================================

def _fit_richardson(x: np.ndarray, y: np.ndarray, n_terms: int,
                    future_x: float, cfg: dict) -> float:
    """
    Fit s(n) = L + c1/n^a1 [+ c2/n^a2 [+ c3/n^a3]] by nonlinear LS.
    """
    L0 = cfg.get("L_inf", 0.01)
    x_safe = np.maximum(x, 1.0)

    if n_terms == 1:
        def model(n, L, c, a):
            return L + c / n**a
        p0     = [L0, max(float(y[-1]) - L0, 1e-4), 0.8]
        bounds = ([0.0, -5.0, 0.05], [2.0, 5.0, 4.0])
        n_req  = 4
    elif n_terms == 2:
        def model(n, L, c1, a1, c2, a2):
            return L + c1 / n**a1 + c2 / n**a2
        p0     = [L0, 0.4, 0.8, 0.2, 1.5]
        bounds = ([0.0, -5.0, 0.05, -5.0, 0.05],
                  [2.0,  5.0, 4.00,  5.0, 4.00])
        n_req  = 7
    else:
        def model(n, L, c1, a1, c2, a2, c3, a3):
            return L + c1 / n**a1 + c2 / n**a2 + c3 / n**a3
        p0     = [L0, 0.3, 0.5, 0.2, 1.0, 0.1, 2.0]
        bounds = ([0.0, -5.0, 0.05, -5.0, 0.05, -5.0, 0.05],
                  [2.0,  5.0, 4.00,  5.0, 4.00,  5.0, 4.00])
        n_req  = 10

    if len(x_safe) < n_req:
        return np.nan
    try:
        popt, _ = curve_fit(model, x_safe, y, p0=p0, bounds=bounds,
                            maxfev=3000)
        return _safe(float(model(max(future_x, 1.0), *popt)), cfg)
    except Exception:
        return np.nan


def accel_richardson_1(seq, indices, future_x: float, cfg: dict) -> float:
    """Richardson 1-term: L + c/n^alpha  [RI11]."""
    y, x = _to_arrays(seq, indices)
    return _fit_richardson(x, y, 1, future_x, cfg)


def accel_richardson_2(seq, indices, future_x: float, cfg: dict) -> float:
    """Richardson 2-term: L + c1/n^a1 + c2/n^a2  [RI11]."""
    y, x = _to_arrays(seq, indices)
    return _fit_richardson(x, y, 2, future_x, cfg)


def accel_richardson_3(seq, indices, future_x: float, cfg: dict) -> float:
    """Richardson 3-term: L + c1/n^a1 + c2/n^a2 + c3/n^a3  [RI11]."""
    y, x = _to_arrays(seq, indices)
    return _fit_richardson(x, y, 3, future_x, cfg)


def _richardson_fixed(seq, indices, future_x: float, cfg: dict,
                      alpha: float) -> float:
    """Richardson with fixed exponent: fits L + c/n^alpha by linear LS."""
    y, x = _to_arrays(seq, indices)
    L0 = cfg.get("L_inf", 0.01)
    x_s = np.maximum(x, 1.0)
    phi = 1.0 / x_s ** alpha
    A = np.column_stack([np.ones_like(phi), phi])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        L_fit, c_fit = coeffs
        pred = L_fit + c_fit / max(future_x, 1.0) ** alpha
        return _safe(float(pred), cfg)
    except Exception:
        return np.nan


def accel_richardson_half(seq, indices, future_x: float, cfg: dict) -> float:
    """Richardson fixed alpha=0.5: L + c/sqrt(n)  [RI11]."""
    return _richardson_fixed(seq, indices, future_x, cfg, alpha=0.5)


def accel_richardson_one(seq, indices, future_x: float, cfg: dict) -> float:
    """Richardson fixed alpha=1.0: L + c/n  [RI11]."""
    return _richardson_fixed(seq, indices, future_x, cfg, alpha=1.0)


def accel_richardson_two(seq, indices, future_x: float, cfg: dict) -> float:
    """Richardson fixed alpha=2.0: L + c/n^2  [RI11]."""
    return _richardson_fixed(seq, indices, future_x, cfg, alpha=2.0)


# =============================================================================
# FAMILY 3 — Parametric Functional Fits
# =============================================================================

def _parametric_fit(seq, indices, future_x: float, cfg: dict,
                    model_fn, p0, bounds) -> float:
    """Shared fitting engine for Family 3."""
    y, x = _to_arrays(seq, indices)
    if len(x) < len(p0) + 1:
        return np.nan
    try:
        popt, _ = curve_fit(model_fn, x, y, p0=p0, bounds=bounds,
                            maxfev=3000)
        return _safe(float(model_fn(float(future_x), *popt)), cfg)
    except Exception:
        return np.nan


def accel_single_exp_fit(seq, indices, future_x: float, cfg: dict) -> float:
    """Fit L + A*exp(-lambda*n) by nonlinear LS."""
    L0 = cfg.get("L_inf", 0.01)
    def model(n, L, A, lam): return L + A * np.exp(-lam * n)
    return _parametric_fit(seq, indices, future_x, cfg, model,
                           p0=[L0, 0.5, 0.05],
                           bounds=([0, 0, 1e-4], [1, 2, 5]))


def accel_double_exp_fit(seq, indices, future_x: float, cfg: dict) -> float:
    """Fit L + A1*exp(-l1*n) + A2*exp(-l2*n) by nonlinear LS."""
    L0 = cfg.get("L_inf", 0.01)
    def model(n, L, A1, l1, A2, l2):
        return L + A1 * np.exp(-l1 * n) + A2 * np.exp(-l2 * n)
    return _parametric_fit(seq, indices, future_x, cfg, model,
                           p0=[L0, 0.4, 0.02, 0.2, 0.15],
                           bounds=([0, 0, 1e-4, 0, 1e-4], [1, 2, 5, 2, 5]))


def accel_rational_fit(seq, indices, future_x: float, cfg: dict) -> float:
    """Fit L + A/(1 + B*n) by nonlinear LS."""
    L0 = cfg.get("L_inf", 0.01)
    def model(n, L, A, B): return L + A / (1.0 + B * n)
    return _parametric_fit(seq, indices, future_x, cfg, model,
                           p0=[L0, 0.5, 0.05],
                           bounds=([0, 0, 1e-5], [1, 2, 10]))


def accel_log_fit(seq, indices, future_x: float, cfg: dict) -> float:
    """Fit L + A/log(n + e) by nonlinear LS."""
    L0 = cfg.get("L_inf", 0.01)
    def model(n, L, A): return L + A / np.log(n + np.e)
    return _parametric_fit(seq, indices, future_x, cfg, model,
                           p0=[L0, 0.5],
                           bounds=([0, 0], [1, 5]))


# =============================================================================
# FAMILY 4 — Shanks / Aitken Delta^2  [SH55]
# =============================================================================

def _aitken_step(s: np.ndarray, tol: float = 1e-14) -> np.ndarray:
    """
    One pass of Aitken's Delta^2 process over array s.
    Returns array of length len(s) - 2.

    Formula (Aitken 1926 / Shanks 1955):
        T_n = (s_{n+2}*s_n - s_{n+1}^2) / (s_{n+2} - 2*s_{n+1} + s_n)
    """
    num = s[2:] * s[:-2] - s[1:-1] ** 2
    den = s[2:] - 2.0 * s[1:-1] + s[:-2]
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(np.abs(den) < tol, np.nan, num / den)
    return out


def _accel_shanks(seq, order: int, tol: float = 1e-14) -> float:
    """Apply Aitken Delta^2 transform `order` times; return last finite value."""
    s = np.asarray(seq, dtype=float)
    for _ in range(order):
        s = _aitken_step(s, tol)
        finite = s[np.isfinite(s)]
        if len(finite) == 0:
            return np.nan
        s = finite
    return float(s[-1]) if len(s) > 0 else np.nan


def accel_shanks_1(seq, indices, future_x: float, cfg: dict) -> float:
    """Shanks order 1 (single Aitken Delta^2 pass)  [SH55]."""
    return _safe(_accel_shanks(seq, 1, cfg.get("denom_tol", 1e-14)), cfg)


def accel_shanks_2(seq, indices, future_x: float, cfg: dict) -> float:
    """Shanks order 2 (double Aitken Delta^2 pass)  [SH55]."""
    return _safe(_accel_shanks(seq, 2, cfg.get("denom_tol", 1e-14)), cfg)


def accel_shanks_3(seq, indices, future_x: float, cfg: dict) -> float:
    """Shanks order 3  [SH55]."""
    return _safe(_accel_shanks(seq, 3, cfg.get("denom_tol", 1e-14)), cfg)


def accel_shanks_4(seq, indices, future_x: float, cfg: dict) -> float:
    """Shanks order 4  [SH55]. Tests degradation boundary."""
    return _safe(_accel_shanks(seq, 4, cfg.get("denom_tol", 1e-14)), cfg)


# =============================================================================
# FAMILY 5 — Wynn Epsilon Algorithm  [WY56]
# =============================================================================

def _wynn_epsilon(seq, order: int, tol: float = 1e-14) -> float:
    """
    Wynn epsilon algorithm returning epsilon_{2*order}^{(0)}.

    Recursion (Wynn 1956)  [WY56]:
        epsilon_{-1}^{(n)} = 0
        epsilon_0^{(n)}    = s_n
        epsilon_{k+1}^{(n)} = epsilon_{k-1}^{(n+1)}
                              + 1 / (epsilon_k^{(n+1)} - epsilon_k^{(n)})

    Requires at least 2*order + 1 sequence values.
    """
    needed = 2 * order + 1
    s = list(seq[-needed:]) if len(seq) >= needed else list(seq)
    if len(s) < needed:
        return np.nan

    T_pp: List[float] = [0.0] * len(s)
    T_p:  List[float] = [float(v) for v in s]

    for _ in range(2 * order):
        n_new = len(T_p) - 1
        if n_new <= 0:
            return np.nan
        T_c: List[float] = []
        for j in range(n_new):
            d = T_p[j + 1] - T_p[j]
            if abs(d) < tol:
                return np.nan
            T_c.append(T_pp[j + 1] + 1.0 / d)
        T_pp, T_p = T_p, T_c

    v = T_p[0] if T_p else np.nan
    return float(v) if np.isfinite(v) else np.nan


def accel_wynn_eps_1(seq, indices, future_x: float, cfg: dict) -> float:
    """Wynn epsilon order 1  [WY56]."""
    return _safe(_wynn_epsilon(seq, 1, cfg.get("denom_tol", 1e-14)), cfg)


def accel_wynn_eps_2(seq, indices, future_x: float, cfg: dict) -> float:
    """Wynn epsilon order 2  [WY56]."""
    return _safe(_wynn_epsilon(seq, 2, cfg.get("denom_tol", 1e-14)), cfg)


def accel_wynn_eps_3(seq, indices, future_x: float, cfg: dict) -> float:
    """Wynn epsilon order 3  [WY56]."""
    return _safe(_wynn_epsilon(seq, 3, cfg.get("denom_tol", 1e-14)), cfg)


# =============================================================================
# FAMILY 6 — Wynn Rho Algorithm  [WY56, BRZ91]
# =============================================================================

def _wynn_rho(seq, indices, order: int, tol: float = 1e-14) -> float:
    """
    Wynn rho algorithm returning rho_{2*order}^{(0)}.

    Designed for power-law and logarithmically convergent sequences.
    Uses x_n = float(index_n) as the auxiliary sequence, which avoids
    the degeneracy that occurs with 1/(n+1) when the error is exactly
    proportional to the auxiliary.

    Recursion (Brezinski & Redivo-Zaglia 1991)  [BRZ91]:
        rho_{-1}^{(n)} = 0
        rho_0^{(n)}    = s_n
        rho_{k+1}^{(n)} = rho_{k-1}^{(n+1)}
                         + (x_{n+k+1} - x_n) / (rho_k^{(n+1)} - rho_k^{(n)})

    where x_n = float(index_n) and indices in the superscript refer to
    positions in the original (fixed) auxiliary array x_full.

    Requires 2*order + 1 terms.
    """
    needed = 2 * order + 1
    if len(seq) < needed:
        return np.nan

    s_use   = list(seq[-needed:])
    idx_use = list(indices[-needed:])
    # Use the actual sequence indices as auxiliary values.
    # This choice avoids degeneracy for power-law sequences and gives
    # (x_{n+k+1} - x_n) = (k+1) * step where step is the index spacing.
    x_full  = [float(i) for i in idx_use]   # fixed reference, never shrunk
    N       = len(s_use)

    T_pp: List[float] = [0.0] * N
    T_p:  List[float] = [float(v) for v in s_use]

    for k in range(2 * order):
        n_new = len(T_p) - 1
        if n_new <= 0:
            return np.nan
        T_c: List[float] = []
        for j in range(n_new):
            d = T_p[j + 1] - T_p[j]
            if abs(d) < tol:
                return np.nan
            # x_{n+k+1} - x_n always references the original fixed array
            dx = x_full[j + k + 1] - x_full[j]
            T_c.append(T_pp[j + 1] + dx / d)
        T_pp = T_p
        T_p  = T_c

    v = T_p[0] if T_p else np.nan
    return float(v) if np.isfinite(v) else np.nan


def accel_wynn_rho_1(seq, indices, future_x: float, cfg: dict) -> float:
    """Wynn rho order 1  [WY56, BRZ91]."""
    return _safe(_wynn_rho(seq, indices, 1, cfg.get("denom_tol", 1e-14)), cfg)


def accel_wynn_rho_2(seq, indices, future_x: float, cfg: dict) -> float:
    """Wynn rho order 2  [WY56, BRZ91]."""
    return _safe(_wynn_rho(seq, indices, 2, cfg.get("denom_tol", 1e-14)), cfg)


def accel_wynn_rho_3(seq, indices, future_x: float, cfg: dict) -> float:
    """Wynn rho order 3  [WY56, BRZ91]."""
    return _safe(_wynn_rho(seq, indices, 3, cfg.get("denom_tol", 1e-14)), cfg)


# =============================================================================
# FAMILY 7 — Pade Approximants  [BRZ91]
# =============================================================================

def _pade_fit(seq, indices, future_x: float, p: int, q: int,
              cfg: dict) -> float:
    """
    Fit rational function R(x) = P_p(x) / Q_q(x) with Q(0) = 1 normalised.

    Uses an over-determined linear system with Tikhonov regularisation.
    X is normalised to [0,1] within the window to improve conditioning.

    System (linear in coefficients a_0..a_p, b_1..b_q):
        a_0 + a_1*x_i + ... - b_1*x_i*y_i - ... = y_i
    """
    y, x_raw = _to_arrays(seq, indices)
    n_pts    = len(y)
    n_params = (p + 1) + q

    if n_pts < n_params + 2:
        return np.nan

    xn, xf, _ = _normalise_x(x_raw, future_x)
    if xn is None:
        return np.nan

    ridge = cfg.get("ridge", 1e-8)
    A = np.zeros((n_pts, n_params))
    for j in range(p + 1):
        A[:, j] = xn ** j
    for j in range(1, q + 1):
        A[:, p + j] = -(xn ** j) * y

    A_aug = np.vstack([A, ridge * np.eye(n_params)])
    b_aug = np.concatenate([y, np.zeros(n_params)])

    try:
        coeffs, _, _, _ = np.linalg.lstsq(A_aug, b_aug, rcond=None)
    except Exception:
        return np.nan

    a_c = coeffs[:p + 1]
    b_c = np.concatenate([[1.0], coeffs[p + 1:]])

    Pv  = sum(a_c[j] * xf ** j for j in range(p + 1))
    Qv  = sum(b_c[j] * xf ** j for j in range(q + 1))
    tol = cfg.get("denom_tol", 1e-14)

    if abs(Qv) < tol:
        return np.nan
    return _safe(float(Pv / Qv), cfg)


def accel_pade_11(seq, indices, future_x: float, cfg: dict) -> float:
    """Pade [1,1]  [BRZ91]."""
    return _pade_fit(seq, indices, future_x, 1, 1, cfg)

def accel_pade_12(seq, indices, future_x: float, cfg: dict) -> float:
    """Pade [1,2] — denominator-heavy, historically safer  [BRZ91]."""
    return _pade_fit(seq, indices, future_x, 1, 2, cfg)

def accel_pade_13(seq, indices, future_x: float, cfg: dict) -> float:
    """Pade [1,3]  [BRZ91]."""
    return _pade_fit(seq, indices, future_x, 1, 3, cfg)

def accel_pade_21(seq, indices, future_x: float, cfg: dict) -> float:
    """Pade [2,1] — numerator-heavy, historically unstable  [BRZ91]."""
    return _pade_fit(seq, indices, future_x, 2, 1, cfg)

def accel_pade_22(seq, indices, future_x: float, cfg: dict) -> float:
    """Pade [2,2]  [BRZ91]."""
    return _pade_fit(seq, indices, future_x, 2, 2, cfg)

def accel_pade_23(seq, indices, future_x: float, cfg: dict) -> float:
    """Pade [2,3]  [BRZ91]."""
    return _pade_fit(seq, indices, future_x, 2, 3, cfg)

def accel_pade_31(seq, indices, future_x: float, cfg: dict) -> float:
    """Pade [3,1] — highly unstable in prior runs  [BRZ91]."""
    return _pade_fit(seq, indices, future_x, 3, 1, cfg)

def accel_pade_32(seq, indices, future_x: float, cfg: dict) -> float:
    """Pade [3,2]  [BRZ91]."""
    return _pade_fit(seq, indices, future_x, 3, 2, cfg)


# =============================================================================
# FAMILY 8 — Levin Transforms  [LE73, WE89]
# =============================================================================

def _levin_transform(seq, indices, order: int, variant: str,
                     tol: float = 1e-14) -> float:
    """
    Levin sequence transformation of order k.

    Formula (Levin 1973, Weniger 1989)  [LE73, WE89]:

        L_k = N_k / D_k  where

        N_k = sum_{j=0}^{k} (-1)^j C(k,j) beta(j,k) s_{n+j} / w_{n+j}
        D_k = sum_{j=0}^{k} (-1)^j C(k,j) beta(j,k)       / w_{n+j}

    with  beta(j,k) = (n0+j+1)^{k-1}  for k > 1, else 1.0

    Remainder estimates (variants):
        't': w_n = Delta(s_n) = s_{n+1} - s_n
        'u': w_n = (n+1) * Delta(s_n)
        'v': w_n = Delta(s_n)*Delta(s_{n+1}) / (Delta(s_{n+1}) - Delta(s_n))

    Point counts per variant:
        't', 'u': need = order + 2   (order+1 diffs from order+2 values)
        'v'     : need = order + 3   (one extra to form ratio-of-diffs weights)
    """
    tol_v = tol

    if variant in ("t", "u"):
        need = order + 2
    elif variant == "v":
        need = order + 3   # FIX: v-variant requires one additional point
    else:
        raise ValueError(f"Unknown Levin variant '{variant}'.")

    if len(seq) < need:
        return np.nan

    s   = np.asarray(seq[-need:], dtype=float)
    idx = np.asarray(indices[-need:], dtype=float)
    n0  = float(idx[0])
    ds  = np.diff(s)

    if variant == "t":
        w = ds                          # length order+1
        k = order

    elif variant == "u":
        n_idx = idx[:-1] + 1.0         # (n+1) for each difference position
        w = n_idx * ds                  # length order+1
        k = order

    elif variant == "v":
        # Ratio-of-diffs remainder estimate  [WE89]
        # ds has length order+2 (from need=order+3 values)
        # denom_v = Delta(ds) = ds[j+1] - ds[j], length order+1
        denom_v = ds[1:] - ds[:-1]     # length order+1
        near_z  = np.abs(denom_v) < tol_v
        if np.any(near_z):
            return np.nan
        w   = ds[:-1] * ds[1:] / denom_v   # length order+1
        s   = s[:-1]                        # align: drop last element
        idx = idx[:-1]
        n0  = float(idx[0])
        k   = order                         # now len(w) = order+1 >= k+1

    # Safety: ensure we have enough w values for order k
    if len(w) < k + 1:
        return np.nan

    N_num = 0.0
    D_num = 0.0
    for j in range(k + 1):
        sign  = (-1.0) ** j
        binom = float(sp_comb(k, j, exact=True))
        beta  = (n0 + j + 1.0) ** (k - 1) if k > 1 else 1.0
        wj    = w[j]
        if abs(wj) < tol:
            return np.nan
        sj    = s[j]
        coeff = sign * binom * beta / wj
        N_num += coeff * sj
        D_num += coeff

    if abs(D_num) < tol:
        return np.nan
    return float(N_num / D_num)


def accel_levin_t1(seq, indices, future_x: float, cfg: dict) -> float:
    """Levin t-transform order 1  [LE73]."""
    return _safe(_levin_transform(seq, indices, 1, "t",
                                  cfg.get("denom_tol", 1e-14)), cfg)


def accel_levin_t2(seq, indices, future_x: float, cfg: dict) -> float:
    """Levin t-transform order 2  [LE73]."""
    return _safe(_levin_transform(seq, indices, 2, "t",
                                  cfg.get("denom_tol", 1e-14)), cfg)


def accel_levin_u1(seq, indices, future_x: float, cfg: dict) -> float:
    """Levin u-transform order 1  [LE73]."""
    return _safe(_levin_transform(seq, indices, 1, "u",
                                  cfg.get("denom_tol", 1e-14)), cfg)


def accel_levin_u2(seq, indices, future_x: float, cfg: dict) -> float:
    """Levin u-transform order 2  [LE73]."""
    return _safe(_levin_transform(seq, indices, 2, "u",
                                  cfg.get("denom_tol", 1e-14)), cfg)


def accel_levin_v1(seq, indices, future_x: float, cfg: dict) -> float:
    """Levin v-transform order 1 (Weniger variant)  [WE89]."""
    return _safe(_levin_transform(seq, indices, 1, "v",
                                  cfg.get("denom_tol", 1e-14)), cfg)


def accel_levin_v2(seq, indices, future_x: float, cfg: dict) -> float:
    """Levin v-transform order 2 (Weniger variant)  [WE89]."""
    return _safe(_levin_transform(seq, indices, 2, "v",
                                  cfg.get("denom_tol", 1e-14)), cfg)


# =============================================================================
# FAMILY 9 — Weniger's d-Transform  [WE89]
# =============================================================================

def _weniger_delta(seq, indices, order: int, tol: float = 1e-14) -> float:
    """
    Weniger delta transformation.
    Uses the sequence values themselves as remainder estimates: w_n = s_n.
    Same formula as Levin with w_n = s_n.  [WE89]
    """
    need = order + 2
    if len(seq) < need:
        return np.nan

    s   = np.asarray(seq[-need:], dtype=float)
    idx = np.asarray(indices[-need:], dtype=float)
    n0  = float(idx[0])
    k   = order

    N_num = 0.0
    D_num = 0.0
    for j in range(k + 1):
        sign  = (-1.0) ** j
        binom = float(sp_comb(k, j, exact=True))
        beta  = (n0 + j + 1.0) ** (k - 1) if k > 1 else 1.0
        wj    = s[j]
        if abs(wj) < tol:
            return np.nan
        coeff = sign * binom * beta / wj
        N_num += coeff * s[j]
        D_num += coeff

    if abs(D_num) < tol:
        return np.nan
    return float(N_num / D_num)


def accel_weniger_d1(seq, indices, future_x: float, cfg: dict) -> float:
    """Weniger delta-transform order 1  [WE89]."""
    return _safe(_weniger_delta(seq, indices, 1,
                                cfg.get("denom_tol", 1e-14)), cfg)


def accel_weniger_d2(seq, indices, future_x: float, cfg: dict) -> float:
    """Weniger delta-transform order 2  [WE89]."""
    return _safe(_weniger_delta(seq, indices, 2,
                                cfg.get("denom_tol", 1e-14)), cfg)


# =============================================================================
# FAMILY 10 — Brezinski Theta Algorithm  [BR71, BRZ91]
# =============================================================================

def _brezinski_theta(seq, order: int, tol: float = 1e-14) -> float:
    """
    Brezinski theta algorithm for logarithmically convergent sequences.

    Recursion (Brezinski 1971)  [BR71]:
        theta_0^{(n)} = s_n
        theta_{2k+2}^{(n)} = theta_{2k}^{(n+1)}
                             - [theta_{2k}^{(n+1)} - theta_{2k}^{(n)}]^2
                             / [theta_{2k}^{(n+2)} - 2*theta_{2k}^{(n+1)}
                                + theta_{2k}^{(n)}]

    Even-level columns give the accelerated estimates.
    Requires 2*order + 1 values.
    """
    needed = 2 * order + 1
    if len(seq) < needed:
        return np.nan

    col = np.asarray(seq[-needed:], dtype=float)

    for _ in range(order):
        n_new = len(col) - 2
        if n_new <= 0:
            return np.nan
        new_col = np.zeros(n_new)
        for j in range(n_new):
            num = (col[j + 1] - col[j]) ** 2
            den = col[j + 2] - 2.0 * col[j + 1] + col[j]
            if abs(den) < tol:
                return np.nan
            new_col[j] = col[j + 1] - num / den
        col = new_col

    v = float(col[0]) if len(col) > 0 else np.nan
    return v if np.isfinite(v) else np.nan


def accel_brezinski_theta1(seq, indices, future_x: float, cfg: dict) -> float:
    """Brezinski theta order 1  [BR71]."""
    return _safe(_brezinski_theta(seq, 1, cfg.get("denom_tol", 1e-14)), cfg)


def accel_brezinski_theta2(seq, indices, future_x: float, cfg: dict) -> float:
    """Brezinski theta order 2  [BR71]."""
    return _safe(_brezinski_theta(seq, 2, cfg.get("denom_tol", 1e-14)), cfg)


# =============================================================================
# FAMILY 11 — Neville-Aitken Polynomial Extrapolation  [NE66]
# =============================================================================

def _neville(seq, indices, future_x: float, degree: int) -> float:
    """
    Neville iterated interpolation in the variable 1/n.

    Fits a polynomial of given degree to {(1/x_i, s_i)} and evaluates
    at 1/future_x ~ 0, extrapolating to the limit.  [NE66]
    """
    need = degree + 1
    if len(seq) < need:
        return np.nan

    y   = np.asarray(seq[-need:], dtype=float)
    x_r = np.asarray(indices[-need:], dtype=float)
    x_r = np.maximum(x_r, 1.0)
    xi  = 1.0 / x_r
    xf  = 1.0 / max(float(future_x), 1.0)

    Q = y.copy()
    for k in range(1, degree + 1):
        for j in range(degree, k - 1, -1):
            denom = xi[j] - xi[j - k]
            if abs(denom) < 1e-15:
                return np.nan
            Q[j] = ((xf - xi[j - k]) * Q[j]
                    - (xf - xi[j]) * Q[j - 1]) / denom

    return float(Q[degree]) if np.isfinite(Q[degree]) else np.nan


def accel_neville_2(seq, indices, future_x: float, cfg: dict) -> float:
    """Neville degree 2 (quadratic in 1/n, 3 points)  [NE66]."""
    return _safe(_neville(seq, indices, future_x, 2), cfg)


def accel_neville_3(seq, indices, future_x: float, cfg: dict) -> float:
    """Neville degree 3 (cubic in 1/n, 4 points)  [NE66]."""
    return _safe(_neville(seq, indices, future_x, 3), cfg)


def accel_neville_4(seq, indices, future_x: float, cfg: dict) -> float:
    """Neville degree 4 (quartic in 1/n, 5 points)  [NE66]."""
    return _safe(_neville(seq, indices, future_x, 4), cfg)


# =============================================================================
# FAMILY 12 — Anderson Acceleration  [AN65]
# =============================================================================

def _anderson(seq, indices, future_x: float, order: int,
              ridge: float = 1e-8) -> float:
    """
    Anderson acceleration of depth m  [AN65].

    For scalar 1D sequences, Anderson of depth m finds the optimal
    convex combination of the last m+1 iterates that minimises the
    weighted residual.

    Minimisation problem (1D, depth m):
        min_{gamma in R^m} | f_m + DeltaF @ gamma |^2
        where DeltaF[j] = F[j+1] - F[j]  (j = 0..m-1)
        and   F[j] = s[j+1] - s[j]  (residuals)

    This gives a (1 x m) system which is underdetermined for m > 1;
    the minimum-norm solution is found via ridge-regularised lstsq.

    Reconstruction:
        theta[0]   = -gamma[0]
        theta[j]   =  gamma[j-1] - gamma[j]   for 1 <= j <= m-1
        theta[m]   =  1 + gamma[m-1]

    The Anderson estimate is  sum_j theta[j] * s[-(m+1)+j].

    Note: depth 1 is algebraically equivalent to Aitken's Delta^2.
    """
    m    = order
    need = m + 2
    if len(seq) < need or m < 1:
        return np.nan

    s = np.asarray(seq[-need:], dtype=float)   # m+2 values
    F = np.diff(s)                              # m+1 residuals

    # DeltaF: differences of consecutive residuals, length m
    delta_F = np.diff(F)                        # shape (m,)

    if len(delta_F) == 0:
        return np.nan

    f_last = F[-1]

    # Build system: minimize | delta_F @ gamma + f_last |^2
    # A has shape (1, m): single row = delta_F
    # b has shape (1,)  : single entry = -f_last
    A_mat = delta_F.reshape(1, -1)              # (1, m)
    b_vec = np.array([-f_last])                 # (1,)

    # Ridge-augmented system: (1+m, m)
    sq_ridge = np.sqrt(ridge)
    A_aug = np.vstack([A_mat, sq_ridge * np.eye(m)])
    b_aug = np.concatenate([b_vec, np.zeros(m)])

    try:
        gamma, _, _, _ = np.linalg.lstsq(A_aug, b_aug, rcond=None)
    except Exception:
        return np.nan

    # Reconstruct convex weights theta (sum = 1)
    theta = np.zeros(m + 1)
    theta[0] = -gamma[0]
    for j in range(1, m):
        theta[j] = gamma[j - 1] - gamma[j]
    theta[m] = 1.0 + gamma[m - 1]

    limit_est = float(np.dot(theta, s[-(m + 1):]))
    return limit_est


def accel_anderson_1(seq, indices, future_x: float, cfg: dict) -> float:
    """Anderson acceleration depth 1  [AN65]."""
    return _safe(_anderson(seq, indices, future_x, 1,
                           cfg.get("ridge", 1e-8)), cfg)


def accel_anderson_2(seq, indices, future_x: float, cfg: dict) -> float:
    """Anderson acceleration depth 2  [AN65]."""
    return _safe(_anderson(seq, indices, future_x, 2,
                           cfg.get("ridge", 1e-8)), cfg)


def accel_anderson_3(seq, indices, future_x: float, cfg: dict) -> float:
    """Anderson acceleration depth 3  [AN65]."""
    return _safe(_anderson(seq, indices, future_x, 3,
                           cfg.get("ridge", 1e-8)), cfg)


# =============================================================================
# FAMILY 13 — Ensemble and Diagnostic Methods
# =============================================================================

def _shift_iqr(method_fn, seq, indices, future_x: float,
               shifts, cfg: dict) -> float:
    """Window-shift IQR for a given base method."""
    ests = []
    for sh in shifts:
        sh_seq = seq[max(0, sh):]
        sh_idx = indices[max(0, sh):]
        if len(sh_seq) < 3:
            continue
        v = method_fn(sh_seq, sh_idx, future_x, cfg)
        if np.isfinite(v) and cfg["min_valid"] <= v <= cfg["max_valid"]:
            ests.append(v)
    if len(ests) < 2:
        return float("inf")
    return float(np.subtract(*np.percentile(ests, [75, 25])))


def accel_median_ensemble(seq, indices, future_x: float, cfg: dict) -> float:
    """
    Median of valid estimates from a fixed candidate pool.

    Pool: richardson_1, shanks_1/2/3, wynn_eps_1/2, pade_12, pade_13.
    Returns median if >= 3 valid estimates exist, else np.nan.
    """
    pool_fns = [
        accel_richardson_1, accel_shanks_1, accel_shanks_2, accel_shanks_3,
        accel_wynn_eps_1,   accel_wynn_eps_2,
        accel_pade_12,      accel_pade_13,
    ]
    valids = []
    for fn in pool_fns:
        v = fn(seq, indices, future_x, cfg)
        if np.isfinite(v) and cfg["min_valid"] <= v <= cfg["max_valid"]:
            valids.append(v)
    if len(valids) < 3:
        return np.nan
    return float(np.median(valids))


def accel_stability_weighted(seq, indices, future_x: float,
                             cfg: dict) -> float:
    """
    Stability-weighted average over the same pool as the median ensemble.
    Weight = 1 / (shift_IQR + epsilon); consistent methods get higher weight.
    """
    pool_fns = [
        accel_richardson_1, accel_shanks_1, accel_shanks_2, accel_shanks_3,
        accel_wynn_eps_1,   accel_wynn_eps_2,
        accel_pade_12,      accel_pade_13,
    ]
    shifts = cfg.get("win_shifts", [-2, -1, 0, 1, 2])
    eps    = 1e-6

    ests:    List[float] = []
    weights: List[float] = []
    for fn in pool_fns:
        v = fn(seq, indices, future_x, cfg)
        if not (np.isfinite(v) and cfg["min_valid"] <= v <= cfg["max_valid"]):
            continue
        iqr = _shift_iqr(fn, seq, indices, future_x, shifts, cfg)
        ests.append(v)
        weights.append(1.0 / (iqr + eps))

    if len(ests) < 2:
        return np.nan
    w = np.array(weights)
    w /= w.sum()
    return float(np.dot(w, ests))


def accel_best_shanks_wynn(seq, indices, future_x: float,
                           cfg: dict) -> float:
    """
    Runtime selector: return the Shanks or Wynn-epsilon estimate with
    the smallest window-shift IQR.  Falls back to current_value if all
    are invalid.
    """
    candidates = {
        "sh1": accel_shanks_1, "sh2": accel_shanks_2, "sh3": accel_shanks_3,
        "wy1": accel_wynn_eps_1, "wy2": accel_wynn_eps_2,
        "wy3": accel_wynn_eps_3,
    }
    shifts   = cfg.get("win_shifts", [-2, -1, 0, 1, 2])
    best_v   = np.nan
    best_iqr = float("inf")

    for fn in candidates.values():
        v = fn(seq, indices, future_x, cfg)
        if not (np.isfinite(v) and cfg["min_valid"] <= v <= cfg["max_valid"]):
            continue
        iqr = _shift_iqr(fn, seq, indices, future_x, shifts, cfg)
        if iqr < best_iqr:
            best_iqr = iqr
            best_v   = v

    return best_v if np.isfinite(best_v) else float(seq[-1])


# =============================================================================
# METHOD REGISTRY
# =============================================================================

METHODS = {
    # Family 1 — Baselines
    "current_value":      accel_current_value,
    "linear":             accel_linear,
    "log_linear":         accel_log_linear,
    "geom_avg_diff":      accel_geom_avg_diff,
    # Family 2 — Richardson / Power-Law
    "richardson_1":       accel_richardson_1,
    "richardson_2":       accel_richardson_2,
    "richardson_3":       accel_richardson_3,
    "richardson_a05":     accel_richardson_half,
    "richardson_a10":     accel_richardson_one,
    "richardson_a20":     accel_richardson_two,
    # Family 3 — Parametric Fits
    "single_exp_fit":     accel_single_exp_fit,
    "double_exp_fit":     accel_double_exp_fit,
    "rational_fit":       accel_rational_fit,
    "log_fit":            accel_log_fit,
    # Family 4 — Shanks
    "shanks_1":           accel_shanks_1,
    "shanks_2":           accel_shanks_2,
    "shanks_3":           accel_shanks_3,
    "shanks_4":           accel_shanks_4,
    # Family 5 — Wynn Epsilon
    "wynn_eps_1":         accel_wynn_eps_1,
    "wynn_eps_2":         accel_wynn_eps_2,
    "wynn_eps_3":         accel_wynn_eps_3,
    # Family 6 — Wynn Rho
    "wynn_rho_1":         accel_wynn_rho_1,
    "wynn_rho_2":         accel_wynn_rho_2,
    "wynn_rho_3":         accel_wynn_rho_3,
    # Family 7 — Pade
    "pade_11":            accel_pade_11,
    "pade_12":            accel_pade_12,
    "pade_13":            accel_pade_13,
    "pade_21":            accel_pade_21,
    "pade_22":            accel_pade_22,
    "pade_23":            accel_pade_23,
    "pade_31":            accel_pade_31,
    "pade_32":            accel_pade_32,
    # Family 8 — Levin
    "levin_t1":           accel_levin_t1,
    "levin_t2":           accel_levin_t2,
    "levin_u1":           accel_levin_u1,
    "levin_u2":           accel_levin_u2,
    "levin_v1":           accel_levin_v1,
    "levin_v2":           accel_levin_v2,
    # Family 9 — Weniger
    "weniger_d1":         accel_weniger_d1,
    "weniger_d2":         accel_weniger_d2,
    # Family 10 — Brezinski Theta
    "brezinski_theta1":   accel_brezinski_theta1,
    "brezinski_theta2":   accel_brezinski_theta2,
    # Family 11 — Neville
    "neville_2":          accel_neville_2,
    "neville_3":          accel_neville_3,
    "neville_4":          accel_neville_4,
    # Family 12 — Anderson
    "anderson_1":         accel_anderson_1,
    "anderson_2":         accel_anderson_2,
    "anderson_3":         accel_anderson_3,
    # Family 13 — Ensemble
    "median_ensemble":    accel_median_ensemble,
    "stability_weighted": accel_stability_weighted,
    "best_shanks_wynn":   accel_best_shanks_wynn,
}

METHOD_NAMES = list(METHODS.keys())
