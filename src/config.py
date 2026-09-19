"""
config.py
=========
Central configuration for the synthetic acceleration study.
All experiments import from here so that changing one value propagates
everywhere.  Nothing in this file has side effects.

Redesign v2 (2026-09)
---------------------
The rejected design had (a) one true asymptote L* = 0.01 shared by all 18
regimes, (b) that exact value handed to every method as the "assumed"
asymptote, and (c) headline horizons where the target had already converged
to L*.  The three knobs below remove each ingredient:

    ASYMPTOTE_MODE          hidden, heterogeneous L_true per (regime, seed)
    ASSUMED_L_MODE          what methods are told about the asymptote (L_hat)
    HORIZON_GAP_FRACTIONS   horizons defined by remaining gap, not by index

Reference
---------
Maleki, K. (2026). Finite-Horizon Learning-Curve Prediction for Gradient
Boosting. Machine Learning (in revision).
"""

# ── True asymptote (hidden from every method) ──────────────────────────────────
# "hetero": each (regime, seed) has its own L_true, drawn log-uniformly from
#           L_TRUE_RANGE by src.generators.true_asymptote().  Deterministic,
#           reproducible, never shared across seeds.
# "legacy": the rejected design (L_true = LEGACY_L_INF for every regime and
#           seed).  Exists only so tests/test_redesign.py can lock in the
#           referee's finding; no experiment should run in this mode.
ASYMPTOTE_MODE: str = "hetero"
ASYMPTOTE_MODES = ("hetero", "legacy")
L_TRUE_RANGE = (0.005, 0.5)          # log-uniform support of L_true
LEGACY_L_INF: float = 0.01           # the old shared asymptote; legacy mode only

# ── Assumed asymptote supplied to methods (L_hat) ──────────────────────────────
# Every accelerator, feature extractor and cascade input that needs an
# asymptote receives L_hat = assumed_asymptote(L_true, window, mode), never
# L_true itself.  See src/asymptote.py.
#   zero   : L_hat = 0                          deployment-honest default
#   half   : L_hat = 0.5 * L_true               under-estimate
#   oracle : L_hat = L_true                     reference only, labelled oracle
#   double : L_hat = 2.0 * L_true               over-estimate
#   winmin : L_hat = max(0, 0.9 * min(window))  data-driven, no oracle access
ASSUMED_L_MODE: str = "zero"
ASSUMED_L_MODES = ("zero", "half", "oracle", "double", "winmin")

# ── Sequence generation ────────────────────────────────────────────────────────
N_TERMS: int   = 300         # total length of each generated sequence
N_SEEDS: int   = 30          # independent replicates per (regime, noise) pair
NOISE_LEVELS   = [0.0, 0.001, 0.005, 0.010, 0.020]   # Phase 2 uses the full set;
                                                        # Phase 0/1 use first three

# ── Observation window ─────────────────────────────────────────────────────────
OBS_IDX:    int = 90         # last observed index (0-based)
WINDOW_LEN: int = 60         # number of terms fed to each accelerator

# ── Prediction horizon ─────────────────────────────────────────────────────────
# Redesign v2: the main benchmark uses gap-stratified horizons.  For a target
# remaining-gap fraction g, n_f(regime, g) is the first n > OBS_IDX at which
# the noiseless gap has shrunk to g * gap(OBS_IDX), capped at HORIZON_N_CAP.
# When the cap binds the achieved fraction is recorded (src/horizons.py).
HORIZON_GAP_FRACTIONS = [0.5, 0.1, 0.02]
HORIZON_N_CAP: int    = 50_000
HEADLINE_G: float     = 0.1          # stratum used for console summaries / figures

# Fixed-index horizons of the rejected design.  Still consumed by the legacy
# phase runners (2, 4, 5a, 5b) and by the flaw regression test.  Do not use
# them for headline claims: at n = 5000 most targets have converged.
FUTURE_IDX_DEFAULT: int = 5000
FUTURE_IDX_NEAR:    int = 150
FUTURE_IDX_MID:     int = 1000

# ── Numerical safety ───────────────────────────────────────────────────────────
RIDGE:     float = 1e-8      # Tikhonov regularisation for Padé least-squares
MIN_VALID: float = -0.5      # estimates below this are invalid
MAX_VALID: float = 500.0     # estimates above this are invalid
DENOM_TOL: float = 1e-14     # denominator near-zero threshold

# ── Stability diagnostics ──────────────────────────────────────────────────────
CAT_MULT:       float = 5.0  # catastrophic = error > CAT_MULT × baseline error
WIN_SHIFTS      = [-2, -1, 0, 1, 2]   # window-start offsets for shift IQR
PERTURB_TRIALS: int   = 5
PERTURB_SCALE:  float = 0.02           # relative perturbation magnitude

# ── Stability score weights ────────────────────────────────────────────────────
# score = valid_rate - W_CAT * cat_rate + W_BEATS * beats_rate
W_CAT:   float = 2.0
W_BEATS: float = 0.4

# ── Phase 1 derived classification ─────────────────────────────────────────────
# Methods that Phase 1 classifies as dangerous: their pooled stability score is
# negative at one or more horizons, meaning they fail worse than making no
# prediction at all.  Phases 5a and 5b both consume this set.
#
# IMPORTANT: this set is *derived* from Phase 1 output, not independent of it.
# After any change that alters Phase 1 results, re-derive it from
# results/phase1/phase1_global.csv (any method with stability < 0 at any
# horizon) and update it here.  scripts/check_dangerous.py does this check.
# It has NOT yet been re-derived under the redesign (pending the Prompt 2 run).
DANGEROUS_METHODS = frozenset({
    "neville_2", "neville_3", "neville_4",
    "pade_21", "pade_31", "pade_32",
    "linear", "geom_avg_diff",
})

# ── Output ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR: str = "results"   # relative to the working directory
FIG_DPI:    int = 150

# ── Unit-test tolerances ───────────────────────────────────────────────────────
# An accelerator passes its unit test if it reduces |error| by at least this
# factor relative to the raw sequence value at OBS_IDX.
UNIT_TEST_MIN_IMPROVEMENT: float = 2.0    # must at least halve the error
UNIT_TEST_GEOM_IMPROVEMENT: float = 10.0  # geometric series: demand strong improvement
