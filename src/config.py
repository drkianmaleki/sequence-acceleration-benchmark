"""
config.py
=========
Central configuration for the synthetic acceleration study (config v2).
All experiments import from here so that changing one value propagates
everywhere.  Nothing in this file has side effects.

Redesign v2 (2026-09)
---------------------
The rejected design had (a) one true asymptote L* = 0.01 shared by all 18
regimes, (b) that exact value handed to every method as the "assumed"
asymptote, and (c) headline horizons where the target had already converged
to L*.  The knobs below remove each ingredient:

    ASYMPTOTE_MODE          hidden, heterogeneous L_true per (regime, seed)
    ASSUMED_L_MODE          what methods are told about the asymptote (L_hat)
    HORIZON_GAP_FRACTIONS   horizons defined by remaining gap, not by index

Binding decisions from the Report-1 review (2026-09-19)
-------------------------------------------------------
    headline stratum       g = 0.1, all three strata reported
    capped cells           excluded from pooled cross-regime rankings and
                           reported in a separate "capped" block with achieved_g
    main-run mode          "zero" only
    skill                  best-of-four trivial reference per cell
    per-regime column      best_by_skill is primary; median error sorts
                           global tables
    Phase-0 harness        trivial comparators excluded
    dangerous set          re-derived from Phase 1 into an artifact that
                           phases 2-5 read; the hard-coded list is legacy only

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
# Redesign v2: every phase uses gap-stratified horizons.  For a target
# remaining-gap fraction g, n_f(regime, n_obs, g) is the first n > n_obs at
# which the noiseless gap has shrunk to g * gap(n_obs), capped at
# HORIZON_N_CAP.  When the cap binds the achieved fraction is recorded
# (src/horizons.py) and the cell is excluded from pooled rankings.
HORIZON_GAP_FRACTIONS = [0.5, 0.1, 0.02]
HORIZON_N_CAP: int    = 50_000
HEADLINE_G: float     = 0.1          # headline stratum (all strata reported)
PHASE5B_GAP_FRACTIONS = [0.5, 0.1]   # sensitivity sweeps run at these strata

EXCLUDE_CAPPED_FROM_POOLED: bool = True   # the capped-exclusion rule
RANK_METRIC: str = "med_error"            # sorts global tables (ascending)

# Report-2 review: a method is eligible for a rank in a pooled ranked table
# only when its pooled valid_rate is at least this floor.  Below-floor
# methods are still shown (unranked, in a separate block with valid_rate)
# so a rarely-valid method cannot take a rank slot on the cells where it
# happened to return a value.  Applied by src.pipeline.assign_ranks.
RANK_MIN_VALID: float = 0.9

# Fixed-index horizons of the rejected design.  Retained only for the flaw
# regression test; no phase evaluates at them any more.
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

# ── Dangerous-method set ───────────────────────────────────────────────────────
# Redesign v2: the dangerous set (pooled stability S < 0 over the core
# regimes, pooled over strata, capped cells excluded, oracle excluded) is
# RE-DERIVED from Phase 1 output by scripts/derive_dangerous.py and written
# to DANGEROUS_ARTIFACT.  Phases 2-5 read the artifact through
# src.dangerous.load_dangerous(); reproduce_all.py enforces the ordering
# Phase 1 -> derivation -> Phases 2-5.
DANGEROUS_ARTIFACT: str = "results/phase1/dangerous_methods.json"

# LEGACY ONLY.  The hard-coded set of the rejected design (shared L* = 0.01,
# fixed horizons).  Nothing consumes it; it is kept so the old value stays
# on record.
LEGACY_DANGEROUS_METHODS = frozenset({
    "neville_2", "neville_3", "neville_4",
    "pade_21", "pade_31", "pade_32",
    "linear", "geom_avg_diff",
})

# ── Output ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR: str = "results"   # relative to the working directory
FIG_DPI:    int = 150

# ── Unit-test tolerances (Phase 0 analytic harness) ────────────────────────────
# An accelerator passes its unit test if it reduces |error| by at least this
# factor relative to the raw sequence value at OBS_IDX.  The trivial
# comparators are excluded from the harness (they are not accelerators).
UNIT_TEST_MIN_IMPROVEMENT: float = 2.0    # must at least halve the error
UNIT_TEST_GEOM_IMPROVEMENT: float = 10.0  # geometric series: demand strong improvement

# ══════════════════════════════════════════════════════════════════════════════
# Pipeline grids (config v2).  Each phase runner reads PHASEx["full"] or
# PHASEx["quick"]; reproduce_all.py --plan counts evaluations from them.
# ══════════════════════════════════════════════════════════════════════════════

# --quick: 2 seeds, 2 regimes per group, 2 noise levels.  The core pair holds
# one fast regime and one that hits the horizon cap; the held-out pair holds
# one closed-form shape and the per-seed random_knots family, so the quick
# run exercises the capped and seed-dependent code paths end to end.
QUICK_N_SEEDS          = 2
QUICK_CORE_REGIMES     = ["single_exp", "log_slow"]
QUICK_HOLDOUT_REGIMES  = ["stretched_exp", "random_knots"]
QUICK_NOISE_LEVELS     = [0.0, 0.005]

PHASE1 = {
    "full": dict(n_seeds=N_SEEDS, noise_levels=NOISE_LEVELS[:3],
                 gap_fractions=HORIZON_GAP_FRACTIONS, obs_idx=OBS_IDX,
                 window_len=WINDOW_LEN, core_regimes=None, holdout_regimes=None),
    "quick": dict(n_seeds=QUICK_N_SEEDS, noise_levels=QUICK_NOISE_LEVELS,
                  gap_fractions=HORIZON_GAP_FRACTIONS, obs_idx=OBS_IDX,
                  window_len=WINDOW_LEN, core_regimes=QUICK_CORE_REGIMES,
                  holdout_regimes=QUICK_HOLDOUT_REGIMES),
}

PHASE2_OBS_DEPTHS = [20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140]
PHASE2 = {   # core 18 only (selector / cascade training data)
    "full": dict(obs_idx_list=PHASE2_OBS_DEPTHS,
                 noise_list=[0.0, 0.003, 0.005, 0.010, 0.020],
                 gap_fractions=HORIZON_GAP_FRACTIONS, n_seeds=20,
                 window_len=WINDOW_LEN, core_regimes=None),
    "quick": dict(obs_idx_list=PHASE2_OBS_DEPTHS, noise_list=QUICK_NOISE_LEVELS,
                  gap_fractions=HORIZON_GAP_FRACTIONS, n_seeds=QUICK_N_SEEDS,
                  window_len=WINDOW_LEN, core_regimes=QUICK_CORE_REGIMES),
}

PHASE4 = {
    "full": dict(obs_idx_list=[30, 60, 90, 120], noise_list=[0.0, 0.005, 0.020],
                 gap_fractions=HORIZON_GAP_FRACTIONS, n_seeds=20,
                 window_len=WINDOW_LEN, shifts=WIN_SHIFTS,
                 perturb_trials=PERTURB_TRIALS, perturb_scale=PERTURB_SCALE,
                 core_regimes=None, holdout_regimes=None),
    "quick": dict(obs_idx_list=[60, 90], noise_list=QUICK_NOISE_LEVELS,
                  gap_fractions=HORIZON_GAP_FRACTIONS, n_seeds=QUICK_N_SEEDS,
                  window_len=WINDOW_LEN, shifts=[-1, 0, 1],
                  perturb_trials=3, perturb_scale=PERTURB_SCALE,
                  core_regimes=QUICK_CORE_REGIMES,
                  holdout_regimes=QUICK_HOLDOUT_REGIMES),
}

PHASE5A = {
    "full": dict(obs_idx_list=[30, 60, 90, 120], noise_list=[0.0, 0.005, 0.020],
                 gap_fractions=HORIZON_GAP_FRACTIONS, n_seeds=20,
                 window_len=WINDOW_LEN, perturb_trials=PERTURB_TRIALS,
                 perturb_scale=PERTURB_SCALE,
                 core_regimes=None, holdout_regimes=None),
    "quick": dict(obs_idx_list=[60, 90], noise_list=QUICK_NOISE_LEVELS,
                  gap_fractions=HORIZON_GAP_FRACTIONS, n_seeds=QUICK_N_SEEDS,
                  window_len=WINDOW_LEN, perturb_trials=3,
                  perturb_scale=PERTURB_SCALE,
                  core_regimes=QUICK_CORE_REGIMES,
                  holdout_regimes=QUICK_HOLDOUT_REGIMES),
}

PHASE5B = {
    "full": dict(assumed_modes=list(ASSUMED_L_MODES),
                 window_lengths=[20, 40, 60, 80, 100],
                 catmult_values=[2.0, 5.0, 10.0],
                 obs_idx=OBS_IDX, window_len_default=WINDOW_LEN,
                 noise_list=[0.0, 0.005, 0.020],
                 gap_fractions=PHASE5B_GAP_FRACTIONS, n_seeds=20,
                 core_regimes=None, holdout_regimes=None),
    "quick": dict(assumed_modes=list(ASSUMED_L_MODES),
                  window_lengths=[20, 60, 100],
                  catmult_values=[2.0, 10.0],
                  obs_idx=OBS_IDX, window_len_default=WINDOW_LEN,
                  noise_list=QUICK_NOISE_LEVELS,
                  gap_fractions=PHASE5B_GAP_FRACTIONS, n_seeds=QUICK_N_SEEDS,
                  core_regimes=QUICK_CORE_REGIMES,
                  holdout_regimes=QUICK_HOLDOUT_REGIMES),
}

# Real data: re-evaluation of the RECORDED curves only (no retraining).
REAL_DATA = dict(
    curves_csv="results/real_data/real_data_curves.csv",
    depths=[30, 60, 90, 120, 150],
    targets=[300, 400, 500],          # target round; cells need depth < target
    window_len=WINDOW_LEN,
)
