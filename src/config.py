"""
config.py
=========
Central configuration for the synthetic acceleration study.
All experiments import from here so that changing one value propagates
everywhere.  Nothing in this file has side effects.

Reference
---------
Maleki, K. (2026). Finite-Horizon Learning-Curve Prediction for Gradient
Boosting: Regime Dependence, Failure Detection, and Conservative
Extrapolation Rules. Machine Learning (submitted).
"""

# ── Asymptotic target ──────────────────────────────────────────────────────────
L_INF: float = 0.01          # true limit all synthetic regimes converge to

# ── Sequence generation ────────────────────────────────────────────────────────
N_TERMS: int   = 300         # total length of each generated sequence
N_SEEDS: int   = 30          # independent replicates per (regime, noise) pair
NOISE_LEVELS   = [0.0, 0.001, 0.005, 0.010, 0.020]   # Phase 2 uses the full set;
                                                        # Phase 0/1 use first three

# ── Observation window ─────────────────────────────────────────────────────────
OBS_IDX:    int = 90         # last observed index (0-based)
WINDOW_LEN: int = 60         # number of terms fed to each accelerator

# ── Prediction horizon ─────────────────────────────────────────────────────────
# Multiple horizons are tested in Phase 2; Phase 0/1 use FUTURE_IDX_DEFAULT
FUTURE_IDX_DEFAULT: int = 5000
FUTURE_IDX_NEAR:    int = 150    # near-horizon test
FUTURE_IDX_MID:     int = 1000   # mid-horizon test

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

# ── Output ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR: str = "results"   # relative to the working directory
FIG_DPI:    int = 150

# ── Unit-test tolerances ───────────────────────────────────────────────────────
# An accelerator passes its unit test if it reduces |error| by at least this
# factor relative to the raw sequence value at OBS_IDX.
UNIT_TEST_MIN_IMPROVEMENT: float = 2.0    # must at least halve the error
UNIT_TEST_GEOM_IMPROVEMENT: float = 10.0  # geometric series: demand strong improvement
