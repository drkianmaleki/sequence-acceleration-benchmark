"""
datasets.py
===========
Real-data pipeline: OpenML dataset loading and XGBoost training.

Downloads tabular classification datasets from OpenML CC-18,
trains XGBoost for 500 rounds recording validation log-loss every round,
and returns the loss curves for use in the real-data experiment.
"""

import os
import numpy as np
import pandas as pd
import openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ── Dataset registry ───────────────────────────────────────────────────────────

DATASET_IDS = {
    "covertype":      1596,
    "higgs":          23512,
    "adult":          1590,
    "jannis":         41168,
    "miniboone":      41150,
    "bank_marketing": 1461,
}

N_ROUNDS      = 500
VAL_FRACTION  = 0.2
RANDOM_STATE  = 42


# ── Loaders ────────────────────────────────────────────────────────────────────

def load_datasets() -> dict:
    """
    Download all datasets from OpenML and return as {name: (X, y)} dict.
    X is a numpy array of floats; y is a numpy array of integers.
    """
    datasets = {}
    for name, did in DATASET_IDS.items():
        print(f"  Loading {name} (OpenML id={did}) ...")
        ds = openml.datasets.get_dataset(
            did,
            download_data=True,
            download_qualities=False,
            download_features_meta_data=False,
        )
        X, y, _, _ = ds.get_data(target=ds.default_target_attribute)

        # Convert to numpy, encode categoricals and labels
        if hasattr(X, 'to_numpy'):
            # Handle categorical columns
            for col in X.select_dtypes(include='category').columns:
                X[col] = X[col].cat.codes
            X = X.to_numpy(dtype=float)
        else:
            X = np.array(X, dtype=float)

        if hasattr(y, 'to_numpy'):
            y = y.to_numpy()
        le = LabelEncoder()
        y = le.fit_transform(y).astype(int)

        # Replace NaNs with column medians
        col_medians = np.nanmedian(X, axis=0)
        nan_mask = np.isnan(X)
        X[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])

        print(f"    {name}: X={X.shape}, classes={len(np.unique(y))}")
        datasets[name] = (X, y)

    return datasets


# ── XGBoost training ───────────────────────────────────────────────────────────

def train_xgboost(
    X: np.ndarray,
    y: np.ndarray,
    n_rounds:     int   = N_ROUNDS,
    val_fraction: float = VAL_FRACTION,
    seed:         int   = RANDOM_STATE,
) -> np.ndarray:
    """
    Train XGBoost for n_rounds, recording validation log-loss every round.

    Returns
    -------
    curve : np.ndarray, shape (n_rounds,)
        Validation log-loss at each boosting round.
    """
    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError("xgboost is required: pip install xgboost")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_fraction, random_state=seed, stratify=y
    )

    n_classes = len(np.unique(y))
    if n_classes == 2:
        objective = 'binary:logistic'
        eval_metric = 'logloss'
    else:
        objective = 'multi:softprob'
        eval_metric = 'mlogloss'

    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval   = xgb.DMatrix(X_val, label=y_val)

    evals_result = {}
    params = dict(
        objective        = objective,
        eval_metric      = eval_metric,
        num_class        = n_classes if n_classes > 2 else None,
        learning_rate    = 0.05,
        max_depth        = 6,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        seed             = seed,
        verbosity        = 0,
    )
    # Remove num_class for binary
    if n_classes == 2:
        params.pop('num_class')

    xgb.train(
        params,
        dtrain,
        num_boost_round = n_rounds,
        evals           = [(dval, 'val')],
        evals_result    = evals_result,
        verbose_eval    = False,
    )

    metric_key = 'logloss' if n_classes == 2 else 'mlogloss'
    curve = np.array(evals_result['val'][metric_key], dtype=float)
    return curve


# ── Full experiment pipeline ───────────────────────────────────────────────────

def run_real_data_experiment(
    out_dir:  str = 'results/real_data',
    n_rounds: int = N_ROUNDS,
    seed:     int = RANDOM_STATE,
) -> dict:
    """
    Full pipeline: load datasets → train XGBoost → save curves.

    Returns
    -------
    curves : dict  {dataset_name: np.ndarray of shape (n_rounds,)}
    """
    os.makedirs(out_dir, exist_ok=True)

    datasets = load_datasets()
    curves   = {}

    for name, (X, y) in datasets.items():
        print(f"  Training XGBoost on {name} ...")
        curve = train_xgboost(X, y, n_rounds=n_rounds, seed=seed)
        curves[name] = curve
        print(f"    Final val loss: {curve[-1]:.6f}  "
              f"Best: {curve.min():.6f} at round {curve.argmin()+1}")

    # Save raw curves
    df = pd.DataFrame(curves)
    df.index.name = 'round'
    df.to_csv(os.path.join(out_dir, 'real_data_curves.csv'))
    print(f"\n  Curves saved to {out_dir}/real_data_curves.csv")

    return curves