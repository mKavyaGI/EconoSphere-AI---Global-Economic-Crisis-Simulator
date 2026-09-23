"""
EconoSphere AI — Phase 19
models.py : Pipeline definitions for all architectural candidates.

GOVERNANCE:
  - All pipelines are EXPERIMENTAL_ONLY.
  - None of these replace the Phase 11 production artifact.
  - Model filenames must end with EXPERIMENTAL_ONLY.
  - Production artifact is never loaded or modified here.
"""
from __future__ import annotations

from pathlib import Path

from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, HuberRegressor, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ── Phase 11 locked hyperparameters (reconstruction reference only) ────────────
PHASE11_PARAMS = {
    "max_depth":         5,
    "l2_regularization": 5.0,
    "learning_rate":     0.05,
    "max_iter":          300,
    "random_state":      42,
}

# ── Candidate hyperparameter grids ────────────────────────────────────────────
# Selection within these grids must happen using validation data only.
# The test/forecast observations must never influence hyperparameter choice.

RIDGE_ALPHAS        = [0.1, 1.0, 10.0, 100.0]
ELASTICNET_PARAMS   = [
    {"alpha": 0.1,  "l1_ratio": 0.2},
    {"alpha": 0.1,  "l1_ratio": 0.5},
    {"alpha": 0.1,  "l1_ratio": 0.8},
    {"alpha": 1.0,  "l1_ratio": 0.2},
    {"alpha": 1.0,  "l1_ratio": 0.5},
    {"alpha": 1.0,  "l1_ratio": 0.8},
    {"alpha": 10.0, "l1_ratio": 0.5},
]
HUBER_EPSILONS     = [1.35, 1.5, 2.0]   # epsilon=1.35 ≈ Huber standard
RF_PARAMS          = {                    # fixed conservative RF config
    "n_estimators":  300,
    "max_depth":     8,
    "min_samples_leaf": 5,
    "random_state":  42,
    "n_jobs":        -1,
}


# ── Builder functions ──────────────────────────────────────────────────────────

def build_m0() -> Pipeline:
    """
    M0 — Phase 11 Reconstruction.
    Exact Phase 11 hyperparameters re-implemented for chronological backtesting.
    Label: PHASE11_RECONSTRUCTION_ONLY — NOT the production artifact.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model",   HistGradientBoostingRegressor(**PHASE11_PARAMS)),
    ])


def build_ridge(alpha: float = 10.0) -> Pipeline:
    """A1 — Ridge regression with StandardScaler."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   Ridge(alpha=alpha)),
    ])


def build_elasticnet(alpha: float = 1.0, l1_ratio: float = 0.5) -> Pipeline:
    """A2 — ElasticNet with StandardScaler."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000)),
    ])


def build_huber(epsilon: float = 1.35) -> Pipeline:
    """A3 — HuberRegressor with StandardScaler."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   HuberRegressor(epsilon=epsilon, max_iter=500)),
    ])


def build_random_forest() -> Pipeline:
    """A4 — RandomForest with fixed conservative parameters."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model",   RandomForestRegressor(**RF_PARAMS)),
    ])


# ── Candidate registry ─────────────────────────────────────────────────────────
# Used by the evaluation loop.  A5 (ensemble) and A6 (residual) are constructed
# dynamically inside evaluation.py after base model predictions are available.

BASE_CANDIDATES = {
    "M0_Phase11":    build_m0,
    "A1_Ridge":      lambda: build_ridge(alpha=10.0),      # default; tuned in eval
    "A2_ElasticNet": lambda: build_elasticnet(),           # default; tuned in eval
    "A3_Huber":      lambda: build_huber(),                # default; tuned in eval
    "A4_RandomForest": build_random_forest,
}

CANDIDATE_DISPLAY_NAMES = {
    "M0_Phase11":      "Phase 11 (M0)",
    "A1_Ridge":        "Ridge (A1)",
    "A2_ElasticNet":   "Elastic Net (A2)",
    "A3_Huber":        "Huber (A3)",
    "A4_RandomForest": "Random Forest (A4)",
    "A5a_Ensemble":    "Ensemble Equal (A5a)",
    "A5b_Ensemble":    "Ensemble Val-Weighted (A5b)",
    "A6_Residual":     "Residual Model (A6)",
}

# ── Artifact naming rule ───────────────────────────────────────────────────────

def experimental_model_filename(candidate_key: str) -> str:
    """
    Return the required filename for a saved experimental model.
    Must end with EXPERIMENTAL_ONLY.
    """
    safe = candidate_key.replace(" ", "_")
    return f"phase19_{safe}_EXPERIMENTAL_ONLY.joblib"


def validate_experimental_filename(filename: str) -> bool:
    """Return True if filename ends with EXPERIMENTAL_ONLY (with any extension)."""
    stem = Path(filename).stem
    return stem.endswith("EXPERIMENTAL_ONLY")
