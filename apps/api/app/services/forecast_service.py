"""
EconoSphere AI — Phase 11 Step 13 Forecast Service
====================================================
Production service for the locked T+1 GDP growth forecast pipeline.

Architecture:
  Raw Data → ML Pipeline → Validated Artifact → THIS SERVICE → FastAPI → Frontend

The service reads the pre-generated forecast artifact produced by Step 12
(train_t1_adaptive_uncertainty.py). It does NOT retrain or re-invoke the model.

Official locked configuration:
  - Model: HistGradientBoostingRegressor (Step 10)
  - Test RMSE: 3.9113
  - Uncertainty: Global Q90 residual-calibrated interval (Step 11/12)
  - Q90: 11.4507 percentage points
  - Artifact: data/processed/t1_step12_2026_forecasts.csv

Path resolution:
  This file is at: apps/api/app/services/forecast_service.py
  parents[4] = project root (D:/Development/EconoSphere AI)
  CSV is at:   data/processed/t1_step12_2026_forecasts.csv

Caching:
  Forecasts are loaded once on first access and stored in _CACHE (module-level dict).
  No Redis required — this is a static artifact that changes only when the ML pipeline
  is re-run. Use reload_cache() to force a fresh load.

Duplicate handling:
  The artifact may contain duplicate country rows (artifact of the Step 12 run).
  Duplicates are detected on load. If duplicates have CONFLICTING values, the service
  raises ValueError and refuses to serve stale/ambiguous data. Non-conflicting duplicates
  are deduplicated silently with a logged warning.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Official locked constants (must match Step 10/11/12 experimental results)
# ---------------------------------------------------------------------------
_OFFICIAL_MODEL_VERSION = "phase11_step10_locked"
_OFFICIAL_ALGORITHM = "HistGradientBoostingRegressor"
_OFFICIAL_TEST_RMSE = 3.9113
_OFFICIAL_Q90 = 11.4507
_OFFICIAL_COVERAGE_TARGET = 0.90
_OFFICIAL_TRAIN_PERIOD = "2000-2018"
_OFFICIAL_VALIDATION_PERIOD = "2019-2022"
_OFFICIAL_TEST_PERIOD = "2023-2024"
_OFFICIAL_FEATURE_YEAR = 2025
_OFFICIAL_FORECAST_YEAR = 2026
_OFFICIAL_UNCERTAINTY_METHOD = "global_q90_residual"
_OFFICIAL_UNCERTAINTY_DESCRIPTION = (
    "Empirical 90% prediction interval calibrated from absolute out-of-sample "
    "validation residuals (2019-2022). Step 12 adaptive experiments confirmed "
    "the global Q90 as the optimal method."
)
_OFFICIAL_STEP12_DECISION = "KEEP STEP 11 GLOBAL Q90 CONTROL"
_OFFICIAL_UNCERTAINTY_METHOD_CSV = "step11_global_q90_control"

# Required columns in the forecast CSV
_REQUIRED_COLUMNS = {
    "country_code",
    "country_name",
    "feature_year",
    "target_year",
    "predicted_gdp_growth",
    "lower_bound_90",
    "upper_bound_90",
    "interval_width",
    "uncertainty_method",
}

# ---------------------------------------------------------------------------
# Path resolution — verified at Step 13 design time
# parents[4] of this file = project root
# ---------------------------------------------------------------------------
def _get_project_root() -> Path:
    """Resolve the project root directory from this file's location."""
    return Path(__file__).resolve().parents[4]


def _get_artifact_path() -> Path:
    return _get_project_root() / "data" / "processed" / "t1_step12_2026_forecasts.csv"


# ---------------------------------------------------------------------------
# Module-level cache
# ---------------------------------------------------------------------------
_CACHE: Optional[Dict[str, dict]] = None  # country_code → forecast dict


def _load_and_validate() -> Dict[str, dict]:
    """
    Load, validate, deduplicate, and cache the forecast artifact.

    Returns a dict keyed by uppercase country_code.

    Raises:
        FileNotFoundError: if the CSV artifact does not exist.
        ValueError: if schema validation or numeric integrity checks fail,
                    or if duplicate countries have conflicting forecast values.
    """
    path = _get_artifact_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Forecast artifact not found. Expected at: data/processed/t1_step12_2026_forecasts.csv. "
            f"Run apps/api/ml/train_t1_adaptive_uncertainty.py to regenerate."
        )

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Forecast artifact could not be parsed as CSV: {exc}") from exc

    # --- Column presence ---
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Forecast artifact is missing required columns: {sorted(missing)}")

    # --- target_year and feature_year integrity ---
    if not (df["target_year"] == _OFFICIAL_FORECAST_YEAR).all():
        bad = df[df["target_year"] != _OFFICIAL_FORECAST_YEAR]["target_year"].unique().tolist()
        raise ValueError(f"target_year must be {_OFFICIAL_FORECAST_YEAR} for all rows. Found: {bad}")

    if not (df["feature_year"] == _OFFICIAL_FEATURE_YEAR).all():
        bad = df[df["feature_year"] != _OFFICIAL_FEATURE_YEAR]["feature_year"].unique().tolist()
        raise ValueError(f"feature_year must be {_OFFICIAL_FEATURE_YEAR} for all rows. Found: {bad}")

    # --- uncertainty_method integrity ---
    if not (df["uncertainty_method"] == _OFFICIAL_UNCERTAINTY_METHOD_CSV).all():
        bad = df[df["uncertainty_method"] != _OFFICIAL_UNCERTAINTY_METHOD_CSV]["uncertainty_method"].unique().tolist()
        raise ValueError(
            f"uncertainty_method must be '{_OFFICIAL_UNCERTAINTY_METHOD_CSV}'. Found: {bad}"
        )

    # --- Numeric finiteness ---
    numeric_cols = ["predicted_gdp_growth", "lower_bound_90", "upper_bound_90", "interval_width"]
    for col in numeric_cols:
        if df[col].isna().any() or not df[col].apply(lambda v: math.isfinite(v)).all():
            raise ValueError(f"Column '{col}' contains non-finite values (NaN or Inf).")

    # --- lower <= predicted <= upper ---
    bad_lower = df[df["lower_bound_90"] > df["predicted_gdp_growth"]]
    if not bad_lower.empty:
        raise ValueError(
            f"lower_bound_90 > predicted_gdp_growth for countries: {bad_lower['country_code'].tolist()}"
        )
    bad_upper = df[df["predicted_gdp_growth"] > df["upper_bound_90"]]
    if not bad_upper.empty:
        raise ValueError(
            f"predicted_gdp_growth > upper_bound_90 for countries: {bad_upper['country_code'].tolist()}"
        )

    # --- Duplicate handling ---
    dup_mask = df.duplicated(subset=["country_code"], keep=False)
    if dup_mask.any():
        dup_codes = df[dup_mask]["country_code"].unique().tolist()
        logger.warning(
            "Forecast artifact contains duplicate rows for: %s. Checking for conflicts...",
            dup_codes,
        )
        for code in dup_codes:
            rows = df[df["country_code"] == code]
            for col in numeric_cols:
                values = rows[col].tolist()
                if len(set(values)) > 1:
                    raise ValueError(
                        f"Duplicate rows for country '{code}' have CONFLICTING values for '{col}': {values}. "
                        f"Re-run the ML pipeline to generate a clean artifact."
                    )
        # All duplicates are non-conflicting — drop extras
        df = df.drop_duplicates(subset=["country_code"], keep="first")
        logger.info("Non-conflicting duplicates removed. %d unique countries loaded.", len(df))

    # --- Build result dict ---
    result: Dict[str, dict] = {}
    for _, row in df.iterrows():
        code = str(row["country_code"]).upper().strip()
        result[code] = {
            "country_code": code,
            "country_name": str(row["country_name"]),
            "feature_year": int(row["feature_year"]),
            "forecast_year": int(row["target_year"]),
            "predicted_gdp_growth": round(float(row["predicted_gdp_growth"]), 4),
            "lower_bound_90": round(float(row["lower_bound_90"]), 4),
            "upper_bound_90": round(float(row["upper_bound_90"]), 4),
            "interval_width": round(float(row["interval_width"]), 4),
            "forecast_status": "forecast",
        }

    logger.info("Forecast artifact loaded successfully. %d countries available: %s", len(result), sorted(result.keys()))
    return result


def _get_cache() -> Dict[str, dict]:
    """Return the cache, loading it on first access."""
    global _CACHE
    if _CACHE is None:
        _CACHE = _load_and_validate()
    return _CACHE


def reload_cache() -> Dict[str, dict]:
    """Force a fresh load of the forecast artifact, replacing the cache."""
    global _CACHE
    _CACHE = _load_and_validate()
    return _CACHE


# ---------------------------------------------------------------------------
# Public service interface
# ---------------------------------------------------------------------------

def _make_model_info() -> dict:
    return {
        "version": _OFFICIAL_MODEL_VERSION,
        "algorithm": _OFFICIAL_ALGORITHM,
        "test_rmse": _OFFICIAL_TEST_RMSE,
        "train_period": _OFFICIAL_TRAIN_PERIOD,
        "validation_period": _OFFICIAL_VALIDATION_PERIOD,
        "test_period": _OFFICIAL_TEST_PERIOD,
    }


def _make_uncertainty_info() -> dict:
    return {
        "method": _OFFICIAL_UNCERTAINTY_METHOD,
        "method_description": _OFFICIAL_UNCERTAINTY_DESCRIPTION,
        "q90": _OFFICIAL_Q90,
        "coverage_target": _OFFICIAL_COVERAGE_TARGET,
        "step12_decision": _OFFICIAL_STEP12_DECISION,
    }


def get_all_forecasts() -> List[dict]:
    """
    Return a list of all available country forecasts.

    Each item includes the full forecast data plus model and uncertainty provenance.
    Raises FileNotFoundError or ValueError if the artifact is missing or invalid.
    """
    cache = _get_cache()
    model_info = _make_model_info()
    uncertainty_info = _make_uncertainty_info()
    return [
        {**data, "model": model_info, "uncertainty": uncertainty_info}
        for data in sorted(cache.values(), key=lambda x: x["country_code"])
    ]


def get_forecast(country_code: str) -> Optional[dict]:
    """
    Return the forecast for a specific country code, or None if not available.

    Args:
        country_code: ISO 3166-1 alpha-3 country code (case-insensitive).

    Returns:
        Forecast dict with model and uncertainty provenance, or None if not found.
    """
    code = country_code.upper().strip()
    cache = _get_cache()
    data = cache.get(code)
    if data is None:
        return None
    return {
        **data,
        "model": _make_model_info(),
        "uncertainty": _make_uncertainty_info(),
    }


def get_metadata() -> dict:
    """
    Return static metadata about the locked forecast model and artifact.

    Does not expose filesystem paths.
    """
    cache = _get_cache()
    available = sorted(cache.keys())
    return {
        "model_version": _OFFICIAL_MODEL_VERSION,
        "algorithm": _OFFICIAL_ALGORITHM,
        "test_rmse": _OFFICIAL_TEST_RMSE,
        "train_period": _OFFICIAL_TRAIN_PERIOD,
        "validation_period": _OFFICIAL_VALIDATION_PERIOD,
        "test_period": _OFFICIAL_TEST_PERIOD,
        "feature_year": _OFFICIAL_FEATURE_YEAR,
        "forecast_year": _OFFICIAL_FORECAST_YEAR,
        "uncertainty_method": _OFFICIAL_UNCERTAINTY_METHOD,
        "q90": _OFFICIAL_Q90,
        "coverage_target": _OFFICIAL_COVERAGE_TARGET,
        "step12_decision": _OFFICIAL_STEP12_DECISION,
        "artifact_file": "t1_step12_2026_forecasts.csv",
        "available_countries": available,
        "n_countries": len(available),
    }


def is_artifact_available() -> bool:
    """Return True if the forecast artifact file exists on disk."""
    return _get_artifact_path().exists()
