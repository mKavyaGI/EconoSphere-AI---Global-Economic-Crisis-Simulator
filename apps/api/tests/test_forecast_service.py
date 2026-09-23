"""
EconoSphere AI — Phase 11 Step 13 Forecast Service Tests
==========================================================
Tests for apps/api/app/services/forecast_service.py

Verifies:
  - Artifact loads successfully
  - Required columns are present
  - Values are numeric and finite
  - lower_bound_90 <= predicted_gdp_growth <= upper_bound_90
  - target_year = 2026
  - feature_year = 2025
  - Q90 matches official value
  - Test RMSE metadata is correct
  - Unknown country returns None
  - Known countries return data
  - No filesystem paths leak through the service interface
  - Forecast status is "forecast" for all entries
  - Duplicate conflict detection works correctly
"""
import math
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Ensure the service module is importable
import app.services.forecast_service as svc

# ---------------------------------------------------------------------------
# Constants for assertions
# ---------------------------------------------------------------------------
EXPECTED_TEST_RMSE = 3.9113
EXPECTED_Q90 = 11.4507
EXPECTED_FEATURE_YEAR = 2025
EXPECTED_FORECAST_YEAR = 2026
EXPECTED_COUNTRIES = {"CHN", "GBR", "IND", "JPN", "USA"}
RMSE_TOLERANCE = 0.001
Q90_TOLERANCE = 0.001


# ---------------------------------------------------------------------------
# Helper: Reset cache before each test
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def reset_cache():
    """Reset the module-level cache before each test to avoid state leakage."""
    original = svc._CACHE
    svc._CACHE = None
    yield
    svc._CACHE = original


# ===========================================================================
# 1. Artifact Loading
# ===========================================================================

class TestArtifactLoading:
    def test_artifact_file_exists(self):
        """The official Step 12 forecast artifact must exist on disk."""
        assert svc.is_artifact_available(), (
            "data/processed/t1_step12_2026_forecasts.csv not found. "
            "Run train_t1_adaptive_uncertainty.py to regenerate."
        )

    def test_all_forecasts_loads_without_error(self):
        """get_all_forecasts() must not raise an exception."""
        result = svc.get_all_forecasts()
        assert isinstance(result, list)
        assert len(result) > 0

    def test_metadata_loads_without_error(self):
        """get_metadata() must not raise an exception."""
        meta = svc.get_metadata()
        assert isinstance(meta, dict)

    def test_expected_country_count(self):
        """Artifact should contain exactly 5 unique countries after deduplication."""
        result = svc.get_all_forecasts()
        codes = {r["country_code"] for r in result}
        assert len(codes) == 5, f"Expected 5 countries, got {len(codes)}: {codes}"

    def test_expected_countries_present(self):
        """All five focal countries must be present in the artifact."""
        result = svc.get_all_forecasts()
        codes = {r["country_code"] for r in result}
        assert codes == EXPECTED_COUNTRIES, f"Expected {EXPECTED_COUNTRIES}, got {codes}"


# ===========================================================================
# 2. Schema / Column Validation
# ===========================================================================

class TestSchemaValidation:
    def test_required_fields_present(self):
        """Each forecast dict must contain all required fields."""
        required = {
            "country_code", "country_name", "feature_year", "forecast_year",
            "predicted_gdp_growth", "lower_bound_90", "upper_bound_90",
            "interval_width", "forecast_status", "model", "uncertainty",
        }
        for forecast in svc.get_all_forecasts():
            missing = required - set(forecast.keys())
            assert not missing, f"Missing fields in {forecast['country_code']}: {missing}"

    def test_model_info_fields(self):
        """Model info sub-dict must contain required provenance fields."""
        for forecast in svc.get_all_forecasts():
            model = forecast["model"]
            assert "version" in model
            assert "algorithm" in model
            assert "test_rmse" in model
            assert "train_period" in model
            assert "validation_period" in model
            assert "test_period" in model

    def test_uncertainty_info_fields(self):
        """Uncertainty info sub-dict must contain required calibration fields."""
        for forecast in svc.get_all_forecasts():
            unc = forecast["uncertainty"]
            assert "method" in unc
            assert "q90" in unc
            assert "coverage_target" in unc
            assert "step12_decision" in unc


# ===========================================================================
# 3. Numeric Integrity
# ===========================================================================

class TestNumericIntegrity:
    def test_predicted_gdp_growth_is_finite(self):
        """predicted_gdp_growth must be a finite float for all countries."""
        for f in svc.get_all_forecasts():
            v = f["predicted_gdp_growth"]
            assert isinstance(v, float), f"{f['country_code']}: expected float, got {type(v)}"
            assert math.isfinite(v), f"{f['country_code']}: predicted_gdp_growth is not finite: {v}"

    def test_lower_bound_is_finite(self):
        """lower_bound_90 must be finite for all countries."""
        for f in svc.get_all_forecasts():
            assert math.isfinite(f["lower_bound_90"]), f"{f['country_code']}: lower_bound_90 not finite"

    def test_upper_bound_is_finite(self):
        """upper_bound_90 must be finite for all countries."""
        for f in svc.get_all_forecasts():
            assert math.isfinite(f["upper_bound_90"]), f"{f['country_code']}: upper_bound_90 not finite"

    def test_interval_width_is_positive(self):
        """interval_width must be strictly positive for all countries."""
        for f in svc.get_all_forecasts():
            assert f["interval_width"] > 0, f"{f['country_code']}: non-positive interval_width"

    def test_lower_le_predicted_le_upper(self):
        """lower_bound_90 <= predicted_gdp_growth <= upper_bound_90 for all countries."""
        for f in svc.get_all_forecasts():
            c = f["country_code"]
            assert f["lower_bound_90"] <= f["predicted_gdp_growth"], (
                f"{c}: lower_bound_90={f['lower_bound_90']} > predicted={f['predicted_gdp_growth']}"
            )
            assert f["predicted_gdp_growth"] <= f["upper_bound_90"], (
                f"{c}: predicted={f['predicted_gdp_growth']} > upper_bound_90={f['upper_bound_90']}"
            )

    def test_interval_width_matches_bounds(self):
        """interval_width must equal upper_bound_90 - lower_bound_90 (within float tolerance)."""
        for f in svc.get_all_forecasts():
            expected_width = f["upper_bound_90"] - f["lower_bound_90"]
            assert abs(f["interval_width"] - expected_width) < 0.01, (
                f"{f['country_code']}: interval_width mismatch: "
                f"{f['interval_width']} vs expected {expected_width}"
            )


# ===========================================================================
# 4. Temporal Integrity
# ===========================================================================

class TestTemporalIntegrity:
    def test_forecast_year_is_2026(self):
        """forecast_year must be 2026 for all countries."""
        for f in svc.get_all_forecasts():
            assert f["forecast_year"] == EXPECTED_FORECAST_YEAR, (
                f"{f['country_code']}: forecast_year={f['forecast_year']} != {EXPECTED_FORECAST_YEAR}"
            )

    def test_feature_year_is_2025(self):
        """feature_year must be 2025 for all countries."""
        for f in svc.get_all_forecasts():
            assert f["feature_year"] == EXPECTED_FEATURE_YEAR, (
                f"{f['country_code']}: feature_year={f['feature_year']} != {EXPECTED_FEATURE_YEAR}"
            )


# ===========================================================================
# 5. Country Lookup
# ===========================================================================

class TestCountryLookup:
    def test_known_country_returns_forecast(self):
        """get_forecast('IND') must return India's forecast."""
        result = svc.get_forecast("IND")
        assert result is not None
        assert result["country_code"] == "IND"
        assert result["country_name"] == "India"

    def test_known_country_case_insensitive(self):
        """Country lookup must be case-insensitive."""
        assert svc.get_forecast("ind") is not None
        assert svc.get_forecast("Ind") is not None
        assert svc.get_forecast("IND") is not None

    def test_unknown_country_returns_none(self):
        """get_forecast('XYZ') must return None for unknown country."""
        assert svc.get_forecast("XYZ") is None

    def test_empty_string_returns_none(self):
        """Empty string country code must return None."""
        assert svc.get_forecast("") is None

    def test_all_five_countries_return_forecast(self):
        """Each of the five focal countries must return a forecast."""
        for code in EXPECTED_COUNTRIES:
            result = svc.get_forecast(code)
            assert result is not None, f"Expected forecast for {code}, got None"


# ===========================================================================
# 6. Official Metadata Values
# ===========================================================================

class TestOfficialMetadataValues:
    def test_test_rmse_is_correct(self):
        """model.test_rmse must match the official locked value within tolerance."""
        meta = svc.get_metadata()
        assert abs(meta["test_rmse"] - EXPECTED_TEST_RMSE) < RMSE_TOLERANCE, (
            f"test_rmse={meta['test_rmse']} differs from expected {EXPECTED_TEST_RMSE}"
        )

    def test_q90_is_correct(self):
        """uncertainty q90 must match the official locked value within tolerance."""
        meta = svc.get_metadata()
        assert abs(meta["q90"] - EXPECTED_Q90) < Q90_TOLERANCE, (
            f"q90={meta['q90']} differs from expected {EXPECTED_Q90}"
        )

    def test_algorithm_is_histgb(self):
        """Algorithm must be HistGradientBoostingRegressor."""
        meta = svc.get_metadata()
        assert "HistGradientBoosting" in meta["algorithm"], (
            f"Algorithm must be HistGradientBoostingRegressor, got: {meta['algorithm']}"
        )

    def test_coverage_target_is_90_percent(self):
        """Coverage target must be 0.90 (90%)."""
        meta = svc.get_metadata()
        assert abs(meta["coverage_target"] - 0.90) < 1e-6

    def test_step12_decision_is_keep(self):
        """Step 12 decision must record KEEP STEP 11 GLOBAL Q90 CONTROL."""
        meta = svc.get_metadata()
        assert "KEEP" in meta["step12_decision"].upper()

    def test_feature_year_metadata(self):
        """Metadata feature_year must be 2025."""
        meta = svc.get_metadata()
        assert meta["feature_year"] == EXPECTED_FEATURE_YEAR

    def test_forecast_year_metadata(self):
        """Metadata forecast_year must be 2026."""
        meta = svc.get_metadata()
        assert meta["forecast_year"] == EXPECTED_FORECAST_YEAR

    def test_available_countries_in_metadata(self):
        """Metadata must list the five available countries."""
        meta = svc.get_metadata()
        assert set(meta["available_countries"]) == EXPECTED_COUNTRIES

    def test_n_countries_metadata(self):
        """Metadata n_countries must equal 5."""
        meta = svc.get_metadata()
        assert meta["n_countries"] == 5


# ===========================================================================
# 7. Forecast Status
# ===========================================================================

class TestForecastStatus:
    def test_forecast_status_is_forecast(self):
        """All forecasts must have forecast_status='forecast'."""
        for f in svc.get_all_forecasts():
            assert f["forecast_status"] == "forecast", (
                f"{f['country_code']}: forecast_status must be 'forecast', got '{f['forecast_status']}'"
            )


# ===========================================================================
# 8. No Path Leakage
# ===========================================================================

class TestNoPathLeakage:
    def test_no_filesystem_path_in_metadata(self):
        """Metadata artifact_file field must not contain a filesystem path."""
        meta = svc.get_metadata()
        artifact = meta["artifact_file"]
        assert not artifact.startswith("/"), f"artifact_file exposes absolute path: {artifact}"
        assert ":\\" not in artifact, f"artifact_file exposes Windows path: {artifact}"
        assert "/" not in artifact or artifact == artifact.split("/")[-1], (
            f"artifact_file should be a filename, not a path: {artifact}"
        )

    def test_no_filesystem_path_in_forecast(self):
        """Individual forecast dicts must not contain filesystem paths."""
        for f in svc.get_all_forecasts():
            for key, value in f.items():
                if isinstance(value, str):
                    assert ":\\" not in value, f"{key}='{value}' exposes Windows path"
                    assert not (isinstance(value, str) and value.startswith("/data/")), (
                        f"{key}='{value}' exposes Unix filesystem path"
                    )


# ===========================================================================
# 9. Duplicate Conflict Detection
# ===========================================================================

class TestDuplicateConflictDetection:
    def test_conflicting_duplicates_raise_value_error(self):
        """Service must raise ValueError if duplicates have conflicting values."""
        conflicting_csv = (
            "country_code,country_name,feature_year,target_year,"
            "predicted_gdp_growth,lower_bound_90,upper_bound_90,interval_width,uncertainty_method\n"
            "IND,India,2025,2026,1.02,-10.43,12.47,22.90,step11_global_q90_control\n"
            "IND,India,2025,2026,2.50,-8.00,14.00,22.00,step11_global_q90_control\n"  # conflict!
        )
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(conflicting_csv)
            tmp_path = Path(f.name)
        try:
            with patch.object(svc, "_get_artifact_path", return_value=tmp_path):
                with pytest.raises(ValueError, match="CONFLICTING"):
                    svc._load_and_validate()
        finally:
            os.unlink(tmp_path)

    def test_non_conflicting_duplicates_are_accepted(self):
        """Service must accept non-conflicting duplicate rows silently."""
        good_csv = (
            "country_code,country_name,feature_year,target_year,"
            "predicted_gdp_growth,lower_bound_90,upper_bound_90,interval_width,uncertainty_method\n"
            "IND,India,2025,2026,1.0163,-10.4344,12.467,22.9013,step11_global_q90_control\n"
            "IND,India,2025,2026,1.0163,-10.4344,12.467,22.9013,step11_global_q90_control\n"
        )
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(good_csv)
            tmp_path = Path(f.name)
        try:
            with patch.object(svc, "_get_artifact_path", return_value=tmp_path):
                result = svc._load_and_validate()
                assert "IND" in result
                assert len(result) == 1  # deduped
        finally:
            os.unlink(tmp_path)

    def test_missing_artifact_raises_file_not_found(self):
        """Service must raise FileNotFoundError if artifact is missing."""
        with patch.object(svc, "_get_artifact_path", return_value=Path("/nonexistent/path.csv")):
            with pytest.raises(FileNotFoundError):
                svc._load_and_validate()
