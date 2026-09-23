"""
EconoSphere AI — Phase 11 Step 13 Forecast API Tests
=====================================================
Tests for GET /api/v1/forecasts, /api/v1/forecasts/metadata,
and /api/v1/forecasts/{country_code}.

Uses FastAPI TestClient for synchronous HTTP testing (no real DB needed
because forecast endpoints have no DB dependency).

Verifies:
  - GET /forecasts returns 200 with 5 countries
  - GET /forecasts contains required JSON fields
  - GET /forecasts/metadata returns 200 with RMSE and Q90
  - GET /forecasts/IND returns 200 with India forecast
  - GET /forecasts/UNKNOWN returns 404
  - GET /forecasts/12 returns 400 (invalid code)
  - forecast_status is "forecast" (not "actual")
  - No filesystem paths in any response
  - API does not fabricate 2026 actual GDP values
  - Official metadata values are correct
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app

# ---------------------------------------------------------------------------
# Client fixture
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def client():
    """FastAPI test client shared for all tests in this module."""
    return TestClient(app)


EXPECTED_COUNTRIES = {"CHN", "GBR", "IND", "JPN", "USA"}
EXPECTED_TEST_RMSE = 3.9113
EXPECTED_Q90 = 11.4507
RMSE_TOL = 0.001
Q90_TOL = 0.001


# ===========================================================================
# GET /api/v1/forecasts
# ===========================================================================

class TestGetAllForecasts:
    def test_status_200(self, client):
        """GET /forecasts must return HTTP 200."""
        r = client.get("/api/v1/forecasts")
        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"

    def test_response_has_forecasts_key(self, client):
        """Response must contain a 'forecasts' key."""
        r = client.get("/api/v1/forecasts")
        body = r.json()
        assert "forecasts" in body, f"'forecasts' key missing. Got: {list(body.keys())}"

    def test_five_unique_countries(self, client):
        """Response must contain exactly 5 unique country forecasts."""
        r = client.get("/api/v1/forecasts")
        forecasts = r.json()["forecasts"]
        codes = {f["country_code"] for f in forecasts}
        assert len(codes) == 5, f"Expected 5 countries, got {len(codes)}: {codes}"
        assert codes == EXPECTED_COUNTRIES

    def test_response_has_metadata_key(self, client):
        """Response must contain a 'metadata' key."""
        body = client.get("/api/v1/forecasts").json()
        assert "metadata" in body

    def test_response_has_disclaimer(self, client):
        """Response must contain a 'disclaimer' key (statistical transparency)."""
        body = client.get("/api/v1/forecasts").json()
        assert "disclaimer" in body
        assert len(body["disclaimer"]) > 10

    def test_forecast_status_is_forecast(self, client):
        """All forecast entries must have forecast_status='forecast'."""
        forecasts = client.get("/api/v1/forecasts").json()["forecasts"]
        for f in forecasts:
            assert f["forecast_status"] == "forecast", (
                f"{f['country_code']}: forecast_status must be 'forecast', got '{f['forecast_status']}'"
            )

    def test_required_forecast_fields(self, client):
        """Each forecast must contain all required response fields."""
        required = {
            "country_code", "country_name", "feature_year", "forecast_year",
            "predicted_gdp_growth", "lower_bound_90", "upper_bound_90",
            "interval_width", "forecast_status", "model", "uncertainty",
        }
        for f in client.get("/api/v1/forecasts").json()["forecasts"]:
            missing = required - set(f.keys())
            assert not missing, f"Missing fields for {f.get('country_code')}: {missing}"

    def test_no_filesystem_path_in_response(self, client):
        """No API response field must contain a filesystem path."""
        body = str(client.get("/api/v1/forecasts").json())
        assert ":\\" not in body, "Windows filesystem path exposed in API response"
        assert "/data/processed" not in body, "Unix filesystem path exposed in API response"

    def test_feature_year_is_2025(self, client):
        """All forecasts must have feature_year=2025."""
        for f in client.get("/api/v1/forecasts").json()["forecasts"]:
            assert f["feature_year"] == 2025, f"{f['country_code']}: feature_year != 2025"

    def test_forecast_year_is_2026(self, client):
        """All forecasts must have forecast_year=2026."""
        for f in client.get("/api/v1/forecasts").json()["forecasts"]:
            assert f["forecast_year"] == 2026, f"{f['country_code']}: forecast_year != 2026"

    def test_lower_le_predicted_le_upper(self, client):
        """lower_bound_90 <= predicted_gdp_growth <= upper_bound_90 for all countries."""
        for f in client.get("/api/v1/forecasts").json()["forecasts"]:
            c = f["country_code"]
            assert f["lower_bound_90"] <= f["predicted_gdp_growth"], (
                f"{c}: lower > predicted"
            )
            assert f["predicted_gdp_growth"] <= f["upper_bound_90"], (
                f"{c}: predicted > upper"
            )

    def test_interval_width_positive(self, client):
        """interval_width must be positive for all countries."""
        for f in client.get("/api/v1/forecasts").json()["forecasts"]:
            assert f["interval_width"] > 0, f"{f['country_code']}: non-positive interval_width"


# ===========================================================================
# GET /api/v1/forecasts/metadata
# ===========================================================================

class TestForecastMetadata:
    def test_status_200(self, client):
        """GET /forecasts/metadata must return HTTP 200."""
        r = client.get("/api/v1/forecasts/metadata")
        assert r.status_code == 200, r.text

    def test_rmse_is_correct(self, client):
        """Metadata test_rmse must match the official locked value."""
        meta = client.get("/api/v1/forecasts/metadata").json()
        assert abs(meta["test_rmse"] - EXPECTED_TEST_RMSE) < RMSE_TOL, (
            f"test_rmse={meta['test_rmse']} expected {EXPECTED_TEST_RMSE}"
        )

    def test_q90_is_correct(self, client):
        """Metadata q90 must match the official locked value."""
        meta = client.get("/api/v1/forecasts/metadata").json()
        assert abs(meta["q90"] - EXPECTED_Q90) < Q90_TOL, (
            f"q90={meta['q90']} expected {EXPECTED_Q90}"
        )

    def test_algorithm_is_histgb(self, client):
        """Metadata algorithm must be HistGradientBoostingRegressor."""
        meta = client.get("/api/v1/forecasts/metadata").json()
        assert "HistGradientBoosting" in meta["algorithm"]

    def test_n_countries_is_5(self, client):
        """Metadata n_countries must be 5."""
        meta = client.get("/api/v1/forecasts/metadata").json()
        assert meta["n_countries"] == 5

    def test_available_countries(self, client):
        """Metadata available_countries must match the five focal countries."""
        meta = client.get("/api/v1/forecasts/metadata").json()
        assert set(meta["available_countries"]) == EXPECTED_COUNTRIES

    def test_no_filesystem_path_in_metadata(self, client):
        """Metadata must not expose filesystem paths."""
        body = str(client.get("/api/v1/forecasts/metadata").json())
        assert ":\\" not in body
        assert "/data/processed" not in body

    def test_artifact_file_is_filename_only(self, client):
        """artifact_file must be a filename, not a path."""
        meta = client.get("/api/v1/forecasts/metadata").json()
        artifact = meta["artifact_file"]
        assert "\\" not in artifact, f"artifact_file contains path: {artifact}"
        assert "/" not in artifact, f"artifact_file contains path: {artifact}"

    def test_step12_decision_is_keep(self, client):
        """Metadata step12_decision must record the KEEP decision."""
        meta = client.get("/api/v1/forecasts/metadata").json()
        assert "KEEP" in meta["step12_decision"].upper()

    def test_feature_year_metadata(self, client):
        """Metadata feature_year must be 2025."""
        meta = client.get("/api/v1/forecasts/metadata").json()
        assert meta["feature_year"] == 2025

    def test_forecast_year_metadata(self, client):
        """Metadata forecast_year must be 2026."""
        meta = client.get("/api/v1/forecasts/metadata").json()
        assert meta["forecast_year"] == 2026


# ===========================================================================
# GET /api/v1/forecasts/{country_code}
# ===========================================================================

class TestGetCountryForecast:
    def test_india_returns_200(self, client):
        """GET /forecasts/IND must return HTTP 200."""
        r = client.get("/api/v1/forecasts/IND")
        assert r.status_code == 200, r.text

    def test_india_country_code(self, client):
        """India forecast must have country_code='IND'."""
        f = client.get("/api/v1/forecasts/IND").json()
        assert f["country_code"] == "IND"

    def test_india_country_name(self, client):
        """India forecast must have country_name='India'."""
        f = client.get("/api/v1/forecasts/IND").json()
        assert f["country_name"] == "India"

    def test_china_returns_200(self, client):
        """GET /forecasts/CHN must return HTTP 200."""
        assert client.get("/api/v1/forecasts/CHN").status_code == 200

    def test_usa_returns_200(self, client):
        """GET /forecasts/USA must return HTTP 200."""
        assert client.get("/api/v1/forecasts/USA").status_code == 200

    def test_japan_returns_200(self, client):
        """GET /forecasts/JPN must return HTTP 200."""
        assert client.get("/api/v1/forecasts/JPN").status_code == 200

    def test_uk_returns_200(self, client):
        """GET /forecasts/GBR must return HTTP 200."""
        assert client.get("/api/v1/forecasts/GBR").status_code == 200

    def test_unknown_country_returns_404(self, client):
        """GET /forecasts/XYZ must return HTTP 404."""
        r = client.get("/api/v1/forecasts/XYZ")
        assert r.status_code == 404, f"Expected 404, got {r.status_code}"

    def test_invalid_code_returns_400(self, client):
        """GET /forecasts/12345 (non-alpha) must return HTTP 400."""
        r = client.get("/api/v1/forecasts/12345")
        assert r.status_code == 400, f"Expected 400, got {r.status_code}"

    def test_forecast_status_is_forecast(self, client):
        """Single-country forecast must have forecast_status='forecast'."""
        f = client.get("/api/v1/forecasts/IND").json()
        assert f["forecast_status"] == "forecast"

    def test_model_test_rmse_in_single_forecast(self, client):
        """Single-country forecast model.test_rmse must match official value."""
        f = client.get("/api/v1/forecasts/IND").json()
        assert abs(f["model"]["test_rmse"] - EXPECTED_TEST_RMSE) < RMSE_TOL

    def test_q90_in_single_forecast(self, client):
        """Single-country forecast uncertainty.q90 must match official value."""
        f = client.get("/api/v1/forecasts/IND").json()
        assert abs(f["uncertainty"]["q90"] - EXPECTED_Q90) < Q90_TOL

    def test_no_fabricated_2026_actual_in_response(self, client):
        """
        API must not present 2026 forecast as an observed GDP statistic.
        forecast_status must be 'forecast', not 'actual'.
        """
        f = client.get("/api/v1/forecasts/IND").json()
        assert f["forecast_status"] != "actual", (
            "2026 GDP value must not be presented as 'actual' — it is a model forecast."
        )

    def test_no_filesystem_path_in_single_forecast(self, client):
        """Single-country response must not expose filesystem paths."""
        body = str(client.get("/api/v1/forecasts/IND").json())
        assert ":\\" not in body
        assert "/data/processed" not in body
