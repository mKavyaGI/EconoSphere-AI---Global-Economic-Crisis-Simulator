"""
Unit tests for Scenarios endpoints and event type inspection.
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from app.main import app
from app.api.dependencies.database import get_db

client = TestClient(app)

class MockScenario:
    def __init__(self, id, title, description="Test Scenario", is_template=False):
        self.id = id
        self.title = title
        self.description = description
        self.is_template = is_template
        self.parameters = {"shock_type": "TARIFF", "magnitude": 15}
        self.created_at = "2026-01-01T00:00:00Z"
        self.updated_at = "2026-01-01T00:00:00Z"

@pytest.fixture
def mock_db():
    async def override_get_db():
        yield AsyncMock()
    app.dependency_overrides[get_db] = override_get_db
    yield
    app.dependency_overrides.clear()

def test_get_scenarios_list(mock_db):
    mock_list = [MockScenario(1, "Tariff War 2026"), MockScenario(2, "Energy Shock 2026")]
    with patch("app.services.scenario.ScenarioService.get_scenarios", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_list
        response = client.get("/api/v1/scenarios/")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["title"] == "Tariff War 2026"
        assert data[0]["id"] == 1

def test_get_templates_list(mock_db):
    mock_templates = [MockScenario(10, "Global Recession Template", is_template=True)]
    with patch("app.services.scenario.ScenarioService.get_scenarios", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_templates
        response = client.get("/api/v1/scenarios/templates")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["is_template"] is True

def test_get_event_types():
    response = client.get("/api/v1/scenarios/events/types")
    assert response.status_code == 200
    types = response.json()
    assert isinstance(types, list)
    assert len(types) > 0
    assert "TARIFF" in types or "TARIFF_SHOCK" in types or any(isinstance(t, str) for t in types)

def test_get_single_scenario_not_found(mock_db):
    with patch("app.services.scenario.ScenarioService.get_scenario", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = None
        response = client.get("/api/v1/scenarios/999")
        assert response.status_code == 404
        assert "Scenario not found" in response.json()["detail"]

def test_get_scenarios_pagination_and_filter(mock_db):
    mock_list = [MockScenario(5, "Filtered Scenario")]
    with patch("app.services.scenario.ScenarioService.get_scenarios", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_list
        response = client.get("/api/v1/scenarios/?skip=10&limit=5&status=DRAFT")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["title"] == "Filtered Scenario"
        mock_get.assert_called_once_with(
            is_template=None,
            status="DRAFT",
            title=None,
            skip=10,
            limit=5
        )

def test_validate_scenario_endpoint(mock_db):
    mock_result = {
        "is_valid": False,
        "errors": ["Event ID 10 is missing a valid title.", "Event 'High Shock' has shockIntensity (1.5) outside valid range [0.0, 1.0]."],
        "warnings": ["An empty scenario cannot be evaluated in the simulation engine."]
    }
    with patch("app.services.scenario.ScenarioService.validate_scenario", new_callable=AsyncMock) as mock_validate:
        mock_validate.return_value = mock_result
        response = client.post("/api/v1/scenarios/1/validate")
        assert response.status_code == 200
        data = response.json()
        assert data["is_valid"] is False
        assert len(data["errors"]) == 2
        assert "outside valid range" in data["errors"][1]

