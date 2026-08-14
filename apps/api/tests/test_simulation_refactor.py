"""
Unit tests for Refactored Simulation Module and Clean Architecture API contract.
"""
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from fastapi.testclient import TestClient
from app.main import app
from app.api.v1.endpoints.simulation import get_simulation_service
from app.models.simulation import SimulationStatus

client = TestClient(app)

class MockSimulationRun:
    def __init__(self, id, scenario_id, status, config=None):
        self.id = id
        self.scenario_id = scenario_id
        self.status = status
        self.config = config or {}
        self.started_at = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
        self.completed_at = None

class MockSimulationSnapshot:
    def __init__(self, id, run_id, horizon_days):
        self.id = id
        self.run_id = run_id
        self.horizon_days = horizon_days
        self.timestamp = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
        self.global_state = {"gdp_shock": 0.05}
        self.active_shocks = {"TARIFF": 15.0}

@pytest.fixture
def mock_sim_service():
    mock_svc = MagicMock()
    mock_svc.start_simulation = AsyncMock()
    mock_svc.get_simulation_status = AsyncMock()
    mock_svc.stop_simulation = AsyncMock()
    mock_svc.list_simulation_runs = AsyncMock()
    mock_svc.get_run_snapshots = AsyncMock()
    app.dependency_overrides[get_simulation_service] = lambda: mock_svc
    yield mock_svc
    app.dependency_overrides.clear()

def test_start_simulation_success(mock_sim_service):
    mock_run = MockSimulationRun(101, 1, SimulationStatus.PENDING)
    mock_sim_service.start_simulation.return_value = mock_run
    
    response = client.post("/api/v1/simulation/start", json={"scenario_id": 1, "config": {}})
    assert response.status_code == 201
    data = response.json()
    assert data["id"] == 101
    assert data["status"] == "PENDING"
    mock_sim_service.start_simulation.assert_called_once()

def test_start_simulation_not_found(mock_sim_service):
    mock_sim_service.start_simulation.side_effect = ValueError("Scenario not found")
    response = client.post("/api/v1/simulation/start", json={"scenario_id": 999, "config": {}})
    assert response.status_code == 404
    assert "Scenario not found" in response.json()["detail"]

def test_get_simulation_status(mock_sim_service):
    mock_run = MockSimulationRun(101, 1, SimulationStatus.RUNNING)
    mock_sim_service.get_simulation_status.return_value = mock_run
    
    response = client.get("/api/v1/simulation/101/status")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == 101
    assert data["status"] == "RUNNING"

def test_get_simulation_snapshots(mock_sim_service):
    mock_run = MockSimulationRun(101, 1, SimulationStatus.COMPLETED)
    mock_snapshot = MockSimulationSnapshot(1, 101, 30)
    mock_sim_service.get_simulation_status.return_value = mock_run
    mock_sim_service.get_run_snapshots.return_value = [mock_snapshot]
    
    response = client.get("/api/v1/simulation/101/snapshots")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["id"] == 1
    assert data[0]["horizon_days"] == 30

def test_get_simulation_not_found(mock_sim_service):
    mock_sim_service.get_simulation_status.return_value = None
    response = client.get("/api/v1/simulation/999/status")
    assert response.status_code == 404
    assert "Simulation run not found" in response.json()["detail"]
