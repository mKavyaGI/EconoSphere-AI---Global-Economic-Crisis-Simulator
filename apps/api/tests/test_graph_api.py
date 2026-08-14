"""
Unit tests for Trade Network Graph endpoints and service integration.
"""
import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_get_global_network():
    mock_network = {
        "nodes": [{"id": "USA", "label": "United States", "properties": {"region": "Americas"}}],
        "edges": [{"id": "e1", "source": "USA", "target": "CAN", "type": "TRADE", "properties": {"weight": 500.0}}]
    }
    with patch("app.services.graph.graph_service.get_network", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_network
        response = client.get("/api/v1/graph/network")
        assert response.status_code == 200
        data = response.json()
        assert len(data["nodes"]) == 1
        assert len(data["edges"]) == 1
        assert data["nodes"][0]["id"] == "USA"
        assert data["edges"][0]["id"] == "e1"

def test_get_neighbors():
    mock_network = {
        "nodes": [{"id": "USA", "label": "United States", "properties": {}}, {"id": "MEX", "label": "Mexico", "properties": {}}],
        "edges": [{"id": "e2", "source": "USA", "target": "MEX", "type": "TRADE", "properties": {"weight": 300.0}}]
    }
    with patch("app.services.graph.graph_service.get_neighbors", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_network
        response = client.get("/api/v1/graph/neighbors/USA?depth=2")
        assert response.status_code == 200
        data = response.json()
        assert len(data["nodes"]) == 2
        assert len(data["edges"]) == 1

def test_get_shortest_path():
    mock_path = {
        "nodes": [{"id": "USA", "label": "United States", "properties": {}}, {"id": "CHN", "label": "China", "properties": {}}],
        "edges": [{"id": "e3", "source": "USA", "target": "CHN", "type": "TRADE", "properties": {"weight": 700.0}}]
    }
    with patch("app.services.graph.graph_service.get_shortest_path", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_path
        response = client.get("/api/v1/graph/path?source=USA&target=CHN")
        assert response.status_code == 200
        data = response.json()
        assert data["nodes"][1]["id"] == "CHN"

def test_get_centrality():
    mock_centrality = [
        {"node_id": "USA", "score": 0.95},
        {"node_id": "CHN", "score": 0.89}
    ]
    with patch("app.services.graph.graph_service.get_centrality", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_centrality
        response = client.get("/api/v1/graph/centrality")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["node_id"] == "USA"
        assert data[0]["score"] == 0.95
