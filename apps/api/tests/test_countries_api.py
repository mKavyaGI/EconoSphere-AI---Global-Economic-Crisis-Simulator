"""
Unit tests for Countries endpoints and service integration.
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from app.main import app
from app.api.dependencies.database import get_db

client = TestClient(app)

class MockCountry:
    def __init__(self, iso3, name, region="Americas", income_group="High income"):
        self.iso3 = iso3
        self.name = name
        self.region = region
        self.income_group = income_group
        self.metadata_info = None

class MockIndicator:
    def __init__(self, indicator_name, year, value, source="World Bank"):
        self.indicator_name = indicator_name
        self.year = year
        self.value = value
        self.source = source

@pytest.fixture
def mock_db():
    async def override_get_db():
        yield AsyncMock()
    app.dependency_overrides[get_db] = override_get_db
    yield
    app.dependency_overrides.clear()

def test_get_countries(mock_db):
    mock_countries = [MockCountry("USA", "United States"), MockCountry("CAN", "Canada")]
    with patch("app.services.country.country_service.get_countries", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = (mock_countries, 2)
        response = client.get("/api/v1/countries?limit=10&offset=0")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
        assert len(data["data"]) == 2
        assert data["data"][0]["iso3"] == "USA"

def test_search_countries(mock_db):
    mock_countries = [MockCountry("USA", "United States")]
    with patch("app.services.country.country_service.search_countries", new_callable=AsyncMock) as mock_search:
        mock_search.return_value = mock_countries
        response = client.get("/api/v1/countries/search?query=United")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["iso3"] == "USA"
        
        # Test short query returns empty list without calling service
        short_response = client.get("/api/v1/countries/search?query=U")
        assert short_response.status_code == 200
        assert short_response.json() == []

def test_get_country_by_iso3(mock_db):
    mock_country = MockCountry("DEU", "Germany", region="Europe")
    with patch("app.services.country.country_service.get_country_by_iso3", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_country
        response = client.get("/api/v1/countries/DEU")
        assert response.status_code == 200
        assert response.json()["name"] == "Germany"

def test_get_country_not_found(mock_db):
    with patch("app.services.country.country_service.get_country_by_iso3", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = None
        response = client.get("/api/v1/countries/UNKNOWN")
        assert response.status_code == 404
        assert "Country not found" in response.json()["detail"]

def test_get_country_history(mock_db):
    mock_history = [MockIndicator("GDP", 2023, 25000000000.0)]
    with patch("app.services.country.country_service.get_country_by_iso3", new_callable=AsyncMock) as mock_get_c, \
         patch("app.services.country.country_service.get_country_history", new_callable=AsyncMock) as mock_hist:
        mock_get_c.return_value = MockCountry("USA", "United States")
        mock_hist.return_value = mock_history
        response = client.get("/api/v1/countries/USA/history?indicator=GDP")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["indicator_name"] == "GDP"
        assert data[0]["year"] == 2023
