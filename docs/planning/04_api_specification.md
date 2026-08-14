# API Specification Document
**Project:** EconoSphere AI - AI-Powered Economic Decision Intelligence Platform
**Role:** Principal Backend Architect & API Designer
**Version:** v1.0

---

## 1. API Overview
This document defines the complete REST API contract for EconoSphere AI. It adheres to OpenAPI 3.1 standards and acts as the strict interface boundary between the Next.js frontend and the FastAPI backend. It integrates structured PostgreSQL querying, Neo4j graph traversals, and async AI simulation polling.

## 2. Versioning Strategy
URI-based versioning is enforced across all endpoints to ensure backward compatibility. 
Example: `https://api.econosphere.ai/api/v1/...`

## 3. Authentication Flow
JWT (JSON Web Token) via Bearer scheme.
1. `POST /api/v1/auth/login` returns `{ "access_token": "...", "refresh_token": "..." }`.
2. Frontend attaches `Authorization: Bearer <access_token>` to all subsequent requests.

## 4. Authorization Strategy
Role-Based Access Control (RBAC). Standard endpoints check for `Role.USER`. Destructive operations or global model changes require `Role.ADMIN` or `Role.RESEARCHER`.

## 5. Base URL
* **Development:** `http://localhost:8000/api/v1`
* **Production:** `https://api.econosphere.ai/api/v1`

## 6. Status Codes
* `200 OK` - Success.
* `201 Created` - Resource generated.
* `202 Accepted` - Async background task accepted (e.g., Simulations).
* `400 Bad Request` - Pydantic validation failure.
* `401 Unauthorized` - Missing/invalid JWT.
* `403 Forbidden` - Insufficient RBAC privileges.
* `404 Not Found` - Resource does not exist.
* `429 Too Many Requests` - Rate limit exceeded.
* `500 Internal Server Error` - Unhandled backend exception.

## 7. Standard Error Response (Error Model)
All API errors return a standard enterprise error envelope:
```json
{
  "error": {
    "code": "RESOURCE_NOT_FOUND",
    "message": "The requested country ISO code was not found.",
    "details": {
      "field": "iso3",
      "value": "ZZZ"
    },
    "timestamp": "2026-07-21T20:10:00Z",
    "trace_id": "req-9a8b7c6d5e4f"
  }
}
```

## 8. Standard Success Response
Responses return the raw object or an array of objects directly at the root, unless wrapped by pagination.

## 9. Pagination Strategy
Offset-based pagination for structured lists.
* **Query Params:** `?limit=50&offset=0`
* **Response Envelope:**
```json
{
  "data": [ ... ],
  "meta": {
    "total": 1450,
    "limit": 50,
    "offset": 0,
    "has_more": true
  }
}
```

## 10. Sorting Strategy
* **Query Params:** `?sort=+gdp,-inflation` (Prefix `+` for ascending, `-` for descending).

## 11. Filtering Strategy
Filters are passed as explicit query parameters or comma-separated lists.
* Example: `?regions=EU,NA&min_gdp=1000000`

## 12. Rate Limiting Strategy
* Implemented via Redis.
* Standard Endpoints: 100 req / minute.
* Heavy Simulation Endpoints: 5 req / minute.

## 13. API Naming Conventions
* Kebab-case for URLs (`/api/v1/economic-indicators`).
* Snake_case for JSON payloads to align with Python/PostgreSQL standards.

## 14. Request Validation Rules
All incoming request bodies are validated using Pydantic v2. Extra fields are rejected (400 Bad Request). Types are strictly coerced.

## 15. Response Standards
* Dates/Times: ISO 8601 UTC formats (`YYYY-MM-DDThh:mm:ssZ`).
* Currency/Financials: Handled as exact strings or floats depending on precision requirements.

## 16. API Security
* CORS restricted to frontend domains.
* Secure, HttpOnly cookies for `refresh_token`.
* Rate limiting to prevent DDoS.

---

## API Groups & Endpoints

### A. Authentication
* **Method:** `POST`
* **Endpoint:** `/api/v1/auth/login`
* **Purpose:** Authenticate user and return JWT.
* **Auth Required:** No
* **Body:** `{ "email": "user@example.com", "password": "secure_password" }`
* **Response:** `{ "access_token": "jwt...", "refresh_token": "jwt..." }` (200 OK, 401 Unauthorized)

### B. Countries & Search (Global Filtering)
* **Method:** `GET`
* **Endpoint:** `/api/v1/countries`
* **Purpose:** List and filter global economies.
* **Query Params:** `?region=Asia&min_population=50000000`
* **Response:** Array of Country objects including Metadata.

* **Method:** `GET`
* **Endpoint:** `/api/v1/search`
* **Purpose:** Omni-search across countries, indicators, and scenarios.
* **Query Params:** `?q=Oil Shock`

### C. Trade Network (Neo4j Graph API)
* **Method:** `GET`
* **Endpoint:** `/api/v1/trade/dependency-graph`
* **Purpose:** Get a node/edge map of trade dependencies for React Flow.
* **Query Params:** `?source_iso3=USA&depth=3&commodity=Semiconductors`
* **Response:**
```json
{
  "nodes": [{ "id": "USA", "type": "Country", "gdp": ... }],
  "edges": [{ "source": "TWN", "target": "USA", "type": "EXPORTS", "volume": 50000 }]
}
```

### D. Scenarios & Events
* **Method:** `POST`
* **Endpoint:** `/api/v1/scenarios`
* **Purpose:** Create a new scenario template.
* **Auth Required:** Yes
* **Body:** `{ "name": "Global Pandemic", "description": "High severity." }`
* **Response:** `201 Created`, Scenario Object.

* **Method:** `POST`
* **Endpoint:** `/api/v1/scenarios/{scenario_id}/events`
* **Purpose:** Add a shock to a scenario (e.g., Oil Price Spike).
* **Body:** `{ "target_entity": "Oil", "impact_multiplier": 1.40 }`

### E. Simulation Engine (The Heavy Lifter)
* **Method:** `POST`
* **Endpoint:** `/api/v1/simulations/run`
* **Purpose:** Triggers the AI pipeline in the background.
* **Auth Required:** Yes
* **Body:** `{ "scenario_version_id": "uuid..." }`
* **Response:** `202 Accepted`
```json
{
  "run_id": "uuid-1234",
  "status": "PENDING",
  "message": "Simulation queued. Connect to WebSocket /ws/simulations/uuid-1234 for progress."
}
```
* **DB Tables:** `simulation_runs`, `simulation_metrics`

### F. Simulation Results & AI Forecasting
* **Method:** `GET`
* **Endpoint:** `/api/v1/simulations/{run_id}/results`
* **Purpose:** Fetch the final aggregated JSON metrics post-simulation.

* **Method:** `GET`
* **Endpoint:** `/api/v1/forecasts/gdp`
* **Purpose:** Fetch time-series predictions generated by AI models.
* **Query Params:** `?run_id=uuid&iso3=USA&start_year=2024&end_year=2030`
* **Response:** Array of Data Points `[{ "year": 2024, "value": 25.4, "confidence_lower": 24.8, "confidence_upper": 26.0 }]`

### G. Compare APIs (Analytics)
* **Method:** `GET`
* **Endpoint:** `/api/v1/analytics/compare-countries`
* **Purpose:** Compare KPIs for multiple countries.
* **Query Params:** `?iso3_codes=USA,CHN,IND&indicator=Inflation`

* **Method:** `GET`
* **Endpoint:** `/api/v1/analytics/compare-simulations`
* **Purpose:** Compare outputs of two different runs.
* **Query Params:** `?run_a=uuid1&run_b=uuid2`

### H. AI Policy Engine
* **Method:** `POST`
* **Endpoint:** `/api/v1/policy/generate`
* **Purpose:** Trigger the LLM advisor to write mitigation strategies for a run.
* **Body:** `{ "run_id": "uuid..." }`
* **Response:**
```json
{
  "recommendations": [
    {
      "title": "Subsidize Semiconductor Imports",
      "rationale": "Because the shock drops supply by 40%, subsidizing will stabilize auto manufacturing.",
      "impact_score": 85
    }
  ]
}
```

---

## WebSockets
Standard REST is insufficient for long-running AI simulation tasks. We use WebSockets for real-time telemetry.

* **Endpoint:** `ws://api.econosphere.ai/ws/simulations/{run_id}`
* **Purpose:** Streams progress of the AI Engine.
* **Messages:**
  * `{ "event": "PROGRESS", "percentage": 25, "step": "Calculating Graph Topology..." }`
  * `{ "event": "PROGRESS", "percentage": 75, "step": "Running XGBoost Models..." }`
  * `{ "event": "COMPLETED", "result_url": "/api/v1/simulations/{run_id}/results" }`

* **Endpoint:** `ws://api.econosphere.ai/ws/notifications`
* **Purpose:** Global user notification stream (alerts, shared scenario invites).
