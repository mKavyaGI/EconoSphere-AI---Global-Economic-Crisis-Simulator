# Phase 14 Step 3: Production Deployment Documentation & Operational Runbook

## 1. Document Purpose

This runbook provides the standard operating procedures for the EconoSphere AI production deployment.

Phase 14 Step 2 verified the following baseline:
**FULL CONTAINERIZED RUNTIME VERIFIED**

Phase 14 Step 3 accomplishes:
**PRODUCTION DEPLOYMENT DOCUMENTATION & OPERATIONAL RUNBOOK**

This runbook operationalizes and documents the exact deployment path already verified in Step 2. It **does not modify** the frozen ML model, nor any code logic. The Phase 11 production baseline remains **immutable** and unchanged. Any updates to models require a formal new phase governance procedure.

---

## 2. Production Architecture

The verified EconoSphere AI production deployment consists of a containerized architecture orchestrated via Docker Compose (`infrastructure/docker/docker-compose.yml`).

```text
Client
  |
  v
FastAPI API
  |
  +--> PostgreSQL
  |
  +--> Redis
  |
  +--> Neo4j
  |
  +--> Read-Only Phase 11 Model
  |
  +--> Read-Only Dataset / Production Artifacts
```

### Service Responsibilities
*   **API (`api`)**: The core FastAPI application serving predictions and handling business logic. Port: `8000`.
*   **PostgreSQL (`postgres`)**: Relational database for structured application data. Port: `5432`.
*   **Redis (`redis`)**: In-memory data store for caching and rate-limiting. Port: `6379`.
*   **Neo4j (`neo4j`)**: Graph database for economic relationship analysis. Ports: `7474`, `7687`.

### Infrastructure Protections
*   **Networking**: Services communicate via the internal Docker network. Only necessary ports are mapped to the host.
*   **Read-Only Mounts**: The `models/` and `data/` directories are mounted as strictly read-only (`:ro`) to prevent accidental runtime mutation of the Phase 11 baseline.
*   **Startup Dependencies**: The API waits for `postgres`, `redis`, and `neo4j` to be healthy/started before fully initializing.
*   **Health/Readiness**: The API exposes readiness gates that perform deep preflight integrity checks on startup.

---

## 3. Prerequisites

The deployment environment must have the following tools available.

### Verification Commands
```bash
# Verify Docker is running
docker version

# Verify Docker Compose is available (V2)
docker compose version

# Verify Git is available
git --version
```

The Docker Desktop daemon or an equivalent Docker Engine MUST be running prior to deployment. If `docker version` fails to connect to the daemon, start Docker first.

---

## 4. Repository Preparation

The safe repository preparation workflow ensures the correct artifacts are present before starting the infrastructure.

1.  **Clone repository and navigate to root**:
    ```bash
    git clone <repository_url> econosphere-ai
    cd econosphere-ai
    ```

2.  **Confirm required production artifacts exist**:
    Verify the existence of the following frozen Phase 11 artifacts. **Do not modify or attempt to regenerate them.**
    ```text
    models/phase11/best_t1_gdp_growth_model.joblib
    models/phase11/phase11_production_release_manifest.json
    data/raw/master_panel.csv
    ```

3.  **Confirm Docker configuration exists**:
    Ensure `infrastructure/docker/docker-compose.yml` and `apps/api/Dockerfile` are present.

---

## 5. Production Environment Configuration

The production environment is configured through environment variables passed to the `api` container in `docker-compose.yml`.

| Variable | Purpose | Required | Safe Example | Unsafe Behavior |
| :--- | :--- | :--- | :--- | :--- |
| `ENV` | Dictates the application mode. | Yes | `production` | Setting to `development` disables strict preflight checks. |
| `PRODUCTION_SAFETY_MODE` | Enforces fail-closed security mechanics. | Yes | `true` | Setting to `false` bypasses CORS and model integrity checks. |
| `ALLOWED_ORIGINS` | JSON list of permitted CORS origins. | Yes | `["https://api.econosphere.ai"]` | Wildcard `["*"]` or empty `[]` will cause container startup to crash in safety mode. |
| `POSTGRES_HOST` | Internal hostname for PostgreSQL. | Yes | `postgres` | Hardcoding external IPs incorrectly routes internal traffic. |
| `REDIS_HOST` | Internal hostname for Redis. | Yes | `redis` | - |
| `NEO4J_URI` | Internal URI for Neo4j. | Yes | `bolt://neo4j:7687` | - |

### ALLOWED_ORIGINS Configuration
The production configuration **MUST NOT** use the wildcard `["*"]` and **MUST NOT** be empty. The verified implementation enforces this.

```json
["https://api.econosphere.ai"]
```

---

## 6. Pre-Deployment Checklist

Before triggering a deployment, operators must verify:

- [ ] Docker daemon is running.
- [ ] Docker Compose is available.
- [ ] Required Phase 11 model exists (`best_t1_gdp_growth_model.joblib`).
- [ ] Required release manifest exists.
- [ ] Required dataset exists (`master_panel.csv`).
- [ ] Production environment variables are configured in `docker-compose.yml`.
- [ ] `ENV` is set to `production`.
- [ ] `PRODUCTION_SAFETY_MODE` is `true` if required by deployment policy.
- [ ] `ALLOWED_ORIGINS` is explicit and non-empty.
- [ ] `ALLOWED_ORIGINS` does not contain wildcard `*`.
- [ ] Models volume is configured read-only (`:ro`).
- [ ] Data volume is configured read-only (`:ro`).
- [ ] `.dockerignore` excludes local `.venv`.
- [ ] No unapproved Candidate B promotion exists.
- [ ] Production schema remains exactly 31 features.

---

## 7. Build Procedure

### Standard Build
For normal operational deployments where the cache is trusted:
```bash
docker compose -f infrastructure/docker/docker-compose.yml build api
```

### Clean Validation Build
If dependency pollution is suspected or a strict verifiable build is required, use `--no-cache`.
This will significantly increase build time as dependencies may need to be re-downloaded.

```bash
docker compose -f infrastructure/docker/docker-compose.yml build --no-cache api
```

---

## 8. Production Startup Procedure

The exact startup workflow using Docker Compose:

1.  **Build services** (as per Section 7).
2.  **Start services** in detached mode:
    ```bash
    docker compose -f infrastructure/docker/docker-compose.yml up -d
    ```
3.  **Check service status**:
    ```bash
    docker compose -f infrastructure/docker/docker-compose.yml ps
    ```
4.  **Inspect logs and wait for cold-start readiness**:
    ```bash
    docker compose -f infrastructure/docker/docker-compose.yml logs -f api
    ```

A container being created does not automatically mean that the application is production-ready. The application is only ready once the readiness endpoint passes.

---

## 9. Post-Deployment Verification Procedure

Verify the following endpoints in sequence:

1.  **Container status**: Checked via `docker compose ps`.
2.  **Infrastructure health**: `/api/v1/health`
    *   Verifies basic connectivity to Postgres, Redis, and Neo4j.
3.  **Model health**: `/api/v1/forecasts/health`
    *   Verifies the internal model registry state.
4.  **Production readiness**: `/api/v1/forecasts/readiness`
    *   **The final gate.** Evaluates all integrity constraints. Must return `ready: true`.
5.  **Model metadata**: `/api/v1/forecasts/model-info`
    *   Verifies the loaded model is the frozen Phase 11 baseline.
6.  **Inference**: `/api/v1/forecasts/gdp`
    *   (See Section 11)

---

## 10. Production Model Verification

Verify the expected production release metadata via the `/api/v1/forecasts/model-info` endpoint.

**Expected Baseline Attributes:**
*   Production Release ID: `ECONOSPHERE-PHASE11-PROD-2026-08-21`
*   Expected Feature Count: `31`
*   Status: `PHASE 11 PRODUCTION FROZEN`

---

## 11. Inference Verification Procedure

To verify live prediction consistency against the approved baseline, execute a request with the known deterministic validation payload.

**Endpoint**: `POST /api/v1/forecasts/gdp`

**Payload**:
```json
{
    "country": "USA",
    "year": 2025,
    "exchange_rate_lcu_usd": 1.0,
    "tariff_rate_pct": 2.5,
    "remittances_usd": 1000000000.0,
    "fdi_net_inflow_usd": 5000000000.0,
    "unemployment_pct": 4.0,
    "imports_pct_gdp": 15.0,
    "tax_revenue_pct_gdp": 20.0,
    "exports_pct_gdp": 12.0,
    "interest_rate_pct": 5.0,
    "reserves_usd": 3000000000.0,
    "current_account_pct_gdp": -2.5,
    "inflation_cpi_pct": 2.5,
    "population_total": 330000000.0,
    "gdp_current_usd": 25000000000000.0,
    "gdp_growth_lag1": 2.0,
    "gdp_growth_lag2": 2.2,
    "gdp_growth_lag3": 1.9,
    "inflation_lag1": 2.4,
    "unemployment_lag1": 4.1,
    "exports_lag1": 11.5,
    "imports_lag1": 14.5,
    "gdp_growth_rolling_mean_3": 2.03,
    "gdp_growth_rolling_std_3": 0.15,
    "gdp_growth_rolling_mean_5": 2.1,
    "inflation_rolling_mean_3": 2.3,
    "trade_openness": 27.0,
    "trade_balance_ratio": 0.8,
    "log_gdp_usd": 30.8,
    "log_population": 19.6,
    "gdp_growth_rolling_std_5": 0.2,
    "inflation_rolling_std_3": 0.1
}
```

**Expected Prediction**:
`2.6995696601100905`

The required comparison tolerance is `<= 1e-12`. This exact value is explicitly tied to the validation payload above from the existing audit implementation.

---

## 12. Security Verification

### Candidate B Isolation
The verified implementation strictly rejects Candidate B features. Requests containing `growth_regime_num` or `stress_regime_num` will be rejected with an `HTTP 422 Unprocessable Entity` response.

### Target Leakage Protection
Requests containing target leakage fields (`target_year`, `target`, `gdp_growth_next_year`) will be rejected with an `HTTP 422 Unprocessable Entity` response.

### Production CORS
*   Explicit allowed origins are enforced.
*   Wildcard origins (`*`) are prohibited in production.
*   Empty origin lists are rejected in production/safety mode.
*   Unauthorized preflight `OPTIONS` requests fail with `HTTP 400 Bad Request` and lack `access-control-allow-origin` headers.

---

## 13. Fail-Closed Behavior

EconoSphere AI embraces a strict fail-closed philosophy. The application refuses to serve traffic when critical validation fails.

**Startup Failures (Container Exits)**:
*   Missing production model file.
*   Artifact hash mismatch during preflight.
*   Invalid production CORS configuration (e.g., wildcard).

**Runtime Validation Failures (HTTP 422/400/500)**:
*   Candidate B schema contamination.
*   Target leakage fields present.
*   Schema feature-count violation.
*   CORS origin mismatch.

---

## 14. Monitoring and Operational Checks

Minimum recommended operational checks (no external platform required):

*   `docker compose ps` (Check container state)
*   `docker compose logs api` (Check for preflight logs)
*   `/api/v1/health` (Database connectivity)
*   `/api/v1/forecasts/readiness` (Preflight integrity gate)
*   `/api/v1/forecasts/health` (Model state)

**Structured Lifecycle Events (in logs):**
*   `application_startup_attempt`
*   `production_preflight_success`
*   `production_preflight_failure`
*   `application_shutdown`

---

## 15. Troubleshooting Guide

| Symptom | Likely Cause | Diagnostic Command | Safe Resolution |
| :--- | :--- | :--- | :--- |
| Cannot connect to Docker daemon | Docker Engine not running | `docker version` | Start Docker Desktop or daemon service. |
| API container exits immediately | Invalid CORS, missing model, artifact hash mismatch, or failed preflight | `docker compose logs api` | Verify artifacts exist and hashes match. Fix CORS configuration. Do not bypass checks. |
| API remains unhealthy during startup | Waiting for databases (cold start) | `docker compose logs api` | Wait for dependencies to start. Do not disable readiness checks. |
| `.venv` cross-platform issues | Local `.venv` leaked into build context | `cat apps/api/.dockerignore` | Ensure `.dockerignore` excludes `.venv`. Run a clean build (`--no-cache`). |
| Port 8000 already in use | Another service is using port 8000 | Check active ports | Stop conflicting service. Do not modify production compose ports. |
| Prediction differs from baseline | Incorrect schema, unauthorized model change, or incorrect validation payload | Compare prediction payload | Confirm exact validation payload and Phase 11 release ID. Do not retrain the model. |
| CORS request blocked | Origin not explicitly allowed in configuration | Check `ALLOWED_ORIGINS` | Update `ALLOWED_ORIGINS` with the required explicit URI. Do not use wildcard. |

---

## 16. Safe Shutdown Procedure

To safely stop the infrastructure and clean up network resources without destroying data:

```bash
docker compose -f infrastructure/docker/docker-compose.yml down --remove-orphans
```

*   **Graceful stopping**: Sends SIGTERM to containers.
*   **Container removal**: Removes container instances and default networks.
*   **What is NOT deleted**: Named volumes (`postgres_data`, `redis_data`, `neo4j_data`) and read-only host mounts (`models`, `data`) remain intact.

---

## 17. Recovery Procedure

If the API fails to start or serve traffic:

1.  Stop the failed stack: `docker compose -f infrastructure/docker/docker-compose.yml down`
2.  Inspect API logs: `docker compose -f infrastructure/docker/docker-compose.yml logs api`
3.  Identify the failure category (e.g., hash mismatch, CORS error).
4.  Verify environment configuration (`docker-compose.yml`).
5.  Verify required artifact existence.
6.  Verify artifact hashes against the approved Phase 11 baseline.
7.  Verify Compose read-only mounts are intact.
8.  Correct only configuration or infrastructure issues.
9.  Restart the stack.
10. Re-run health and readiness checks.
11. Run deterministic validation if appropriate.

**EXPLICITLY PROHIBITED RECOVERY ACTIONS:**
*   Retraining the model as a recovery mechanism.
*   Replacing the model with Candidate B.
*   Regenerating the release manifest without authorization.
*   Ignoring or bypassing hash mismatches.
*   Disabling production safety mode.
*   Changing wildcard CORS to bypass validation.

---

## 18. Artifact Integrity Audit Procedure

Operators can safely audit protected artifacts using a read-only script. The provided script calculates MD5 hashes of the production artifacts and compares them without modifying files.

**Execute Audit:**
```bash
python -X utf8 apps/api/ml/audit_phase14_step3_operational_runbook.py
```

This verifies the integrity of:
*   `best_t1_gdp_growth_model.joblib`
*   `phase11_production_release_manifest.json`
*   `master_panel.csv`

---

## 19. Final Production Readiness Checklist

The final operator verification before declaring the deployment operational:

- [ ] Docker daemon operational.
- [ ] Docker Compose configuration valid.
- [ ] Production environment enabled (`ENV=production`).
- [ ] Explicit CORS origins configured.
- [ ] No wildcard CORS (`*`).
- [ ] API image built successfully.
- [ ] All required services running (API, Postgres, Neo4j, Redis).
- [ ] Infrastructure health passed (`/health`).
- [ ] Production model health passed (`/forecasts/health`).
- [ ] Production readiness passed (`/forecasts/readiness`).
- [ ] Production release ID verified (`ECONOSPHERE-PHASE11-PROD-2026-08-21`).
- [ ] Production schema verified as exactly 31 features.
- [ ] Inference endpoint verified against 1e-12 tolerance.
- [ ] Candidate B features rejected (HTTP 422).
- [ ] Target leakage fields rejected (HTTP 422).
- [ ] Artifact integrity verified.
- [ ] Read-only model mount verified (`:ro`).
- [ ] Read-only data mount verified (`:ro`).
- [ ] Protected artifact hashes unchanged.
- [ ] Graceful shutdown procedure documented and verified.
