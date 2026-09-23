# Phase 14 Step 3: Production Deployment Quickstart Guide

This quickstart is a fast-path deployment guide for standard operations. For full details, see the complete [Production Deployment Runbook](step3_production_deployment_runbook.md).

## 1. Prerequisites
*   Docker Desktop / Daemon is running.
*   Docker Compose is available.
*   You are in the repository root directory.

## 2. Verify Protected Artifacts
Ensure the frozen Phase 11 production artifacts exist:
```text
models/phase11/best_t1_gdp_growth_model.joblib
models/phase11/phase11_production_release_manifest.json
data/raw/master_panel.csv
```
> [!IMPORTANT]
> Do not modify these files. They are protected and immutable.

## 3. Configure Production Environment
Verify `infrastructure/docker/docker-compose.yml` has the following in the `api` service:
```yaml
    environment:
      ENV: "production"
      PRODUCTION_SAFETY_MODE: "true"
      ALLOWED_ORIGINS: '["https://api.econosphere.ai"]' # NO WILDCARDS ALLOWED
```

## 4. Build
Build the API image. (Use `--no-cache` only if a clean validation build is explicitly required).
```bash
docker compose -f infrastructure/docker/docker-compose.yml build api
```

## 5. Start
Start the container stack in detached mode:
```bash
docker compose -f infrastructure/docker/docker-compose.yml up -d
```

## 6. Check Containers
Verify all services (`api`, `postgres`, `redis`, `neo4j`) are running:
```bash
docker compose -f infrastructure/docker/docker-compose.yml ps
```

## 7. Check Health
Verify database connectivity:
```bash
curl http://localhost:8000/api/v1/health
```

## 8. Check Readiness
Verify the strict production preflight gates pass (this may take a few seconds on cold start):
```bash
curl http://localhost:8000/api/v1/forecasts/readiness
```
> [!NOTE]
> Must return `{"ready": true, ...}` before proceeding.

## 9. Verify Model
Confirm the correct Phase 11 production baseline is loaded:
```bash
curl http://localhost:8000/api/v1/forecasts/model-info
```

## 10. Stop Safely
Gracefully shut down the stack when required, leaving data volumes intact:
```bash
docker compose -f infrastructure/docker/docker-compose.yml down --remove-orphans
```
