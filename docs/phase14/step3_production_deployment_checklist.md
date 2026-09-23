# Phase 14 Step 3: Production Deployment Checklist

This operator checklist ensures strict adherence to EconoSphere AI production safety and governance policies. Complete all relevant sections for a deployment.

## PRE-DEPLOYMENT
- [ ] Docker daemon is running and accessible.
- [ ] Docker Compose is available.
- [ ] Required Phase 11 model exists: `models/phase11/best_t1_gdp_growth_model.joblib`.
- [ ] Required release manifest exists: `models/phase11/phase11_production_release_manifest.json`.
- [ ] Required dataset exists: `data/raw/master_panel.csv`.
- [ ] `infrastructure/docker/docker-compose.yml` has `ENV` set to `production`.
- [ ] `infrastructure/docker/docker-compose.yml` has `PRODUCTION_SAFETY_MODE` set to `true`.
- [ ] `ALLOWED_ORIGINS` is explicitly configured (e.g., `["https://api.econosphere.ai"]`).
- [ ] `ALLOWED_ORIGINS` **does not** contain the wildcard `*`.
- [ ] `apps/api/.dockerignore` correctly excludes `.venv`.

## DEPLOYMENT
- [ ] API image built successfully (`docker compose build api`).
- [ ] Stack started successfully (`docker compose up -d`).
- [ ] Containers verified as running (`docker compose ps`).
- [ ] Cold-start dependencies are healthy (Postgres, Redis, Neo4j).

## POST-DEPLOYMENT
- [ ] Infrastructure health endpoint returns OK (`/api/v1/health`).
- [ ] Production model health endpoint returns OK (`/api/v1/forecasts/health`).
- [ ] Production readiness gate passes (`/api/v1/forecasts/readiness` -> `ready: true`).
- [ ] Production release ID verified as `ECONOSPHERE-PHASE11-PROD-2026-08-21` (`/api/v1/forecasts/model-info`).
- [ ] Expected Feature Count verified as 31 (`/api/v1/forecasts/model-info`).
- [ ] Test inference succeeds and produces expected deterministic result.

## SECURITY & ISOLATION
- [ ] Candidate B `growth_regime_num` feature correctly rejected (HTTP 422).
- [ ] Candidate B `stress_regime_num` feature correctly rejected (HTTP 422).
- [ ] Target leakage field `target_year` correctly rejected (HTTP 422).
- [ ] Target leakage field `gdp_growth_next_year` correctly rejected (HTTP 422).

## IMMUTABILITY
- [ ] `models/` volume is mounted read-only (`:ro`).
- [ ] `data/` volume is mounted read-only (`:ro`).
- [ ] MD5/SHA-256 hashes of the model, dataset, and manifest match the approved baseline.

## SHUTDOWN
- [ ] Stack stopped gracefully (`docker compose down --remove-orphans`).
- [ ] Read-only artifacts remain unchanged post-shutdown.

## INCIDENT RECOVERY
- [ ] API logs inspected for root cause (e.g., hash mismatch, CORS failure).
- [ ] Configuration corrected without disabling safety mode.
- [ ] No unauthorized retraining or model replacement performed.
