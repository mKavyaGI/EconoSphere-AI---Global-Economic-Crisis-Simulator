# Phase 20 — Rollback Plan

## Current Production
```
Model:    HistGradientBoostingRegressor (Phase 11)
Artifact: models/phase11/best_t1_gdp_growth_model.joblib
Status:   FROZEN — MD5 and SHA256 verified before/after Phase 20
```

## Candidate
```
Model:    A5a Equal-Weight Ensemble
Status:   EXPERIMENTAL_ONLY
Artifacts: apps/api/ml/phase20/artifacts/phase20_*_EXPERIMENTAL_ONLY.joblib
```

## Rollback Mechanism
Phase 11 is never overwritten. If A5a is deployed and needs to be rolled back:

1. Stop serving A5a ensemble
2. Reload `models/phase11/best_t1_gdp_growth_model.joblib`
3. Verify MD5 matches Phase 11 manifest value
4. Resume serving Phase 11

No database migration, no feature store change, no schema change is required.
The rollback is instantaneous.

## Design Principle
A5a is deployed ALONGSIDE Phase 11 (not instead of it) until a formal release
decision is made. If A5a fails in production, Phase 11 is always available.

## Gate 10 Implication
Rollback safety is a mandatory gate. If the Phase 11 artifact has been
accidentally mutated, Gate 2 will fail and EXPERIMENT_FAILED_GOVERNANCE
is declared before any deployment.
