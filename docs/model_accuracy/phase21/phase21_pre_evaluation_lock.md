# Phase 21 — Pre-Evaluation Lock

Generated: 2026-09-16T15:59:01.635412+00:00

## Lock Purpose
This document records the complete frozen experimental state BEFORE any OOS target
value was read. The lock was created in Stage A and cannot be modified by OOS discovery.

**Lock SHA256:** `dbd94b58be1d86baa66678cef4fc78c6a89817af15202dd6b107488568cc6317`
**Lock Timestamp:** `2026-09-16T15:59:01.686472+00:00`
**Git Commit:** `9d312bc4fe50cfd60e3dc30213ccda6f952407b9`

## Production Model
- Path: `models/phase11/best_t1_gdp_growth_model.joblib`
- Status: FROZEN — Phase 11 production

## Candidate Model
- Architecture: `Equal-weight ensemble: (M0 + Ridge + RandomForest) / 3`
- Status: EXPERIMENTAL_ONLY

## Frozen Feature Set
- Feature count: `31`
- Manifest feature count: `31`
- Lists match: `True`

## Ensemble Weights (frozen)
```
M0      = 1/3 = 0.3333...
Ridge   = 1/3 = 0.3333...
RF      = 1/3 = 0.3333...
Formula = A5a = (M0 + Ridge + RF) / 3
```

## Clipping Policy (frozen)
```
Ridge clip low  = -40.0
Ridge clip high = 60.0
M0  clipped     = False
RF  clipped     = False
```

## Promotion Thresholds (frozen — cannot change after OOS labels revealed)
```json
{
  "min_rmse_improvement_pct": 1.0,
  "max_guardrail_delta_rmse": 0.5,
  "max_priority_degraded": 1,
  "max_p95_ae_increase": 1.0,
  "ind_max_delta_rmse": 0.2,
  "min_oos_observations": 10
}
```

## Governance Rules Applied
- Lock does NOT contain any OOS-derived metric
- OOS discovery cannot modify this lock
- Post-hoc tuning is prohibited after OOS reveal
- No automatic promotion regardless of result
