# Phase 21 — Promotion Gate Results

Generated: 2026-09-16T15:59:01.635412+00:00

## Gate Summary

| Gate | Status |
|---|---|
| G1 | PASS |
| G2 | PASS |
| G3 | PASS |
| G4 | INCONCLUSIVE (No genuine new OOS data found by programmatic discovery) |
| G5 | PASS |
| G6 | INCONCLUSIVE (No OOS data — cannot evaluate) |
| G7 | INCONCLUSIVE (No OOS data) |
| G8 | INCONCLUSIVE (No OOS data) |
| G9 | INCONCLUSIVE (No OOS data) |
| G10 | INCONCLUSIVE (No OOS data) |
| G11 | PASS |
| G12 | INCONCLUSIVE — Mandatory gates inconclusive: ['G4', 'G6', 'G7', 'G8', 'G9'] |

## Notes
None

## Gate Definitions (Phase 21)
| Gate | Requirement |
|---|---|
| G1 | Production artifact hashes unchanged |
| G2 | Model and manifest hashes unchanged |
| G3 | All leakage rules L1-L12 pass |
| G4 | OOS data is genuinely new and authenticated |
| G5 | Reproducibility: Run1 == Run2 (atol=1e-10) |
| G6 | A5a RMSE < Phase 11 RMSE on OOS data |
| G7 | No unacceptable MAE/tail error increase |
| G8 | No guardrail country exceeds ΔRMSE tolerance |
| G9 | Priority countries not systematically degraded |
| G10 | Statistical and practical evidence supports promotion |
| G11 | Operational safety (artifact load, schema, NaN, determinism) |
| G12 | Final promotion decision (all G1-G11 considered) |

## Promotion Thresholds Applied
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
