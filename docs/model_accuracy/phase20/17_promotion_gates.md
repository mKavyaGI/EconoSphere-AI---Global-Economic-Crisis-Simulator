# Phase 20 — Promotion Gates

A5a can only be considered for production if ALL mandatory gates pass.

| Gate | Status |
|---|---|
| G1 | ✅ PASS |
| G2 | ✅ PASS |
| G3 | ✅ PASS |
| G4 | ✅ PASS |
| G5 | ✅ PASS |
| G6 | ✅ PASS |
| G7 | ⚠️ MARGINAL (IND degraded slightly, consistent with P19. Passed.) |
| G8 | ⚠️ MARGINAL (Degraded: ['BRA']) |
| G9 | ✅ PASS |
| G10 | ⚠️ MARGINAL |
| G11 | ❌ INCONCLUSIVE (New data unavailable) |


## Gate Definitions

| Gate | Requirement |
|---|---|
| G1 | Dataset MD5/SHA256 unchanged |
| G2 | Production model+manifest unchanged after experiment |
| G3 | All leakage rules L1–L12 pass |
| G4 | Reproducibility: Run1 == Run2 (atol=1e-10) |
| G5 | A5a advantage not solely an artifact of clipping choice |
| G6 | Broad improvement (not concentrated in few years) |
| G7 | No serious guardrail degradation (IND assessed explicitly) |
| G8 | No systematic priority country failure |
| G9 | No unacceptable increase in extreme errors (p95/max AE) |
| G10 | Operational readiness + rollback-safe |
| G11 | New OOS evidence (INCONCLUSIVE if no new data) |
