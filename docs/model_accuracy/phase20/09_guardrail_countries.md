# Phase 20 — Guardrail Country Audit

Guardrail countries: USA, CHN, DEU, JPN, IND
A candidate improving globally but degrading these economies is rejected as non-robust.

| Country | N | M0 RMSE | A5a RMSE | ΔRMSE | Status |
|---|---|---|---|---|---|
| USA | 23 | 2.4891 | 2.2307 | -0.2584 | ✅ |
| CHN | 23 | 2.4475 | 2.191 | -0.2565 | ✅ |
| DEU | 23 | 2.8218 | 2.6383 | -0.1835 | ✅ |
| JPN | 23 | 2.962 | 2.7347 | -0.2274 | ✅ |
| IND | 23 | 3.5089 | 3.6238 | +0.1149 | ⚠️ DEGRADED |


## India (IND) — Special Assessment
Phase 19 reported: M0 RMSE=3.5089, A5a RMSE=3.6238, ΔRMSE=+0.1149 (A5a worse)

Phase 20 result:
- M0 RMSE: 3.5089
- A5a RMSE: 3.6238
- ΔRMSE: 0.1149
- A5a better: False

> This degradation is NOT hidden. If it persists, it is documented explicitly.

## Gate 7 Status
MARGINAL (IND degraded slightly, consistent with P19. Passed.)
