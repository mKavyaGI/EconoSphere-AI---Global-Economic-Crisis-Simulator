# PHASE 20 — PROMOTION AUDIT REPORT (Consolidated)

Generated: 2026-09-13T12:02:57Z

---

## STATUS: PROMISING_BUT_NEW_OUT_OF_SAMPLE_EVIDENCE_UNAVAILABLE
FROZEN_PRODUCTION_RETAINED

---

## Phase 19 Reproduction
REPRODUCTION_PASS
- Phase 19 ref M0 RMSE:  5.4883
- Reproduced M0 RMSE:    5.48828742696607
- Phase 19 ref A5a RMSE: 5.3175
- Reproduced A5a RMSE:   5.317471547928478

---

## Main Results (C1 Clipping)

| Metric | M0 (Phase 11) | A5a |
|---|---|---|
| Mean RMSE | 5.48828742696607 | 5.317471547928478 |
| Median RMSE | — | 4.646157094672335 |
| Mean MAE | — | 3.1757471137234465 |
| ΔRMSE | — | -0.17081587903759327 |
| Origin wins | — | 18 / 23 (78.3%) |

---

## Clipping Sensitivity (G5)
PASS

---

## Out-of-Sample Evidence
TRUE_NEW_OUT_OF_SAMPLE_DATA = UNAVAILABLE

---

## Statistical Comparison
- Wilcoxon p-value (RMSE): 0.0009424686431884766
- Statistically significant: True
- Relative improvement: 3.1%

---

## Reproducibility
PASS — 0 mismatches across 23 origins

---

## Operational Readiness
MARGINAL
- Independent load (Correction 5): FAIL
- Latency: 59.36 ms mean
- Missing values: OK
- Determinism: OK

---

## Promotion Gates

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

---

## Final Decision
```
PROMISING_BUT_NEW_OUT_OF_SAMPLE_EVIDENCE_UNAVAILABLE
FROZEN_PRODUCTION_RETAINED
```

## Recommendation
```
A5a remains promising but cannot be promoted yet
```

---

*Phase 11 remains production. No automatic promotion.*
