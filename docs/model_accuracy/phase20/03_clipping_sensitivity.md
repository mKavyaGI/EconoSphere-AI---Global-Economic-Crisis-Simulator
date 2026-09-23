# Phase 20 — Clipping Sensitivity Analysis

## Configurations
| Config | Description |
|---|---|
| C0 | No clipping — raw linear predictions |
| C1 | Phase 19 bounds [-40%, +60%] |
| C2 | Alternative bounds [-50%, +75%] |

> Configurations are pre-specified. No test-set clipping selection.

## Results

| Config | Label | A5a RMSE | M0 RMSE | ΔRMSE | A5a Wins |
|---|---|---|---|---|---|
| C0 | No Clipping | 123.07428706349644 | 5.48828742696607 | +117.5860 | 17 |
| C1 | Phase 19 Clipping [-40, +60] | 5.317471547928478 | 5.48828742696607 | -0.1708 | 18 |
| C2 | Alternative Bounds [-50, +75] | 5.318332453766746 | 5.48828742696607 | -0.1700 | 18 |


## Gate 5 Assessment
PASS

## Interpretation
C0 is expected to show degraded performance for linear models due to known
extrapolation behavior (extreme exchange rates, micro-state GDP swings).
The key question is whether the A5a advantage under C1 is an artifact of
the clipping boundary or a genuine architectural benefit.

> A5a improvement is not solely an artifact of clipping if it survives C1 and C2.
