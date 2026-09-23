# Phase 18 Step 4: Independent Structural Feature-Reduction Robustness

## Models Evaluated
- Phase 11 baseline (M0)
- A1 Historical-Structure (10 features)

## Backtest origins
2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024

## RMSE Summary
- Mean ΔRMSE (A1 - M0): +0.0101

## Win rate
- A1 wins: 3
- Phase 11 wins: 7
- Ties: 0

## Governance

> [!CAUTION]
> **Final Decision:** `A1_NOT_ROBUST` | `FROZEN_PRODUCTION_RETAINED`
> 
> The Phase 11 model remains frozen and immutable. The A1 model's performance improvement in Step 3 did not generalize robustly across multiple expanding-window historical origins. Phase 11 won 7 out of 10 periods. No production files were mutated. All leakage checks passed. All targeted and full regression tests passed successfully.
