# Phase 12 Step 3: Forward-Looking Economic Indicators & Recession Signal Experiment

## 1. Objective
Investigate whether genuinely forward-looking macroeconomic indicators can improve EconoSphere AI's ability to forecast next-year GDP growth, particularly during recessionary periods, without compromising chronological discipline or model stability.

## 2. Phase 11 Control Definition
The Phase 11 model is a strict `HistGradientBoostingRegressor` evaluated chronologically:
- **Validation RMSE**: 8.2853
- **Test RMSE**: 3.9113
- **Walk-forward RMSE**: 6.0967

## 3. Data Availability Audit
A dataset availability audit was conducted to identify forward-looking macroeconomic indicators:
- **Available**: `inflation_cpi_pct`, `interest_rate_pct`, `unemployment_pct`, `exports_pct_gdp`, `imports_pct_gdp`, `exchange_rate_lcu_usd`, `reserves_usd`, `fdi_net_inflow_usd`, `govt_debt_pct_gdp`.
- **Temporally Safe (Category A)**: `inflation_cpi_pct`, `interest_rate_pct`, `unemployment_pct`, `exports_pct_gdp`, `imports_pct_gdp`, `exchange_rate_lcu_usd`, `reserves_usd`. (Published immediately or with minimal delay, safe to use early in Year t+1 for predicting Year t+1).
- **Rejected (Category B)**: `fdi_net_inflow_usd`, `govt_debt_pct_gdp` (Published with significant lag and subject to heavy revisions).

## 4. Temporal Availability Assumptions
- Indicators sourced for Year $t$ are assumed to be finalized or robustly estimated by early Year $t+1$. 
- No information from Year $t+1$ was permitted to leak into the prediction features for Year $t+1$.

## 5. Feature Definitions
- **Macro Changes**: Year-over-year changes (e.g., `inflation_change = inflation(t) - inflation(t-1)`).
- **Macro Stress**: Threshold-based stress indicators (e.g., `unemployment_deterioration` where unemployment change > 0.5%).

## 6. Experiments A–F
- **A. Control**: Phase 11 feature set (31 features).
- **C. Macro Changes**: Control + 4 YoY change features.
- **D. Macro Stress**: Control + 4 stress indicator features.
- **E. Forward + Momentum**: Control + Macro Changes + Macro Stress + GDP growth change.
- **F. Minimal Signal**: Control + compact subset of engineered signals.

## 7. Results
| Experiment | Val RMSE | Test RMSE | Val Rec RMSE |
| :--- | :--- | :--- | :--- |
| A. Control | 8.2853 | 3.9113 | 11.5653 |
| C. Macro Changes | 8.3672 | 3.8809 | 11.6638 |
| D. Macro Stress | 8.3603 | 3.9374 | 11.5771 |
| E. Forward + Momentum | 8.3074 | 3.7214 | 11.4088 |
| F. Minimal Signal | 8.3635 | 3.9457 | 11.6511 |

## 8. Failure Modes & Conclusion
None of the candidate experiments managed to surpass the Phase 11 Control on Validation RMSE. Even though Experiment E achieved a lower Test RMSE (3.7214 vs 3.9113), promoting it would violate our strict chronological evaluation rules (Criterion 1), risking overfitting. 

**Diagnostic importance does not imply causality**, but feature analysis indicates that while some macro stress signals slightly improved the final test split, they lacked robustness across the broader 2019-2022 validation window.

## 9. Promotion Decision
**Decision**: KEEP PHASE 11 CONTROL.

## 10. Recommendation for Phase 12 Step 4
Given that external forward-looking indicators and backward-looking momentum have both failed to predict sudden regime shifts, the next step must address the model's structural ability to adapt to distinct growth environments. We recommend exploring regime-switching models or conditional learning techniques that explicitly treat shock/recession regimes separately from normal growth regimes.
