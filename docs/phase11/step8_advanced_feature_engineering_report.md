# Phase 11 Step 8: Advanced Feature Engineering Report

## 1. Objective
Determine whether additional economic features and stronger model configurations can improve T+1 GDP growth forecasting compared with the verified Step 7 baseline.

## 2. Existing Step 7 baseline
- Best Validation Model: HistGradientBoosting
- Validation RMSE: 8.3342
- Test RMSE: 4.0176

## 3. Files inspected
- `apps/api/ml/train_t1_baseline.py`
- `apps/api/ml/forecasting_target.py`
- `apps/api/ml/data_preprocessing.py`
- `data/processed/master_panel_forecasting.csv`

## 4. Feature inventory & 5. New features
- Baseline: 29 features
- Added Momentum: `gdp_growth_momentum_1_2`, `inflation_momentum_1_2`, `unemployment_momentum_1_2`, `exports_momentum`, `imports_momentum`
- Added Rolling: `gdp_growth_rolling_std_5`, `inflation_rolling_std_3`
- Added Country/Global: `country_historical_avg_gdp_growth`, `country_historical_avg_inflation`, `global_gdp_growth_lag1`, `global_inflation_lag1`

## 6. Mathematical definitions
- Momentum: `feature_lag1` - `feature_lag2`
- Rolling Std: `shift(1).rolling(window=X, min_periods=2).std()`

## 7. Leakage checks
Tested automatically using pytest suite. Passed all checks including target leakage, raw dataset mutation, and country aggregate bounds.

## 8. Dataset dimensions
- Rows: 6,084
- Columns: 52

## 9. Feature experiments & 10. Model comparison
- Exp A (Baseline): Val RMSE = 8.4128
- Exp B (+Momentum): Val RMSE = 8.4358
- Exp C (+Rolling): Val RMSE = 8.3427 (BEST)
- Exp D (+Country/Global): Val RMSE = 8.6690
- Exp E (All Advanced): Val RMSE = 8.6584

Best Model on Exp C:
- HistGradientBoosting: Val RMSE = 8.3087
- RandomForest: Val RMSE = 8.6099
- ExtraTrees: Val RMSE = 8.5035

## 11. Hyperparameter optimization
- Tuned HistGradientBoosting using Time-Aware Validations.
- Best Tuned Val RMSE: 8.2699

## 12. Validation results & 13. Test results
- Val RMSE: 8.2699
- Test RMSE: 4.0045

## 14. Baseline vs improved comparison
- Baseline Test RMSE: 4.0176
- Improved Test RMSE: 4.0045
- Improvement: 0.326%

## 15. Five-country analysis
| Country | Year | Actual Growth | Predicted | Error |
|---|---|---|---|---|
| IND | 2024 | 7.567% | 6.950% | -0.617% |
| CHN | 2024 | 4.960% | 5.275% | 0.315% |
| USA | 2024 | 2.161% | 2.500% | 0.338% |
| JPN | 2024 | 1.193% | 2.425% | 1.232% |
| GBR | 2024 | 1.388% | 1.483% | 0.095% |

## 16. Feature importance (Permutation Importance)
| Feature | Importance |
|---|---|
| gdp_current_usd | 0.0866 |
| reserves_usd | 0.0327 |
| gdp_growth_lag1 | 0.0253 |
| trade_balance_ratio | 0.0176 |
| inflation_cpi_pct | 0.0157 |
| exchange_rate_lcu_usd | 0.0136 |
| population_total | 0.0136 |
| fdi_net_inflow_usd | 0.0117 |
| gdp_growth_rolling_std_5 | 0.0114 |
| current_account_pct_gdp | 0.0112 |
*Note: Feature importance is NOT proof of causality.*

## 17. 2026 forecasts
* **IND**: 6.802%
* **CHN**: 4.708%
* **USA**: 2.817%
* **JPN**: 2.305%
* **GBR**: 1.411%

## 18. Limitations
- Macro shocks (like COVID-19 or wars) remain difficult to capture purely through momentum/rolling statistics.

## 19. Reproducibility information
- Script: `apps/api/ml/train_t1_advanced.py`

## 20. Final recommendation
IMPROVED. The model demonstrates a 0.326% improvement on the test set while safely adhering to the strict T+1 framework.
