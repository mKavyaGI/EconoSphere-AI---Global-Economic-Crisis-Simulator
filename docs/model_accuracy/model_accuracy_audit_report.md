# EconoSphere AI: Production Model Accuracy Audit Report

**Audit Date:** 2026-09-01 10:55:13
**Target Model:** `best_t1_gdp_growth_model.joblib`
**Manifest ID:** `ECONOSPHERE-PHASE11-PROD-2026-08-21`

## 1. Executive Summary
The frozen Phase 11 production model underwent a deep, evidence-based accuracy audit. All artifacts remained completely unmodified. The audit successfully reconstructed the exact training splits and found that the model fails to predict high-variance economic shocks (like COVID-19).

## 2. Model Evaluation (Reproduced)
*Note: Evaluated strictly on the recreated dataset splits from Phase 11.*

### Production Model
| Metric | Train (2000-2018) | Validation (2019-2022) | Test (2023-2024) |
|---|---|---|---|
| RMSE | 3.3072 | 8.2853 | 3.9113 |
| MAE | 1.9661 | 5.1732 | 2.2886 |
| R² | 0.6175 | 0.0260 | 0.0084 |
| Bias | 0.0000 | 0.7612 | 0.2641 |

### Baselines (Test Set)
| Baseline | RMSE | MAE | R² |
|---|---|---|---|
| Production Model | 3.9113 | 2.2886 | 0.0084 |
| Dummy Mean Regressor | 3.9323 | 2.2655 | -0.0023 |
| Naive Last-Year GDP | 5.0193 | 2.0690 | -0.6329 |

## 3. Country-wise Error Analysis (Test Set 2023-2024)

### Best Performing Countries (Lowest MAE)
| country_code | Observations | MAE | RMSE | Bias |
| --- | --- | --- | --- | --- |
| THA | 2.0000 | 0.0459 | 0.0492 | 0.0177 |
| POL | 2.0000 | 0.1183 | 0.1413 | -0.0773 |
| COG | 2.0000 | 0.1187 | 0.1402 | 0.0747 |
| GMB | 2.0000 | 0.1395 | 0.1910 | -0.1395 |
| KEN | 2.0000 | 0.1912 | 0.2509 | 0.1625 |
| BHS | 1.0000 | 0.1940 | 0.1940 | 0.1940 |
| SWE | 2.0000 | 0.2531 | 0.2862 | 0.2531 |
| GTM | 2.0000 | 0.3131 | 0.3998 | 0.3131 |
| SLB | 2.0000 | 0.3433 | 0.3681 | 0.3433 |
| IND | 2.0000 | 0.3439 | 0.4751 | -0.3277 |

### Worst Performing Countries (Highest MAE)
| country_code | Observations | MAE | RMSE | Bias |
| --- | --- | --- | --- | --- |
| GUY | 2.0000 | 28.4861 | 31.2961 | -28.4861 |
| NCL | 1.0000 | 15.3724 | 15.3724 | 15.3724 |
| PSE | 2.0000 | 15.3186 | 18.3700 | 10.1390 |
| LBN | 1.0000 | 8.9933 | 8.9933 | 8.9933 |
| SDN | 2.0000 | 7.5004 | 8.5454 | 4.0949 |
| VEN | 2.0000 | 7.2035 | 7.5180 | -7.2035 |
| BWA | 2.0000 | 6.9734 | 7.1756 | 6.9734 |
| TLS | 2.0000 | 6.8760 | 9.5483 | 6.6249 |
| KGZ | 2.0000 | 6.5943 | 6.5943 | -6.5943 |
| PLW | 2.0000 | 6.0260 | 7.1565 | -6.0260 |

## 4. Year-wise Error Analysis (Validation + Test)
| target_year | Observations | MAE | RMSE | Bias |
| --- | --- | --- | --- | --- |
| 2020.0000 | 209.0000 | 9.0854 | 11.6646 | 8.3419 |
| 2021.0000 | 209.0000 | 4.5928 | 6.6609 | -2.7061 |
| 2022.0000 | 208.0000 | 4.1213 | 7.0877 | -1.7994 |
| 2023.0000 | 203.0000 | 2.8206 | 6.5648 | -0.8504 |
| 2024.0000 | 199.0000 | 2.5504 | 4.7175 | 0.4196 |
| 2025.0000 | 185.0000 | 2.0070 | 2.7955 | 0.0968 |

## 5. Residual Distribution (Validation + Test)
| Metric | Value |
|---|---|
| Mean Residual | 0.6038 |
| Median Residual | 0.1093 |
| Std Deviation | 7.1689 |
| Max Positive Residual | 55.7243 |
| Min Negative Residual | -69.8770 |
| Max Absolute Error | 69.8770 |

## 6. Raw Data Quality Audit
| Metric | Value |
|---|---|
| Total Raw Observations | 6136 |
| Missing Targets | 340 |
| Duplicate Rows | 26 |

## 7. API Payload vs Historical Training Distribution
The live Phase 14 Step 2 API payload was compared against the training set distributions:
| Feature | API Payload Value | Train Mean | Train Std | Z-Score | Status |
| --- | --- | --- | --- | --- | --- |
| exchange_rate_lcu_usd | 1.0000 | 10994422.0527 | 227610466.2449 | -0.0483 | IN DISTRIBUTION |
| tariff_rate_pct | 2.5000 | 6.9974 | 8.3825 | -0.5365 | IN DISTRIBUTION |
| remittances_usd | 1000000000.0000 | 1926142970.3625 | 5119827162.5526 | -0.1809 | IN DISTRIBUTION |
| fdi_net_inflow_usd | 5000000000.0000 | 8746831734.9033 | 35773240040.8248 | -0.1047 | IN DISTRIBUTION |
| unemployment_pct | 4.0000 | 7.9265 | 5.6526 | -0.6946 | IN DISTRIBUTION |
| imports_pct_gdp | 15.0000 | 47.7916 | 28.8514 | -1.1366 | IN DISTRIBUTION |
| tax_revenue_pct_gdp | 20.0000 | 16.1700 | 8.1969 | 0.4672 | IN DISTRIBUTION |
| exports_pct_gdp | 12.0000 | 42.2283 | 31.4711 | -0.9605 | IN DISTRIBUTION |
| interest_rate_pct | 5.0000 | 5.8139 | 7.0305 | -0.1158 | IN DISTRIBUTION |
| reserves_usd | 3000000000.0000 | 40399566027.2218 | 197899439914.2537 | -0.1890 | IN DISTRIBUTION |
| current_account_pct_gdp | -2.5000 | -1.9002 | 15.4357 | -0.0389 | IN DISTRIBUTION |
| inflation_cpi_pct | 2.5000 | 6.0859 | 15.6317 | -0.2294 | IN DISTRIBUTION |
| population_total | 330000000.0000 | 33236462.4743 | 130173441.7647 | 2.2798 | IN DISTRIBUTION |
| gdp_current_usd | 25000000000000.0000 | 295420778804.9080 | 1295171971711.6421 | 19.0744 | OUT OF DISTRIBUTION |
| gdp_growth_lag1 | 2.0000 | 3.6850 | 5.5003 | -0.3063 | IN DISTRIBUTION |
| gdp_growth_lag2 | 2.2000 | 3.6922 | 5.4281 | -0.2749 | IN DISTRIBUTION |
| gdp_growth_lag3 | 1.9000 | 3.7592 | 5.4072 | -0.3438 | IN DISTRIBUTION |
| inflation_lag1 | 2.4000 | 6.1234 | 15.4446 | -0.2411 | IN DISTRIBUTION |
| unemployment_lag1 | 4.1000 | 7.9656 | 5.6667 | -0.6822 | IN DISTRIBUTION |
| exports_lag1 | 11.5000 | 42.2750 | 31.6450 | -0.9725 | IN DISTRIBUTION |
| imports_lag1 | 14.5000 | 47.8631 | 29.0798 | -1.1473 | IN DISTRIBUTION |
| gdp_growth_rolling_mean_3 | 2.0300 | 3.7011 | 3.9264 | -0.4256 | IN DISTRIBUTION |
| gdp_growth_rolling_std_3 | 0.1500 | 2.7605 | 3.7686 | -0.6927 | IN DISTRIBUTION |
| gdp_growth_rolling_mean_5 | 2.1000 | 3.7489 | 3.5404 | -0.4657 | IN DISTRIBUTION |
| inflation_rolling_mean_3 | 2.3000 | 6.1337 | 13.3536 | -0.2871 | IN DISTRIBUTION |
| trade_openness | 27.0000 | 90.0199 | 57.2616 | -1.1006 | IN DISTRIBUTION |
| trade_balance_ratio | 0.8000 | -5.5633 | 19.1513 | 0.3323 | IN DISTRIBUTION |
| log_gdp_usd | 30.8000 | 23.7473 | 2.3934 | 2.9468 | IN DISTRIBUTION |
| log_population | 19.6000 | 15.2430 | 2.3309 | 1.8693 | IN DISTRIBUTION |
| gdp_growth_rolling_std_5 | 0.2000 | 3.1521 | 3.6355 | -0.8120 | IN DISTRIBUTION |
| inflation_rolling_std_3 | 0.1000 | 2.5993 | 7.6780 | -0.3255 | IN DISTRIBUTION |

## 8. API Offline Consistency
- **Offline Model Prediction**: `2.699569660110`
- **API Runtime Prediction**: `2.699569660110`
- **Consistency Validated**: `PASS` (Tolerance 1e-12)

## 9. Immutability Verification
- Pre-audit hashes matched authoritative baseline: **PASS**
- Post-audit hashes unchanged: **PASS**
