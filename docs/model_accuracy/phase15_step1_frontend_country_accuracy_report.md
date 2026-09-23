# Phase 15 Step 1: Frontend Country Accuracy Audit Report

**Audit Date:** 2026-09-01 11:12:57
**Target Model:** `best_t1_gdp_growth_model.joblib`

## 1. Executive Summary
The frontend dynamically exposes two sets of countries. The 10 explicitly highlighted `SUPPORTED_COUNTRIES` in the primary dashboard widgets (Tier A) were subjected to a deep accuracy, baseline, and feature distribution audit.

## 2. Discovered Countries

### Tier A (Primary Frontend Forecast Countries)
The following 10 countries are hardcoded in `apps/web/src/app/dashboard/forecast/page.tsx`:
USA, CHN, DEU, JPN, IND, GBR, BRA, FRA, CAN, AUS

### Tier B (Dynamically Discoverable Countries)
The API mechanisms (`/forecasts` and `/countries/search`) dynamically expose the remaining 207 countries present in the dataset. These are available in the database but are not the primary highlighted paths.

## 3. Tier A: Dataset Coverage
| Country | Total Obs | Train Obs | Val Obs | Test Obs | Missing Targets | Feat Miss % |
| --- | --- | --- | --- | --- | --- | --- |
| USA | 26 | 19 | 4 | 2 | 1 | 2.7295 |
| CHN | 26 | 19 | 4 | 2 | 1 | 2.7295 |
| DEU | 26 | 19 | 4 | 2 | 1 | 2.7295 |
| JPN | 26 | 19 | 4 | 2 | 1 | 2.7295 |
| IND | 26 | 19 | 4 | 2 | 1 | 2.7295 |
| GBR | 26 | 19 | 4 | 2 | 1 | 2.7295 |
| BRA | 26 | 19 | 4 | 2 | 1 | 2.7295 |
| FRA | 26 | 19 | 4 | 2 | 1 | 2.7295 |
| CAN | 26 | 19 | 4 | 2 | 1 | 2.7295 |
| AUS | 26 | 19 | 4 | 2 | 1 | 2.7295 |

## 4. Tier A: Baseline Comparison & Model Performance
*Warning: Many countries only have 1-2 test observations. R² is omitted. Interpret 'PRELIMINARY' metrics with caution.*
| Country | Test MAE | Test RMSE | Dummy MAE | Dummy RMSE | Naive MAE | Naive RMSE | Beats Dummy | Beats Naive | Reliable Stats |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| USA | 0.7644 | 0.7900 | 1.0590 | 1.1051 | 0.3865 | 0.4578 | True | False | PRELIMINARY |
| CHN | 0.4368 | 0.5292 | 1.4228 | 1.4228 | 0.2297 | 0.3236 | True | False | PRELIMINARY |
| DEU | 1.8104 | 1.8279 | 3.6644 | 3.6828 | 0.5546 | 0.5833 | True | False | PRELIMINARY |
| JPN | 1.2336 | 1.4144 | 3.0598 | 3.1426 | 1.1970 | 1.2201 | True | False | PRELIMINARY |
| IND | 0.3439 | 0.4751 | 3.7967 | 3.8039 | 0.2892 | 0.3397 | True | False | PRELIMINARY |
| GBR | 0.7943 | 0.8222 | 2.3019 | 2.3071 | 0.5584 | 0.6119 | True | False | PRELIMINARY |
| BRA | 0.7975 | 0.9473 | 0.6838 | 0.8881 | 0.6556 | 0.8113 | False | False | PRELIMINARY |
| FRA | 1.2648 | 1.3193 | 2.5206 | 2.5266 | 0.2991 | 0.3033 | True | False | PRELIMINARY |
| CAN | 0.9894 | 1.2547 | 1.6418 | 1.6488 | 0.1984 | 0.2245 | True | False | PRELIMINARY |
| AUS | 1.0121 | 1.1038 | 2.1744 | 2.1744 | 1.1167 | 1.5631 | True | True | PRELIMINARY |

## 5. Tier A: Root Cause Classification
| Country | Test RMSE | Val RMSE | Root Causes |
| --- | --- | --- | --- |
| USA | 0.7900 | 3.1273 | BASELINE_NOT_BEATEN (Naive) |
| CHN | 0.5292 | 2.5649 | BASELINE_NOT_BEATEN (Naive) |
| DEU | 1.8279 | 3.4560 | BASELINE_NOT_BEATEN (Naive) |
| JPN | 1.4144 | 3.1305 | BASELINE_NOT_BEATEN (Naive) |
| IND | 0.4751 | 6.8551 | BASELINE_NOT_BEATEN (Naive) |
| GBR | 0.8222 | 7.3658 | BASELINE_NOT_BEATEN (Naive) |
| BRA | 0.9473 | 3.4589 | BASELINE_NOT_BEATEN (Dummy), BASELINE_NOT_BEATEN (Naive) |
| FRA | 1.3193 | 5.3958 | BASELINE_NOT_BEATEN (Naive) |
| CAN | 1.2547 | 4.3269 | BASELINE_NOT_BEATEN (Naive) |
| AUS | 1.1038 | 2.3469 | ACCEPTABLE_CURRENT_PERFORMANCE |

## 6. Tier A: Crisis Sensitivity (Mean Abs Error by Target Year)
| Country | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
| --- | --- | --- | --- | --- | --- | --- |
| AUS | 2.3063 | 0.3339 | 2.7682 | 2.9896 | 1.4526 | 0.5715 |
| BRA | 5.3355 | 3.2820 | 2.1623 | 1.9853 | 1.3087 | 0.2864 |
| CAN | 7.3546 | 3.7015 | 2.6629 | 0.0741 | 1.7610 | 0.2178 |
| CHN | 4.1597 | 2.4740 | 1.6918 | 0.1728 | 0.1381 | 0.7355 |
| DEU | 6.2773 | 2.2691 | 0.7879 | 1.6132 | 2.0631 | 1.5576 |
| FRA | 9.4895 | 5.1193 | 0.1648 | 0.4184 | 0.8898 | 1.6399 |
| GBR | 12.2324 | 6.9120 | 4.2243 | 1.3300 | 1.0067 | 0.5819 |
| IND | 12.4586 | 4.8836 | 2.6451 | 1.3811 | 0.0163 | 0.6716 |
| JPN | 5.7306 | 2.3853 | 0.0275 | 0.8190 | 1.9256 | 0.5416 |
| USA | 4.8767 | 3.7062 | 0.5230 | 1.1532 | 0.9636 | 0.5653 |

## 7. Feature Distribution & Out-of-Distribution (OOD) Risk
Analysis of historical feature distributions for Tier A, identifying skewness and OOD risk compared to global training distributions. Pay special attention to features with extreme standard deviation or skewness like `gdp_current_usd` and `population_total`.
| Feature | Mean | Std | Min | Max | Skew | High OOD Risk |
| --- | --- | --- | --- | --- | --- | --- |
| exchange_rate_lcu_usd | 18.7780 | 36.3702 | 0.4998 | 151.3663 | 2.0203 | NO |
| tariff_rate_pct | 3.7654 | 3.8758 | 0.7100 | 26.5100 | 3.0806 | NO |
| remittances_usd | 13607259254.4623 | 22007234294.3237 | 465722533.9779 | 150711596171.4930 | 3.3517 | YES |
| fdi_net_inflow_usd | 87296142203.2296 | 97909021829.6018 | -25093141435.1895 | 511434000000.0000 | 1.9886 | YES |
| unemployment_pct | 6.3175 | 2.2809 | 2.3510 | 13.6970 | 0.6282 | NO |
| imports_pct_gdp | 23.8063 | 7.9779 | 8.9653 | 43.1573 | 0.0875 | NO |
| tax_revenue_pct_gdp | 15.4363 | 5.9168 | 6.9337 | 26.8890 | 0.5738 | NO |
| exports_pct_gdp | 23.8651 | 9.1978 | 9.0357 | 45.6284 | 0.3171 | NO |
| interest_rate_pct | 3.7270 | 3.3602 | 0.0356 | 21.9707 | 2.3214 | NO |
| reserves_usd | 518202351685.2714 | 830775092115.3052 | 18663980677.4372 | 3900039302991.0898 | 2.6904 | YES |
| current_account_pct_gdp | -0.2740 | 3.4432 | -7.5509 | 9.7996 | 0.6406 | NO |
| inflation_cpi_pct | 2.8432 | 2.4309 | -1.3528 | 14.7149 | 1.4379 | NO |
| population_total | 353841597.9769 | 490960183.9005 | 19028802.0000 | 1463865525.0000 | 1.4397 | YES |
| gdp_current_usd | 4732698814681.5195 | 5663973268393.9248 | 380360222861.1970 | 30769700000000.0000 | 2.3500 | YES |
| gdp_growth_lag1 | 2.8846 | 3.3236 | -10.0479 | 14.1508 | 0.0109 | NO |
| gdp_growth_lag2 | 2.9081 | 3.3597 | -10.0479 | 14.1508 | -0.0101 | NO |
| gdp_growth_lag3 | 2.9219 | 3.3974 | -10.0479 | 14.1508 | -0.0236 | NO |
| inflation_lag1 | 2.8548 | 2.4641 | -1.3528 | 14.7149 | 1.4270 | NO |
| unemployment_lag1 | 6.3764 | 2.2882 | 2.3510 | 13.6970 | 0.6075 | NO |
| exports_lag1 | 23.8072 | 9.2331 | 9.0357 | 45.6284 | 0.3258 | NO |
| imports_lag1 | 23.7431 | 7.9807 | 8.9653 | 43.1573 | 0.0870 | NO |
| gdp_growth_rolling_mean_3 | 2.8913 | 2.7034 | -2.4134 | 12.7599 | 1.2188 | NO |
| gdp_growth_rolling_std_3 | 1.7433 | 1.7162 | 0.0363 | 9.9004 | 2.0404 | NO |
| gdp_growth_rolling_mean_5 | 2.8838 | 2.5744 | -0.6422 | 11.7068 | 1.4828 | NO |
| inflation_rolling_mean_3 | 2.8326 | 2.2352 | -0.7845 | 10.5945 | 1.3226 | NO |
| trade_openness | 47.6713 | 16.9652 | 19.2752 | 88.7858 | 0.2004 | NO |
| trade_balance_ratio | 0.0588 | 2.9455 | -6.7249 | 8.5469 | 0.5320 | NO |
| log_gdp_usd | 28.7093 | 0.9287 | 26.6644 | 31.0576 | 0.4543 | YES |
| log_population | 18.7814 | 1.3257 | 16.7615 | 21.1043 | 0.5201 | NO |
| gdp_growth_rolling_std_5 | 1.9411 | 1.4276 | 0.1455 | 7.0136 | 1.4358 | NO |
| inflation_rolling_std_3 | 1.0038 | 0.8907 | 0.0291 | 4.6102 | 1.5926 | NO |

## 8. Data Quality Findings
- Significant skewness was observed in raw scale features like `gdp_current_usd` and `population_total`.
- The `High OOD Risk` flags indicate that countries like the USA and China submit data values drastically larger than the global mean, pushing predictions into unreliable leaf nodes.
- Using standardized normalizations globally exposes tail ends of distributions to severe inaccuracies.

## 9. Immutability Verification
- Pre-audit hashes matched authoritative baseline: **PASS**
- Post-audit hashes unchanged: **PASS**
