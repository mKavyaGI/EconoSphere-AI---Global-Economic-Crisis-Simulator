# Phase 19 — Error Tail Analysis

## Tail Error Percentiles
| candidate | p90_ae | p95_ae | max_ae | n_above_10 | n_above_15 |
| --- | --- | --- | --- | --- | --- |
| Ridge (A1) | 7.1192 | 10.5112 | 83.3355 | 262 | 119 |
| Elastic Net (A2) | 7.1905 | 10.4131 | 83.3875 | 257 | 117 |
| Huber (A3) | 7.2341 | 10.4702 | 82.1622 | 261 | 119 |
| Random Forest (A4) | 7.2534 | 10.5782 | 85.8632 | 262 | 115 |
| Ensemble Equal (A5a) | 7.1730 | 10.1295 | 86.7574 | 250 | 112 |
| Ensemble Val-Weighted (A5b) | 7.5021 | 10.6249 | 91.0735 | 271 | 110 |
| Residual Model (A6) | 7.6857 | 10.7847 | 91.3673 | 282 | 114 |
| Phase 11 (M0) | 7.5021 | 10.6249 | 91.0735 | 271 | 110 |


## Top 20 Worst Predictions (all candidates)
| target_year | country_code | candidate | actual_gdp_growth | predicted_gdp_growth | absolute_error |
| --- | --- | --- | --- | --- | --- |
| 2012 | LBY | A6_Residual | 86.8270 | -4.5410 | 91.3670 |
| 2012 | LBY | A5b_Ensemble | 86.8270 | -4.2470 | 91.0730 |
| 2012 | LBY | M0_Phase11 | 86.8270 | -4.2470 | 91.0730 |
| 2012 | LBY | A5a_Ensemble | 86.8270 | 0.0690 | 86.7570 |
| 2012 | LBY | A4_RandomForest | 86.8270 | 0.9640 | 85.8630 |
| 2012 | LBY | A2_ElasticNet | 86.8270 | 3.4390 | 83.3870 |
| 2012 | LBY | A1_Ridge | 86.8270 | 3.4910 | 83.3350 |
| 2012 | LBY | A3_Huber | 86.8270 | 4.6650 | 82.1620 |
| 2023 | MAC | A3_Huber | 75.3080 | 0.1850 | 75.1230 |
| 2023 | MAC | A1_Ridge | 75.3080 | 0.6680 | 74.6410 |
| 2023 | MAC | A6_Residual | 75.3080 | 1.1950 | 74.1140 |
| 2023 | MAC | A2_ElasticNet | 75.3080 | 1.4900 | 73.8180 |
| 2023 | MAC | A5a_Ensemble | 75.3080 | 1.6580 | 73.6500 |
| 2023 | MAC | A5b_Ensemble | 75.3080 | 1.8460 | 73.4620 |
| 2023 | MAC | M0_Phase11 | 75.3080 | 1.8460 | 73.4620 |
| 2023 | MAC | A4_RandomForest | 75.3080 | 2.4600 | 72.8480 |
| 2017 | TCA | A4_RandomForest | 74.5880 | 3.0040 | 71.5840 |
| 2017 | TCA | A1_Ridge | 74.5880 | 3.6140 | 70.9740 |
| 2017 | TCA | A5a_Ensemble | 74.5880 | 3.6310 | 70.9570 |
| 2017 | TCA | A3_Huber | 74.5880 | 3.8070 | 70.7800 |


## Interpretation
A model that improves mean RMSE by concentrating large errors in a few observations
is not considered robust. Examine the p95 and max_ae columns alongside mean_rmse.
