# Phase 12 Step 6: Statistical Robustness & Model Stability Validation

## 1. Objective
Determine whether Candidate B's improvement over Phase 11 Control is statistically meaningful, stable, and practically significant.

## 2. Locked Phase 11 Control
- **Model**: HistGradientBoostingRegressor
- **Features**: 31
- **Validation RMSE**: 8.2853
- **Test RMSE**: 3.9113
- **Walk-forward RMSE**: 6.0967

## 3. Candidate B Definition
- **Model**: HistGradientBoostingRegressor
- **Features**: 33 (Base + Regime)
- **Validation RMSE**: 8.2598
- **Test RMSE**: 3.8640
- **Walk-forward RMSE**: 6.0943

## 4. Dataset Integrity
- **Raw MD5**: 8ac7e0b2bf09fbe89289f82d0c7cf25e
- **Integrity**: Verified unchanged.

## 5. Experimental Methodology
We performed rigorous paired tests, bootstrap sampling, walk-forward analysis, country/regime disaggregation, seed stability, and window sensitivity tests.

## 6. Year-by-Year Comparison
- Candidate B won 1 years.
- Control won 2 years.
- Tied 9 years.

## 7. Paired Statistical Significance
- **Wilcoxon P-value (Squared Error)**: 0.449
- **Permutation P-value (Squared Error)**: 0.5654

## 8. Bootstrap Confidence Intervals
- **95% CI (RMSE Diff)**: [-0.0105, 0.0054]
- **Probability Candidate Improves**: 72.28%

## 9. Walk-forward Stability
- Candidate B improved 1 out of 12 expanding windows.

## 10. Temporal Subperiod Robustness
| period      |   sample_count |   control_rmse |   candidate_rmse |   rmse_difference |   control_mae |   candidate_mae |   mae_difference | candidate_better   |
|:------------|---------------:|---------------:|-----------------:|------------------:|--------------:|----------------:|-----------------:|:-------------------|
| Pre-shock   |           1464 |        4.46165 |          4.46318 |        0.00153083 |       2.40647 |         2.40761 |       0.00113626 | False              |
| COVID/shock |            418 |        9.92511 |          9.91385 |       -0.0112684  |       7.25613 |         7.26482 |       0.00868956 | True               |
| Post-shock  |            610 |        6.04747 |          6.04747 |        0          |       3.14174 |         3.14174 |       0          | False              |

## 11. Country Robustness
- **Countries won**: 100
- **Countries lost**: 110

## 12. Regime Robustness
| growth_regime   | stress_regime   |   sample_count |   control_rmse |   candidate_rmse |   rmse_difference |   control_mae |   candidate_mae |   mae_difference | candidate_better   |
|:----------------|:----------------|---------------:|---------------:|-----------------:|------------------:|--------------:|----------------:|-----------------:|:-------------------|
| LOW             | NORMAL          |            735 |        4.62201 |          4.62159 |      -0.000421734 |       2.93261 |         2.93659 |       0.00398534 | True               |
| LOW             | STRESS          |            260 |        7.1022  |          7.11394 |       0.0117385   |       4.84047 |         4.84713 |       0.00665943 | False              |
| MODERATE        | NORMAL          |            741 |        5.39262 |          5.39345 |       0.000823273 |       3.15952 |         3.1632  |       0.00368395 | False              |
| MODERATE        | STRESS          |            129 |        6.53668 |          6.54706 |       0.0103779   |       3.24282 |         3.24139 |      -0.00142422 | False              |
| HIGH            | NORMAL          |            372 |        4.66843 |          4.66762 |      -0.000811833 |       2.68999 |         2.68857 |      -0.0014145  | True               |
| HIGH            | STRESS          |            255 |       10.5985  |         10.5734  |      -0.0250503   |       5.09179 |         5.08635 |      -0.00543077 | True               |
| ALL (RECESSION) | ALL (RECESSION) |            444 |       10.1908  |         10.1815  |      -0.00927855  |       7.5983  |         7.60285 |       0.00455499 | True               |

## 13. Random-seed Robustness
- Candidate B won 5 out of 5 random seeds.

## 14. Training-window Sensitivity
| configuration   |   training_start |   training_end |   control_val_rmse |   candidate_val_rmse |   rmse_difference |   control_val_mae |   candidate_val_mae |   mae_difference | candidate_better   | decision   |
|:----------------|-----------------:|---------------:|-------------------:|---------------------:|------------------:|------------------:|--------------------:|-----------------:|:-------------------|:-----------|
| full_historical |             1960 |           2018 |            8.28527 |              8.25979 |         -0.025482 |           5.17315 |             5.16828 |      -0.00487863 | True               | PROMOTE    |
| recent_history  |             2000 |           2018 |            8.28527 |              8.25979 |         -0.025482 |           5.17315 |             5.16828 |      -0.00487863 | True               | PROMOTE    |
| medium_history  |             1990 |           2018 |            8.28527 |              8.25979 |         -0.025482 |           5.17315 |             5.16828 |      -0.00487863 | True               | PROMOTE    |

## 15. Practical Significance
- **RMSE Improvement**: 0.31%
- **Assessment**: Negligible/Small

## 16. Failure Modes
- Improvement is too small or unstable across conditions.

## 17. Statistical Limitations
- Paired tests can be overconfident with highly correlated time-series errors. Bootstrap intervals provide better context.

## 18. Final Decision
**ROBUSTNESS INCONCLUSIVE**

## 19. Recommendation for Phase 12 Step 7
If rejected or inconclusive, keep Phase 11 locked and do not proceed with regime features.
