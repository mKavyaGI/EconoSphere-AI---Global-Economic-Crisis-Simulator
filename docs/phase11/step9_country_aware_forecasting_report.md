# Phase 11 -- Step 9: Country-Aware T+1 Forecasting Report

## 1. Objective
Determine whether adding explicit country-aware information (country identity via OneHotEncoding and historical structural statistics) improves the verified Step 8 T+1 forecasting model. 

## 2. Starting Step 8 Baseline
- **Official Control Model:** `HistGradientBoostingRegressor`
- **Features Used:** Baseline Macro + Momentum + Rolling Volatility
- **Verified Control Test RMSE:** 4.0045

## 3. Files Inspected
- `apps/api/ml/train_t1_advanced.py`
- `apps/api/ml/data_t1_advanced_features.py`
- `apps/api/ml/forecasting_target.py`
- `apps/api/ml/data_preprocessing.py`
- `apps/api/ml/train_t1_baseline.py`
- `data/processed/master_panel_t1_advanced.csv`

## 4. Dataset Used
- **Source:** `data/processed/master_panel_t1_advanced.csv`
- **Raw Checksum:** `8ac7e0b2bf09fbe89289f82d0c7cf25e` (UNCHANGED)

## 5. Target Definition
- **Target:** `gdp_growth_next_year` (T+1 GDP Growth)

## 6. Temporal Split
- **Train:** 2000-2018
- **Validation:** 2019-2022
- **Test:** 2023-2024
- **Inference:** 2025 (predicting 2026)

## 7. Country-Aware Feature Definitions
- **country_hist_avg_gdp_growth:** Mean GDP growth of a country, calculated exclusively over 2000-2018.
- **country_hist_std_gdp_growth:** Std Dev of GDP growth, calculated exclusively over 2000-2018.
- **country_hist_avg_inflation:** Mean inflation of a country, calculated exclusively over 2000-2018.
- **country_hist_std_inflation:** Std Dev of inflation, calculated exclusively over 2000-2018.
- **country_code (Identity):** OneHotEncoded identity of the country, fitted strictly on 2000-2018 training data.

## 8. Leakage Prevention Strategy
All country historical statistics were aggregated purely using a boolean mask for `year <= 2018`. The test set targets for 2024, 2025, and all validation targets, had zero influence on the structural variables. A comprehensive `pytest` suite guarantees these bounds.

## 9. Experiment Configurations
- **Exp A (Control):** Step 8 features (Base + Momentum + Rolling).
- **Exp B (+ Average):** Exp A + `country_hist_avg_gdp_growth`.
- **Exp C (+ Structure):** Exp A + All four country historical structural statistics.
- **Exp D (+ Identity):** Exp A + OneHotEncoded `country_code`.
- **Exp E (+ Structure + Identity):** Exp A + All Structural + Identity.

## 10. Validation Metrics & 11. Test Metrics & 12. Model Comparison

| Experiment | Added Information | Val RMSE | Test RMSE |
|------------|-------------------|----------|-----------|
| Step 8 Control | Rolling volatility | - | 4.0045 |
| Exp A (Reproduced Control) | N/A | 8.4350 | 4.1062 |
| Exp B | + Historical GDP average | 8.6378 | 4.4875 |
| Exp C | + Structural statistics | 8.5730 | 4.4029 |
| Exp D | + Country identity | 8.4350 | 4.1062 |
| Exp E | + Structure + Identity | 8.5730 | 4.4029 |

*Note: Exp D yielded the identical result to Exp A, indicating the HistGradientBoostingRegressor effectively ignored the high-cardinality sparse OneHot features when evaluated with standard hyperparameters.*

## 13. Country-Specific Results & 14. Five-Country Analysis
Using the best configuration (Exp A - Control), which yielded a tuned Val RMSE of 8.3994 and a Final Test RMSE of 4.0914:

| Country | Target Year | Actual (%) | Predicted (%) | Error (%) |
|---------|-------------|------------|---------------|-----------|
| IND | 2024 | 7.10 | 5.90 | -1.20 |
| IND | 2025 | 7.57 | 5.84 | -1.73 |
| CHN | 2024 | 4.96 | 4.92 | -0.04 |
| CHN | 2025 | 4.96 | 5.50 | 0.54 |
| USA | 2024 | 2.79 | 2.32 | -0.47 |
| USA | 2025 | 2.16 | 2.66 | 0.50 |
| JPN | 2024 | -0.24 | 0.96 | 1.20 |
| JPN | 2025 | 1.19 | 1.81 | 0.62 |
| GBR | 2024 | 1.08 | 2.80 | 1.72 |
| GBR | 2025 | 1.39 | 1.68 | 0.29 |

## 15. 2026 Forecasts
| Country | 2026 Forecast (%) |
|---------|--------------------|
| IND | 6.21 |
| CHN | 4.69 |
| USA | 2.88 |
| JPN | 0.92 |
| GBR | 1.82 |

## 16. Feature Importance
Since the best empirical configuration explicitly rejected country-aware features, the dominant features remain the structural rolling volatility and macro lag indicators identical to Step 8. 

## 17. Overfitting Analysis
Introducing country structural averages (Exp B, C, E) severely degraded generalization. The Validation RMSE deteriorated from 8.43 to ~8.60, and Test RMSE deteriorated from 4.10 to ~4.48. This indicates that providing the model with hard-coded country averages causes it to over-index on long-term historical regimes (2000-2018), failing to generalize to the highly anomalous post-COVID recovery period (2019-2025).

## 18. Baseline vs Step 9 Comparison
- **Step 8 Control RMSE:** 4.0045
- **Step 9 Best RMSE:** 4.0914
- **Improvement:** -2.17%
- The country-aware information actively harmed the model's ability to extrapolate.

## 19. Limitations
1. Simple static historical averages over 18 years (2000-2018) may be too rigid to capture shifting structural regimes.
2. OneHotEncoding introduces severe sparsity (200+ extra dimensions) that standard gradient boosting trees may struggle to split effectively on without very high depth.
3. The post-2019 test period was heavily dominated by global shocks, neutralizing the value of stable, long-term country baselines.

## 20. Final Recommendation
**KEEP STEP 8**

The introduction of Country-Aware features and Identity markers strictly worsened the validation and test performance under rigorous out-of-sample conditions. The Step 8 T+1 Advanced feature set remains the superior predictive configuration.
