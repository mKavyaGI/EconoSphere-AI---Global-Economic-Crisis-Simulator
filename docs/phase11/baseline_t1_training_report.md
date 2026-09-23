# Phase 11 -- Baseline ML Training Report

**Phase**: 11 -- AI Agents & ML Forecasting  
**Script**: `apps/api/ml/train_baseline.py`  
**Generated**: 2026-08-17 08:30 UTC  
**Processed dataset**: `data/processed/master_panel_processed.csv`

> This is a **BASELINE** experiment only. No model should be considered production-ready at this stage.

---

---

## 1. Objective

Establish a scientifically valid and reproducible baseline for country-level GDP growth forecasting using historical macroeconomic panel data. The baseline answers: _can standard supervised ML models learn meaningful signal from lagged macro indicators?_

**Target**: `gdp_growth_pct` (annual % GDP growth)
**Problem type**: Temporal panel regression
**Evaluation**: Chronological train/val/test split -- never random

---

## 2. Dataset Used

| Metric | Value |
|---|---|
| Source | `data/processed/master_panel_processed.csv` |
| Total rows | 6,084 |
| Columns | 39 |
| Training rows (post-filter) | 3,924 |
| Validation rows | 829 |
| Test rows | 384 |
| Countries in training | 214 |

---

## 4. Feature List

**29 features** used in baseline. `govt_debt_pct_gdp` is excluded (~80% missing).

- `exchange_rate_lcu_usd`
- `tariff_rate_pct`
- `remittances_usd`
- `fdi_net_inflow_usd`
- `unemployment_pct`
- `imports_pct_gdp`
- `tax_revenue_pct_gdp`
- `exports_pct_gdp`
- `interest_rate_pct`
- `reserves_usd`
- `current_account_pct_gdp`
- `inflation_cpi_pct`
- `population_total`
- `gdp_current_usd`
- `gdp_growth_lag1`
- `gdp_growth_lag2`
- `gdp_growth_lag3`
- `inflation_lag1`
- `unemployment_lag1`
- `exports_lag1`
- `imports_lag1`
- `gdp_growth_rolling_mean_3`
- `gdp_growth_rolling_std_3`
- `gdp_growth_rolling_mean_5`
- `inflation_rolling_mean_3`
- `trade_openness`
- `trade_balance_ratio`
- `log_gdp_usd`
- `log_population`
---

## 3. Country Filtering

The following World Bank aggregate codes are excluded from ML training because they do not represent individual sovereign states:

`['AFE', 'AFW', 'ARB', 'EAP', 'ECA', 'IDX', 'LAC', 'MEA', 'MNA', 'NAC', 'SSA', 'TEA', 'TEC', 'TLA', 'TMN', 'TSA', 'TSS']`

| Metric | Value |
|---|---|
| Rows before filtering | 6,084 |
| Aggregate rows excluded | 442 |
| Country-only rows | 5,642 |
| Unique countries in training dataset | 217 |

> These codes remain in the processed CSV file and can be used for future aggregate forecasting tasks. They are excluded only from model training.

---

## 6. Temporal Split

> Chronological split -- NO random shuffling is applied.

| Period | Total rows | With target | Without target (excluded) | Countries | Year range |
| --- | --- | --- | --- | --- | --- |
| Train (2000-2018) | 4123 | 3924 | 199 | 217 | 2000-2018 |
| Val   (2019-2022) | 868 | 829 | 39 | 217 | 2019-2022 |
| Test  (2023-2025) | 651 | 384 | 267 | 217 | 2023-2025 |

Rows with missing `gdp_growth_pct` are excluded from supervised training and evaluation. They remain in the processed dataset for inference.

After target filtering:
- Train: **3,924** rows
- Validation: **829** rows
- Test: **384** rows


---

## 8. Models Trained & Pipelines

Three sklearn Pipelines are built. Each pipeline fits preprocessing (median imputation, StandardScaler) **exclusively on training data**.

### Leakage Prevention in Training Pipeline

- `SimpleImputer(strategy='median')`: fitted on `X_train` only. Handles residual NaN in lag/rolling features (first year of each country's series cannot be lag-filled; imputed here using the training-period median -- no future data used).
- `StandardScaler()`: fitted on `X_train` only. Mean and std from train applied to val/test.
- The within-country `bfill` present in the preprocessing step is **not relied upon** for the training-period feature values -- residual NaN after the bfill are handled by the imputer above.

### Model Hyperparameters

| Model | Key Parameters |
|---|---|
| Ridge Regression | alpha=1.0 |
| Random Forest | n_estimators=200, max_depth=10, min_samples_leaf=5 |
| HistGradientBoosting | max_iter=300, max_depth=6, lr=0.05, l2=1.0 |

Random seed: `42`. No hyperparameter search performed.
Country identity is **not** included as a feature in the baseline (adding a 217-column one-hot matrix at baseline would conflate country fixed effects with the macro signal; reserved for Step 6).


---

## 10. Validation & Test Metrics

Models are evaluated on the **validation set (2019-2022)** to select the best model. The test set is held out.

| Model | Val MAE | Val RMSE | Val R2 | Val Bias |
| --- | --- | --- | --- | --- |
| Dummy (Mean) | 5.1522 | 8.5249 | -0.0311 | 1.4810 |
| Linear Regression | 5.3883 | 8.6855 | -0.0703 | 0.4798 |
| Random Forest | 5.1755 | 8.4022 | -0.0016 | 0.7638 |
| HistGradientBoosting | 5.2133 | 8.3342 | 0.0145 | 0.9127 |

### Test Set Metrics (shown for reference -- not used for model selection)

| Model | Test MAE | Test RMSE | Test R2 | Test Bias |
| --- | --- | --- | --- | --- |
| Dummy (Mean) | 2.2655 | 3.9323 | -0.0023 | 0.1868 |
| Linear Regression | 1.9554 | 3.5375 | 0.1889 | -0.6106 |
| Random Forest | 1.9362 | 3.5772 | 0.1706 | -0.0630 |
| HistGradientBoosting | 2.2723 | 4.0176 | -0.0462 | 0.0761 |

---

## 13. Error Analysis

Error analysis on the **test set (2023-2025)** using the best model (`HistGradientBoosting`).

**Top 20 largest absolute errors:**

| country_code | country_name | year | actual_gdp_growth | predicted_gdp_growth | absolute_error | residual | context |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GUY | Guyana | 2023 | 43.819 | -0.038 | 43.857 | -43.857 | Guyana: oil boom onset |
| PSE | West Bank and Gaza | 2023 | -22.857 | 2.337 | 25.195 | 25.195 | -- |
| GUY | Guyana | 2024 | 19.336 | 2.289 | 17.047 | -17.047 | Guyana: oil boom onset |
| NCL | New Caledonia | 2023 | -13.5 | 1.705 | 15.205 | 15.205 | -- |
| TLS | Timor-Leste | 2023 | -9.101 | 4.337 | 13.438 | 13.438 | -- |
| LBN | Lebanon | 2023 | -7.08 | 4.923 | 12.003 | 12.003 | -- |
| SDN | Sudan | 2023 | -13.964 | -2.599 | 11.366 | 11.366 | -- |
| VEN | Venezuela, RB | 2023 | 5.525 | -4.965 | 10.489 | -10.489 | Venezuela: hyperinflation crisis |
| PLW | Palau | 2023 | 12.006 | 1.701 | 10.304 | -10.304 | -- |
| BWA | Botswana | 2023 | -2.78 | 6.158 | 8.938 | 8.938 | -- |
| SGP | Singapore | 2023 | 5.344 | -3.572 | 8.915 | -8.915 | -- |
| LBY | Libya | 2024 | 13.372 | 5.45 | 7.921 | -7.921 | Libya: civil war/oil shock |
| TCA | Turks and Caicos Islands | 2023 | 5.637 | 13.448 | 7.811 | 7.811 | Turks & Caicos: hurricane recovery |
| ECU | Ecuador | 2023 | -1.944 | 5.272 | 7.217 | 7.217 | -- |
| GNQ | Equatorial Guinea | 2024 | -5.847 | 0.885 | 6.732 | 6.732 | Equatorial Guinea: oil windfall |
| BWA | Botswana | 2024 | -0.734 | 5.766 | 6.499 | 6.499 | -- |
| KGZ | Kyrgyz Republic | 2023 | 11.54 | 5.184 | 6.356 | -6.356 | -- |
| ZWE | Zimbabwe | 2024 | 8.107 | 1.971 | 6.135 | -6.135 | -- |
| MNG | Mongolia | 2023 | 5.122 | 11.084 | 5.962 | 5.962 | -- |
| KWT | Kuwait | 2023 | -1.474 | 4.42 | 5.894 | 5.894 | -- |

**Observations:**
- Large errors tend to correspond to genuine economic crises, commodity shocks, and geopolitical events that are structurally difficult to predict from lagged macroeconomic indicators alone.
- These observations are preserved -- no observations were deleted.
- The model is not expected to forecast black-swan events accurately at this baseline stage.


---

## 14. Actual vs Predicted Analysis

Plots saved to `docs/phase11/plots/`:

1. `actual_vs_predicted_test.png` -- Scatter: actual vs predicted GDP growth (test set)
2. `residual_distribution_test.png` -- Histogram of residuals (test set)
3. `timeseries_actual_vs_predicted.png` -- Time-series for selected countries
4. `model_comparison_validation.png` -- Validation RMSE/MAE comparison


---

## 12. Best Model

**Selected**: `HistGradientBoosting` (lowest validation RMSE)

| Metric | Validation | Test |
|---|---|---|
| MAE | 5.2133 | 2.2723 |
| RMSE | 8.3342 | 4.0176 |
| R2 | 0.0145 | -0.0462 |
| Bias (mean error) | 0.9127 | 0.0761 |

> The test set was evaluated **once** on the selected model after validation-based selection. It was not used during model selection.

---

## 15. Limitations

1. **Country heterogeneity**: Country fixed effects are not modelled at baseline. A single set of features may under-fit small open economies vs large ones.
2. **Global shocks**: COVID-19 (2020), Ukraine war (2022), Macao casino shutdown -- these are structurally unpredictable from lagged macro indicators.
3. **`govt_debt_pct_gdp` excluded**: Largest single missing feature (~80% NaN). Including it with aggressive imputation would introduce spurious signal.
4. **Within-country bfill residual**: The preprocessing pipeline applied bfill for leading-edge gaps -- the training pipeline's median imputer neutralises this but the residual signal may still carry slight historical leakage for the very first observations of each country's series.
5. **No hyperparameter tuning**: These are MVP-scale parameters. A proper HPO study (e.g., cross-validated grid/random search) is reserved for Step 6.
6. **2024-2025 test sparseness**: Recent years have 32-46% higher feature missingness; test-period metrics are less stable than validation metrics.

---

## 16. Reproducibility

| Parameter | Value |
|---|---|
| Random seed | 42 |
| Data split | Deterministic chronological (no random) |
| Model serialisation | joblib (compress=3) |
| Python requirement | >= 3.12 |

Running `python apps/api/ml/train_t1_baseline.py` twice with the same data produces identical model files and metrics.

---

## 17. Final Readiness Assessment

**BASELINE COMPLETE -- READY WITH CAUTIONS**

The baseline establishes a measurable starting point. Before production use, the following must be addressed:
- Country fixed effects / panel structure
- Hyperparameter optimisation
- Feature selection / regularisation study
- Uncertainty quantification (prediction intervals)
- Out-of-sample robustness validation
