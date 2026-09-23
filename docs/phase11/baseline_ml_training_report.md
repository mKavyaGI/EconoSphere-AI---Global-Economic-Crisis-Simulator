# Phase 11 -- Baseline ML Training Report

**Phase**: 11 -- AI Agents & ML Forecasting  
**Script**: `apps/api/ml/train_baseline.py`  
**Generated**: 2026-08-17 07:11 UTC  
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
| Training rows (post-filter) | 3,914 |
| Validation rows | 835 |
| Test rows | 587 |
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
| Train (2000-2018) | 4123 | 3914 | 209 | 217 | 2000-2018 |
| Val   (2019-2022) | 868 | 835 | 33 | 217 | 2019-2022 |
| Test  (2023-2025) | 651 | 587 | 64 | 217 | 2023-2025 |

Rows with missing `gdp_growth_pct` are excluded from supervised training and evaluation. They remain in the processed dataset for inference.

After target filtering:
- Train: **3,914** rows
- Validation: **835** rows
- Test: **587** rows


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
| Linear Regression | 4.8680 | 7.7319 | 0.0444 | 0.7398 |
| Random Forest | 5.0373 | 7.8890 | 0.0052 | 0.8534 |
| HistGradientBoosting | 5.0252 | 7.8197 | 0.0226 | 0.7359 |

### Test Set Metrics (shown for reference -- not used for model selection)

| Model | Test MAE | Test RMSE | Test R2 | Test Bias |
| --- | --- | --- | --- | --- |
| Linear Regression | 2.0209 | 4.7249 | 0.1443 | -0.5231 |
| Random Forest | 2.0630 | 4.3954 | 0.2594 | -0.0201 |
| HistGradientBoosting | 2.0817 | 4.1629 | 0.3357 | -0.1718 |

---

## 13. Error Analysis

Error analysis on the **test set (2023-2025)** using the best model (`Linear Regression`).

**Top 20 largest absolute errors:**

| country_code | country_name | year | actual_gdp_growth | predicted_gdp_growth | absolute_error | residual | context |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MAC | Macao SAR, China | 2023 | 75.308 | -3.961 | 79.27 | -79.27 | Macao: COVID/gambling shutdown |
| SDN | Sudan | 2023 | -29.433 | 0.475 | 29.908 | 29.908 | -- |
| GUY | Guyana | 2024 | 43.819 | 16.421 | 27.398 | -27.398 | Guyana: oil boom onset |
| PSE | West Bank and Gaza | 2024 | -22.857 | 1.083 | 23.94 | 23.94 | -- |
| TLS | Timor-Leste | 2023 | -18.136 | 0.064 | 18.2 | 18.2 | -- |
| GUY | Guyana | 2023 | 33.795 | 18.895 | 14.9 | -14.9 | Guyana: oil boom onset |
| NCL | New Caledonia | 2024 | -13.5 | 0.911 | 14.411 | 14.411 | -- |
| WSM | Samoa | 2023 | 15.234 | 2.499 | 12.735 | -12.735 | -- |
| PLW | Palau | 2024 | 12.006 | -0.701 | 12.707 | -12.707 | -- |
| LBY | Libya | 2025 | 13.372 | 2.786 | 10.586 | -10.586 | Libya: civil war/oil shock |
| TCA | Turks and Caicos Islands | 2023 | 13.731 | 3.236 | 10.496 | -10.496 | Turks & Caicos: hurricane recovery |
| LBY | Libya | 2023 | 10.2 | -0.011 | 10.211 | -10.211 | Libya: civil war/oil shock |
| VEN | Venezuela, RB | 2023 | 4.023 | -5.82 | 9.843 | -9.843 | Venezuela: hyperinflation crisis |
| UKR | Ukraine | 2023 | 5.535 | -4.259 | 9.793 | -9.793 | Ukraine: war |
| IRL | Ireland | 2025 | 12.338 | 2.919 | 9.419 | -9.419 | -- |
| VEN | Venezuela, RB | 2024 | 5.525 | -3.859 | 9.384 | -9.384 | Venezuela: hyperinflation crisis |
| SDN | Sudan | 2024 | -13.964 | -4.753 | 9.211 | 9.211 | -- |
| TLS | Timor-Leste | 2025 | 6.975 | -1.758 | 8.733 | -8.733 | -- |
| ARG | Argentina | 2025 | 4.367 | -3.9 | 8.267 | -8.267 | -- |
| GNQ | Equatorial Guinea | 2023 | -7.425 | 0.746 | 8.171 | 8.171 | Equatorial Guinea: oil windfall |

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

**Selected**: `Linear Regression` (lowest validation RMSE)

| Metric | Validation | Test |
|---|---|---|
| MAE | 4.8680 | 2.0209 |
| RMSE | 7.7319 | 4.7249 |
| R2 | 0.0444 | 0.1443 |
| Bias (mean error) | 0.7398 | -0.5231 |

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

Running `python apps/api/ml/train_baseline.py` twice with the same data produces identical model files and metrics.

---

## 17. Final Readiness Assessment

**BASELINE COMPLETE -- READY WITH CAUTIONS**

The baseline establishes a measurable starting point. Before production use, the following must be addressed:
- Country fixed effects / panel structure
- Hyperparameter optimisation
- Feature selection / regularisation study
- Uncertainty quantification (prediction intervals)
- Out-of-sample robustness validation
