# Phase 11 -- Next-Year Forecasting Design Report

**Phase**: 11 -- AI Agents & ML Forecasting  
**Script**: `apps/api/ml/forecasting_target.py`  
**Generated**: 2026-08-17 07:30 UTC  
**Processed dataset**: `data/processed/master_panel_processed.csv` (UNCHANGED)  
**Forecasting dataset**: `data/processed/master_panel_forecasting.csv` (NEW)

> **NO ML MODELS WERE TRAINED IN THIS STEP.**
> This step exclusively prepares the dataset for true T+1 forecasting.

---

## 1. Objective
Convert the dataset into a strict "next-year" forecasting formulation. 
We predict `gdp_growth_pct` for year T+1 using features from year T. This prevents 
accidental leakage of target-year information into the model.

## 2. Existing Project State & Files Inspected
- **Inspected files**: 
  - `apps/api/ml/data_preprocessing.py` (verified shift(1) logic for lags and rolling features)
  - `apps/api/ml/train_baseline.py` (verified current baseline predicted same-year GDP)
  - `data/processed/master_panel_processed.csv` (verified existing features)
- **Current Baseline Formulation**: Evaluated same-year features to predict same-year `gdp_growth_pct`. 

## 3. Why Next-Year Forecasting is Required
The baseline results revealed high test performance for stable economies (Japan, UK) but 
significant errors for volatile periods (COVID, US/India 2021 rebounds). A true forecasting 
system requires predicting T+1 using ONLY information available at T.

## 4. Exact Target Definition
New target variable: `gdp_growth_next_year`
Logic: `gdp_growth_next_year(T) = gdp_growth_pct(T+1)`

Computed per-country via pandas `groupby("country_code")["gdp_growth_pct"].shift(-1)`.

## 5. Feature Availability Assumptions
**Assumption**: All original economic features for year T (e.g., `inflation_cpi_pct`, `unemployment_pct`) 
are assumed to be known at the end of year T. 
- Example: We use inflation for 2023 to predict GDP growth for 2024.
- **Limitation**: In reality, macroeconomic data is subject to publication lags and revisions. We assume the final revised value of year T is available on Jan 1st of T+1.

## 6. Leakage Audit & Country Panel Integrity
The following automated tests were performed during dataset construction:

| Test Description | Result |
|---|---|
| Next-year target exactly equals T+1 actual GDP growth within country | PASS |
| 2025 rows have missing next-year target | PASS |
| No feature is directly equal to gdp_growth_next_year | PASS |
| Raw dataset checksum verified (MD5: 8ac7e0b2bf09fbe89289f82d0c7cf25e) | PASS |
| No country boundary crossing (target shifted within country) | PASS |
| Lags use historical values (verified via `shift(1)` logic in preprocessing) | PASS |
| Rolling features use historical values (verified via `shift(1).rolling` in preprocessing) | PASS |

## 7. Temporal Split Recommendation
Because the target is shifted, the chronological alignment changes:

- **Train Features**: 2000-2018 -> Predicts **Target Years**: 2001-2019
- **Validation Features**: 2019-2022 -> Predicts **Target Years**: 2020-2023
- **Test Features**: 2023-2024 -> Predicts **Target Years**: 2024-2025

The 2025 feature rows have a `NaN` target (2026 data unavailable) and will be excluded 
from supervised evaluation, serving only as inference rows.

**Recommendation**: Retain the feature-based split years (Train: 2000-2018, Val: 2019-2022, 
Test: 2023-2024). The evaluation effectively tests performance up to the target year 2025.

## 8. Missing Target Handling
Rows where `gdp_growth_next_year` is NaN (e.g., all 2025 rows, and earlier rows where T+1 
data is missing) are left as NaN. **THE TARGET IS NEVER IMPUTED.** A convenience indicator 
`next_year_target_available` (1=available, 0=missing) was added.

## 9. Dataset Dimensions
| Metric | Value |
|---|---|
| Rows | 6084 |
| Columns | 41 |
| Missing `gdp_growth_next_year` | 522 |
| 2025 Inference Rows (NaN target) | 234 |

## 10. Manual Verification (Country Samples)

The following tables show feature values for T alongside the target for T+1.

### IND
| year | gdp_growth_pct | gdp_growth_next_year | gdp_growth_lag1 | gdp_growth_lag2 | gdp_growth_lag3 |
| --- | --- | --- | --- | --- | --- |
| 2021.0 | 9.69 | 7.609 | -5.778 | 3.871 | 6.454 |
| 2022.0 | 7.609 | 7.21 | 9.69 | -5.778 | 3.871 |
| 2023.0 | 7.21 | 7.099 | 7.609 | 9.69 | -5.778 |
| 2024.0 | 7.099 | 7.567 | 7.21 | 7.609 | 9.69 |
| 2025.0 | 7.567 | nan | 7.099 | 7.21 | 7.609 |

### CHN
| year | gdp_growth_pct | gdp_growth_next_year | gdp_growth_lag1 | gdp_growth_lag2 | gdp_growth_lag3 |
| --- | --- | --- | --- | --- | --- |
| 2021.0 | 8.57 | 3.134 | 2.34 | 6.067 | 6.758 |
| 2022.0 | 3.134 | 5.416 | 8.57 | 2.34 | 6.067 |
| 2023.0 | 5.416 | 4.958 | 3.134 | 8.57 | 2.34 |
| 2024.0 | 4.958 | 4.96 | 5.416 | 3.134 | 8.57 |
| 2025.0 | 4.96 | nan | 4.958 | 5.416 | 3.134 |

### USA
| year | gdp_growth_pct | gdp_growth_next_year | gdp_growth_lag1 | gdp_growth_lag2 | gdp_growth_lag3 |
| --- | --- | --- | --- | --- | --- |
| 2021.0 | 6.152 | 2.524 | -2.081 | 2.584 | 2.967 |
| 2022.0 | 2.524 | 2.934 | 6.152 | -2.081 | 2.584 |
| 2023.0 | 2.934 | 2.793 | 2.524 | 6.152 | -2.081 |
| 2024.0 | 2.793 | 2.161 | 2.934 | 2.524 | 6.152 |
| 2025.0 | 2.161 | nan | 2.793 | 2.934 | 2.524 |

### JPN
| year | gdp_growth_pct | gdp_growth_next_year | gdp_growth_lag1 | gdp_growth_lag2 | gdp_growth_lag3 |
| --- | --- | --- | --- | --- | --- |
| 2021.0 | 3.564 | 1.332 | -4.283 | -0.308 | 0.834 |
| 2022.0 | 1.332 | 0.721 | 3.564 | -4.283 | -0.308 |
| 2023.0 | 0.721 | -0.24 | 1.332 | 3.564 | -4.283 |
| 2024.0 | -0.24 | 1.193 | 0.721 | 1.332 | 3.564 |
| 2025.0 | 1.193 | nan | -0.24 | 0.721 | 1.332 |

### GBR
| year | gdp_growth_pct | gdp_growth_next_year | gdp_growth_lag1 | gdp_growth_lag2 | gdp_growth_lag3 |
| --- | --- | --- | --- | --- | --- |
| 2021.0 | 8.543 | 5.15 | -10.048 | 1.256 | 1.551 |
| 2022.0 | 5.15 | 0.272 | 8.543 | -10.048 | 1.256 |
| 2023.0 | 0.272 | 1.08 | 5.15 | 8.543 | -10.048 |
| 2024.0 | 1.08 | 1.388 | 0.272 | 5.15 | 8.543 |
| 2025.0 | 1.388 | nan | 1.08 | 0.272 | 5.15 |


## 11. Final Readiness Assessment
The forecasting dataset `master_panel_forecasting.csv` is formulated correctly for T+1 prediction 
and is **READY** for the next ML training step. 
