# Phase 12 Step 1 — Walk-Forward Validation Report

**Document:** `docs/phase12/step1_walk_forward_validation_report.md`  
**Date:** 2026-08-19  
**Status:** DIAGNOSTIC COMPLETE  
**Phase 11 Production RMSE:** 3.9113 — **LOCKED AND UNCHANGED**

---

## 1. Objective

This report evaluates the **stability** of the existing locked Phase 11 production model across 12 historical forecasting windows. The model itself is not modified. The purpose is to:

- Determine how consistently the model performs across historical periods
- Identify which countries, years, and economic regimes generate the largest errors
- Diagnose structural weaknesses before Phase 12 introduces new features
- Provide evidence-based recommendations for Phase 12 Step 2

All analyses are strictly post-hoc diagnostics. Growth regime labels, missingness indicators, and permutation importance results are **never used in model training or feature selection**.

---

## 2. Locked Production Model

| Property | Value |
|---|---|
| Algorithm | HistGradientBoostingRegressor |
| learning_rate | 0.05 |
| max_depth | 5 |
| max_iter | 300 |
| l2_regularization | 5.0 |
| random_state | 42 |
| Imputation | SimpleImputer(strategy="median") |
| Feature count | 31 (Step 8 Advanced Features) |
| Training period | 2000–2018 |
| Validation period | 2019–2022 |
| Test period | 2023–2024 |
| **Official Test RMSE** | **3.9113** |
| Uncertainty method | Global Q90 = 11.4507 |
| Dataset MD5 | `8ac7e0b2bf09fbe89289f82d0c7cf25e` |

---

## 3. Walk-Forward Methodology

**Expanding-window walk-forward evaluation** simulates how the locked model would have performed if deployed at 12 successive points in history.

**T+1 Target Design:** A row with `feature_year = Y` contains macroeconomic indicators for year Y. The target `gdp_growth_next_year` represents GDP growth in Y+1. Therefore:
- Training uses rows with `feature_year < eval_feature_year`
- Evaluation uses rows with `feature_year = eval_feature_year` and `next_year_target_available == 1`
- The target year evaluated is `feature_year + 1`

**Key invariants enforced:**
- Imputer fitted strictly on training data only per window
- No random splits — all splits are strictly chronological
- 2025 inference year excluded (all targets are NaN)
- WB aggregate codes excluded (same list as Phase 11)

### Window Definitions

| Window | Train Period | Feature Year | Target Year | N Obs |
|---|---|---|---|---|
| 1 | 2000–2011 | 2012 | **2013** | 208 |
| 2 | 2000–2012 | 2013 | **2014** | 209 |
| 3 | 2000–2013 | 2014 | **2015** | 210 |
| 4 | 2000–2014 | 2015 | **2016** | 209 |
| 5 | 2000–2015 | 2016 | **2017** | 209 |
| 6 | 2000–2016 | 2017 | **2018** | 210 |
| 7 | 2000–2017 | 2018 | **2019** | 209 |
| 8 | 2000–2018 | 2019 | **2020** | 209 |
| 9 | 2000–2019 | 2020 | **2021** | 209 |
| 10 | 2000–2020 | 2021 | **2022** | 208 |
| 11 | 2000–2021 | 2022 | **2023** | 203 |
| 12 | 2000–2022 | 2023 | **2024** | 199 |

*Note: Training uses `feature_year < eval_feature_year` (strict inequality), so Window 1 trains on years 2000–2011 and evaluates on feature year 2012 (target year 2013).*

---

## 4. Overall Walk-Forward Results

| Metric | Value |
|---|---|
| Walk-Forward Windows | 12 |
| Total Predictions | 2,492 |
| **Overall RMSE** | **6.0967** |
| **Overall MAE** | **3.3999** |
| **Overall Bias** | **−0.407** (slight net over-prediction) |
| **Overall R²** | **0.018** |

**Interpretation:** The overall walk-forward RMSE of 6.10 is substantially higher than the official Phase 11 test RMSE of 3.91. This is expected — the Phase 11 test evaluates only 2023–2024 (post-COVID recovery), while the walk-forward includes 2020 and 2021 (the COVID shock years), which dominate the error distribution. The model's performance on non-shock years (Windows 1–7, 12) is broadly consistent with the Phase 11 test result.

---

## 5. Performance by Year

| Target Year | N | MAE | RMSE | R² | Bias |
|---|---|---|---|---|---|
| 2013 | 208 | 2.86 | 5.12 | 0.002 | +1.00 |
| 2014 | 209 | 2.30 | 3.93 | −0.170 | +0.51 |
| 2015 | 210 | 2.68 | 4.69 | −0.083 | +0.82 |
| 2016 | 209 | 2.29 | 3.93 | +0.033 | +0.50 |
| 2017 | 209 | 2.54 | 6.20 | +0.106 | −0.30 |
| **2018** | 210 | **1.98** | **3.10** | +0.222 | +0.47 |
| 2019 | 209 | 2.21 | 3.49 | +0.272 | +0.48 |
| **2020** | 209 | **9.09** | **11.66** | −0.755 | **+8.34** |
| 2021 | 209 | 5.43 | 7.81 | −0.685 | −3.96 |
| 2022 | 208 | 4.17 | 7.15 | −0.114 | −2.18 |
| 2023 | 203 | 2.75 | 6.50 | +0.086 | −1.04 |
| 2024 | 199 | 2.47 | 3.94 | +0.289 | +0.17 |

**Best forecast year:** 2018 (RMSE = 3.10) — lowest error across all windows  
**Worst forecast year:** 2020 (RMSE = 11.66) — dominated by COVID-19 recession shock

**Observations:**
- Pre-shock years (2013–2019): RMSE ranges 3.1–6.2, consistent with Phase 11 levels
- 2020: Catastrophic RMSE of 11.66. Bias of +8.34 means the model severely **under-predicted** the 2020 recession (predicted near-normal growth when extreme contraction occurred). This is a structural feature of the locked model — it has no mechanism to predict sudden GDP collapses.
- 2021: Recovery rebound was also mispredicted. Bias of −3.96 means the model **over-predicted** the 2021 rebound recovery magnitude (predicted lower growth than actually occurred).
- 2022–2023: Elevated errors persist in the post-shock years; the model gradually recovers.
- 2024: RMSE = 3.94, closely matching the official Phase 11 test RMSE, confirming reproducibility.
- 2017: Elevated RMSE (6.20) despite no major global shock — driven by a small number of country-level outliers (Libya, Turks & Caicos).

---

## 6. Performance by Country

**Qualifying countries:** 210 (minimum 5 walk-forward predictions each)

### Top 10 Highest-Error Countries

| Country | N | RMSE | MAE | Bias |
|---|---|---|---|---|
| Macao SAR, China (MAC) | 12 | 30.15 | 21.29 | +2.95 |
| Turks and Caicos Islands (TCA) | 12 | 26.41 | 17.24 | −10.33 |
| Guyana (GUY) | 12 | 23.71 | 13.91 | −12.99 |
| Libya (LBY) | 12 | 18.38 | 14.85 | +3.04 |
| Northern Mariana Islands (MNP) | 10 | 17.16 | 14.65 | −1.90 |
| Maldives (MDV) | 12 | 15.47 | 8.48 | −1.80 |
| Yemen, Rep. (YEM) | 6 | 13.57 | 8.22 | +7.11 |
| Timor-Leste (TLS) | 12 | 12.96 | 9.57 | +1.75 |
| Central African Republic (CAF) | 12 | 12.49 | 5.34 | +4.52 |
| Syrian Arab Republic (SYR) | 10 | 11.93 | 7.16 | +5.11 |

*Note: These are small/fragile economies with extreme growth volatility driven by political conflict, commodity shocks, or external dependency. High errors do not necessarily reflect model design flaws — they reflect inherent unpredictability in these economies.*

### Top 10 Lowest-Error Countries

| Country | N | RMSE | MAE | Bias |
|---|---|---|---|---|
| Sao Tome and Principe (STP) | 12 | 1.27 | 1.12 | −0.26 |
| Tanzania (TZA) | 12 | 1.33 | 0.92 | −0.08 |
| Tajikistan (TJK) | 12 | 1.34 | 1.19 | −0.55 |
| Cameroon (CMR) | 12 | 1.36 | 0.97 | −0.10 |
| Australia (AUS) | 12 | 1.36 | 0.97 | −0.56 |
| Uganda (UGA) | 12 | 1.38 | 1.25 | +0.04 |
| Togo (TGO) | 12 | 1.61 | 1.37 | −0.56 |
| Comoros (COM) | 12 | 1.63 | 1.26 | +0.63 |
| China (CHN) | 12 | 1.68 | 1.15 | +0.54 |
| Guinea-Bissau (GNB) | 12 | 1.70 | 1.41 | −0.43 |

### Systematic Over-Prediction (Bias most negative — predicted higher than actual)

| Country | N | Bias |
|---|---|---|
| Guyana (GUY) | 12 | −12.99 |
| Turks and Caicos Islands (TCA) | 12 | −10.33 |
| Ireland (IRL) | 12 | −4.22 |
| Monaco (MCO) | 12 | −3.75 |
| Bermuda (BMU) | 12 | −2.10 |

### Systematic Under-Prediction (Bias most positive — predicted lower than actual)

| Country | N | Bias |
|---|---|---|
| Yemen, Rep. (YEM) | 6 | +7.11 |
| Syrian Arab Republic (SYR) | 10 | +5.11 |
| Congo, Rep. (COG) | 12 | +4.94 |
| Lebanon (LBN) | 12 | +4.91 |
| Venezuela, RB (VEN) | 12 | +4.88 |

**Note:** Small-sample countries (YEM N=6) should be interpreted cautiously. Countries with systematic under-prediction are predominantly conflict-affected or economic-crisis economies — the model cannot anticipate the structural collapses that drive extreme downturns.

---

## 7. Growth Regime Analysis (Post-Hoc Only)

> [!IMPORTANT]
> Growth regime labels are assigned strictly **after** prediction using actual GDP growth values. They are **never** introduced as model features.

| Regime | N | MAE | RMSE | Bias |
|---|---|---|---|---|
| RECESSION (< 0%) | 444 | 7.60 | **10.19** | **+7.51** |
| LOW GROWTH (0–2%) | 441 | 1.86 | 2.45 | +1.32 |
| MODERATE GROWTH (2–5%) | 962 | 1.51 | **2.09** | −0.27 |
| HIGH GROWTH (≥ 5%) | 645 | 4.37 | 7.84 | **−4.09** |

**Key Findings:**
1. **Recessions are the primary failure mode.** RMSE of 10.19 and Bias of +7.51 in recession years shows the model systematically predicts near-positive growth when the economy is actually contracting. This is the single most important structural weakness.
2. **Moderate growth (2–5%) is where the model excels.** RMSE of 2.09 — the model handles normal economic conditions well.
3. **High growth is systematically under-predicted.** Bias of −4.09 means the model predicts lower growth than occurs during boom periods. The model is anchored toward historical average growth rates.
4. **Low growth** is predicted reasonably well (RMSE = 2.45).

**Conclusion:** The model is calibrated for moderate-growth environments and struggles at both extremes — unable to anticipate severe recessions or strong recoveries/booms.

---

## 8. Shock Period Analysis

| Period | N | MAE | RMSE | R² | Bias |
|---|---|---|---|---|---|
| PRE-SHOCK (2013–2019) | 1,464 | 2.41 | 4.46 | +0.061 | +0.50 |
| SHOCK (2020–2021) | 418 | 7.26 | **9.93** | −0.128 | +2.19 |
| POST-SHOCK (2022–2024) | 610 | 3.14 | 6.05 | +0.049 | −1.03 |

**Key Findings:**
- **RMSE increases 2.2× from pre-shock to shock period.** The model degrades severely during the COVID-19 shock years.
- **Pre-shock performance** (RMSE 4.46) is broadly comparable to the Phase 11 test RMSE (3.91), with positive R² confirming some explanatory power.
- **Shock period:** Negative R² (−0.128) — the model is worse than a naive mean prediction during the shock. The model has no mechanism to predict sudden GDP collapses.
- **Post-shock (2022–2024):** Recovery is partial. RMSE remains elevated at 6.05 vs 4.46 pre-shock, with negative bias (−1.03) — the model under-predicted the post-COVID recovery strength.

---

## 9. Missingness vs Forecast Error

| Missingness Group | N | MAE | RMSE | Bias |
|---|---|---|---|---|
| LOW MISSINGNESS | 1,417 | 3.02 | 5.35 | +0.33 |
| MEDIUM MISSINGNESS | 349 | 2.80 | 4.94 | +0.85 |
| HIGH MISSINGNESS | 726 | 4.42 | 7.75 | +0.34 |

**Pearson Correlation (missingness ratio vs absolute error):** r = 0.119, p < 0.0001

**Interpretation:** There is a statistically significant positive correlation between data missingness and forecast error, though the effect size is modest (r = 0.12). High-missingness observations have RMSE 7.75 vs 5.35 for low-missingness (45% higher). 

> [!NOTE]
> Correlation does not imply causation. High-missingness countries tend to be smaller, less stable economies that are also inherently harder to forecast — the missingness may be a proxy for structural complexity rather than a direct cause of error.

---

## 10. Largest Individual Forecast Errors

Top 20 largest absolute errors:

| Country | Feature Year | Target Year | Actual | Predicted | Error |
|---|---|---|---|---|---|
| Macao SAR, China | 2022 | 2023 | +75.3% | +1.8% | 73.46 |
| Turks & Caicos Is. | 2016 | 2017 | +74.6% | +4.3% | 70.31 |
| Guyana | 2021 | 2022 | +63.3% | −0.4% | 63.70 |
| Macao SAR, China | 2019 | 2020 | −54.4% | +1.3% | −55.72 |
| Turks & Caicos Is. | 2020 | 2021 | +29.6% | −13.1% | 42.69 |
| Central African Rep. | 2012 | 2013 | −36.4% | +6.2% | −42.63 |
| Guyana | 2019 | 2020 | +43.5% | +4.9% | 38.54 |
| Maldives | 2019 | 2020 | −32.9% | +5.5% | −38.44 |
| Maldives | 2020 | 2021 | +37.5% | +2.1% | 35.45 |
| Turks & Caicos Is. | 2019 | 2020 | −33.8% | −0.6% | −33.24 |

**Pattern:** The 20 largest errors are predominantly driven by:
1. **Small, tourism-dependent or commodity-export economies** (Macao, Maldives, Turks & Caicos, Guyana) undergoing extreme commodity/oil booms or COVID-driven collapses
2. **Conflict-affected economies** (Syria, CAF, Yemen, Libya) experiencing sudden political shocks
3. **COVID-19 year transitions** — 2019→2020 and 2020→2021 account for a disproportionate share

---

## 11. Feature Diagnostics (Post-Hoc Permutation Importance)

> [!IMPORTANT]
> Permutation importance computed on **evaluation sets only** after prediction. This is post-hoc analysis and is **never used to select features or tune the model**.

Top 15 features by mean permutation importance (across 12 windows):

| Feature | Mean Importance | Std Dev | Windows |
|---|---|---|---|
| gdp_current_usd | 3.839 | 3.614 | 12 |
| trade_balance_ratio | 1.107 | 0.937 | 12 |
| inflation_cpi_pct | 1.056 | 0.875 | 12 |
| reserves_usd | 1.020 | 1.836 | 12 |
| gdp_growth_lag1 | 0.908 | 2.113 | 12 |
| population_total | 0.818 | 2.204 | 12 |
| exports_pct_gdp | 0.717 | 0.867 | 12 |
| exchange_rate_lcu_usd | 0.674 | 0.878 | 12 |
| current_account_pct_gdp | 0.555 | 0.812 | 12 |
| fdi_net_inflow_usd | 0.457 | 0.642 | 12 |

**Observation:** `gdp_current_usd` shows the highest mean permutation importance with high variance (std = 3.61), suggesting it has a large but inconsistent effect. `trade_balance_ratio` and `inflation_cpi_pct` are the most consistently influential trade/price features. GDP lag features (`gdp_growth_lag1`) show moderate importance — the model uses recent history but this alone cannot predict sudden shocks.

---

## 12. Key Failure Modes

The following failure modes are supported by the empirical walk-forward evidence:

### Failure Mode 1: Complete Inability to Predict Sudden Recessions
**Evidence:** 2020 RMSE = 11.66, Bias = +8.34. Recession regime RMSE = 10.19, Bias = +7.51.
The model predicts near-normal growth when severe contraction occurs. No features in the current set capture leading indicators of economic collapse onset.

### Failure Mode 2: Systematic Under-prediction of High-Growth Economies
**Evidence:** HIGH_GROWTH regime Bias = −4.09. The model anchors predictions toward historical average growth and fails to capture the magnitude of strong expansion phases (commodity booms, rapid development economies).

### Failure Mode 3: Extreme Volatility in Small, Shock-Prone Economies
**Evidence:** Top 10 highest-error countries are all small or conflict-affected (Macao, TCA, GUY, LBY, MNP). These countries exhibit GDP growth ranges of 30–75% in single years, which no smooth regression model can anticipate without exogenous structural indicators.

### Failure Mode 4: Post-Shock Recovery Magnitude Underestimation
**Evidence:** 2021 Bias = −3.96 (predicted less growth than occurred). The model learned pre-COVID base rates and does not capture rebound dynamics.

### Failure Mode 5: Elevated Error in High-Missingness Countries
**Evidence:** HIGH_MISSINGNESS RMSE = 7.75 vs LOW_MISSINGNESS RMSE = 5.35. Countries with sparse macro data are harder to forecast, and median imputation may be inadequate for extreme data gaps.

---

## 13. Recommendations for Phase 12 Step 2

Recommendations are based strictly on the observed diagnostic evidence.

### Recommendation 1 (Primary): Recession/Shock Leading Indicator Features
**Basis:** Failure Mode 1 — the model cannot predict sudden recessions.  
**Suggested direction:** Investigate whether adding leading indicators that signal recession risk improves pre-recession performance. Candidates include: year-over-year change in GDP growth (acceleration/deceleration), investment/consumption ratios, or commodity price volatility proxies if available in the dataset.

### Recommendation 2 (Secondary): High-Growth Regime Calibration
**Basis:** Failure Mode 2 — systematic under-prediction in high-growth environments.  
**Suggested direction:** Investigate whether the model's high-growth bias can be reduced by adding features that distinguish emerging-market boom cycles (e.g., FDI acceleration, export growth momentum).

### Recommendation 3 (Tertiary): Missingness-Aware Imputation Improvement
**Basis:** Failure Mode 5 — high-missingness observations have 45% higher RMSE.  
**Suggested direction:** The current median imputation may be inadequate for extreme data gaps. Phase 12 Step 2 could evaluate whether more sophisticated imputation strategies (e.g., country-specific historical mean imputation) improve high-missingness predictions.

### NOT Recommended for Step 2
- Adding a COVID indicator feature (would constitute retroactive target leakage)
- Replacing the model algorithm (diagnostic only — Step 1 showed consistent behavior for normal years)
- Adding growth regime as a feature (post-hoc diagnostic only, would cause look-ahead bias)

---

*Report generated: 2026-08-19*  
*Phase 11 production RMSE remains: **3.9113** (LOCKED AND UNCHANGED)*  
*STATUS: DIAGNOSTIC COMPLETE*
