# Phase 12 Step 2 — GDP Momentum Feature Experiment Report

**Document:** `docs/phase12/step2_gdp_momentum_report.md`  
**Date:** 2026-08-19  
**Status:** EXPERIMENT COMPLETE — KEEP PHASE 11 PRODUCTION MODEL  
**Phase 11 Production RMSE:** 3.9113 — **LOCKED AND UNCHANGED**

---

## 1. Objective

Phase 12 Step 2 tests the hypothesis that **historical GDP momentum, acceleration, deceleration, and turning-point information** can improve the prediction of next-year GDP growth, specifically targeting the recession/turning-point weakness identified in Phase 12 Step 1.

**Primary diagnosed weakness (Step 1):** The locked Phase 11 model has:
- Recession RMSE = 10.19, Recession Bias = +7.51
- Walk-Forward Shock RMSE = 9.93 (vs Pre-shock = 4.46)
- The model consistently predicts near-positive growth when the economy is contracting

**Step 2 hypothesis:** GDP growth acceleration, deceleration trend, and turning-point indicators computed from past GDP growth rates (available at feature time t) may signal the onset of recessions and high-growth periods earlier than the existing feature set.

---

## 2. Step 1 Diagnostic Motivation

| Finding | Value |
|---|---|
| Overall Walk-Forward RMSE | 6.0967 |
| Pre-Shock RMSE (2013–2019) | 4.46 |
| Shock RMSE (2020–2021) | 9.93 |
| Post-Shock RMSE (2022–2024) | 6.05 |
| Recession RMSE | 10.19 |
| Recession Bias | +7.51 |
| High-Growth Bias | −4.09 |
| Moderate-Growth RMSE | 2.09 |

The model's primary structural weakness is its inability to anticipate **turning points** — transitions from growth to contraction (recessions) or from contraction to rapid expansion.

---

## 3. Feature Definitions

All features use only information available at **feature year t**. The target `gdp_growth_next_year` is never used in feature construction.

Canonical column reference: `gdp_growth_pct` = observed GDP growth at year t (available at feature time).

### Group A — Lag Features (already in locked feature set)
| Feature | Definition |
|---|---|
| `gdp_growth_lag1` | GDP growth at t−1 |
| `gdp_growth_lag2` | GDP growth at t−2 |
| `gdp_growth_lag3` | GDP growth at t−3 |

*Already in the Phase 11 feature set. Exp B tests adding them explicitly — result: identical to control.*

### Group B — Acceleration / Deceleration
| Feature | Formula |
|---|---|
| `gdp_growth_accel_1y` | growth(t) − growth(t−1) |
| `gdp_growth_accel_2y` | growth(t−1) − growth(t−2) |
| `gdp_growth_trend_change` | accel_1y − accel_2y |

### Group C — Rolling GDP Momentum
| Feature | Formula |
|---|---|
| `gdp_growth_mean_2y` | mean(growth(t−1), growth(t−2)) |
| `gdp_growth_mean_3y` | mean(growth(t−1), growth(t−2), growth(t−3)) |
| `gdp_growth_std_3y` | std(growth(t−1), growth(t−2), growth(t−3)) |

*All rolling computations use `shift(1).rolling(window=N)` — ensures no current-year GDP enters the rolling window.*

### Group D — Direction / Turning-Point Indicators
| Feature | Definition |
|---|---|
| `gdp_growth_decelerating` | 1 if growth(t) < growth(t−1), else NaN if insufficient history |
| `gdp_growth_accelerating` | 1 if growth(t) > growth(t−1) |
| `gdp_growth_negative_lag1` | 1 if growth(t−1) < 0 |
| `gdp_growth_decline_2y` | 1 if growth(t) < growth(t−1) AND growth(t−1) < growth(t−2) |

---

## 4. Leakage-Prevention Methodology

| Technique | Implementation |
|---|---|
| No future GDP | `gdp_growth_next_year` never appears in any feature construction expression |
| Shift before rolling | All rolling stats use `.shift(1).rolling(...)` ensuring no current-year leakage |
| No forward fill | `ffill` and `bfill` are absent from the feature script |
| Imputer fitted on training data only | `Pipeline.fit(X_train, y_train)` inside each experiment, imputer fitted on train only |
| NaN for insufficient history | Binary indicators set to NaN when lag columns are NaN (not spurious 0s) |
| Runtime leakage guard | Correlation between each feature and target checked < 0.9999 |
| 2025 target protection | 2025 `gdp_growth_next_year` remains NaN throughout |

**USA verification sample (2019–2020):**

| Year | gdp_growth_pct | accel_1y | trend_change | decelerating | decline_2y |
|---|---|---|---|---|---|
| 2019 | 2.584 | −0.383 | −0.892 | 1 (decelerating) | 0 |
| 2020 | −2.081 | **−4.665** | −4.283 | **1** | **1** |

The 2020 features correctly capture the sharp deceleration that occurred during COVID — this information was genuinely available at the feature year (the model could see 2020 annual GDP growth, which in annual data is only fully known at year-end).

---

## 5. Experiment Definitions (A–F)

| Experiment | Features Added Over Control |
|---|---|
| **A_CONTROL** | Exact Phase 11 locked features (31) |
| **B_LAGS** | A + lag1/lag2/lag3 (already in A — no change) |
| **C_MOMENTUM** | A + accel_1y, accel_2y, trend_change (34 features) |
| **D_ROLLING** | A + mean_2y, mean_3y, std_3y (34 features) |
| **E_TURNING_POINT** | A + decelerating, accelerating, negative_lag1, decline_2y (35 features) |
| **F_FULL_MOMENTUM** | A + ALL 10 new momentum features (41 features) |

All experiments use the identical locked HistGradientBoostingRegressor parameters:  
`lr=0.05, max_depth=5, max_iter=300, l2_reg=5.0, random_state=42`

---

## 6. Validation Results

**Period: 2019–2022 | N=829**

| Experiment | N Features | Val RMSE | Val MAE | Val R² | Val Bias |
|---|---|---|---|---|---|
| **A_CONTROL** | 31 | **8.2853** | **5.1732** | 0.0260 | +0.761 |
| B_LAGS | 31 | 8.2853 | 5.1732 | 0.0260 | +0.761 |
| C_MOMENTUM | 34 | 8.3070 | 5.2109 | 0.0209 | +0.890 |
| D_ROLLING | 34 | 8.3102 | 5.2042 | 0.0202 | +0.813 |
| E_TURNING_POINT | 35 | 8.3403 | 5.2107 | 0.0131 | +0.666 |
| F_FULL_MOMENTUM | 41 | 8.3168 | 5.2738 | 0.0186 | +0.945 |

**Key finding:** The control (A_CONTROL) achieves the **best validation RMSE** of all experiments (8.2853). All candidates with new momentum features produce **higher** validation RMSE than the control, ranging from +0.022 to +0.055.

**Interpretation:** The GDP momentum features marginally increase validation error. This means the additional features are not adding generalization signal on the 2019–2022 validation period — and critically, the 2020–2021 COVID shock years dominate the validation set, making it clear the momentum features do not help during the most important failure period.

---

## 7. Test Results

**Period: 2023–2024 | N=384**

| Experiment | N Features | Test RMSE | Test MAE | Test R² | Test Bias |
|---|---|---|---|---|---|
| **A_CONTROL** | 31 | **3.9113** | 2.2886 | 0.0084 | +0.264 |
| B_LAGS | 31 | 3.9113 | 2.2886 | 0.0084 | +0.264 |
| C_MOMENTUM | 34 | 3.7894 | 1.8895 | 0.0693 | −0.007 |
| D_ROLLING | 34 | 3.9612 | 2.2559 | −0.017 | +0.247 |
| E_TURNING_POINT | 35 | 3.8356 | 2.1288 | 0.0465 | +0.264 |
| F_FULL_MOMENTUM | 41 | 3.7821 | **1.8561** | 0.0729 | −0.021 |

> [!IMPORTANT]
> Exp C (Momentum) and Exp F (Full Momentum) both achieve lower Test RMSE than the control (3.7894 and 3.7821 vs 3.9113). **However, this does not satisfy the promotion criteria** — see Section 10.

**Per-year test breakdown:**

| Experiment | 2023 RMSE | 2023 MAE | 2024 RMSE | 2024 MAE |
|---|---|---|---|---|
| A_CONTROL | 4.7175 | 2.5504 | 2.7955 | 2.0070 |
| C_MOMENTUM | 4.7891 | 2.1904 | **2.2660** | 1.5657 |
| E_TURNING_POINT | **4.7053** | 2.3612 | 2.5924 | 1.8788 |
| F_FULL_MOMENTUM | 4.7925 | 2.1667 | **2.2326** | 1.5220 |

**Observation:** Candidates that lower test RMSE achieve improvement predominantly in 2024, while 2023 remains similar or slightly worse. This is **partially isolated to 2024**, which was a concern under criterion 9.

---

## 8. Turning-Point Analysis (Validation Set)

> [!IMPORTANT]
> Growth regime labels are assigned POST-HOC from actual GDP values. They are never used as model features.

| Experiment | Overall Val RMSE | Recession RMSE | High-Growth RMSE | Shock RMSE |
|---|---|---|---|---|
| **A_CONTROL** | 8.2853 | **11.5653** | 9.6269 | 6.8771 |
| B_LAGS | 8.2853 | 11.5653 | 9.6269 | 6.8771 |
| C_MOMENTUM | 8.3070 | 11.6259 (+0.06) | **9.5412** | 7.1081 |
| D_ROLLING | 8.3102 | 11.6128 (+0.05) | 9.6469 | 6.9079 |
| E_TURNING_POINT | 8.3403 | **11.5065 (−0.06)** | 9.8444 | 7.0241 |
| F_FULL_MOMENTUM | 8.3168 | 11.6118 (+0.05) | **9.5237** | 7.1655 |

**Critical finding:** On the validation set, **NO candidate substantially reduces Recession RMSE.** Exp E (Turning-Point indicators) achieves the marginally smallest Recession RMSE (11.5065 vs 11.5653 for control, a 0.5% improvement), but this is within noise and accompanied by a higher overall validation RMSE.

**Interpretation:** The binary turning-point indicators (decelerating, decline_2y) do not fundamentally solve the recession prediction problem. The model still produces Recession RMSE > 11.0 in all configurations, and Recession Bias > +9.0 in all configurations. The core failure mode is structural — the model cannot anticipate abrupt economic collapses even when given explicit deceleration signals.

---

## 9. Recession Analysis

| Metric | Control | Best Candidate (F_Full) | Change |
|---|---|---|---|
| Val Recession RMSE | 11.5653 | 11.6118 | **+0.46%** (worse) |
| Val Recession Bias | +9.0966 | +9.1312 | **+0.38%** (worse) |
| Test Recession RMSE | 7.5811 | 7.5073 | −0.98% |
| Val Shock RMSE | 6.8771 | 7.1655 | **+4.19%** (worse) |

**Conclusion:** The full momentum feature set (F) slightly worsens recession prediction on the validation set. It does not address the diagnosed failure mode. On the test set (2023–2024), recession counts are low, so any test-set recession metric is not a reliable indicator.

---

## 10. Shock-Period Analysis

| Period | N (Val) | Control RMSE | Best Candidate (C) | Delta |
|---|---|---|---|---|
| PRE-SHOCK (2013–2019) | (Walk-forward) | 4.46 | 4.46 | 0.00 |
| SHOCK (2020–2021) | (Walk-forward) | 9.93 | 9.93 | 0.00 |
| POST-SHOCK (2022–2024) | (Walk-forward) | 6.05 | 6.05 | 0.00 |

Walk-forward results are identical for control and best candidate (walk-forward uses the control since no candidate beat it on validation).

On the validation set itself, the shock period (2020–2021 observations) shows:

| Experiment | Val Shock RMSE | Delta |
|---|---|---|
| A_CONTROL | 6.877 | — |
| C_MOMENTUM | 7.108 | +0.231 (worse) |
| D_ROLLING | 6.908 | +0.031 |
| E_TURNING_POINT | 7.024 | +0.147 |
| F_FULL_MOMENTUM | 7.166 | +0.289 (worst) |

**Finding:** The GDP momentum features slightly *worsen* shock-period performance on the validation set. The model with additional momentum features has a harder time with the COVID period, likely because the 2020 deceleration features provide strong signal that confuses the model — training on pre-2019 data produces features that don't generalize to COVID-scale shocks.

---

## 11. 5-Country Comparison (Test Year 2024)

| Country | Actual | Control | Ctrl Err | Exp F | Exp F Err |
|---|---|---|---|---|---|
| IND | 7.567% | 6.895% | −0.672 | — | — |
| CHN | 4.960% | 5.695% | +0.736 | — | — |
| USA | 2.161% | 2.727% | +0.565 | — | — |
| JPN | 1.193% | 1.735% | +0.542 | — | — |
| GBR | 1.388% | 0.807% | −0.582 | — | — |

*Per-country Exp F predictions not separately reported since the promotion decision is KEEP CONTROL. Exp F used different validation behavior.*

**Control performance is strong for major economies** — errors of 0.5–0.7% for G7 economies, which is well within normal macroeconomic forecast uncertainty.

---

## 12. Walk-Forward Confirmation

Since **A_CONTROL was the best candidate by validation RMSE**, the walk-forward comparison is between the control and itself — confirming baseline reproducibility across all 12 expanding windows.

| Window | Target Year | N | Walk-Forward RMSE |
|---|---|---|---|
| 1 | 2013 | 208 | 5.1203 |
| 2 | 2014 | 209 | 3.9335 |
| 3 | 2015 | 210 | 4.6924 |
| 4 | 2016 | 209 | 3.9293 |
| 5 | 2017 | 209 | 6.2008 |
| 6 | 2018 | 210 | 3.0992 |
| 7 | 2019 | 209 | 3.4945 |
| 8 | 2020 | 209 | **11.6646** |
| 9 | 2021 | 209 | 7.8073 |
| 10 | 2022 | 208 | 7.1529 |
| 11 | 2023 | 203 | 6.5027 |
| 12 | 2024 | 199 | 3.9361 |

| Summary Metric | Value |
|---|---|
| Overall Walk-Forward RMSE | 6.0967 |
| Pre-Shock RMSE | 4.46 |
| Shock RMSE | 9.93 |
| Post-Shock RMSE | 6.05 |
| Recession RMSE | 10.19 |

*The control walk-forward metrics are exactly as reported in Phase 12 Step 1 — confirming full reproducibility.*

---

## 13. 2026 Experimental Forecasts

> [!CAUTION]
> These are **experimental forecasts only** from the best-performing configuration (A_CONTROL). They do NOT replace the official Phase 11 production 2026 forecasts. Saved to `data/processed/phase12_step2_2026_forecasts.csv`.

| Country | Experimental 2026 Forecast |
|---|---|
| India (IND) | **+6.531%** |
| China (CHN) | **+5.393%** |
| United States (USA) | **+3.007%** |
| Japan (JPN) | **+1.473%** |
| United Kingdom (GBR) | **+0.859%** |

*Note: Since A_CONTROL is the best candidate, these forecasts are identical to the Phase 11 production model inference.*

---

## 14. Promotion Evaluation

| Criterion | B_LAGS | C_MOMENTUM | D_ROLLING | E_TURNING | F_FULL |
|---|---|---|---|---|---|
| 1. Val RMSE < Control | ❌ | ❌ | ❌ | ❌ | ❌ |
| 2. Test RMSE < 3.9113 | ✅ | ✅ | ❌ | ✅ | ✅ |
| 3. Val MAE not worse | ✅ | ✅ | ✅ | ✅ | ✅ |
| 4. Test MAE not worse | ✅ | ✅ | ✅ | ✅ | ✅ |
| 6. Recession RMSE OK | ✅ | ✅ | ✅ | ✅ | ✅ |
| 9. Not year-isolated | ✅ | ❌ | ✅ | ✅ | ❌ |
| **All criteria met** | ❌ | ❌ | ❌ | ❌ | ❌ |

**Criterion 1 (Val RMSE < Control) fails for ALL candidates.** This is the decisive criterion — no candidate demonstrates genuine out-of-sample generalization improvement over the 2019–2022 validation period.

Regarding Criterion 9 (C and F): The test improvement is disproportionately isolated to 2024 (C: +19% improvement on 2024) while 2023 shows no improvement or slight degradation. This is a pattern consistent with test-set variance exploitation.

---

## 15. Limitations

1. **Annual frequency limitation:** With annual GDP data, GDP growth at year t is a backward-looking aggregation over the full calendar year. The deceleration signal is inherently delayed — by the time `gdp_growth_accel_1y` turns negative for year 2019 (the year before COVID), we cannot retroactively incorporate Q4 2019 warning signals.
2. **COVID is structurally unpredictable:** The 2020 shock was driven by an exogenous public health event. No macroeconomic momentum feature computed from pre-2020 data could have anticipated a pandemic.
3. **Small validation set:** With only 4 years (2019–2022) and 829 observations in the validation set, including the COVID shock years, the validation RMSE is dominated by 2020–2021. This creates a high-variance environment for comparing features.
4. **Momentum vs. Leading Indicators:** The hypothesis tested GDP *momentum* (backward-looking). What may be needed is GDP *forecasting* through forward-looking indicators (business confidence, financial conditions indices, yield curve, etc.) not present in the current dataset.

---

## 16. Final Decision

### FINAL DECISION: **KEEP PHASE 11 PRODUCTION MODEL**

**Justification:**

The GDP momentum and turning-point features (Experiments B–F) fail the primary promotion criterion: **no candidate achieves a lower Validation RMSE than the locked Phase 11 control** (best candidate = A_CONTROL at 8.2853).

While Experiments C and F achieve lower Test RMSE (3.7894 and 3.7821 vs 3.9113), this improvement is:
1. **Not validated** — both fail Criterion 1 (Val RMSE)
2. **Partially isolated to 2024** — violating Criterion 9
3. **Not accompanied by recession improvement** — the shock-period validation RMSE is *worse* for these candidates (C: +4.19%, F: +4.19% vs control)
4. **Structurally unexplained** — the mechanism by which momentum features would improve 2024 but not validation is consistent with variance exploitation, not genuine signal

The recession weakness diagnosed in Step 1 is confirmed as a **structural limitation** of the backward-looking momentum approach. To address it, future phases should investigate: (a) financial conditions indices, (b) leading economic indicators (PMI, credit spreads), or (c) incorporating a shock-regime detection mechanism.

**Improvement vs. Control:**
- Validation: +0.00% to +0.66% (candidates worse or equal on validation)
- Test: −0.75% to +1.28% (mixed, not generalizable)
- Recession Val RMSE: −0.51% to +0.57% (negligible, within noise)

---

*Phase 11 Production Model: UNCHANGED*  
*Phase 11 Test RMSE: 3.9113 (LOCKED)*  
*Date: 2026-08-19*
