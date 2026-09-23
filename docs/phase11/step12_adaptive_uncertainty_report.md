# Phase 11 — Step 12: Adaptive T+1 Prediction Uncertainty Report

**Date:** 2026-08-18  
**Step:** Phase 11, Step 12  
**Objective:** Investigate whether strictly leakage-free adaptive uncertainty methods can produce better-calibrated or meaningfully narrower 90% residual-calibrated prediction intervals, while preserving the official locked Step 10 point model.

---

## 1. Objective

The Step 11 pipeline produces a uniform 90% residual-calibrated prediction interval of ±11.4507 percentage points (~22.90pp total width) for every country regardless of its historical economic volatility. This step investigates whether country-specific or volatility-aware calibration can improve uncertainty quality.

**Critical Constraint:** The Step 10/Step 11 HistGradientBoosting point model is LOCKED and must not change. This experiment modifies only the uncertainty method.

---

## 2. Step 11 Control Methodology

### 2.1 Point Model (Locked)

| Parameter | Value |
|:---|:---|
| Algorithm | `HistGradientBoostingRegressor` |
| Imputation | `SimpleImputer(strategy="median")` |
| `l2_regularization` | 5.0 |
| `learning_rate` | 0.05 |
| `max_depth` | 5 |
| `max_iter` | 300 |
| `random_state` | 42 |
| Feature set | 31 locked Step 8 features |
| Train period | 2000–2018 |
| **Test RMSE** | **3.9113** |

### 2.2 Step 11 Uncertainty Method

1. Train the locked pipeline on TRAIN data (2000–2018).
2. Generate strictly out-of-sample predictions on VALIDATION set (2019–2022).
3. Compute absolute residuals: `abs_res = |y_val - pred_val|`.
4. Compute: `Q90_global = quantile(abs_res, 0.90) = 11.4507`.
5. Apply symmetrically to all predictions: `[prediction ± 11.4507]`.

---

## 3. Why Global Q90 Produces Identical Widths

The Step 11 method uses a single global quantile computed over all validation observations regardless of country. The result is a **uniform interval width** of `2 × 11.4507 = 22.9013` percentage points for every country, every year.

This design has two documented limitations:
1. **Homoscedasticity assumption**: It implicitly assumes error variance is identical across all economies.
2. **No asymmetry**: GDP growth distributions are typically left-skewed (crashes > booms), but the symmetric design does not capture this.

---

## 4. Missing Limitations of the Global Interval

1. **Over-coverage for stable economies**: Developed economies with low GDP growth variance (e.g., Japan ≈ 1.7% historical σ) receive the same wide interval as fragile states with σ > 10%.
2. **Under-coverage risk for volatile states**: The global Q90 is driven in part by COVID-2020 shock. Countries that were disproportionately affected receive no wider interval than stable ones.
3. **No asymmetry**: Economic shocks are negatively skewed — intervals should arguably extend further below than above the forecast.
4. **Calibration set size**: With only 4 validation years (2019–2022), country-level calibration is severely constrained.

---

## 5. Experiment Definitions

### Exp A — Global Residual Q90 (Step 11 Control)

- Compute: `Q90 = quantile(|y_val - pred_val|, 0.90)`
- Apply: `[prediction - Q90, prediction + Q90]`
- Symmetric, global, identical widths everywhere.
- **Serves as the control baseline for all comparisons.**

### Exp B — Country-Conditional Q90

- For each country compute `country_Q90` if the country has ≥ 20 validation observations.
- Fall back to global Q90 otherwise.
- **MIN_COUNTRY_CALIBRATION_N = 20**
- Uses only validation residuals for calibration.

### Exp C — Volatility-Group Calibration

- Compute per-country training-period GDP growth σ from **TRAIN data only**.
- Assign countries to Low / Medium / High σ groups using p33 and p67 thresholds from TRAIN σ distribution.
- Compute validation Q90 separately for each group.
- Apply the relevant group Q90 to each prediction.

### Exp D — Asymmetric Global Residual Intervals

- Compute **signed** residuals: `r = y_val - pred_val`.
- Compute: `q10 = quantile(r, 0.10)`, `q90 = quantile(r, 0.90)`.
- Apply: `lower = prediction + q10`, `upper = prediction + q90`.
- Asymmetric, global, narrower than symmetric control.
- Explicitly described as a **residual-calibrated asymmetric interval**, not a formal conformal interval.

### Exp E — Country-Conditional Asymmetric Intervals

- Per-country signed residual q10/q90 if ≥ 20 val observations.
- Fall back to global q10/q90 otherwise.

### Exp F — Volatility-Scaled Interval

- Compute training-period σ per country.
- Compute: `scale = clip(country_σ / median_σ, 0.75, 1.50)`.
- Apply: `adaptive_Q90 = global_Q90 × scale`.
- Conservative scaling — prevents extreme widths.

---

## 6. Calibration Methodology

**Strict leakage-free protocol:**

| Data | Permitted Use |
|:---|:---|
| TRAIN (2000–2018) | Model training; volatility statistics derivation |
| VALIDATION (2019–2022) | Residual calibration of all intervals |
| TEST (2023–2024) | Final evaluation only; used once |
| INFERENCE (2025) | Point forecast generation; targets remain NaN |

All volatility statistics (σ, group thresholds) are computed from TRAIN. All quantiles are computed from VALIDATION residuals. No test or inference target enters any calibration function.

---

## 7. Calibration Parameters

| Parameter | Value |
|:---|:---|
| Exp A Global Q90 | 11.4507 (verified ≈ Step 11 control) |
| Exp C Vol p33 threshold | 2.3612 |
| Exp C Vol p67 threshold | 3.7236 |
| Exp C Low group Q90 | 9.1108 |
| Exp C Medium group Q90 | 10.2320 |
| Exp C High group Q90 | 15.3129 |
| Exp D signed q10 (lower offset) | −9.4923 |
| Exp D signed q90 (upper offset) | +5.9487 |
| Exp F median training σ | 3.0312 |
| Exp F scale bounds | [0.75, 1.50] |

> **Note on Exp B and Exp E:** With only 4 validation years (2019–2022), every country in the dataset has fewer than 20 validation observations. All countries therefore fall back to the global calibration parameter. Exp B is numerically identical to Exp A, and Exp E is numerically identical to Exp D.

---

## 8. Validation Results

| Method | Val Coverage | Val Mean Width | Val |coverage Gap| | N Outside |
|:---|---:|---:|---:|---:|
| Exp A Global Q90 (Control) | 89.99% | 22.9013 | 0.01% | 83 / 829 |
| Exp B Country Q90 | 89.99% | 22.9013 | 0.01% | 83 / 829 |
| Exp C Volatility Group Q90 | 89.87% | 22.9115 | 0.13% | 84 / 829 |
| Exp D Asymmetric Global | **79.98%** | 15.4410 | **10.02%** | 166 / 829 |
| Exp E Country Asymmetric | **79.98%** | 15.4410 | **10.02%** | 166 / 829 |
| Exp F Volatility Scaled | 90.23% | 24.1328 | 0.23% | 81 / 829 |

> [!WARNING]
> **Exp D and Exp E** achieve a narrower mean interval width of 15.44pp but at the cost of severe under-coverage: only 79.98% of validation observations lie within the interval, far below the nominal 90%. These methods are **REJECTED** due to unacceptable coverage sacrifice.

---

## 9. Test Results (Final Evaluation)

Test data was used once, after all calibration decisions were frozen.

| Method | Test Coverage | Test Mean Width | Test |Coverage Gap| | N Outside |
|:---|---:|---:|---:|---:|
| Exp A Global Q90 (Control) | 98.44% | 22.9013 | 8.44% | 6 / 384 |
| Exp B Country Q90 | 98.44% | 22.9013 | 8.44% | 6 / 384 |
| Exp C Volatility Group Q90 | 98.96% | 22.7749 | 8.96% | 4 / 384 |
| Exp D Asymmetric Global | 96.09% | 15.4410 | 6.09% | 15 / 384 |
| Exp E Country Asymmetric | 96.09% | 15.4410 | 6.09% | 15 / 384 |
| Exp F Volatility Scaled | 98.96% | 23.9755 | 8.96% | 4 / 384 |

> [!NOTE]
> High test coverage across all methods is expected: the 2023–2024 test period (excluding the COVID shock period captured in validation) had relatively moderate GDP growth variability, and the globally-calibrated 22.90pp interval was quite conservative for this period.

---

## 10. Coverage Comparison

The coverage target is **90% empirical coverage at nominal 90%** (i.e., coverage gap ≈ 0).

| Method | Val Coverage | Test Coverage | Val Calibration Quality |
|:---|---:|---:|:---|
| Exp A (Control) | 89.99% | 98.44% | **Excellent** — 0.01% gap |
| Exp B | 89.99% | 98.44% | Identical to control |
| Exp C | 89.87% | 98.96% | Good — 0.13% gap |
| Exp D | 79.98% | 96.09% | **Unacceptable** — 10.02% gap |
| Exp E | 79.98% | 96.09% | **Unacceptable** — 10.02% gap |
| Exp F | 90.23% | 98.96% | Good — 0.23% over |

**Key observation:** Exp A (Control) achieves the best validation calibration at 89.99%, a 0.01% absolute gap from the nominal 90% target. This is nearly ideal.

---

## 11. Interval-Width Comparison

| Method | Val Mean Width | Val Median Width | Narrower than Control? |
|:---|---:|---:|:---|
| Exp A (Control) | 22.90 pp | 22.90 pp | — |
| Exp B | 22.90 pp | 22.90 pp | No (identical) |
| Exp C | 22.91 pp | 20.46 pp | Slightly (median), not mean |
| Exp D | **15.44 pp** | **15.44 pp** | **Yes — but 79.98% val coverage** |
| Exp E | **15.44 pp** | **15.44 pp** | **Yes — but 79.98% val coverage** |
| Exp F | 24.13 pp | 22.46 pp | No (wider than control) |

No method achieves both narrower mean interval AND adequate validation coverage (≥85%) compared to the control.

---

## 12. Five-Country 2026 Forecasts

*Feature Year: 2025 | Target Year: 2026*

Since the final decision is to keep the Step 11 Global Q90 control, both the "Step 11" and "Adaptive" columns use the same method.

| Country | Point Forecast | Step 11 Lower | Step 11 Upper | Width |
|:---|---:|---:|---:|---:|
| **India (IND)** | 1.02% | −10.43% | +12.47% | 22.90% |
| **China (CHN)** | 3.45% | −8.01% | +14.90% | 22.90% |
| **USA** | 3.78% | −7.67% | +15.23% | 22.90% |
| **Japan (JPN)** | 3.43% | −8.02% | +14.88% | 22.90% |
| **UK (GBR)** | 0.60% | −10.85% | +12.05% | 22.90% |

> [!IMPORTANT]
> The 2026 point forecasts differ slightly from the Step 11 published values. This is because the 2025 input feature data has been updated since Step 11 was originally run. The uncertainty Q90 = 11.4507 is unchanged.

---

## 13. Selected Uncertainty Method

**FINAL DECISION: KEEP STEP 11 GLOBAL Q90 CONTROL**

**Q90 = 11.4507  |  Interval Width = 22.9013 pp**

### Rejection Rationale for Each Candidate

| Method | Rejection Reason |
|:---|:---|
| Exp B (Country Q90) | Numerically identical to control — no improvement possible with < 20 val obs per country |
| Exp C (Volatility Group) | Val coverage slightly lower (89.87%) with slightly wider mean width — no improvement |
| Exp D (Asymmetric Global) | Val coverage 79.98% — severe under-coverage; **REJECTED** |
| Exp E (Country Asymmetric) | Numerically identical to Exp D; same rejection |
| Exp F (Volatility Scaled) | Wider mean intervals (24.13 pp) — strictly worse than control |

---

## 14. Statistical Limitations

1. **Not formal confidence intervals:** All intervals are empirical, residual-calibrated 90% intervals. They are NOT formally conformal prediction intervals unless the exchangeability assumption is satisfied.

2. **Calibration sample size:** With only 4 validation years (2019–2022), each country has at most 4 observations. Per-country calibration is statistically infeasible — the MIN_COUNTRY_CALIBRATION_N = 20 threshold is correctly enforced.

3. **COVID-19 inflation:** The 2019–2022 validation set includes the 2020 global COVID shock, which strongly inflates the 90th percentile of absolute residuals. The resulting wide interval may be overly conservative during normal macro periods.

4. **Asymmetric intervals fail in validation:** The validation period (2019–2022) is heavily influenced by COVID downward shocks, creating a large asymmetry in the signed residual distribution. However, this asymmetry is insufficient to achieve adequate 90% coverage in the validation set.

5. **Temporal distributional shift:** High test coverage (98.44%) vs validation calibration (89.99%) reflects distributional differences between 2019–2022 (COVID included) and 2023–2024 (post-recovery).

---

## 15. Reproducibility Information

- **Script:** `apps/api/ml/train_t1_adaptive_uncertainty.py`
- **Random seeds:** All randomness controlled by `random_state=42`
- **Deterministic:** Two identical runs produce identical calibration parameters and forecasts
- **Raw MD5 verified:** `8ac7e0b2bf09fbe89289f82d0c7cf25e` ✓
- **Point model RMSE:** 3.9113 (|diff| < 0.0001 from locked value) ✓

---

## 16. Final Recommendation

> **OUTCOME B: KEEP STEP 11 GLOBAL Q90 CONTROL**

**Q90 = 11.4507 | Interval = prediction ± 11.4507 | Width ≈ 22.90 pp**

The empirical evidence shows that:
- The Step 11 Global Q90 achieves near-perfect validation calibration (89.99% vs nominal 90%).
- No adaptive method improves mean interval width while maintaining ≥85% validation coverage.
- The asymmetric methods (Exp D, Exp E) achieve narrower intervals but at the cost of only 79.98% validation coverage — a 10 percentage-point under-coverage that is statistically unacceptable for a nominal 90% interval.
- The volatility-group method (Exp C) provides an interesting structure (different group Q90s) but fails to materially improve overall calibration metrics.
- Country-conditional calibration (Exp B, Exp E) is not feasible with only 4 validation observations per country.

**The fundamental constraint is the small validation calibration sample** (only 4 years per country). Future improvement of uncertainty estimation should focus on either:
1. Expanding the validation window if additional historical data becomes available, or
2. Implementing formal conformalized quantile regression if sufficient country-level calibration samples can be obtained.
