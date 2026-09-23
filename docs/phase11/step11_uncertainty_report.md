# Phase 11 — Step 11: Prediction Uncertainty Report
## Interval Forecast Generation and Methodological Limitations

**Date:** 2026-08-17
**Objective:** Transition EconoSphere AI from deterministic point-forecasts to interval-forecasts, providing statistical bounds for expected GDP growth using out-of-sample data.

---

### 1. Uncertainty Methodology

EconoSphere AI generates **90% residual-calibrated prediction intervals**.
This is a computationally efficient, conformal-style calibration technique that assumes future errors will follow a similar distribution to historical validation errors.

**Calibration Procedure:**
1. **Train Model:** The Step 10 locked HistGradientBoosting model is trained on data spanning 2000–2018.
2. **Generate Validation Predictions:** The model generates strict out-of-sample point predictions on the Validation set (2019–2022).
3. **Calculate Absolute Residuals:** The absolute differences between actual GDP growth and predicted GDP growth are captured: `abs_res = abs(y_val - pred_val)`.
4. **Determine Quantile:** The 90th percentile of these absolute residuals ($Q_{0.90}$) is calculated.
5. **Construct Intervals:** For any future observation (Test or Inference), the 90% residual-calibrated prediction interval is defined symmetrically around the predicted point forecast: `[Prediction - Q0.90, Prediction + Q0.90]`.

---

### 2. Calibration Parameters and Audit

- **Calibration Set Boundaries:** Feature years 2018–2021 (Predicting target years 2019–2022).
- **Quantile Computed ($Q_{0.90}$):** `11.4507`
- **Interval Formula:** `[Prediction - 11.4507, Prediction + 11.4507]`

**Leakage Audit (PASSED):**
- **Test Set Exclusion:** No data from 2023 or 2024 was used to compute the residual quantile.
- **Inference Target Exclusion:** The actual target for 2025/2026 was strictly excluded.
- **Training Target Exclusion:** The model did not calculate uncertainty on the training set (which would artificially deflate the interval due to over-fitting).

---

### 3. Five-Country 2026 Prediction Intervals

*Feature Year: 2025 | Target Year: 2026*

| Country | Forecast | 90% Lower Bound | 90% Upper Bound | Interval Width |
| :--- | :--- | :--- | :--- | :--- |
| **India (IND)** | 6.53% | -4.92% | +17.98% | 22.90% |
| **China (CHN)** | 5.39% | -6.06% | +16.84% | 22.90% |
| **USA** | 3.01% | -8.44% | +14.46% | 22.90% |
| **Japan (JPN)** | 1.47% | -9.98% | +12.92% | 22.90% |
| **UK (GBR)** | 0.86% | -10.59% | +12.31% | 22.90% |

---

### 4. Limitations and Statistical Assumptions

This interval should be accurately interpreted by downstream applications, noting the following structural limitations:

1. **Not a Formal Confidence Interval:** This is not a parametric confidence interval on a mean parameter. It is an empirical bound on individual future observations based strictly on past error distributions.
2. **Homoscedasticity Assumption:** The interval width ($22.90\%$) is uniform across all countries. The model implicitly assumes that the variance of the error term is identical regardless of the country's economic volatility. Future steps should consider *Conformalized Quantile Regression (CQR)* or conditional variance modeling to generate country-specific dynamic widths.
3. **Black Swan Skew:** The calibration set (2019–2022) includes the 2020 COVID-19 pandemic shock. This heavily inflates the 90th percentile of absolute residuals (to $\pm11.45\%$), leading to extremely wide prediction bounds that may be overly pessimistic for stable macro periods.
4. **Symmetry:** The intervals are strictly symmetric. Economic shocks are typically negatively skewed (crashes are larger than booms), but this methodology does not currently account for skewed error distributions.
