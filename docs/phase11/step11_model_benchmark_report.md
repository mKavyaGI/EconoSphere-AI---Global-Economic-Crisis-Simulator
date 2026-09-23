# Phase 11 — Step 11: Model Benchmarking and Selection Report
## Exact Reproducibility, Algorithm Benchmarking, and Final Promotion Decision

**Date:** 2026-08-17
**Objective:** Lock the Step 10 control model, benchmark it against robust alternative tabular regression algorithms, and decide whether to promote a new model or keep the locked control.

---

### 1. Locked Step 10 Control Reproducibility

Before proceeding to benchmarking, the tuned Step 10 `HistGradientBoosting` model configuration was explicitly extracted and locked:
- **Imputation:** `SimpleImputer(strategy="median")`
- **Hyperparameters:** `{'l2_regularization': 5.0, 'learning_rate': 0.05, 'max_depth': 5, 'max_iter': 300, 'random_state': 42}`

**Reproducibility Verification:**
The script `reproduce_t1_step10_best.py` ran this exact configuration twice out-of-sample:
- **Run 1 Test RMSE:** `3.9113`
- **Run 2 Test RMSE:** `3.9113`
- **Difference:** < 0.0001
- **Status:** **[SUCCESS] EXACT REPRODUCIBILITY ACHIEVED.**

This model was officially designated as `STEP_11_LOCKED_CONTROL`.

---

### 2. Experimental Setup and Benchmark Scope

The Locked Control was evaluated against robust Scikit-Learn tree-based regression algorithms. Third-party packages (XGBoost, LightGBM, CatBoost) were excluded because they are not natively installed in the environment and dependencies were strictly constrained.

**Temporal Splits (Chronological):**
- **TRAIN:** 2000–2018
- **VALIDATION:** 2019–2022
- **TEST:** 2023–2024
- **INFERENCE:** 2025

**Models Benchmarked:**
- **A.** `HistGradientBoosting` (Locked Control)
- **B.** `RandomForestRegressor` (Default, 100 trees)
- **C.** `ExtraTreesRegressor` (Default, 100 trees)
- **D.** `GradientBoostingRegressor` (Default)
- **E.** `RandomForestRegressor` (Bounded Tuning: depth/samples limit)

---

### 3. Empirical Results

All models were evaluated strictly out-of-sample.

| Configuration | Val RMSE (2019-2022) | Test RMSE (2023-2024) |
| :--- | :--- | :--- |
| **A. HistGradientBoosting (Locked Control)** | **8.2853** | 3.9113 |
| B. RandomForestRegressor (Default) | 8.6108 | 3.7645 |
| C. ExtraTreesRegressor (Default) | 8.5000 | 3.7130 |
| D. GradientBoostingRegressor (Default) | 8.7800 | 4.0190 |
| E. RandomForest (Tuned) | 8.3985 | **3.5251** |

---

### 4. Final Model Selection

**Decision Rule:** Promote a candidate ONLY if it beats the locked Step 10 control on Validation RMSE, AND its final Test RMSE is better than the locked control.

**Analysis:**
The Tuned RandomForest (Model E) achieved an astonishing Test RMSE of **3.5251**, vastly outperforming the control on the 2023-2024 period. However, it achieved a Validation RMSE of **8.3985**, which is worse than the Control's Validation RMSE of **8.2853**.

Because the strict selection framework mandates that validation performance must be the primary selection criterion (to prevent blindly chasing Test set variance), the promotion condition failed.

**Final Decision:** **KEEP STEP 10 LOCKED CONTROL.**
The official Model remains `HistGradientBoosting` (`max_depth=5, max_iter=300, learning_rate=0.05, l2=5.0`).

---

### 5. Temporal Robustness Analysis (Locked Control)

Breaking down the Test Set by year to evaluate robustness:

| Year | N Obs | Test RMSE | Test MAE | Test R² |
| :--- | :--- | :--- | :--- | :--- |
| **2023** | 199 | 4.7175 | 2.5504 | -0.0208 |
| **2024** | 185 | 2.7955 | 2.0070 | +0.0790 |

*Note:* 2023 was a highly chaotic post-recovery macro period, driving the majority of the error variance.

---

### 6. Five-Country Verification (Locked Control)

**Target Year 2024 (Feature Year 2023):**
- **IND:** Actual=7.57%, Predicted=6.90% (Error= -0.67%)
- **CHN:** Actual=4.96%, Predicted=5.70% (Error= +0.74%)
- **USA:** Actual=2.16%, Predicted=2.73% (Error= +0.57%)
- **JPN:** Actual=1.19%, Predicted=1.73% (Error= +0.54%)
- **GBR:** Actual=1.39%, Predicted=0.81% (Error= -0.58%)

**2026 Forecasts (Feature Year 2025):**
- **IND 2026 Forecast:** 6.53%
- **CHN 2026 Forecast:** 5.39%
- **USA 2026 Forecast:** 3.01%
- **JPN 2026 Forecast:** 1.47%
- **GBR 2026 Forecast:** 0.86%
