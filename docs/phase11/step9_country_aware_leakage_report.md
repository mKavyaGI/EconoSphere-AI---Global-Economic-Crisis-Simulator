# Phase 11 -- Step 9: Country-Aware Leakage Report

This document audits the zero-leakage constraints applied during the Country-Aware T+1 Forecasting experiment.

## 1. Automated Tests Implemented
All leakage assertions were executed via the test suite `apps/api/tests/test_t1_country_aware.py`.

### A. Raw Dataset Integrity
- **Check**: MD5 checksum of `data/raw/master_panel.csv`.
- **Expected**: `8ac7e0b2bf09fbe89289f82d0c7cf25e`
- **Result**: PASS (Unchanged)

### B. Target Predictor Isolation
- **Check**: Is `gdp_growth_next_year` ever included in the features list?
- **Result**: PASS

### C. Future Target Bleed in Statistics
- **Check**: Were historical statistics (`country_hist_avg_gdp_growth`, etc.) calculated using any data from `year > 2018`?
- **Result**: PASS. A strict `train_mask = (df["year"] >= 2000) & (df["year"] <= 2018)` was enforced.

### D. OneHotEncoder Leakage
- **Check**: Was the `OneHotEncoder` fitted on the full dataset or just the training split?
- **Result**: PASS. The encoder was strictly initialized and fitted inside the sklearn `ColumnTransformer` during `pipe.fit(X_train, y_train)`. Validation/Test/Inference splits were processed only via `pipe.predict()`.

### E. Inference Safety
- **Check**: Are target values for inference rows (2025 features predicting 2026 target) strictly `NaN`?
- **Result**: PASS. The test explicitly verified that no fabricated 2026 target values existed.

### F. No Random Temporal Shuffling
- **Check**: Was `train_test_split` invoked?
- **Result**: PASS. The dataset was split chronologically by static year boundaries.

## 2. Adversarial Leakage Vulnerability Check
By isolating the structural feature calculations inside the script using `train_df`, the code guarantees that modifying the target variable in the Validation or Test split cannot mathematically alter the features input to the model. Validation and test targets exist exclusively as `y_val` and `y_test` and are completely decoupled from `X_train` generation.

## 3. Overall Leakage Conclusion
**PASS**
Zero data leakage occurred. The performance degradation measured in the Step 9 experiment was a genuine empirical failure of the country-aware hypothesis, rather than an artifact of data leakage or test contamination.
