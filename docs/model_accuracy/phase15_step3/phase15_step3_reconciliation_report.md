# Phase 15 Step 3: Metric Reconciliation Report

## Executive Summary
This report programmatically traces and explains the discrepancy between the metrics reported in Phase 15 Step 1 and Step 2.

## GBR Discrepancy Breakdown
The discrepancy arose from two factors:
1. **Mislabeling**: The user's prompt cited "~7.02 Test RMSE" for GBR in Step 1. In reality, 7.3658 was the **Validation RMSE** for GBR, while the actual **Test RMSE** was 0.8222.
2. **Methodological Difference**: Step 2 averaged the Test RMSE across two chronological folds (Fold 2018 and Fold 2020) instead of relying solely on the canonical Phase 11 model.

### Programmatic Proof
- **GBR Step 1 Validation RMSE (2019-2022)**: `7.36580`
- **GBR Step 1 Test RMSE (2023-2024)**: `0.82221`
- **Step 2 Fold 2018 GBR Test RMSE**: `0.82221`
- **Step 2 Fold 2020 GBR Test RMSE**: `1.38494`
- **Step 2 Averaged Test RMSE**: `1.10358`

## Conclusion
The single authoritative evaluation methodology must be the **Canonical Fold 2018**, exactly matching the Phase 11 frozen production model. Averaging across folds obfuscates the direct comparability. The remaining evaluation in this step strictly adheres to the Fold 2018 canonical split.
