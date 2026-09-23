# Phase 17 Step 3: Leakage & Reproducibility Audit

- **Temporal Cutoff**: Enforced by strictly using `feature_year <= train_cutoff` for training.
- **Preprocessing**: Imputer fitted solely on `train_df` within each fold.
- **Test Isolation**: Targets from evaluation year are only used for metric calculation.
- **Hash Verification**: Phase 11 artifacts verified via MD5/SHA256 before and after execution.
