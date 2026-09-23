# EconoSphere AI: Model Accuracy Next Steps & Roadmap

Based on the forensic audit of the Phase 11 production model, the following experimental improvements are recommended. **These steps should not mutate the Phase 11 baseline. They belong in a new experimental phase.**

## Priority 1: High-Confidence Fixes
1. **Model Capacity Expansion**: The HistGradientBoostingRegressor is currently constrained to `max_depth=5`. Given the complexity of the economic panel, a deeper tree structure or hyperparameter tuning might capture non-linear interactions better.
2. **Missing Feature Handling**: The model currently uses median imputation for gaps. Implementing adaptive missingness features (e.g., MissingIndicator) or natively handling NaNs inside the tree model will likely improve robustness on sparse rows.

## Priority 2: Experimental Improvements
1. **Regime-Aware Ensembling**: The year-wise error analysis shows large deviations during crises (e.g., 2020 COVID-19 shock). We should introduce `growth_regime` and `stress_regime` as explicit categorical features, or train specialized sub-models for crisis periods (Candidate B isolation).
2. **Advanced Target Encoding**: Implement spatial or temporal embeddings to help the model distinguish between structural economic classes.

## Priority 3: Not Recommended
1. **Target Leakage Risks**: Do NOT include real-time, same-year leading indicators that won't be available at the January 1st forecasting boundary.
2. **Complex Deep Learning**: Given the structured, sparse, and tabular nature of the data, deep learning (LSTMs, Transformers) is likely to overfit compared to optimized gradient boosted trees.
