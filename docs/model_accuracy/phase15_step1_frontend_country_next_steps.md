# Phase 15 Step 1: Model Accuracy Next Steps & Priority

Based on the focused audit of the Tier A frontend countries, we recommend the following experimental strategy.

## Country Prioritization

### Priority 1: Highest user-impact + Poor accuracy
These countries fail to beat basic naive or dummy baselines, or exhibit extreme error during crisis years:
USA, CHN, DEU, JPN, IND, GBR, BRA, FRA, CAN

### Priority 2: Moderate improvement opportunity
(No Tier A countries fall strictly here; they either beat baselines reliably or fail completely).

### Priority 3: Acceptable / Low Priority
These countries show acceptable accuracy and beat baselines:
AUS

## Recommended Experimental Strategy

### Priority A — Data and Feature Fixes
1. **Skewed Raw Feature Transformations**: Features like `gdp_current_usd`, `population_total`, `reserves_usd` have extreme scale and skew. We must introduce robust transformations (e.g. QuantileTransformer) or rely exclusively on their `log_` and ratio equivalents.
2. **Missingness Indicators**: For countries with high feature missingness, explicit NaN indicator columns should be tested.
3. **Target Scaling**: Instead of predicting raw percentages, consider standardizing the target per country or incorporating spatial embeddings.

### Priority B — Model Improvements
1. **XGBoost Comparison**: Evaluate XGBoost with strict monotonic constraints to prevent absurd predictions on out-of-distribution inputs.
2. **LightGBM**: Test as a fast alternative to HistGradientBoostingRegressor, providing native NaN handling and categorical support.

### Priority C — Regime-Aware Approaches
1. Implement a distinct crisis-regime feature (e.g., `is_crisis_year` via external macro factors) to decouple high-variance shocks from normal structural growth.

## Strict Validation Protocol
- Do NOT use the 2023-2024 test set for hyperparameter tuning.
- Use TimeSeriesSplit or fixed Val (2019-2022) for model selection.
- Test set remains protected for final Phase 15 Step 3 evaluation.
