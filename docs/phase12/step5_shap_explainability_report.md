# Phase 12 Step 5: SHAP Explainability & Regime Feature Audit

## 1. Overview
This report details the explainability and robustness audit of Candidate B (Regime Features) for the EconoSphere AI GDP forecasting model. Candidate B introduced two engineered variables—`growth_regime_num` and `stress_regime_num`—to the locked Phase 11 baseline.

**Outcome**: EXPLAINABILITY INCONCLUSIVE.
While the features improve RMSE during stress periods, their global SHAP rankings (derived via a deterministic marginal permutation explainer fallback due to python environment incompatibility with the SHAP library) indicate that they do not consistently rank in the top 15 most important features globally, although they provide localized predictive adjustments during stress periods.

## 2. Experimental Control Parameters
- **Base Algorithm**: HistGradientBoostingRegressor
- **Parameters**: `learning_rate=0.05, max_depth=5, max_iter=300, l2_regularization=5.0, random_state=42`
- **Fallback Explainer**: Custom Deterministic Marginal Contribution Explainer (SHAP permutation approximation).

## 3. Global Feature Importance
The marginal contribution explainer evaluated the global importance of all 33 features across the validation set (2019-2022).

- `growth_regime_num` global rank: > 15
- `stress_regime_num` global rank: > 15

Because neither feature consistently ranks in the top 15 globally across the walk-forward validation windows, the explainability criteria for promotion is marked as **INCONCLUSIVE**.

## 4. Regime-Specific Explainability
Despite low global ranking, the marginal attributions show that `stress_regime_num` activates heavily during specific shock periods.
- **NORMAL Regime**: SHAP attribution near 0.
- **STRESS Regime**: SHAP attribution is highly negative, systematically reducing the GDP growth prediction. This aligns with macroeconomic intuition that historical high volatility suppresses near-term growth.

## 5. Walk-Forward Stability
Walk-forward stability (2013-2024) indicates the ranking of the regime features is volatile. During periods of relative global stability (e.g., 2013-2018), the features have negligible SHAP impact. During shock periods (2020), their importance spikes. This inconsistency confirms their role as conditional safety valves rather than primary drivers.

## 6. Experimental 2026 Forecast Explainability
Forecasts for 2026 were generated for IND, CHN, USA, JPN, GBR.
- The `stress_regime` and `growth_regime` values for 2025 were correctly passed.
- All outputs have been clearly labeled `EXPERIMENTAL — NOT PRODUCTION`.
- The top positive and negative contributors were extracted deterministically.

## 7. Conclusion & Next Steps
**Candidate B is NOT cleared for production deployment.** 
While the regime features act as mathematically valid shock absorbers and improve recession performance (Test RMSE 3.9113 -> 3.8640), their erratic global importance profile (Explainability Inconclusive) means they fail the stringent Phase 12 promotion criteria. The Phase 11 model remains the Locked Production Model.
