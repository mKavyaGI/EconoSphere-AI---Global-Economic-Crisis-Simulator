# Phase 12 Step 4: Regime-Aware / Conditional GDP Forecasting Experiment

## 1. Objective
Investigate whether the model's major weakness — poor performance during recessions, shocks, and unusual growth regimes — can be improved through explicitly modeling regimes (Growth and Stress) using historically available indicators.

## 2. Phase 11 Baseline
- **Model**: HistGradientBoostingRegressor
- **Validation RMSE**: 8.2853
- **Test RMSE**: 3.9113
- **Walk-forward RMSE**: 6.0967
- **Recession Walk-forward RMSE**: 10.1908

## 3. Motivation
Phase 12 Steps 1-3 demonstrated that simply adding ordinary features, backward-looking momentum, or macro forward-looking indicators failed to out-perform the 8.2853 validation RMSE baseline. We hypothesized that conditional/regime-aware modeling would provide the necessary flexibility.

## 4. Regime Definitions
We formulated regimes using purely historical info (data up to `t-1` to predict `t`):
- **Growth Regime**: LOW, MODERATE, HIGH thresholds defined by the 33rd and 67th percentiles of historical GDP growth (`gdp_growth_lag1`) in the training partition.
- **Stress Regime**: NORMAL, STRESS defined by the 80th percentile of rolling 3-year GDP volatility (`gdp_growth_rolling_std_3`) in the training partition.

## 5. Temporal Safety
Target-year leakage was rigorously avoided. Regime assignments for test year `t` never observed the GDP growth, inflation, or unemployment changes of year `t`. Thresholds were re-learned strictly on training partitions during walk-forward.

## 6. Experiments & Validation Results
We evaluated several regime-aware structures:
- **A. Control**: Val RMSE = 8.2853
- **B. Regime Features**: (Adding numeric regime labels as features) **Val RMSE = 8.2598**
- **C. Regime-Specific Models**: (Separate models per Stress regime) Val RMSE = 8.3696
- **D. Regime + Fallback**: (Models per Growth regime with fallback) Val RMSE = 8.5630
- **E. Regime Residual Correction**: Val RMSE = 8.2839
- **F. Conservative Ensemble**: Val RMSE = 8.2840

## 7. Test Results & Walk-forward Evaluation
Candidate B (Regime Features) successfully surpassed the Validation Criterion (8.2598 < 8.2853).
- **Test RMSE**: 3.8640 (beat Control's 3.9113)
- **Walk-forward RMSE**: 6.0943 (beat Control's 6.0967)
- **Recession Walk-forward RMSE**: 10.1815 (beat Control's 10.1908)

## 8. Final Promotion Decision
Because **Candidate B (Regime Features)** consistently beat the Phase 11 Control in Validation RMSE, Test RMSE, overall Walk-forward RMSE, and Recession Walk-forward RMSE—without any temporal leakage—it passes all promotion criteria.

**DECISION: PROMOTE CANDIDATE TO EXPERIMENTAL WINNER**

## 9. Recommendation for Phase 12 Step 5
Having proven that simple regime-aware feature encoding provides a structural improvement in forecasting generalization, Step 5 should focus on **interpretable explainability**. We must confirm exactly how the regime features alter the model's decision pathways (e.g., via SHAP analysis) and determine if this logic aligns with macroeconomic theory prior to considering a full production deployment.
