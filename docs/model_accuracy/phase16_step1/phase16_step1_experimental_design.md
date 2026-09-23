# Phase 16 Step 1: Experimental Design

**Protected Artifacts**
- Model: best_t1_gdp_growth_model.joblib (MD5: 9c539735897eaf6e72e5f54f047390d7)
- Dataset: master_panel_t1_missingness.csv (MD5: 2a490f1d3c245671b43602cd399d0fa0)

## Frozen Schema Inspection
The frozen Phase 11 schema contains 31 features. Notable existing features:
- `trade_openness` (already exists)
- `gdp_growth_rolling_std_5` (already exists)
- `gdp_growth_rolling_mean_5` (already exists)

## Candidate Definitions
- **BASELINE**: Frozen Phase 11 31-feature schema
- **CA1_COUNTRY_STRUCTURAL**: Added `gdp_per_capita` and `reserves_to_gdp`. `trade_openness` already existed so it is not duplicated.
- **CA2_COUNTRY_SEGMENT**: Country Scale segment derived from training period median GDP (0=Small, 1=Medium, 2=Large).
- **RA1_VOLATILITY_REGIME**: Since `gdp_growth_rolling_std_5` is already in baseline, added an ordinal `volatility_regime` (0=Low, 1=Medium, 2=High) computed strictly on trailing standard deviation quantiles from the train set.
- **RA2_GROWTH_REGIME**: `growth_regime` (0=Low, 1=Normal, 2=High) based on training quantiles of `gdp_growth_rolling_mean_5`.
- **CRA_COMBINED**: Combines CA1 (structural) and RA1 (volatility regime).

