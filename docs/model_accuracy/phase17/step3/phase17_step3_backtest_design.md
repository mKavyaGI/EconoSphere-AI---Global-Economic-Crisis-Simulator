# Phase 17 Step 3: Backtest Design

- **Research Question**: Does expanding the training window consistently improve forecasting accuracy?
- **Methodology**: Chronological expanding-window backtests dynamically discovering target availability.
- **Leakage Prevention**: Training only uses features and targets fully available prior to the forecast year. No future data allowed in training or preprocessing.
- **M0**: Frozen Phase 11 model.
- **M1**: Expanding window refit using Phase 11 methodology.
- **N0**: Naive persistence (`gdp_growth_lag1`).
- **N1**: Training set mean.
