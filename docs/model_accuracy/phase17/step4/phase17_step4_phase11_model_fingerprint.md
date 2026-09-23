# Phase 17 Step 4: Phase 11 Model Fingerprint

```json
{
    "EXPECTED_PHASE11_CONFIGURATION": {
        "feature_count": 31,
        "features": [
            "exchange_rate_lcu_usd",
            "tariff_rate_pct",
            "remittances_usd",
            "fdi_net_inflow_usd",
            "unemployment_pct",
            "imports_pct_gdp",
            "tax_revenue_pct_gdp",
            "exports_pct_gdp",
            "interest_rate_pct",
            "reserves_usd",
            "current_account_pct_gdp",
            "inflation_cpi_pct",
            "population_total",
            "gdp_current_usd",
            "gdp_growth_lag1",
            "gdp_growth_lag2",
            "gdp_growth_lag3",
            "inflation_lag1",
            "unemployment_lag1",
            "exports_lag1",
            "imports_lag1",
            "gdp_growth_rolling_mean_3",
            "gdp_growth_rolling_std_3",
            "gdp_growth_rolling_mean_5",
            "inflation_rolling_mean_3",
            "trade_openness",
            "trade_balance_ratio",
            "log_gdp_usd",
            "log_population",
            "gdp_growth_rolling_std_5",
            "inflation_rolling_std_3"
        ],
        "hyperparameters": {
            "max_depth": 5,
            "l2_regularization": 5.0,
            "learning_rate": 0.05,
            "max_iter": 300,
            "random_state": 42
        },
        "target": "t1_gdp_growth",
        "train_period": "<= 2018",
        "test_period": "2023-2024"
    },
    "ACTUAL_EXTRACTED_PHASE11_CONFIGURATION": {
        "model_class": "<class 'sklearn.pipeline.Pipeline'>",
        "feature_count": 31,
        "features": [
            "exchange_rate_lcu_usd",
            "tariff_rate_pct",
            "remittances_usd",
            "fdi_net_inflow_usd",
            "unemployment_pct",
            "imports_pct_gdp",
            "tax_revenue_pct_gdp",
            "exports_pct_gdp",
            "interest_rate_pct",
            "reserves_usd",
            "current_account_pct_gdp",
            "inflation_cpi_pct",
            "population_total",
            "gdp_current_usd",
            "gdp_growth_lag1",
            "gdp_growth_lag2",
            "gdp_growth_lag3",
            "inflation_lag1",
            "unemployment_lag1",
            "exports_lag1",
            "imports_lag1",
            "gdp_growth_rolling_mean_3",
            "gdp_growth_rolling_std_3",
            "gdp_growth_rolling_mean_5",
            "inflation_rolling_mean_3",
            "trade_openness",
            "trade_balance_ratio",
            "log_gdp_usd",
            "log_population",
            "gdp_growth_rolling_std_5",
            "inflation_rolling_std_3"
        ],
        "hyperparameters": {
            "l2_regularization": 5.0,
            "learning_rate": 0.05,
            "max_depth": 5,
            "max_iter": 300,
            "random_state": 42
        },
        "imputer_strategy": "median"
    }
}
```
