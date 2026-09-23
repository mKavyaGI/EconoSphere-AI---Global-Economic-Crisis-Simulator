# Phase 20 — Methodology

## T+1 Design
```
Feature year = Y  →  Target year = Y+1
Training: year < Y  (chronological)
Evaluation: year == Y (with valid gdp_growth_next_year)
```

## Origin Set
Phase 19 origins (loaded from metadata): [2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
Phase 20 dynamically computed:           [np.int64(2002), np.int64(2003), np.int64(2004), np.int64(2005), np.int64(2006), np.int64(2007), np.int64(2008), np.int64(2009), np.int64(2010), np.int64(2011), np.int64(2012), np.int64(2013), np.int64(2014), np.int64(2015), np.int64(2016), np.int64(2017), np.int64(2018), np.int64(2019), np.int64(2020), np.int64(2021), np.int64(2022), np.int64(2023), np.int64(2024)]
Origin set match: False

## Reused from Phase 19 (unchanged)
- 31 locked features
- World Bank aggregate exclusion list
- `get_valid_origins()` with `min_train_rows=50`
- `get_train_eval_split()` (year < feature_year / year == feature_year)
- `_make_val_split()` (last 20% of training years for Ridge alpha)
- `build_m0()`, `build_ridge()`, `build_random_forest()` with Phase 19 configs
- `compute_metrics()` (RMSE, MAE, median AE)
- Prediction clipping: Ridge only, tree models never clipped

## Phase 20 New Analyses
- Clipping sensitivity (C0/C1/C2)
- Component ablation (7 subsets)
- Error diversity (residual + prediction correlations)
- Leave-one-origin-out (LOOO)
- Paired Wilcoxon signed-rank test
- Reproducibility audit (2 runs)
- True OOS evaluation
- Operational readiness + independent artifact load

## Production Isolation
Phase 11 artifacts are FROZEN throughout. Hash verified before and after.
No experimental model is saved to models/phase11/.
