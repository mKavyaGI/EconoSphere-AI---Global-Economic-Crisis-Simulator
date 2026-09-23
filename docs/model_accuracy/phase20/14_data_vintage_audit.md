# Phase 20 — Data Vintage and Leakage Audit

## T+1 Design Verification

For every origin Y:
- Features: year == Y (data available at end of year Y)
- Target: gdp_growth_next_year (= actual GDP growth in Y+1)
- No feature from year Y+1 or later is used

This is structurally enforced by `get_train_eval_split()`:
```python
train_mask = (df[YEAR_COL] < feature_year) & (df[TARGET_AVAIL_COL] == 1)
eval_mask  = (df[YEAR_COL] == feature_year) & (df[TARGET_AVAIL_COL] == 1)
```

## Leakage Rules Status

| Rule | Description | Status |
|---|---|---|
| L1 | Target not in feature set | PASS |
| L2 | Training rows year < feature_year | PASS |
| L3 | StandardScaler fitted on training only | PASS |
| L4 | SimpleImputer fitted on training only | PASS |
| L5 | Fixed 31-feature locked set | PASS |
| L6 | Ridge alpha from val split only | PASS |
| L7 | No test-set model selection | PASS |
| L8 | No test-driven clipping | PASS |
| L9 | OOS not peeked before design frozen | PASS |
| L10 | No random train/test split | PASS |
| L11 | Production artifacts not mutated | PASS |
| L12 | No synthetic/fabricated data | PASS |

## Publication Vintage Limitation
World Bank GDP data is subject to revision. Feature values used here reflect
the snapshot in `master_panel_t1_missingness.csv`. Actual publication-vintage
timestamps are not available in the current dataset. This limitation is
acknowledged but does not invalidate the chronological split design.
