# Phase 17 Step 2: Target Availability Audit

- **Target Column**: `gdp_growth_next_year`
- **LATEST_FULLY_LABELED_TARGET_YEAR**: `2021`
- **LATEST_PARTIALLY_LABELED_TARGET_YEAR**: `2024`
- **LATEST_FEATURE_YEAR**: `2025`

## Year-by-Year Target Coverage

|   year |   total_rows |   labeled_rows |   null_rows |   coverage_pct |
|-------:|-------------:|---------------:|------------:|---------------:|
|   2000 |          217 |            200 |          17 |          92.17 |
|   2001 |          217 |            200 |          17 |          92.17 |
|   2002 |          217 |            204 |          13 |          94.01 |
|   2003 |          217 |            204 |          13 |          94.01 |
|   2004 |          217 |            204 |          13 |          94.01 |
|   2005 |          217 |            204 |          13 |          94.01 |
|   2006 |          217 |            205 |          12 |          94.47 |
|   2007 |          217 |            205 |          12 |          94.47 |
|   2008 |          217 |            208 |           9 |          95.85 |
|   2009 |          217 |            209 |           8 |          96.31 |
|   2010 |          217 |            209 |           8 |          96.31 |
|   2011 |          217 |            208 |           9 |          95.85 |
|   2012 |          217 |            208 |           9 |          95.85 |
|   2013 |          217 |            209 |           8 |          96.31 |
|   2014 |          217 |            210 |           7 |          96.77 |
|   2015 |          217 |            209 |           8 |          96.31 |
|   2016 |          217 |            209 |           8 |          96.31 |
|   2017 |          217 |            210 |           7 |          96.77 |
|   2018 |          217 |            209 |           8 |          96.31 |
|   2019 |          217 |            209 |           8 |          96.31 |
|   2020 |          217 |            209 |           8 |          96.31 |
|   2021 |          217 |            208 |           9 |          95.85 |
|   2022 |          217 |            203 |          14 |          93.55 |
|   2023 |          217 |            199 |          18 |          91.71 |
|   2024 |          217 |            185 |          32 |          85.25 |
|   2025 |          217 |              0 |         217 |           0    |

### Training Window Determination
Supervised fitting will use training observations from **2000 to 2021** only. Rows from years > 2021 with missing target values are strictly excluded from supervised fitting.
