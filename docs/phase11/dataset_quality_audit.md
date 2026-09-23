# EconoSphere AI — Dataset Quality & Coverage Audit

**Phase**: 11 — AI Agents & ML Forecasting  
**Dataset**: `data/raw/master_panel.csv`  
**Generated**: 2026-08-17 05:56 UTC  
**Script**: `apps/api/ml/data_audit.py`

> ⚠️ **Read-Only Audit** — this report documents observed facts only.  
> No rows, columns, or values were modified, imputed, or deleted.

---

## 1. Dataset Dimensions

| Metric | Value |
|---|---|
| Rows | 6,136 |
| Columns | 19 |
| Feature columns (numeric) | 16 |
| Identifier columns | 3 (`country_code`, `country_name`, `year`) |
| File size | 1206.4 KB |

## 2. Schema — Column Inventory

| Column | Dtype | Non-Null | Missing % | Description |
| --- | --- | --- | --- | --- |
| country_code | str | 6,136 | 0.00% | ISO-3166 alpha-3 country code (identifier) |
| country_name | str | 6,136 | 0.00% | Full country name (identifier) |
| year | int64 | 6,136 | 0.00% | Observation year (identifier) |
| exchange_rate_lcu_usd | float64 | 5,355 | 12.73% | Official exchange rate — local currency per 1 USD |
| tariff_rate_pct | float64 | 3,324 | 45.83% | Applied weighted-mean tariff rate on all products (%) |
| remittances_usd | float64 | 4,833 | 21.24% | Personal remittances received (current USD) |
| fdi_net_inflow_usd | float64 | 5,416 | 11.73% | Foreign direct investment, net inflows (BoP, current USD) |
| unemployment_pct | float64 | 5,290 | 13.79% | Unemployment rate (% of total labour force) |
| imports_pct_gdp | float64 | 5,001 | 18.50% | Imports of goods & services (% of GDP) |
| tax_revenue_pct_gdp | float64 | 3,213 | 47.64% | Tax revenue (% of GDP) |
| exports_pct_gdp | float64 | 4,998 | 18.55% | Exports of goods & services (% of GDP) |
| interest_rate_pct | float64 | 3,220 | 47.52% | Real interest rate or lending rate (%) |
| reserves_usd | float64 | 4,442 | 27.61% | Total reserves including gold (current USD) |
| govt_debt_pct_gdp | float64 | 1,229 | 79.97% | Central government debt, total (% of GDP) |
| gdp_growth_pct | float64 | 5,796 | 5.54% | GDP growth rate (annual %, TARGET VARIABLE) |
| population_total | float64 | 6,084 | 0.85% | Total population |
| current_account_pct_gdp | float64 | 4,512 | 26.47% | Current account balance (% of GDP) |
| inflation_cpi_pct | float64 | 5,125 | 16.48% | Inflation, consumer prices (annual %) |
| gdp_current_usd | float64 | 5,870 | 4.34% | GDP in current USD |

## 3. Country Coverage

| Metric | Value |
|---|---|
| Unique country codes | 235 |
| Unique country names | 236 |
| Countries with <= 10 observations | 0 |
| Min observations per country | 26 |
| Max observations per country | 52 |
| Median observations per country | 26 |

### Top 10 Countries by Observation Count

| Code | Country | Obs |
| --- | --- | --- |
| NRU | Naoero | 52 |
| NRU | Nauru | 52 |
| AFE | Africa Eastern and Southern | 26 |
| ABW | Aruba | 26 |
| AFW | Africa Western and Central | 26 |
| AGO | Angola | 26 |
| ALB | Albania | 26 |
| AND | Andorra | 26 |
| ARB | Arab World | 26 |
| ARE | United Arab Emirates | 26 |
| ARG | Argentina | 26 |

### Countries with <= 10 Observations (Warning)

*None found.*

## 4. Time Coverage

| Metric | Value |
|---|---|
| Minimum year | 2000 |
| Maximum year | 2025 |
| Unique years | 26 |
| Expected years (continuous) | 26 |
| Missing global years | None -- fully continuous |
| Panel type | Globally balanced — all years present |

### Observations Per Year (first 5 and last 5)

| Year | Obs |
| --- | --- |
| 2000 | 236 |
| 2001 | 236 |
| 2002 | 236 |
| 2003 | 236 |
| 2004 | 236 |
| 2021 | 236 |
| 2022 | 236 |
| 2023 | 236 |
| 2024 | 236 |
| 2025 | 236 |

### Countries with Largest Year Gaps (>= 5 missing years)

*None found.*

## 5. Panel Integrity

| Check | Result | Status |
|---|---|---|
| Duplicate (country_code, year) pairs | 26 | WARNING Found |
| Duplicate (country_name, year) pairs | 0 | OK None |
| Missing country codes | 0 | OK None |
| Missing country names | 0 | OK None |
| Codes mapping to > 1 country name | 1 | WARNING Found |
| Names mapping to > 1 country code | 0 | OK None |

### Code to Name Mismatches
| Code | Distinct Names |
| --- | --- |
| NRU | 2 |



## 6. Missing-Value Analysis

### Per-Feature Summary (sorted by missingness)

| Feature | Missing | Missing % | Available | Category |
| --- | --- | --- | --- | --- |
| govt_debt_pct_gdp | 4,907 | 79.97% | 1,229 | C |
| tax_revenue_pct_gdp | 2,923 | 47.64% | 3,213 | B |
| interest_rate_pct | 2,916 | 47.52% | 3,220 | B |
| tariff_rate_pct | 2,812 | 45.83% | 3,324 | B |
| reserves_usd | 1,694 | 27.61% | 4,442 | B |
| current_account_pct_gdp | 1,624 | 26.47% | 4,512 | B |
| remittances_usd | 1,303 | 21.24% | 4,833 | B |
| exports_pct_gdp | 1,138 | 18.55% | 4,998 | A |
| imports_pct_gdp | 1,135 | 18.50% | 5,001 | A |
| inflation_cpi_pct | 1,011 | 16.48% | 5,125 | A |
| unemployment_pct | 846 | 13.79% | 5,290 | A |
| exchange_rate_lcu_usd | 781 | 12.73% | 5,355 | A |
| fdi_net_inflow_usd | 720 | 11.73% | 5,416 | A |
| gdp_growth_pct | 340 | 5.54% | 5,796 | A |
| gdp_current_usd | 266 | 4.34% | 5,870 | A |
| population_total | 52 | 0.85% | 6,084 | A |

### Top 15 Countries with Most Missing Feature-Values

| Code | Country | Total Missing Cells | Missing % |
| --- | --- | --- | --- |
| INX | Not classified | 416 | 100.0 |
| VGB | British Virgin Islands | 365 | 87.7 |
| PRK | Korea, Dem. People's Rep. | 364 | 87.5 |
| GIB | Gibraltar | 364 | 87.5 |
| MAF | St. Martin (French part) | 353 | 84.9 |
| NRU | Nauru | 336 | 40.4 |
| IMN | Isle of Man | 316 | 76.0 |
| MCO | Monaco | 314 | 75.5 |
| LIE | Liechtenstein | 310 | 74.5 |
| TCA | Turks and Caicos Islands | 302 | 72.6 |
| CHI | Channel Islands | 301 | 72.4 |
| AND | Andorra | 301 | 72.4 |
| NRU | Naoero | 296 | 35.6 |
| SXM | Sint Maarten (Dutch part) | 290 | 69.7 |
| MNP | Northern Mariana Islands | 281 | 67.5 |

### Top 10 Years with Most Missing Feature-Values

| Year | Total Missing Cells |
| --- | --- |
| 2025 | 1753 |
| 2024 | 1205 |
| 2000 | 1110 |
| 2023 | 1106 |
| 2001 | 1061 |
| 2002 | 1012 |
| 2003 | 1012 |
| 2004 | 1001 |
| 2005 | 947 |
| 2022 | 914 |

## 7. Numerical Validity — Suspicious / Impossible Values

> CRITICAL = likely data error | WARNING = investigate | INFO = extreme but may be legitimate

| Feature | Check | Severity | Count | Example |
| --- | --- | --- | --- | --- |
| tax_revenue_pct_gdp | Tax revenue > 100% GDP | WARNING | 3 | 110.2 |
| tariff_rate_pct | Tariff > 200% | WARNING | 1 | 421.5 |
| gdp_growth_pct | GDP decline > 50% | WARNING | 2 | -50.34 |
| gdp_growth_pct | GDP growth > 50% | WARNING | 7 | 63.38 |
| govt_debt_pct_gdp | Negative debt/GDP | WARNING | 1 | -1.171 |
| exchange_rate_lcu_usd | 3xIQR outlier [-318, 427] | INFO | 965 | 578.3 |
| tariff_rate_pct | 3xIQR outlier [-19.7, 31.2] | INFO | 8 | 103.2 |
| remittances_usd | 3xIQR outlier [-1.05e+10, 1.42e+10] | INFO | 552 | 1.512e+10 |
| fdi_net_inflow_usd | 3xIQR outlier [-2.18e+10, 2.94e+10] | INFO | 728 | 5.439e+10 |
| unemployment_pct | 3xIQR outlier [-15.4, 29.6] | INFO | 33 | 30.22 |
| imports_pct_gdp | 3xIQR outlier [-63.1, 149] | INFO | 69 | 191.3 |
| tax_revenue_pct_gdp | 3xIQR outlier [-16.3, 49.4] | INFO | 5 | 110.2 |
| exports_pct_gdp | 3xIQR outlier [-64.9, 138] | INFO | 105 | 156.4 |
| interest_rate_pct | 3xIQR outlier [-14.5, 24.6] | INFO | 41 | 39.58 |
| reserves_usd | 3xIQR outlier [-8.32e+10, 1.13e+11] | INFO | 423 | 1.311e+11 |
| govt_debt_pct_gdp | 3xIQR outlier [-81.7, 182] | INFO | 2 | 194.7 |
| gdp_growth_pct | 3xIQR outlier [-11.9, 19.1] | INFO | 119 | -23.94 |
| population_total | 3xIQR outlier [-8.78e+07, 1.19e+08] | INFO | 719 | 4.062e+08 |
| current_account_pct_gdp | 3xIQR outlier [-35.2, 30] | INFO | 113 | 33.68 |
| inflation_cpi_pct | 3xIQR outlier [-13.3, 21.9] | INFO | 185 | 26.42 |
| gdp_current_usd | 3xIQR outlier [-6.89e+11, 9.31e+11] | INFO | 727 | 9.597e+11 |

**Note**: 3xIQR outlier detection applied per column. No values were removed.

## 8. Target Variable Analysis — `gdp_growth_pct`

| Metric | Value |
|---|---|
| Total rows | 6,136 |
| Missing values | 340 (5.54%) |
| Available observations | 5,796 |
| Minimum | -54.40 % |
| Maximum | 86.83 % |
| Mean | 3.40 % |
| Median | 3.56 % |
| Std deviation | 5.73 % |
| 5th percentile | -4.37 % |
| 25th percentile | 1.38 % |
| 75th percentile | 5.80 % |
| 95th percentile | 9.94 % |

### Distribution Buckets

| Bucket | Count | Pct % |
| --- | --- | --- |
| (-100, -10] | 96 | 1.66 |
| (-10, -5] | 162 | 2.8 |
| (-5, -2] | 266 | 4.59 |
| (-2, 0] | 367 | 6.33 |
| (0, 2] | 911 | 15.72 |
| (2, 5] | 2103 | 36.28 |
| (5, 10] | 1606 | 27.71 |
| (10, 100] | 285 | 4.92 |

### 5 Highest GDP Growth Observations

| Code | Country | Year | GDP Growth % |
| --- | --- | --- | --- |
| LBY | Libya | 2012 | 86.827 |
| MAC | Macao SAR, China | 2023 | 75.308 |
| TCA | Turks and Caicos Islands | 2017 | 74.588 |
| GNQ | Equatorial Guinea | 2001 | 63.38 |
| GUY | Guyana | 2022 | 63.335 |

### 5 Lowest GDP Growth Observations

| Code | Country | Year | GDP Growth % |
| --- | --- | --- | --- |
| MAC | Macao SAR, China | 2020 | -54.402 |
| LBY | Libya | 2011 | -50.339 |
| SSD | South Sudan | 2012 | -46.082 |
| IRQ | Iraq | 2003 | -36.657 |
| CAF | Central African Republic | 2013 | -36.392 |

### Countries with < 5 Valid GDP-Growth Observations (Insufficient History)

| Code | Country | Valid Obs |
| --- | --- | --- |
| INX | Not classified | 0 |
| GIB | Gibraltar | 0 |
| VGB | British Virgin Islands | 0 |
| PRK | Korea, Dem. People's Rep. | 0 |

## 9. Feature Relationships

> Pearson correlations on available non-missing pairs.  
> No variables removed, even if highly correlated.

### Correlation with Target (`gdp_growth_pct`)

| Feature | Pearson r with gdp_growth_pct |
| --- | --- |
| govt_debt_pct_gdp | -0.1494 |
| unemployment_pct | -0.1005 |
| tax_revenue_pct_gdp | -0.0878 |
| population_total | 0.0841 |
| inflation_cpi_pct | -0.0662 |
| current_account_pct_gdp | 0.0633 |
| exchange_rate_lcu_usd | -0.0504 |
| exports_pct_gdp | 0.0314 |
| tariff_rate_pct | 0.0299 |
| interest_rate_pct | -0.0298 |
| remittances_usd | 0.0203 |
| fdi_net_inflow_usd | 0.0172 |
| imports_pct_gdp | 0.0155 |
| reserves_usd | 0.01 |
| gdp_current_usd | 0.0015 |

### Highly Correlated Feature Pairs (|r| > 0.70)

| Feature A | Feature B | Pearson r | Strength |
| --- | --- | --- | --- |
| imports_pct_gdp | exports_pct_gdp | 0.8133 | High (0.7-0.9) |
| fdi_net_inflow_usd | gdp_current_usd | 0.7895 | High (0.7-0.9) |
| remittances_usd | population_total | 0.7309 | High (0.7-0.9) |

**Note**: High correlation does not automatically disqualify features.

## 10. Country / Year Coverage Examples

### Countries with Excellent Data Coverage

| Code | Country | Coverage |
| --- | --- | --- |
| HUN | Hungary | 98.1% |
| URY | Uruguay | 98.1% |
| GEO | Georgia | 97.6% |
| MDA | Moldova | 97.4% |
| CAN | Canada | 96.9% |
| NZL | New Zealand | 96.6% |
| MYS | Malaysia | 96.6% |
| CHE | Switzerland | 96.6% |

### Countries with Poor Data Coverage

| Code | Country | Coverage |
| --- | --- | --- |
| INX | Not classified | 0.0% |
| VGB | British Virgin Islands | 12.3% |
| PRK | Korea, Dem. People's Rep. | 12.5% |
| GIB | Gibraltar | 12.5% |
| MAF | St. Martin (French part) | 15.1% |
| IMN | Isle of Man | 24.0% |
| NRU | Nauru | 24.0% |
| NRU | Naoero | 24.0% |

### Years with Highest Missing-Value Rate

| Year | Missing Rate % |
| --- | --- |
| 2025.0 | 46.42 |
| 2024.0 | 31.91 |
| 2000.0 | 29.4 |
| 2023.0 | 29.29 |
| 2001.0 | 28.1 |
| 2002.0 | 26.8 |
| 2003.0 | 26.8 |
| 2004.0 | 26.51 |
| 2005.0 | 25.08 |
| 2022.0 | 24.21 |

## 11. Data Readiness Assessment

| Category | Meaning | Features |
|---|---|---|
| A | Ready / high coverage (>= 80%) | ['population_total', 'fdi_net_inflow_usd', 'exchange_rate_lcu_usd', 'unemployment_pct', 'inflation_cpi_pct', 'imports_pct_gdp', 'exports_pct_gdp', 'gdp_current_usd', 'gdp_growth_pct'] |
| B | Usable after cleaning (50-80% coverage) | ['remittances_usd', 'current_account_pct_gdp', 'reserves_usd', 'tariff_rate_pct', 'interest_rate_pct', 'tax_revenue_pct_gdp'] |
| C | High missingness -- handle carefully (20-50% coverage) | ['govt_debt_pct_gdp'] |
| D | Potentially unsuitable (< 20% coverage) | [] |

### Per-Feature Detail

| Feature | Missing % | Category | Assessment |
| --- | --- | --- | --- |
| population_total | 0.85% | A | Ready / high coverage (>= 80%) |
| fdi_net_inflow_usd | 11.73% | A | Ready / high coverage (>= 80%) |
| exchange_rate_lcu_usd | 12.73% | A | Ready / high coverage (>= 80%) |
| unemployment_pct | 13.79% | A | Ready / high coverage (>= 80%) |
| inflation_cpi_pct | 16.48% | A | Ready / high coverage (>= 80%) |
| imports_pct_gdp | 18.50% | A | Ready / high coverage (>= 80%) |
| exports_pct_gdp | 18.55% | A | Ready / high coverage (>= 80%) |
| gdp_current_usd | 4.34% | A | Ready / high coverage (>= 80%) |
| gdp_growth_pct | 5.54% | A | Ready / high coverage (>= 80%) |
| remittances_usd | 21.24% | B | Usable after cleaning / imputation (50-80% coverage) |
| current_account_pct_gdp | 26.47% | B | Usable after cleaning / imputation (50-80% coverage) |
| reserves_usd | 27.61% | B | Usable after cleaning / imputation (50-80% coverage) |
| tariff_rate_pct | 45.83% | B | Usable after cleaning / imputation (50-80% coverage) |
| interest_rate_pct | 47.52% | B | Usable after cleaning / imputation (50-80% coverage) |
| tax_revenue_pct_gdp | 47.64% | B | Usable after cleaning / imputation (50-80% coverage) |
| govt_debt_pct_gdp | 79.97% | C | High missingness -- needs careful treatment (20-50% coverage) |

### Summary Count

| Category | Feature Count |
| --- | --- |
| A | 9 |
| B | 6 |
| C | 1 |

## 12. Recommendations for the Next Step

> IMPORTANT: These are recommendations only. No action was taken automatically.

---

### Confirmed Facts

- Dataset is a panel of **6,136 rows x 19 columns** covering **235 countries** from **2000** to **2025**.
- Panel is globally continuous (no missing global years).
- No duplicate (country_code, year) pairs detected.
- Target `gdp_growth_pct` has **5.54%** missing rate -- manageable.
- Category A features (9): ['population_total', 'fdi_net_inflow_usd', 'exchange_rate_lcu_usd', 'unemployment_pct', 'inflation_cpi_pct', 'imports_pct_gdp', 'exports_pct_gdp', 'gdp_current_usd', 'gdp_growth_pct']

---

### Warnings

- Countries with <=10 observations (0 total) may be unreliable for training.
- Category D features ([]) are very sparsely populated and may add noise.
- Extreme values flagged -- review before winsorising.
- Exchange-rate column has near-zero or zero values in some rows -- verify encoding.

---

### Recommended Next Step: Phase 11.2 — Data Preprocessing Pipeline

1. **Create** `apps/api/ml/data_preprocessing.py` (do NOT modify master_panel.csv).
2. **Imputation strategy**:
   - Category A: forward-fill / backward-fill within each country panel.
   - Category B: country-level mean or linear interpolation.
   - Category C: global median or indicator-variable encoding.
   - Category D ([]): consider dropping from initial feature set.
3. **Outlier handling**:
   - Review GDP-growth extremes (growth > 50% or < -50%).
   - Winsorise at 1st/99th percentile only after explicit review.
4. **Feature engineering**:
   - Lag features (t-1, t-2) for GDP growth, inflation, unemployment.
   - Rolling means (3-year, 5-year windows).
   - Trade openness: imports_pct_gdp + exports_pct_gdp.
5. **Temporal train/val/test split** (NEVER random):
   - Train: 2000-2018 | Validate: 2019-2022 | Test: 2023-2025.
6. **Baseline model**: LightGBM or XGBoost on Category A features only.

---

*Audit complete. Source dataset `data/raw/master_panel.csv` was NOT modified.*
