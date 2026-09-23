# Phase 11 -- Data Preprocessing Report

**Phase**: 11 -- AI Agents & ML Forecasting  
**Script**: `apps/api/ml/data_preprocessing.py`  
**Generated**: 2026-08-17 06:52 UTC  
**Raw dataset**: `data/raw/master_panel.csv`  
**Processed dataset**: `data/processed/master_panel_processed.csv`

> No ML models were trained in this step.  
> The raw dataset (`master_panel.csv`) was **not modified**.

---

---

## 1. Input Dataset

| Metric | Value |
|---|---|
| File | `D:\Development\EconoSphere AI\data\raw\master_panel.csv` |
| Rows | 6,136 |
| Columns | 19 |
| MD5 checksum | `8ac7e0b2bf09fbe89289f82d0c7cf25e` |
| SHA-256 checksum | `0bad18504ede43e2d2dbfdecab275c00ee84ab97f5ac4cee22857d6170515db0` |

### Column Inventory (post-load validation)

| Column | Dtype | Missing Count | Missing % |
| --- | --- | --- | --- |
| country_code | str | 0 | 0.00% |
| country_name | str | 0 | 0.00% |
| year | int64 | 0 | 0.00% |
| exchange_rate_lcu_usd | float64 | 781 | 12.73% |
| tariff_rate_pct | float64 | 2812 | 45.83% |
| remittances_usd | float64 | 1303 | 21.24% |
| fdi_net_inflow_usd | float64 | 720 | 11.73% |
| unemployment_pct | float64 | 846 | 13.79% |
| imports_pct_gdp | float64 | 1135 | 18.50% |
| tax_revenue_pct_gdp | float64 | 2923 | 47.64% |
| exports_pct_gdp | float64 | 1138 | 18.55% |
| interest_rate_pct | float64 | 2916 | 47.52% |
| reserves_usd | float64 | 1694 | 27.61% |
| govt_debt_pct_gdp | float64 | 4907 | 79.97% |
| gdp_growth_pct | float64 | 340 | 5.54% |
| population_total | float64 | 52 | 0.85% |
| current_account_pct_gdp | float64 | 1624 | 26.47% |
| inflation_cpi_pct | float64 | 1011 | 16.48% |
| gdp_current_usd | float64 | 266 | 4.34% |

---

## 2. Country Identity Cleaning

**Problem**: NRU appeared under two country-name variants: `Nauru` and `Naoero`.

| Metric | Value |
|---|---|
| NRU rows before correction | 52 |
| NRU name variants found | ['Naoero', 'Nauru'] |
| Years affected | 2000-2025 |
| Normalised name | Nauru |
| NRU duplicate country-year pairs | 52 |

### Duplicate Resolution

Retained the observation with greatest feature completeness per year. Ties broken by lowest original DataFrame index (deterministic). 26 row(s) removed.

| year | orig_index | completeness | action |
| --- | --- | --- | --- |
| 2000 | 4108 | 1 | DROP |
| 2000 | 4134 | 3 | KEEP |
| 2001 | 4109 | 1 | DROP |
| 2001 | 4135 | 3 | KEEP |
| 2002 | 4110 | 1 | DROP |
| 2002 | 4136 | 3 | KEEP |
| 2003 | 4111 | 1 | DROP |
| 2003 | 4137 | 3 | KEEP |
| 2004 | 4112 | 1 | DROP |
| 2004 | 4138 | 3 | KEEP |
| 2005 | 4113 | 1 | DROP |
| 2005 | 4139 | 3 | KEEP |
| 2006 | 4114 | 1 | DROP |
| 2006 | 4140 | 3 | KEEP |
| 2007 | 4115 | 1 | DROP |
| 2007 | 4141 | 3 | KEEP |
| 2008 | 4116 | 5 | KEEP |
| 2008 | 4142 | 3 | DROP |
| 2009 | 4117 | 5 | KEEP |
| 2009 | 4143 | 3 | DROP |
| 2010 | 4118 | 5 | KEEP |
| 2010 | 4144 | 3 | DROP |
| 2011 | 4119 | 5 | KEEP |
| 2011 | 4145 | 4 | DROP |
| 2012 | 4120 | 5 | KEEP |
| 2012 | 4146 | 4 | DROP |
| 2013 | 4121 | 5 | KEEP |
| 2013 | 4147 | 3 | DROP |
| 2014 | 4122 | 6 | KEEP |
| 2014 | 4148 | 3 | DROP |
| 2015 | 4123 | 6 | KEEP |
| 2015 | 4149 | 3 | DROP |
| 2016 | 4124 | 7 | KEEP |
| 2016 | 4150 | 3 | DROP |
| 2017 | 4125 | 7 | KEEP |
| 2017 | 4151 | 3 | DROP |
| 2018 | 4126 | 8 | KEEP |
| 2018 | 4152 | 3 | DROP |
| 2019 | 4127 | 8 | KEEP |
| 2019 | 4153 | 3 | DROP |
| 2020 | 4128 | 8 | KEEP |
| 2020 | 4154 | 3 | DROP |
| 2021 | 4129 | 8 | KEEP |
| 2021 | 4155 | 3 | DROP |
| 2022 | 4130 | 7 | KEEP |
| 2022 | 4156 | 3 | DROP |
| 2023 | 4131 | 7 | KEEP |
| 2023 | 4157 | 3 | DROP |
| 2024 | 4132 | 7 | KEEP |
| 2024 | 4158 | 3 | DROP |
| 2025 | 4133 | 3 | KEEP |
| 2025 | 4159 | 3 | DROP |

---

## 3. Administrative Records

### INX Removal

| Metric | Value |
|---|---|
| Rows before INX removal | 6,110 |
| INX rows removed | 26 |
| Rows after removal | 6,084 |

**Reason**: `INX` (Not classified) had 0% feature coverage and 0 valid GDP-growth observations.

### Other Aggregate/Non-Country Codes

| Code | Name | Obs | Target valid | Avg feature miss % |
| --- | --- | --- | --- | --- |
| AFE | Africa Eastern and Southern | 26 | 26 | 38.0% |
| AFW | Africa Western and Central | 26 | 26 | 56.2% |
| ARB | Arab World | 26 | 26 | 44.2% |
| EAP | East Asia & Pacific (excluding high income) | 26 | 26 | 38.9% |
| ECA | Europe & Central Asia (excluding high income) | 26 | 26 | 35.8% |
| IDX | IDA only | 26 | 26 | 40.6% |
| INX | Not classified | 26 | 0 | 100.0% |
| LAC | Latin America & Caribbean (excluding high income) | 26 | 26 | 38.7% |
| MEA | Middle East, North Africa, Afghanistan & Pakistan | 26 | 26 | 44.5% |
| MNA | Middle East, North Africa, Afghanistan & Pakistan (excluding high income) | 26 | 26 | 43.8% |
| NAC | North America | 26 | 26 | 32.5% |
| SSA | Sub-Saharan Africa (excluding high income) | 26 | 26 | 43.8% |
| TEA | East Asia & Pacific (IDA & IBRD countries) | 26 | 26 | 38.9% |
| TEC | Europe & Central Asia (IDA & IBRD countries) | 26 | 26 | 34.4% |
| TLA | Latin America & the Caribbean (IDA & IBRD countries) | 26 | 26 | 38.7% |
| TMN | Middle East, North Africa, Afghanistan & Pakistan (IDA & IBRD) | 26 | 26 | 43.8% |
| TSA | South Asia (IDA & IBRD) | 26 | 26 | 34.4% |
| TSS | Sub-Saharan Africa (IDA & IBRD countries) | 26 | 26 | 43.8% |

Other World Bank aggregate codes retained because they carry valid GDP-growth observations: ['AFE', 'AFW', 'ARB', 'EAP', 'ECA', 'IDX', 'LAC', 'MEA', 'MNA', 'NAC', 'SSA', 'TEA', 'TEC', 'TLA', 'TMN', 'TSA', 'TSS']. They can be filtered out at the model-training step if desired.

---

## 4. Duplicate Analysis

| Metric | Value |
|---|---|
| Duplicates before step 4.4 | 0 |
| Duplicates after resolution | 0 |
| Resolution method | No duplicates found -- no action required. |


---

## 4b. Numerical Validity

> Values are **preserved** and **not deleted**. Flagged for downstream inspection.

| Feature | Check | Country | Code | Year | Value | Action |
| --- | --- | --- | --- | --- | --- | --- |
| tax_revenue_pct_gdp | tax_revenue_pct_gdp > 100 | Timor-Leste | TLS | 2010 | 110.173 | FLAGGED -- preserved |
| tax_revenue_pct_gdp | tax_revenue_pct_gdp > 100 | Timor-Leste | TLS | 2011 | 135.484 | FLAGGED -- preserved |
| tax_revenue_pct_gdp | tax_revenue_pct_gdp > 100 | Timor-Leste | TLS | 2012 | 147.64 | FLAGGED -- preserved |
| tariff_rate_pct | tariff_rate_pct > 200 | Nepal | NPL | 2008 | 421.5 | FLAGGED -- preserved |
| govt_debt_pct_gdp | govt_debt_pct_gdp < 0 | Namibia | NAM | 2012 | -1.171 | FLAGGED -- preserved |

**tariff_rate_pct = 421.5%**: Retained. Extreme but not impossible in some trade regimes.

**tax_revenue_pct_gdp > 100%**: Three observations retained but flagged as possible data artefacts.

**govt_debt_pct_gdp < 0** (-1.171): Possibly a net-creditor reporting convention. Retained.

---

## 6. Target Variable Analysis

| Metric | Value |
|---|---|
| Total rows | 6,084 |
| Missing values | 306 (5.03%) |
| Available observations | 5,778 |
| Minimum | -54.40 % |
| Maximum | 86.83 % |
| Mean | 3.39 % |
| Median | 3.56 % |
| Std deviation | 5.72 % |
| 5th percentile | -4.36 % |
| 25th percentile | 1.39 % |
| 75th percentile | 5.80 % |
| 95th percentile | 9.92 % |

> The target variable is **NEVER imputed**. Rows with missing `gdp_growth_pct` are retained for inference.

### Distribution Buckets

| Bucket | Count | Pct % |
| --- | --- | --- |
| (-100, -10] | 96 | 1.66 |
| (-10, -5] | 160 | 2.77 |
| (-5, -2] | 266 | 4.60 |
| (-2, 0] | 363 | 6.28 |
| (0, 2] | 909 | 15.73 |
| (2, 5] | 2099 | 36.33 |
| (5, 10] | 1604 | 27.76 |
| (10, 100] | 281 | 4.86 |

---

## 5. Missing Value Analysis

| Feature | Cat | Missing | Missing % | Countries w/ data | Years w/ data | Min | Max | Median | Treatment |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| govt_debt_pct_gdp | C | 4855 | 79.80% | 93 | 25 | -1.17 | 194.68 | 50.15 | Interior interpolation only; missingness flag created; excluded from baseline |
| gdp_growth_pct | TARGET | 306 | 5.03% | 231 | 26 | -54.40 | 86.83 | 3.56 | NEVER IMPUTED (target) |
| tax_revenue_pct_gdp | B | 2871 | 47.19% | 166 | 25 | 0.04 | 147.64 | 16.13 | Country interpolation + group-median fallback |
| interest_rate_pct | B | 2864 | 47.07% | 150 | 26 | -0.42 | 203.38 | 4.25 | Country interpolation + group-median fallback |
| tariff_rate_pct | B | 2760 | 45.36% | 188 | 23 | 0.00 | 421.50 | 4.66 | Country interpolation + group-median fallback |
| gdp_current_usd | A | 232 | 3.81% | 231 | 26 | 14,139,705.25 | 33,099,237,425,506.20 | 25,529,071,343.33 | Country interpolation + fwd/bwd fill |
| reserves_usd | B | 1642 | 26.99% | 178 | 26 | 267,707.37 | 3,900,039,302,991.09 | 4,120,588,319.89 | Country interpolation + fwd/bwd fill |
| current_account_pct_gdp | B | 1572 | 25.84% | 197 | 26 | -60.70 | 311.75 | -2.60 | Country interpolation + fwd/bwd fill |
| remittances_usd | B | 1251 | 20.56% | 211 | 26 | 6,038.03 | 184,590,644,303.10 | 785,135,446.15 | Country interpolation + fwd/bwd fill |
| exports_pct_gdp | A | 1086 | 17.85% | 209 | 26 | 0.00 | 433.84 | 33.31 | Country interpolation + fwd/bwd fill |
| imports_pct_gdp | A | 1083 | 17.80% | 209 | 26 | 0.00 | 429.36 | 38.53 | Country interpolation + fwd/bwd fill |
| inflation_cpi_pct | A | 961 | 15.80% | 209 | 26 | -16.86 | 557.20 | 3.56 | Country interpolation + fwd/bwd fill |
| unemployment_pct | A | 794 | 13.05% | 204 | 26 | 0.10 | 37.32 | 6.18 | Country interpolation + fwd/bwd fill |
| exchange_rate_lcu_usd | A | 747 | 12.28% | 213 | 26 | 0.04 | 6,723,052,073.34 | 6.95 | Country interpolation + fwd/bwd fill |
| fdi_net_inflow_usd | A | 668 | 10.98% | 220 | 26 | -343,402,798,888.19 | 733,826,501,994.52 | 911,880,885.59 | Country interpolation + fwd/bwd fill |
| population_total | A | 8 | 0.13% | 234 | 26 | 9,492.00 | 2,143,908,411.00 | 7,155,052.50 | Country interpolation + fwd/bwd fill |

> `govt_debt_pct_gdp` (~80% missing): will NOT be aggressively imputed. A missingness flag is created. Excluded from initial baseline model.

---

## 7. Outlier Analysis

> Extreme observations were **preserved and not deleted**.

### GDP-Growth Outlier Flag (`gdp_growth_outlier_flag`)

Rule: IQR x 3.0 -- outside [Q1-3.0*IQR, Q3+3.0*IQR]

| Metric | Value |
|---|---|
| Q1 | 1.39 % |
| Q3 | 5.80 % |
| IQR | 4.42 % |
| Lower fence | -11.86 % |
| Upper fence | 19.05 % |
| Observations flagged | 117 |

### Extreme Observations (flagged, preserved)

| country_code | country_name | year | gdp_growth_pct |
| --- | --- | --- | --- |
| LBY | Libya | 2012 | 86.827 |
| MAC | Macao SAR, China | 2023 | 75.308 |
| TCA | Turks and Caicos Islands | 2017 | 74.588 |
| GNQ | Equatorial Guinea | 2001 | 63.38 |
| GUY | Guyana | 2022 | 63.335 |
| TLS | Timor-Leste | 2000 | 58.078 |
| IRQ | Iraq | 2004 | 53.386 |
| GUY | Guyana | 2024 | 43.819 |
| GUY | Guyana | 2020 | 43.48 |
| GNQ | Equatorial Guinea | 2004 | 37.999 |
| MDV | Maldives | 2021 | 37.508 |
| AZE | Azerbaijan | 2006 | 34.466 |
| GUY | Guyana | 2023 | 33.795 |
| TCD | Chad | 2004 | 33.629 |
| LBY | Libya | 2017 | 32.492 |
| TLS | Timor-Leste | 2020 | 31.726 |
| TCA | Turks and Caicos Islands | 2021 | 29.629 |
| MNP | Northern Mariana Islands | 2016 | 29.212 |
| LBR | Liberia | 2000 | 28.616 |
| AFG | Afghanistan | 2002 | 28.6 |

### Suspicious Feature Flags

| Flag | Condition | Count |
|---|---|---|
| `tariff_suspicious_flag` | tariff_rate_pct > 200.0% | 1 |
| `tax_revenue_suspicious_flag` | tax_revenue_pct_gdp > 100% | 3 |


---

## 4c. Missingness Indicators

| Flag | Source | Flagged rows | Reason |
| --- | --- | --- | --- |
| govt_debt_missing_flag | govt_debt_pct_gdp | 4741 | ~80% missing; excluded from baseline; flag carries signal about data availability |
| gdp_target_available | gdp_growth_pct | 5778 | 1 = target available for supervised learning; 0 = inference-only row |

**Note**: Missingness flags are only created where post-imputation NaN count remains meaningful. Category A/B features are imputed to near-completion so their flags would be trivially zero. Only `govt_debt_pct_gdp` retains substantial post-imputation missingness.

---

## 12. Train / Validation / Test Periods

| Period | Years | Rows | Target available | Target miss % |
| --- | --- | --- | --- | --- |
| Train | 2000-2018 | 4446 | 4237 | 4.7% |
| Validation | 2019-2022 | 936 | 903 | 3.5% |
| Test | 2023-2025 | 702 | 638 | 9.1% |

**Why chronological splitting is mandatory**: Random splitting would allow future observations into the training set, producing falsely optimistic validation metrics and violating the operational deployment constraint (the model always predicts years it has never seen).

**2024-2025 missingness**: 32% and 46% respectively -- these years are retained in the processed dataset but their high missingness must be acknowledged in test evaluation.

---

## 8. Imputation Strategy

| Feature | Strategy |
| --- | --- |
| current_account_pct_gdp | Country interpolation+fwd/bwd fill (610 values filled); training-period global median fallback (962 values filled). Caution: fallback imputed 962 cells from a global median. |
| exchange_rate_lcu_usd | Country interpolation+fwd/bwd fill (201 values filled); training-period global median fallback (546 values filled) |
| exports_pct_gdp | Country interpolation+fwd/bwd fill (436 values filled); training-period global median fallback (650 values filled) |
| fdi_net_inflow_usd | Country interpolation+fwd/bwd fill (304 values filled); training-period global median fallback (364 values filled) |
| gdp_current_usd | Country interpolation+fwd/bwd fill (154 values filled); training-period global median fallback (78 values filled) |
| gdp_growth_pct | NOT IMPUTED -- target variable. |
| govt_debt_pct_gdp | Within-country linear interpolation (interior gaps only): 114 values filled. 4741 values remain NaN. No global fallback. Excluded from initial baseline model. |
| imports_pct_gdp | Country interpolation+fwd/bwd fill (433 values filled); training-period global median fallback (650 values filled) |
| inflation_cpi_pct | Country interpolation+fwd/bwd fill (311 values filled); training-period global median fallback (650 values filled) |
| interest_rate_pct | Country interpolation+fwd/bwd fill (680 values filled); training-period global median fallback (2184 values filled). Caution: fallback imputed 2184 cells from a global median. |
| population_total | Country interpolation+fwd/bwd fill (8 values filled) |
| remittances_usd | Country interpolation+fwd/bwd fill (653 values filled); training-period global median fallback (598 values filled). Caution: fallback imputed 598 cells from a global median. |
| reserves_usd | Country interpolation+fwd/bwd fill (186 values filled); training-period global median fallback (1456 values filled). Caution: fallback imputed 1456 cells from a global median. |
| tariff_rate_pct | Country interpolation+fwd/bwd fill (1564 values filled); training-period global median fallback (1196 values filled). Caution: fallback imputed 1196 cells from a global median. |
| tax_revenue_pct_gdp | Country interpolation+fwd/bwd fill (1103 values filled); training-period global median fallback (1768 values filled). Caution: fallback imputed 1768 cells from a global median. |
| unemployment_pct | Country interpolation+fwd/bwd fill (14 values filled); training-period global median fallback (780 values filled) |

**Key decisions**:
- Target (`gdp_growth_pct`): NEVER imputed.
- Cat-A/B: within-country linear interpolation -> fwd fill -> bwd fill -> training-period global median.
- `govt_debt_pct_gdp` (Cat-C): interior interpolation only; no global fallback; missingness flag created.
- Bfill caveat: within-country backward-fill uses same-country future data for leading-edge gaps   (acknowledged, documented, can be disabled in a future variant).

---

## 9. Generated Features

| Feature | Formula | Source | Purpose |
| --- | --- | --- | --- |
| gdp_growth_lag1 | shift(1) of `gdp_growth_pct` within country | gdp_growth_pct | Historical gdp_growth_pct from 1 year(s) prior |
| gdp_growth_lag2 | shift(2) of `gdp_growth_pct` within country | gdp_growth_pct | Historical gdp_growth_pct from 2 year(s) prior |
| gdp_growth_lag3 | shift(3) of `gdp_growth_pct` within country | gdp_growth_pct | Historical gdp_growth_pct from 3 year(s) prior |
| inflation_lag1 | shift(1) of `inflation_cpi_pct` within country | inflation_cpi_pct | Historical inflation_cpi_pct from 1 year(s) prior |
| unemployment_lag1 | shift(1) of `unemployment_pct` within country | unemployment_pct | Historical unemployment_pct from 1 year(s) prior |
| exports_lag1 | shift(1) of `exports_pct_gdp` within country | exports_pct_gdp | Historical exports_pct_gdp from 1 year(s) prior |
| imports_lag1 | shift(1) of `imports_pct_gdp` within country | imports_pct_gdp | Historical imports_pct_gdp from 1 year(s) prior |
| gdp_growth_rolling_mean_3 | shift(1) -> rolling_mean(w=3) of `gdp_growth_pct` within country | gdp_growth_pct | 3-year rolling mean of gdp_growth_pct. Shift ensures T uses only T-1..T-3 (no target leakage). |
| gdp_growth_rolling_std_3 | shift(1) -> rolling_std(w=3) of `gdp_growth_pct` within country | gdp_growth_pct | 3-year rolling std of gdp_growth_pct. Shift ensures T uses only T-1..T-3 (no target leakage). |
| gdp_growth_rolling_mean_5 | shift(1) -> rolling_mean(w=5) of `gdp_growth_pct` within country | gdp_growth_pct | 5-year rolling mean of gdp_growth_pct. Shift ensures T uses only T-1..T-5 (no target leakage). |
| inflation_rolling_mean_3 | shift(1) -> rolling_mean(w=3) of `inflation_cpi_pct` within country | inflation_cpi_pct | 3-year rolling mean of inflation_cpi_pct. Shift ensures T uses only T-1..T-3 (no target leakage). |
| trade_openness | imports_pct_gdp + exports_pct_gdp | imports_pct_gdp, exports_pct_gdp | Economic openness: higher ratio = more exposure to global demand shocks |
| trade_balance_ratio | exports_pct_gdp - imports_pct_gdp | exports_pct_gdp, imports_pct_gdp | Net trade position. Persistent deficits can signal external vulnerability. |
| log_gdp_usd | log(gdp_current_usd) where gdp_current_usd > 0 | gdp_current_usd | Compresses the ~10^6 to ~10^13 USD GDP scale to a log-linear range |
| log_population | log(population_total) where population_total > 0 | population_total | Log-linearises country size for more stable regression coefficients |
---

## 10. Leakage Audit

**Result: PASS**

| Check | Result | Detail |
| --- | --- | --- |
| Target not directly duplicated as a feature | PASS | gdp_growth_pct not replicated in any other column |
| Lag features are shifted historical values (not current target) | PASS | Checked: ['gdp_growth_outlier_flag', 'tariff_suspicious_flag', 'tax_revenue_suspicious_flag', 'govt_debt_missing_flag', 'gdp_growth_lag1', 'gdp_growth_lag2', 'gdp_growth_lag3', 'inflation_lag1', 'unemployment_lag1', 'exports_lag1', 'imports_lag1'] |
| Rolling features computed on shift(1) series -- current target excluded from window | PASS | Checked: ['gdp_growth_rolling_mean_3', 'gdp_growth_rolling_std_3', 'gdp_growth_rolling_mean_5'] |
| Imputation fallback statistics use only training-period data (years <= 2018) | PASS | global_median computed on df[df['year'] <= 2018] |
| No random shuffle applied -- chronological split only | PASS | All sorts use country_code + year; no random operations used |
| Panel groups respected -- per-country groupby used exclusively for lags/rolling | PASS | groupby('country_code').shift() and .transform() used in all temporal ops |
| Imputation does not use external test-period data (within-country bfill caveat acknowledged) | PASS | Within-country bfill uses same-country future data for leading-edge gaps -- acknowledged limitation documented in report. Global-median fallback uses only training-period observations. |
---

## 11. Dataset Dimensions

| Metric | Raw | Processed |
|---|---|---|
| Rows | 6,136 | 6,084 |
| Columns | 19 | 39 |
| Duplicate country-year pairs | 26 (pre-fix) | 0 |
| Target missingness | 340 (5.54%) | 306 (5.03%) |
| Engineered features added | -- | 15 |

---

## 13. Final Data Readiness Assessment

**READY WITH CAUTIONS**

Key cautions:
- `govt_debt_pct_gdp` (~80% missing post-interpolation) excluded from initial baseline.
- 2024-2025 test-period observations have high residual missingness (32-46%).
- Within-country bfill for leading-edge gaps uses same-country future data (acknowledged).
- World Bank aggregate codes (AFE, AFW, etc.) retained; filter at model-training step if needed.
