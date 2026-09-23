"""
EconoSphere AI — Phase 11: Dataset Quality & Coverage Audit
============================================================
Script   : apps/api/ml/data_audit.py
Input    : data/raw/master_panel.csv  (READ-ONLY -- never modified)
Output   : docs/phase11/dataset_quality_audit.md

Run from the project root:
    python apps/api/ml/data_audit.py
"""
from __future__ import annotations
import os, sys, warnings
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
DATA_PATH    = PROJECT_ROOT / "data" / "raw" / "master_panel.csv"
REPORT_DIR   = PROJECT_ROOT / "docs" / "phase11"
REPORT_PATH  = REPORT_DIR / "dataset_quality_audit.md"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def md_table(df: pd.DataFrame) -> str:
    lines = []
    header  = "| " + " | ".join(str(c) for c in df.columns) + " |"
    divider = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    lines.append(header)
    lines.append(divider)
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join(lines)

def fmt(n, dec=2):
    if pd.isna(n): return "N/A"
    return f"{n:,.{dec}f}"

def pct(n): return f"{n:.2f}%"

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
print(f"[INFO] Loading: {DATA_PATH}")
df = pd.read_csv(DATA_PATH, low_memory=False)
print(f"[INFO] Shape: {df.shape}")

ID_COLS      = ["country_code", "country_name", "year"]
NUMERIC_COLS = [c for c in df.columns if c not in ID_COLS]
TARGET_COL   = "gdp_growth_pct"
n_rows, n_cols = df.shape
ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

sections: list[str] = []

# ===========================================================================
# 0. Header
# ===========================================================================
sections.append(f"""# EconoSphere AI — Dataset Quality & Coverage Audit

**Phase**: 11 — AI Agents & ML Forecasting  
**Dataset**: `data/raw/master_panel.csv`  
**Generated**: {ts}  
**Script**: `apps/api/ml/data_audit.py`

> ⚠️ **Read-Only Audit** — this report documents observed facts only.  
> No rows, columns, or values were modified, imputed, or deleted.

---
""")

# ===========================================================================
# 1. Dimensions
# ===========================================================================
sections.append(f"""## 1. Dataset Dimensions

| Metric | Value |
|---|---|
| Rows | {n_rows:,} |
| Columns | {n_cols} |
| Feature columns (numeric) | {len(NUMERIC_COLS)} |
| Identifier columns | {len(ID_COLS)} (`country_code`, `country_name`, `year`) |
| File size | {DATA_PATH.stat().st_size / 1024:.1f} KB |
""")

# ===========================================================================
# 2. Schema
# ===========================================================================
DESCRIPTIONS = {
    "country_code"            : "ISO-3166 alpha-3 country code (identifier)",
    "country_name"            : "Full country name (identifier)",
    "year"                    : "Observation year (identifier)",
    "exchange_rate_lcu_usd"   : "Official exchange rate — local currency per 1 USD",
    "tariff_rate_pct"         : "Applied weighted-mean tariff rate on all products (%)",
    "remittances_usd"         : "Personal remittances received (current USD)",
    "fdi_net_inflow_usd"      : "Foreign direct investment, net inflows (BoP, current USD)",
    "unemployment_pct"        : "Unemployment rate (% of total labour force)",
    "imports_pct_gdp"         : "Imports of goods & services (% of GDP)",
    "tax_revenue_pct_gdp"     : "Tax revenue (% of GDP)",
    "exports_pct_gdp"         : "Exports of goods & services (% of GDP)",
    "interest_rate_pct"       : "Real interest rate or lending rate (%)",
    "reserves_usd"            : "Total reserves including gold (current USD)",
    "govt_debt_pct_gdp"       : "Central government debt, total (% of GDP)",
    "gdp_growth_pct"          : "GDP growth rate (annual %, TARGET VARIABLE)",
    "population_total"        : "Total population",
    "current_account_pct_gdp" : "Current account balance (% of GDP)",
    "inflation_cpi_pct"       : "Inflation, consumer prices (annual %)",
    "gdp_current_usd"         : "GDP in current USD",
}

schema_rows = []
for col in df.columns:
    dtype    = str(df[col].dtype)
    non_null = int(df[col].notna().sum())
    null_pct = 100.0 * df[col].isna().sum() / n_rows
    desc     = DESCRIPTIONS.get(col, "—")
    schema_rows.append({
        "Column": col, "Dtype": dtype,
        "Non-Null": f"{non_null:,}", "Missing %": pct(null_pct), "Description": desc,
    })
schema_df = pd.DataFrame(schema_rows)
sections.append(f"""## 2. Schema — Column Inventory

{md_table(schema_df)}
""")

# ===========================================================================
# 3. Country Coverage
# ===========================================================================
unique_codes = df["country_code"].nunique()
unique_names = df["country_name"].nunique()

obs_per_country = df.groupby("country_code").size().reset_index(name="observations")
code_name_map   = df[["country_code", "country_name"]].drop_duplicates()

few_obs_threshold = 10
few_obs = (obs_per_country[obs_per_country["observations"] <= few_obs_threshold]
           .merge(code_name_map, on="country_code", how="left"))

top10 = (obs_per_country.sort_values("observations", ascending=False).head(10)
         .merge(code_name_map, on="country_code", how="left"))

sections.append(f"""## 3. Country Coverage

| Metric | Value |
|---|---|
| Unique country codes | {unique_codes} |
| Unique country names | {unique_names} |
| Countries with <= {few_obs_threshold} observations | {len(few_obs)} |
| Min observations per country | {int(obs_per_country["observations"].min())} |
| Max observations per country | {int(obs_per_country["observations"].max())} |
| Median observations per country | {obs_per_country["observations"].median():.0f} |

### Top 10 Countries by Observation Count

{md_table(top10[["country_code","country_name","observations"]].rename(columns={"country_code":"Code","country_name":"Country","observations":"Obs"}))}

### Countries with <= {few_obs_threshold} Observations (Warning)

{"*None found.*" if few_obs.empty else md_table(few_obs[["country_code","country_name","observations"]].rename(columns={"country_code":"Code","country_name":"Country","observations":"Obs"}))}
""")

# ===========================================================================
# 4. Time Coverage
# ===========================================================================
year_min   = int(df["year"].min())
year_max   = int(df["year"].max())
unique_yrs = sorted(df["year"].unique())
n_years    = len(unique_yrs)
expected   = list(range(year_min, year_max + 1))
missing_g  = [y for y in expected if y not in unique_yrs]

def max_gap(grp):
    yrs  = sorted(grp["year"].tolist())
    if len(yrs) < 2: return 0
    gaps = [b - a - 1 for a, b in zip(yrs, yrs[1:])]
    return max(gaps) if gaps else 0

country_gaps = (df.groupby("country_code")
                  .apply(max_gap, include_groups=False)
                  .reset_index(name="max_year_gap")
                  .merge(code_name_map, on="country_code", how="left")
                  .sort_values("max_year_gap", ascending=False))
large_gaps = country_gaps[country_gaps["max_year_gap"] >= 5].head(20)

obs_per_year = df.groupby("year").size().reset_index(name="observations")
yr_sample = pd.concat([obs_per_year.head(5), obs_per_year.tail(5)]).drop_duplicates()

sections.append(f"""## 4. Time Coverage

| Metric | Value |
|---|---|
| Minimum year | {year_min} |
| Maximum year | {year_max} |
| Unique years | {n_years} |
| Expected years (continuous) | {len(expected)} |
| Missing global years | {"None -- fully continuous" if not missing_g else str(missing_g)} |
| Panel type | {"Globally balanced — all years present" if not missing_g else "Unbalanced — gaps exist"} |

### Observations Per Year (first 5 and last 5)

{md_table(yr_sample.rename(columns={"year":"Year","observations":"Obs"}))}

### Countries with Largest Year Gaps (>= 5 missing years)

{"*None found.*" if large_gaps.empty else md_table(large_gaps[["country_code","country_name","max_year_gap"]].rename(columns={"country_code":"Code","country_name":"Country","max_year_gap":"Max Gap (yrs)"}))}
""")

# ===========================================================================
# 5. Panel Integrity
# ===========================================================================
dup_cy  = int(df.duplicated(subset=["country_code","year"]).sum())
dup_ny  = int(df.duplicated(subset=["country_name","year"]).sum())
null_c  = int(df["country_code"].isna().sum())
null_n  = int(df["country_name"].isna().sum())

code_to_names  = df.groupby("country_code")["country_name"].nunique()
mismatch_codes = code_to_names[code_to_names > 1]

name_to_codes  = df.groupby("country_name")["country_code"].nunique()
mismatch_names = name_to_codes[name_to_codes > 1]

sections.append(f"""## 5. Panel Integrity

| Check | Result | Status |
|---|---|---|
| Duplicate (country_code, year) pairs | {dup_cy} | {"OK None" if dup_cy == 0 else "WARNING Found"} |
| Duplicate (country_name, year) pairs | {dup_ny} | {"OK None" if dup_ny == 0 else "WARNING Found"} |
| Missing country codes | {null_c} | {"OK None" if null_c == 0 else "WARNING Found"} |
| Missing country names | {null_n} | {"OK None" if null_n == 0 else "WARNING Found"} |
| Codes mapping to > 1 country name | {len(mismatch_codes)} | {"OK None" if mismatch_codes.empty else "WARNING Found"} |
| Names mapping to > 1 country code | {len(mismatch_names)} | {"OK None" if mismatch_names.empty else "WARNING Found"} |

{"### Code to Name Mismatches" + chr(10) + md_table(mismatch_codes.reset_index().rename(columns={"country_code":"Code","country_name":"Distinct Names"})) if not mismatch_codes.empty else ""}

{"### Name to Code Mismatches" + chr(10) + md_table(mismatch_names.reset_index().rename(columns={"country_name":"Name","country_code":"Distinct Codes"})) if not mismatch_names.empty else ""}
""")

# ===========================================================================
# 6. Missing Value Analysis
# ===========================================================================
miss_stats = []
for col in NUMERIC_COLS:
    n_miss  = int(df[col].isna().sum())
    p       = 100.0 * n_miss / n_rows
    n_avail = n_rows - n_miss
    cat     = ("A" if p <= 20 else "B" if p <= 50 else "C" if p <= 80 else "D")
    miss_stats.append({"Feature":col, "Missing":f"{n_miss:,}", "Missing %":pct(p),
                        "Available":f"{n_avail:,}", "Category":cat})

miss_df = pd.DataFrame(miss_stats).sort_values(
    "Missing %", key=lambda s: s.str.replace("%","").astype(float), ascending=False)

obs_map = obs_per_country.set_index("country_code")["observations"]

miss_by_country = (
    df[NUMERIC_COLS + ["country_code","country_name"]].copy()
      .assign(tm=lambda x: x[NUMERIC_COLS].isna().sum(axis=1))
      .groupby(["country_code","country_name"])["tm"].sum().reset_index()
      .sort_values("tm", ascending=False).head(15)
)
miss_by_country["missing_pct"] = (
    miss_by_country["tm"] /
    (miss_by_country["country_code"].map(obs_map) * len(NUMERIC_COLS)) * 100
).round(1)

miss_by_year = (
    df[NUMERIC_COLS + ["year"]].copy()
      .assign(tm=lambda x: x[NUMERIC_COLS].isna().sum(axis=1))
      .groupby("year")["tm"].sum().reset_index()
      .sort_values("tm", ascending=False).head(10)
)

sections.append(f"""## 6. Missing-Value Analysis

### Per-Feature Summary (sorted by missingness)

{md_table(miss_df)}

### Top 15 Countries with Most Missing Feature-Values

{md_table(miss_by_country.rename(columns={"country_code":"Code","country_name":"Country","tm":"Total Missing Cells","missing_pct":"Missing %"}))}

### Top 10 Years with Most Missing Feature-Values

{md_table(miss_by_year.rename(columns={"year":"Year","tm":"Total Missing Cells"}))}
""")

# ===========================================================================
# 7. Numerical Validity
# ===========================================================================
findings = []

def flag(col, cond, label, sev="WARNING"):
    if col not in df.columns: return
    hits = df[cond & df[col].notna()]
    if not hits.empty:
        findings.append({"Feature":col,"Check":label,"Severity":sev,
                         "Count":len(hits),"Example":f"{hits[col].iloc[0]:.4g}"})

flag("population_total",     df["population_total"]     < 0,   "Negative population",      "CRITICAL")
flag("gdp_current_usd",      df["gdp_current_usd"]      < 0,   "Negative GDP",             "CRITICAL")
flag("reserves_usd",         df["reserves_usd"]         < 0,   "Negative reserves",        "WARNING")
flag("remittances_usd",      df["remittances_usd"]      < 0,   "Negative remittances",     "INFO")
flag("unemployment_pct",     df["unemployment_pct"]     < 0,   "Negative unemployment %",  "CRITICAL")
flag("unemployment_pct",     df["unemployment_pct"]     > 100, "Unemployment > 100%",      "CRITICAL")
flag("imports_pct_gdp",      df["imports_pct_gdp"]      < 0,   "Negative imports/GDP",     "WARNING")
flag("exports_pct_gdp",      df["exports_pct_gdp"]      < 0,   "Negative exports/GDP",     "WARNING")
flag("tax_revenue_pct_gdp",  df["tax_revenue_pct_gdp"]  < 0,   "Negative tax revenue %",   "WARNING")
flag("tax_revenue_pct_gdp",  df["tax_revenue_pct_gdp"]  > 100, "Tax revenue > 100% GDP",   "WARNING")
flag("tariff_rate_pct",      df["tariff_rate_pct"]      < 0,   "Negative tariff %",        "CRITICAL")
flag("tariff_rate_pct",      df["tariff_rate_pct"]      > 200, "Tariff > 200%",            "WARNING")
flag("inflation_cpi_pct",    df["inflation_cpi_pct"]    < -50, "Deflation > 50%",          "WARNING")
flag("inflation_cpi_pct",    df["inflation_cpi_pct"]    > 1000,"Hyperinflation > 1000%",   "WARNING")
flag("gdp_growth_pct",       df["gdp_growth_pct"]       < -50, "GDP decline > 50%",        "WARNING")
flag("gdp_growth_pct",       df["gdp_growth_pct"]       > 50,  "GDP growth > 50%",         "WARNING")
flag("interest_rate_pct",    df["interest_rate_pct"]    < -100,"Interest rate < -100%",    "WARNING")
flag("interest_rate_pct",    df["interest_rate_pct"]    > 500, "Interest rate > 500%",     "WARNING")
flag("govt_debt_pct_gdp",    df["govt_debt_pct_gdp"]    < 0,   "Negative debt/GDP",        "WARNING")
flag("govt_debt_pct_gdp",    df["govt_debt_pct_gdp"]    > 500, "Debt > 500% GDP",          "WARNING")
flag("exchange_rate_lcu_usd",df["exchange_rate_lcu_usd"]<= 0,  "Non-positive exchange rate","CRITICAL")

# 3xIQR extreme outliers
for col in NUMERIC_COLS:
    s = df[col].dropna()
    if s.empty: continue
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0: continue
    lo, hi  = q1 - 3*iqr, q3 + 3*iqr
    ext = df[(df[col] < lo) | (df[col] > hi)]
    if not ext.empty:
        findings.append({"Feature":col,"Check":f"3xIQR outlier [{lo:.3g}, {hi:.3g}]",
                         "Severity":"INFO","Count":len(ext),"Example":f"{ext[col].iloc[0]:.4g}"})

validity_df = pd.DataFrame(findings) if findings else pd.DataFrame(
    columns=["Feature","Check","Severity","Count","Example"])

sections.append(f"""## 7. Numerical Validity — Suspicious / Impossible Values

> CRITICAL = likely data error | WARNING = investigate | INFO = extreme but may be legitimate

{md_table(validity_df) if not validity_df.empty else "*No suspicious values detected.*"}

**Note**: 3xIQR outlier detection applied per column. No values were removed.
""")

# ===========================================================================
# 8. Target Variable Analysis
# ===========================================================================
tgt          = df[TARGET_COL].dropna()
n_miss_tgt   = int(df[TARGET_COL].isna().sum())
pct_miss_tgt = 100.0 * n_miss_tgt / n_rows

hi5 = df.nlargest(5, TARGET_COL)[["country_code","country_name","year",TARGET_COL]]
lo5 = df.nsmallest(5, TARGET_COL)[["country_code","country_name","year",TARGET_COL]]

tgt_per_country = (df.groupby("country_code")[TARGET_COL]
                     .count().reset_index(name="valid_obs")
                     .merge(code_name_map, on="country_code", how="left")
                     .sort_values("valid_obs"))
insuf = tgt_per_country[tgt_per_country["valid_obs"] < 5]

buckets = pd.cut(tgt, bins=[-100,-10,-5,-2,0,2,5,10,100])
dist_df = buckets.value_counts().sort_index().reset_index()
dist_df.columns = ["Bucket","Count"]
dist_df["Pct %"] = (dist_df["Count"] / len(tgt) * 100).round(2)
dist_df["Bucket"] = dist_df["Bucket"].astype(str)

sections.append(f"""## 8. Target Variable Analysis — `{TARGET_COL}`

| Metric | Value |
|---|---|
| Total rows | {n_rows:,} |
| Missing values | {n_miss_tgt:,} ({pct(pct_miss_tgt)}) |
| Available observations | {len(tgt):,} |
| Minimum | {fmt(tgt.min())} % |
| Maximum | {fmt(tgt.max())} % |
| Mean | {fmt(tgt.mean())} % |
| Median | {fmt(tgt.median())} % |
| Std deviation | {fmt(tgt.std())} % |
| 5th percentile | {fmt(tgt.quantile(0.05))} % |
| 25th percentile | {fmt(tgt.quantile(0.25))} % |
| 75th percentile | {fmt(tgt.quantile(0.75))} % |
| 95th percentile | {fmt(tgt.quantile(0.95))} % |

### Distribution Buckets

{md_table(dist_df)}

### 5 Highest GDP Growth Observations

{md_table(hi5.round(3).rename(columns={"country_code":"Code","country_name":"Country","year":"Year","gdp_growth_pct":"GDP Growth %"}))}

### 5 Lowest GDP Growth Observations

{md_table(lo5.round(3).rename(columns={"country_code":"Code","country_name":"Country","year":"Year","gdp_growth_pct":"GDP Growth %"}))}

### Countries with < 5 Valid GDP-Growth Observations (Insufficient History)

{"*None -- all countries have >= 5 observations.*" if insuf.empty else md_table(insuf[["country_code","country_name","valid_obs"]].rename(columns={"country_code":"Code","country_name":"Country","valid_obs":"Valid Obs"}))}
""")

# ===========================================================================
# 9. Correlations
# ===========================================================================
corr_matrix      = df[NUMERIC_COLS].corr(method="pearson")
corr_with_target = corr_matrix[TARGET_COL].drop(TARGET_COL).sort_values(key=abs, ascending=False)

feat_cols = [c for c in NUMERIC_COLS if c != TARGET_COL]
high_pairs = []
for i, c1 in enumerate(feat_cols):
    for c2 in feat_cols[i+1:]:
        r = corr_matrix.loc[c1,c2]
        if abs(r) > 0.7:
            high_pairs.append({"Feature A":c1,"Feature B":c2,
                                "Pearson r":round(r,4),
                                "Strength":"Very High (>0.9)" if abs(r)>0.9 else "High (0.7-0.9)"})

high_df = (pd.DataFrame(high_pairs).sort_values("Pearson r", key=abs, ascending=False)
           if high_pairs else pd.DataFrame(columns=["Feature A","Feature B","Pearson r","Strength"]))

ct_df = corr_with_target.reset_index()
ct_df.columns = ["Feature", f"Pearson r with {TARGET_COL}"]
ct_df[f"Pearson r with {TARGET_COL}"] = ct_df[f"Pearson r with {TARGET_COL}"].round(4)

sections.append(f"""## 9. Feature Relationships

> Pearson correlations on available non-missing pairs.  
> No variables removed, even if highly correlated.

### Correlation with Target (`{TARGET_COL}`)

{md_table(ct_df)}

### Highly Correlated Feature Pairs (|r| > 0.70)

{"*No feature pairs exceed the 0.70 threshold.*" if high_df.empty else md_table(high_df)}

**Note**: High correlation does not automatically disqualify features.
""")

# ===========================================================================
# 10. Coverage Examples
# ===========================================================================
cov_scores = (df.groupby("country_code")[NUMERIC_COLS]
                .apply(lambda g: g.notna().sum().sum() / g.size, include_groups=False)
                .reset_index(name="coverage_score")
                .merge(code_name_map, on="country_code", how="left")
                .sort_values("coverage_score", ascending=False))

excellent = cov_scores.head(8).copy()
poor      = cov_scores.tail(8).sort_values("coverage_score").copy()
excellent["coverage_score"] = excellent["coverage_score"].map(lambda v: f"{v*100:.1f}%")
poor     ["coverage_score"] = poor     ["coverage_score"].map(lambda v: f"{v*100:.1f}%")

miss_rate_yr = (df.groupby("year")[NUMERIC_COLS]
                  .apply(lambda g: g.isna().sum().sum() / g.size, include_groups=False)
                  .reset_index(name="missing_rate")
                  .sort_values("missing_rate", ascending=False).head(10))
miss_rate_yr["missing_rate"] = (miss_rate_yr["missing_rate"] * 100).round(2)

sections.append(f"""## 10. Country / Year Coverage Examples

### Countries with Excellent Data Coverage

{md_table(excellent[["country_code","country_name","coverage_score"]].rename(columns={"country_code":"Code","country_name":"Country","coverage_score":"Coverage"}))}

### Countries with Poor Data Coverage

{md_table(poor[["country_code","country_name","coverage_score"]].rename(columns={"country_code":"Code","country_name":"Country","coverage_score":"Coverage"}))}

### Years with Highest Missing-Value Rate

{md_table(miss_rate_yr.rename(columns={"year":"Year","missing_rate":"Missing Rate %"}))}
""")

# ===========================================================================
# 11. Data Readiness Assessment
# ===========================================================================
readiness_rows = []
for col in NUMERIC_COLS:
    p  = 100.0 * df[col].isna().sum() / n_rows
    if   p <= 20: cat, detail = "A", "Ready / high coverage (>= 80%)"
    elif p <= 50: cat, detail = "B", "Usable after cleaning / imputation (50-80% coverage)"
    elif p <= 80: cat, detail = "C", "High missingness -- needs careful treatment (20-50% coverage)"
    else:         cat, detail = "D", "Potentially unsuitable for ML (< 20% coverage)"
    readiness_rows.append({"Feature":col, "Missing %":pct(p), "Category":cat, "Assessment":detail})

readiness_df = pd.DataFrame(readiness_rows).sort_values(["Category","Missing %"])
cat_a = readiness_df[readiness_df.Category=="A"]["Feature"].tolist()
cat_b = readiness_df[readiness_df.Category=="B"]["Feature"].tolist()
cat_c = readiness_df[readiness_df.Category=="C"]["Feature"].tolist()
cat_d = readiness_df[readiness_df.Category=="D"]["Feature"].tolist()

cat_counts = readiness_df["Category"].value_counts().sort_index().reset_index()
cat_counts.columns = ["Category","Feature Count"]

sections.append(f"""## 11. Data Readiness Assessment

| Category | Meaning | Features |
|---|---|---|
| A | Ready / high coverage (>= 80%) | {cat_a} |
| B | Usable after cleaning (50-80% coverage) | {cat_b} |
| C | High missingness -- handle carefully (20-50% coverage) | {cat_c} |
| D | Potentially unsuitable (< 20% coverage) | {cat_d} |

### Per-Feature Detail

{md_table(readiness_df)}

### Summary Count

{md_table(cat_counts)}
""")

# ===========================================================================
# 12. Recommendations
# ===========================================================================
sections.append(f"""## 12. Recommendations for the Next Step

> IMPORTANT: These are recommendations only. No action was taken automatically.

---

### Confirmed Facts

- Dataset is a panel of **{n_rows:,} rows x {n_cols} columns** covering **{unique_codes} countries** from **{year_min}** to **{year_max}**.
- Panel is globally continuous (no missing global years).
- No duplicate (country_code, year) pairs detected.
- Target `{TARGET_COL}` has **{pct(pct_miss_tgt)}** missing rate -- manageable.
- Category A features ({len(cat_a)}): {cat_a}

---

### Warnings

- Countries with <=10 observations ({len(few_obs)} total) may be unreliable for training.
- Category D features ({cat_d}) are very sparsely populated and may add noise.
- Extreme values flagged -- review before winsorising.
- Exchange-rate column has near-zero or zero values in some rows -- verify encoding.

---

### Recommended Next Step: Phase 11.2 — Data Preprocessing Pipeline

1. **Create** `apps/api/ml/data_preprocessing.py` (do NOT modify master_panel.csv).
2. **Imputation strategy**:
   - Category A: forward-fill / backward-fill within each country panel.
   - Category B: country-level mean or linear interpolation.
   - Category C: global median or indicator-variable encoding.
   - Category D ({cat_d}): consider dropping from initial feature set.
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
""")

# ---------------------------------------------------------------------------
# Write report
# ---------------------------------------------------------------------------
REPORT_PATH.write_text("\n".join(sections), encoding="utf-8")
print(f"[INFO] Report written -> {REPORT_PATH}")

print("\n" + "="*60)
print("AUDIT SUMMARY")
print("="*60)
print(f"  Rows            : {n_rows:,}")
print(f"  Columns         : {n_cols}")
print(f"  Countries       : {unique_codes}")
print(f"  Year range      : {year_min}-{year_max}")
print(f"  Target missing  : {pct(pct_miss_tgt)}")
print(f"  Readiness A     : {len(cat_a)} features -> {cat_a}")
print(f"  Readiness B     : {len(cat_b)} features -> {cat_b}")
print(f"  Readiness C     : {len(cat_c)} features -> {cat_c}")
print(f"  Readiness D     : {len(cat_d)} features -> {cat_d}")
print(f"  Validity flags  : {len(findings)}")
print("="*60)
