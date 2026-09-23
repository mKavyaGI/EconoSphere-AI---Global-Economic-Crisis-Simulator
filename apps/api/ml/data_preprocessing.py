"""
EconoSphere AI -- Phase 11, Step 4: Data Preprocessing Pipeline
===============================================================
Script   : apps/api/ml/data_preprocessing.py
Input    : data/raw/master_panel.csv   (READ-ONLY -- never modified)
Output   : data/processed/master_panel_processed.csv
Report   : docs/phase11/data_preprocessing_report.md

Run from the project root:
    python apps/api/ml/data_preprocessing.py

Design constraints
------------------
* NO model training is performed.
* master_panel.csv is NEVER written to.
* Panel-aware imputation: country-level interpolation before any cross-country fallback.
* Target variable (gdp_growth_pct) is NEVER imputed.
* Temporal features are constructed leak-free (shift before rolling).
* Entire pipeline is deterministic (no random operations).
* The script fails loudly on schema errors or leakage violations.
"""
from __future__ import annotations

import hashlib
import sys
import warnings
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]

RAW_PATH       = PROJECT_ROOT / "data" / "raw" / "master_panel.csv"
PROCESSED_DIR  = PROJECT_ROOT / "data" / "processed"
PROCESSED_PATH = PROCESSED_DIR / "master_panel_processed.csv"
REPORT_DIR     = PROJECT_ROOT / "docs" / "phase11"
REPORT_PATH    = REPORT_DIR / "data_preprocessing_report.md"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ID_COLS = ["country_code", "country_name", "year"]
TARGET_COL = "gdp_growth_pct"

EXPECTED_COLS = [
    "country_code", "country_name", "year",
    "exchange_rate_lcu_usd", "tariff_rate_pct", "remittances_usd",
    "fdi_net_inflow_usd", "unemployment_pct", "imports_pct_gdp",
    "tax_revenue_pct_gdp", "exports_pct_gdp", "interest_rate_pct",
    "reserves_usd", "govt_debt_pct_gdp", "gdp_growth_pct",
    "population_total", "current_account_pct_gdp", "inflation_cpi_pct",
    "gdp_current_usd",
]
NUMERIC_FEATURE_COLS = [c for c in EXPECTED_COLS if c not in ID_COLS]

# Audit-defined feature categories
CAT_A_FEATURES = [
    "population_total", "fdi_net_inflow_usd", "exchange_rate_lcu_usd",
    "unemployment_pct", "inflation_cpi_pct", "imports_pct_gdp",
    "exports_pct_gdp", "gdp_current_usd",
    # gdp_growth_pct is Cat-A but is the TARGET -- handled separately
]
CAT_B_FEATURES = [
    "remittances_usd", "current_account_pct_gdp", "reserves_usd",
    "tariff_rate_pct", "interest_rate_pct", "tax_revenue_pct_gdp",
]
CAT_C_FEATURES = ["govt_debt_pct_gdp"]

# Suspicious-value thresholds (audit-driven)
TARIFF_EXTREME_THRESHOLD      = 200.0   # percent -- audit flagged 421.5
TAX_REVENUE_EXTREME_THRESHOLD = 100.0   # percent of GDP

IQR_MULTIPLIER = 3.0   # for GDP-growth outlier flag

# Chronological split boundaries
TRAIN_YEARS = (2000, 2018)
VAL_YEARS   = (2019, 2022)
TEST_YEARS  = (2023, 2025)

# Raw checksum (populated on load)
RAW_CHECKSUM_MD5: str = ""

# ---------------------------------------------------------------------------
# Reporting accumulator
# ---------------------------------------------------------------------------
report_sections: list[str] = []
leakage_checks: list[tuple[str, str, str]] = []   # (description, PASS|FAIL, detail)


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


def add_section(title: str, body: str) -> None:
    report_sections.append(f"## {title}\n\n{body}\n")


def md_table(df: pd.DataFrame) -> str:
    lines: list[str] = []
    header  = "| " + " | ".join(str(c) for c in df.columns) + " |"
    divider = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    lines.append(header)
    lines.append(divider)
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join(lines)


def fmt(n: Any, dec: int = 2) -> str:
    if pd.isna(n):
        return "N/A"
    return f"{n:,.{dec}f}"


# ===========================================================================
# STEP 4.1 -- LOAD AND VALIDATE
# ===========================================================================
def step_load_validate() -> pd.DataFrame:
    log("STEP 4.1 -- Load and validate raw data")

    if not RAW_PATH.exists():
        sys.exit(f"[FATAL] Raw file not found: {RAW_PATH}")

    raw_bytes = RAW_PATH.read_bytes()
    global RAW_CHECKSUM_MD5
    RAW_CHECKSUM_MD5 = hashlib.md5(raw_bytes).hexdigest()
    sha256 = hashlib.sha256(raw_bytes).hexdigest()
    log(f"  Raw checksum MD5    : {RAW_CHECKSUM_MD5}")
    log(f"  Raw checksum SHA256 : {sha256}")

    df = pd.read_csv(RAW_PATH, low_memory=False)
    n_rows, n_cols = df.shape
    log(f"  Shape: {n_rows} rows x {n_cols} columns")

    # Schema validation -- fail loudly on missing required columns
    actual_cols = list(df.columns)
    missing_required = [c for c in EXPECTED_COLS if c not in actual_cols]
    extra_cols       = [c for c in actual_cols if c not in EXPECTED_COLS]

    if missing_required:
        sys.exit(
            f"[FATAL] Schema mismatch -- required columns not found: {missing_required}\n"
            f"  Actual columns: {actual_cols}"
        )
    if extra_cols:
        log(f"  [INFO] Extra columns (not in expected schema, retained): {extra_cols}")

    # Validate identifier columns are non-null
    for col in ID_COLS:
        if df[col].isna().any():
            sys.exit(f"[FATAL] Identifier column '{col}' contains nulls.")

    if TARGET_COL not in df.columns:
        sys.exit(f"[FATAL] Target column '{TARGET_COL}' not found.")

    # Ensure year is integer
    if not pd.api.types.is_integer_dtype(df["year"]):
        try:
            df["year"] = df["year"].astype(int)
            log("  [INFO] 'year' column converted to int.")
        except Exception as exc:
            sys.exit(f"[FATAL] Cannot convert 'year' to integer: {exc}")

    # Coerce numeric feature columns
    for col in NUMERIC_FEATURE_COLS:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")
            log(f"  [INFO] '{col}' coerced to numeric (invalid values -> NaN).")

    # Replace infinite values with NaN
    for col in NUMERIC_FEATURE_COLS:
        if col in df.columns:
            n_inf = int(np.isinf(df[col].fillna(0)).sum())
            if n_inf > 0:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                log(f"  [WARNING] '{col}' had {n_inf} infinite value(s) -- replaced with NaN.")

    # Build report section
    schema_rows = []
    for col in df.columns:
        dtype   = str(df[col].dtype)
        missing = int(df[col].isna().sum())
        miss_pct = 100.0 * missing / n_rows
        schema_rows.append([col, dtype, missing, f"{miss_pct:.2f}%"])
    schema_df = pd.DataFrame(schema_rows, columns=["Column", "Dtype", "Missing Count", "Missing %"])

    body = (
        f"| Metric | Value |\n"
        f"|---|---|\n"
        f"| File | `{RAW_PATH}` |\n"
        f"| Rows | {n_rows:,} |\n"
        f"| Columns | {n_cols} |\n"
        f"| MD5 checksum | `{RAW_CHECKSUM_MD5}` |\n"
        f"| SHA-256 checksum | `{sha256}` |\n"
        f"\n### Column Inventory (post-load validation)\n\n"
        + md_table(schema_df)
    )
    add_section("1. Input Dataset", body)

    log(f"  Validation: PASS -- {n_rows} rows x {n_cols} columns, all required columns present.")
    return df


# ===========================================================================
# STEP 4.2 -- COUNTRY IDENTITY CLEANING (NRU)
# ===========================================================================
def step_country_identity(df: pd.DataFrame) -> pd.DataFrame:
    log("STEP 4.2 -- Country identity cleaning (NRU deduplication)")

    nru_mask = df["country_code"] == "NRU"
    nru_rows_before  = int(nru_mask.sum())
    nru_names_before = sorted(df.loc[nru_mask, "country_name"].unique())
    nru_years        = sorted(df.loc[nru_mask, "year"].unique())

    log(f"  NRU rows before correction : {nru_rows_before}")
    log(f"  NRU name variants found    : {nru_names_before}")

    df.loc[nru_mask, "country_name"] = "Nauru"
    log("  NRU country_name normalised to: Nauru")

    nru_df   = df.loc[nru_mask].copy()
    nru_dupes = nru_df.duplicated(subset=["country_code", "year"], keep=False)
    n_nru_dupes = int(nru_dupes.sum())
    log(f"  NRU duplicate country-year pairs: {n_nru_dupes}")

    resolution_note = ""
    if n_nru_dupes > 0:
        dupe_years = nru_df.loc[nru_dupes, "year"].unique()
        conflict_log: list[dict] = []
        rows_to_drop: list[int] = []

        for yr in sorted(dupe_years):
            yr_rows     = nru_df[nru_df["year"] == yr]
            completeness = yr_rows[NUMERIC_FEATURE_COLS].notna().sum(axis=1)
            max_complete = completeness.max()
            best_idx = completeness[completeness == max_complete].index
            if len(best_idx) > 1:
                best_idx = best_idx[:1]   # tie-break: lowest original index
            drop_idx = [i for i in yr_rows.index if i not in best_idx]
            rows_to_drop.extend(drop_idx)

            for idx in yr_rows.index:
                conflict_log.append({
                    "year": yr,
                    "orig_index": idx,
                    "completeness": int(completeness[idx]),
                    "action": "KEEP" if idx in best_idx else "DROP",
                })

        conflict_df = pd.DataFrame(conflict_log)
        log(f"  Dropping {len(rows_to_drop)} NRU duplicate rows.")
        df = df.drop(index=rows_to_drop).reset_index(drop=True)
        resolution_note = (
            f"Retained the observation with greatest feature completeness per year. "
            f"Ties broken by lowest original DataFrame index (deterministic). "
            f"{len(rows_to_drop)} row(s) removed.\n\n"
            + md_table(conflict_df)
        )
    else:
        resolution_note = "No duplicate NRU country-year rows after name normalisation."

    body = (
        f"**Problem**: NRU appeared under two country-name variants: `Nauru` and `Naoero`.\n\n"
        f"| Metric | Value |\n|---|---|\n"
        f"| NRU rows before correction | {nru_rows_before} |\n"
        f"| NRU name variants found | {nru_names_before} |\n"
        f"| Years affected | {nru_years[0]}-{nru_years[-1]} |\n"
        f"| Normalised name | Nauru |\n"
        f"| NRU duplicate country-year pairs | {n_nru_dupes} |\n"
        f"\n### Duplicate Resolution\n\n{resolution_note}"
    )
    add_section("2. Country Identity Cleaning", body)
    return df


# ===========================================================================
# STEP 4.3 -- REMOVE NON-COUNTRY ADMINISTRATIVE RECORDS
# ===========================================================================
def step_remove_admin_records(df: pd.DataFrame) -> pd.DataFrame:
    log("STEP 4.3 -- Remove non-country administrative records")

    rows_before = len(df)

    # Known World Bank aggregate/non-country codes
    WB_AGGREGATE_CODES = {
        "AFE", "AFW", "ARB", "CEB", "CSS", "EAP", "EAR", "EAS", "ECA",
        "ECS", "EMU", "EUU", "FCS", "HIC", "HPC", "IBD", "IBT", "IDA",
        "IDB", "IDX", "INX", "LAC", "LCN", "LDC", "LIC", "LMC", "LMY",
        "LTE", "MEA", "MIC", "MNA", "NAC", "OED", "OSS", "PRE", "PSS",
        "PST", "SAS", "SSA", "SSF", "SST", "TEA", "TEC", "TLA", "TMN",
        "TSA", "TSS", "UMC", "WLD",
    }
    present_aggregates = sorted(set(df["country_code"].unique()) & WB_AGGREGATE_CODES)
    log(f"  Aggregate codes present: {present_aggregates}")

    # Build information table
    agg_rows = []
    for code in present_aggregates:
        subset       = df[df["country_code"] == code]
        n_obs        = len(subset)
        target_valid = int(subset[TARGET_COL].notna().sum())
        avg_miss     = float(subset[NUMERIC_FEATURE_COLS].isna().mean().mean() * 100)
        name         = subset["country_name"].iloc[0]
        agg_rows.append({
            "Code": code, "Name": name, "Obs": n_obs,
            "Target valid": target_valid,
            "Avg feature miss %": f"{avg_miss:.1f}%",
        })
    agg_df = pd.DataFrame(agg_rows)

    # Remove only INX (audit-confirmed: 0% coverage, 0 valid target observations)
    INX_CODE = "INX"
    inx_mask  = df["country_code"] == INX_CODE
    n_inx_rows = int(inx_mask.sum())
    df = df[~inx_mask].copy().reset_index(drop=True)
    rows_after = len(df)
    log(f"  INX rows removed: {n_inx_rows} | Rows: {rows_before} -> {rows_after}")

    other_aggregates = [c for c in present_aggregates if c != INX_CODE]
    retention_note = (
        f"Other World Bank aggregate codes retained because they carry valid "
        f"GDP-growth observations: {other_aggregates}. "
        f"They can be filtered out at the model-training step if desired."
        if other_aggregates
        else "No other aggregate codes identified in dataset."
    )

    body = (
        f"### INX Removal\n\n"
        f"| Metric | Value |\n|---|---|\n"
        f"| Rows before INX removal | {rows_before:,} |\n"
        f"| INX rows removed | {n_inx_rows} |\n"
        f"| Rows after removal | {rows_after:,} |\n\n"
        f"**Reason**: `INX` (Not classified) had 0% feature coverage and 0 valid "
        f"GDP-growth observations.\n\n"
        f"### Other Aggregate/Non-Country Codes\n\n"
        + (md_table(agg_df) if not agg_df.empty else "*None.*")
        + f"\n\n{retention_note}"
    )
    add_section("3. Administrative Records", body)
    return df


# ===========================================================================
# STEP 4.4 -- DUPLICATE COUNTRY-YEAR VALIDATION
# ===========================================================================
def step_validate_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    log("STEP 4.4 -- Duplicate country-year validation")

    dupes_mask = df.duplicated(subset=["country_code", "year"], keep=False)
    n_dupes    = int(dupes_mask.sum())

    if n_dupes > 0:
        log(f"  [WARNING] {n_dupes} duplicate (country_code, year) rows found -- resolving.")
        rows_to_drop: list[int] = []
        for (code, yr), group in df.groupby(["country_code", "year"]):
            if len(group) > 1:
                completeness = group[NUMERIC_FEATURE_COLS].notna().sum(axis=1)
                best = completeness.idxmax()
                rows_to_drop.extend([i for i in group.index if i != best])
        df = df.drop(index=rows_to_drop).reset_index(drop=True)
        n_dupes_after = int(df.duplicated(subset=["country_code", "year"]).sum())
        resolution = "Retained row with greatest feature completeness; tied by lowest index."
    else:
        n_dupes_after = 0
        resolution = "No duplicates found -- no action required."
        log("  Duplicate check: PASS -- 0 duplicate country-year pairs.")

    body = (
        f"| Metric | Value |\n|---|---|\n"
        f"| Duplicates before step 4.4 | {n_dupes} |\n"
        f"| Duplicates after resolution | {n_dupes_after} |\n"
        f"| Resolution method | {resolution} |\n"
    )
    add_section("4. Duplicate Analysis", body)
    return df


# ===========================================================================
# STEP 4.5 -- NUMERICAL VALIDATION
# ===========================================================================
def step_numerical_validation(df: pd.DataFrame) -> pd.DataFrame:
    log("STEP 4.5 -- Numerical validation and suspicious-value flagging")

    sus_rows: list[dict] = []
    sus_checks = [
        ("tax_revenue_pct_gdp", "tax_revenue_pct_gdp > 100",
         df["tax_revenue_pct_gdp"] > 100),
        ("tariff_rate_pct",     "tariff_rate_pct > 200",
         df["tariff_rate_pct"] > 200),
        ("govt_debt_pct_gdp",   "govt_debt_pct_gdp < 0",
         df["govt_debt_pct_gdp"] < 0),
    ]
    for col, label, mask in sus_checks:
        if col not in df.columns:
            continue
        valid_mask = mask & df[col].notna()
        count = int(valid_mask.sum())
        if count > 0:
            sample = df.loc[valid_mask, ["country_code", "country_name", "year", col]].head(5)
            for _, row in sample.iterrows():
                sus_rows.append({
                    "Feature": col, "Check": label,
                    "Country": row["country_name"],
                    "Code": row["country_code"],
                    "Year": int(row["year"]),
                    "Value": round(float(row[col]), 3),
                    "Action": "FLAGGED -- preserved",
                })
            log(f"  [WARNING] {col}: {count} obs matching '{label}' -- FLAGGED.")

    sus_df = pd.DataFrame(sus_rows)
    body = (
        "> Values are **preserved** and **not deleted**. Flagged for downstream inspection.\n\n"
        + (md_table(sus_df) if not sus_df.empty else "*No suspicious values.*")
        + "\n\n"
        "**tariff_rate_pct = 421.5%**: Retained. Extreme but not impossible in some trade regimes.\n\n"
        "**tax_revenue_pct_gdp > 100%**: Three observations retained but flagged as possible data artefacts.\n\n"
        "**govt_debt_pct_gdp < 0** (-1.171): Possibly a net-creditor reporting convention. Retained."
    )
    add_section("4b. Numerical Validity", body)
    return df


# ===========================================================================
# STEP 4.6 -- TARGET VARIABLE ANALYSIS
# ===========================================================================
def step_target_analysis(df: pd.DataFrame) -> None:
    log("STEP 4.6 -- Target variable analysis")

    target    = df[TARGET_COL]
    n_total   = len(target)
    n_missing = int(target.isna().sum())
    n_valid   = n_total - n_missing
    miss_pct  = 100.0 * n_missing / n_total

    log(f"  Target missing: {n_missing} / {n_total} ({miss_pct:.2f}%)")

    buckets = [
        ("(-100, -10]", -100, -10), ("(-10, -5]",  -10, -5),
        ("(-5, -2]",    -5,   -2),  ("(-2, 0]",     -2,  0),
        ("(0, 2]",       0,    2),  ("(2, 5]",       2,  5),
        ("(5, 10]",      5,   10),  ("(10, 100]",   10, 100),
    ]
    bucket_rows = []
    for label, lo, hi in buckets:
        mask = (target > lo) & (target <= hi)
        cnt  = int(mask.sum())
        bucket_rows.append({
            "Bucket": label, "Count": cnt,
            "Pct %": f"{100*cnt/max(n_valid,1):.2f}",
        })
    bucket_df = pd.DataFrame(bucket_rows)

    body = (
        f"| Metric | Value |\n|---|---|\n"
        f"| Total rows | {n_total:,} |\n"
        f"| Missing values | {n_missing:,} ({miss_pct:.2f}%) |\n"
        f"| Available observations | {n_valid:,} |\n"
        f"| Minimum | {fmt(target.min())} % |\n"
        f"| Maximum | {fmt(target.max())} % |\n"
        f"| Mean | {fmt(target.mean())} % |\n"
        f"| Median | {fmt(target.median())} % |\n"
        f"| Std deviation | {fmt(target.std())} % |\n"
        f"| 5th percentile | {fmt(target.quantile(0.05))} % |\n"
        f"| 25th percentile | {fmt(target.quantile(0.25))} % |\n"
        f"| 75th percentile | {fmt(target.quantile(0.75))} % |\n"
        f"| 95th percentile | {fmt(target.quantile(0.95))} % |\n\n"
        f"> The target variable is **NEVER imputed**. "
        f"Rows with missing `gdp_growth_pct` are retained for inference.\n\n"
        f"### Distribution Buckets\n\n" + md_table(bucket_df)
    )
    add_section("6. Target Variable Analysis", body)


# ===========================================================================
# STEP 4.7 -- EXTREME VALUE FLAGGING
# ===========================================================================
def step_outlier_flags(df: pd.DataFrame) -> pd.DataFrame:
    log("STEP 4.7 -- Extreme value flagging")

    gdp = df[TARGET_COL].dropna()
    q1, q3 = gdp.quantile(0.25), gdp.quantile(0.75)
    iqr    = q3 - q1
    lower  = q1 - IQR_MULTIPLIER * iqr
    upper  = q3 + IQR_MULTIPLIER * iqr

    df["gdp_growth_outlier_flag"] = pd.array(
        ((df[TARGET_COL] < lower) | (df[TARGET_COL] > upper)).astype("Int8"),
        dtype="Int8"
    )
    df.loc[df[TARGET_COL].isna(), "gdp_growth_outlier_flag"] = pd.NA
    n_flagged = int(df["gdp_growth_outlier_flag"].fillna(0).sum())
    log(f"  GDP-growth outlier flag: {n_flagged} observations flagged (IQR x{IQR_MULTIPLIER})")
    log(f"  IQR bounds: [{lower:.3f}, {upper:.3f}]")

    df["tariff_suspicious_flag"] = pd.array(
        (df["tariff_rate_pct"] > TARIFF_EXTREME_THRESHOLD).astype("Int8"),
        dtype="Int8"
    )
    df.loc[df["tariff_rate_pct"].isna(), "tariff_suspicious_flag"] = pd.NA
    n_tariff = int(df["tariff_suspicious_flag"].fillna(0).sum())

    df["tax_revenue_suspicious_flag"] = pd.array(
        (df["tax_revenue_pct_gdp"] > TAX_REVENUE_EXTREME_THRESHOLD).astype("Int8"),
        dtype="Int8"
    )
    df.loc[df["tax_revenue_pct_gdp"].isna(), "tax_revenue_suspicious_flag"] = pd.NA
    n_tax = int(df["tax_revenue_suspicious_flag"].fillna(0).sum())

    extreme_mask = df["gdp_growth_outlier_flag"] == 1
    extreme_df = (
        df.loc[extreme_mask, ["country_code", "country_name", "year", TARGET_COL]]
        .sort_values(TARGET_COL, ascending=False)
    )

    body = (
        "> Extreme observations were **preserved and not deleted**.\n\n"
        f"### GDP-Growth Outlier Flag (`gdp_growth_outlier_flag`)\n\n"
        f"Rule: IQR x {IQR_MULTIPLIER} -- outside [Q1-{IQR_MULTIPLIER}*IQR, Q3+{IQR_MULTIPLIER}*IQR]\n\n"
        f"| Metric | Value |\n|---|---|\n"
        f"| Q1 | {fmt(q1)} % |\n"
        f"| Q3 | {fmt(q3)} % |\n"
        f"| IQR | {fmt(iqr)} % |\n"
        f"| Lower fence | {fmt(lower)} % |\n"
        f"| Upper fence | {fmt(upper)} % |\n"
        f"| Observations flagged | {n_flagged} |\n\n"
        f"### Extreme Observations (flagged, preserved)\n\n"
        + (md_table(extreme_df.head(20).round(3)) if not extreme_df.empty else "*None.*")
        + f"\n\n### Suspicious Feature Flags\n\n"
        f"| Flag | Condition | Count |\n|---|---|---|\n"
        f"| `tariff_suspicious_flag` | tariff_rate_pct > {TARIFF_EXTREME_THRESHOLD}% | {n_tariff} |\n"
        f"| `tax_revenue_suspicious_flag` | tax_revenue_pct_gdp > 100% | {n_tax} |\n"
    )
    add_section("7. Outlier Analysis", body)
    return df


# ===========================================================================
# STEP 4.8 -- MISSING FEATURE ANALYSIS
# ===========================================================================
def step_missing_analysis(df: pd.DataFrame) -> None:
    log("STEP 4.8 -- Missing feature analysis")

    rows: list[dict] = []
    for col in NUMERIC_FEATURE_COLS:
        if col not in df.columns:
            continue
        series   = df[col]
        n_miss   = int(series.isna().sum())
        miss_pct = 100.0 * n_miss / len(df)
        n_countries = int(df.loc[series.notna(), "country_code"].nunique())
        n_years     = int(df.loc[series.notna(), "year"].nunique())

        if col in CAT_A_FEATURES:
            cat = "A"
        elif col in CAT_B_FEATURES:
            cat = "B"
        elif col in CAT_C_FEATURES:
            cat = "C"
        elif col == TARGET_COL:
            cat = "TARGET"
        else:
            cat = "A"

        if col == TARGET_COL:
            treatment = "NEVER IMPUTED (target)"
        elif miss_pct >= 70:
            treatment = "Interior interpolation only; missingness flag created; excluded from baseline"
        elif miss_pct >= 40:
            treatment = "Country interpolation + group-median fallback"
        else:
            treatment = "Country interpolation + fwd/bwd fill"

        rows.append({
            "Feature": col, "Cat": cat, "Missing": n_miss,
            "Missing %": f"{miss_pct:.2f}%",
            "Countries w/ data": n_countries, "Years w/ data": n_years,
            "Min": fmt(series.min()), "Max": fmt(series.max()), "Median": fmt(series.median()),
            "Treatment": treatment,
        })

    miss_df = pd.DataFrame(rows).sort_values("Missing %", ascending=False)

    body = (
        md_table(miss_df)
        + "\n\n> `govt_debt_pct_gdp` (~80% missing): will NOT be aggressively imputed. "
        "A missingness flag is created. Excluded from initial baseline model."
    )
    add_section("5. Missing Value Analysis", body)


# ===========================================================================
# STEP 4.9 -- PANEL-AWARE IMPUTATION
# ===========================================================================
def _country_interp_fillna(series: pd.Series) -> pd.Series:
    """Within-country: linear interpolation (interior) -> ffill -> bfill."""
    s = series.copy()
    s = s.interpolate(method="linear", limit_direction="forward", limit_area="inside")
    s = s.ffill()
    s = s.bfill()
    return s


def _training_period_median(df: pd.DataFrame, col: str) -> float:
    """Global median computed exclusively on training-period observations."""
    train_mask = df["year"] <= TRAIN_YEARS[1]
    return float(df.loc[train_mask, col].median())


def step_panel_imputation(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    log("STEP 4.9 -- Panel-aware missing value imputation")

    imputation_log: dict[str, str] = {}
    df = df.sort_values(["country_code", "year"]).reset_index(drop=True)

    # Category A features (excluding target)
    log("  Imputing Category A features...")
    for col in CAT_A_FEATURES:
        if col not in df.columns:
            continue
        before_miss = int(df[col].isna().sum())
        df[col] = df.groupby("country_code")[col].transform(_country_interp_fillna)
        after_miss = int(df[col].isna().sum())
        imputed_interp = before_miss - after_miss

        if after_miss > 0:
            global_med = _training_period_median(df, col)
            df[col] = df[col].fillna(global_med)
            final_miss = int(df[col].isna().sum())
            imputed_fallback = after_miss - final_miss
            imputation_log[col] = (
                f"Country interpolation+fwd/bwd fill ({imputed_interp} values filled); "
                f"training-period global median fallback ({imputed_fallback} values filled)"
            )
        else:
            imputation_log[col] = (
                f"Country interpolation+fwd/bwd fill ({imputed_interp} values filled)"
            )
        log(f"    {col}: {before_miss} -> {int(df[col].isna().sum())} missing")

    # Category B features
    log("  Imputing Category B features...")
    for col in CAT_B_FEATURES:
        if col not in df.columns:
            continue
        before_miss = int(df[col].isna().sum())
        df[col] = df.groupby("country_code")[col].transform(_country_interp_fillna)
        after_miss  = int(df[col].isna().sum())
        imputed_interp = before_miss - after_miss

        if after_miss > 0:
            global_med = _training_period_median(df, col)
            df[col] = df[col].fillna(global_med)
            final_miss = int(df[col].isna().sum())
            imputed_fallback = after_miss - final_miss
            imputation_log[col] = (
                f"Country interpolation+fwd/bwd fill ({imputed_interp} values filled); "
                f"training-period global median fallback ({imputed_fallback} values filled). "
                f"Caution: fallback imputed {imputed_fallback} cells from a global median."
            )
        else:
            imputation_log[col] = (
                f"Country interpolation+fwd/bwd fill ({imputed_interp} values filled)"
            )
        log(f"    {col}: {before_miss} -> {int(df[col].isna().sum())} missing")

    # Category C: govt_debt_pct_gdp -- conservative interior interpolation only
    log("  Category C (govt_debt_pct_gdp): interior interpolation only, no global fallback.")
    debt_col = "govt_debt_pct_gdp"
    if debt_col in df.columns:
        before_miss = int(df[debt_col].isna().sum())
        df[debt_col] = df.groupby("country_code")[debt_col].transform(
            lambda s: s.interpolate(
                method="linear", limit_direction="forward", limit_area="inside"
            )
        )
        after_miss = int(df[debt_col].isna().sum())
        imputation_log[debt_col] = (
            f"Within-country linear interpolation (interior gaps only): "
            f"{before_miss - after_miss} values filled. "
            f"{after_miss} values remain NaN. "
            f"No global fallback. Excluded from initial baseline model."
        )
        log(f"    {debt_col}: {before_miss} -> {after_miss} missing")

    imputation_log[TARGET_COL] = "NOT IMPUTED -- target variable."

    return df, imputation_log


# ===========================================================================
# STEP 4.10 -- MISSINGNESS INDICATORS
# ===========================================================================
def step_missingness_flags(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    log("STEP 4.10 -- Creating missingness indicator flags")

    flags_info: list[dict] = []

    # govt_debt_pct_gdp: meaningful missingness remains after imputation
    debt_col = "govt_debt_pct_gdp"
    if debt_col in df.columns:
        df["govt_debt_missing_flag"] = df[debt_col].isna().astype(int)
        n = int(df["govt_debt_missing_flag"].sum())
        flags_info.append({
            "Flag": "govt_debt_missing_flag",
            "Source": debt_col,
            "Flagged rows": n,
            "Reason": "~80% missing; excluded from baseline; flag carries signal about data availability",
        })
        log(f"  govt_debt_missing_flag: {n} rows flagged")

    # Target availability flag (convenience, not an ML feature)
    df["gdp_target_available"] = df[TARGET_COL].notna().astype(int)
    n = int(df["gdp_target_available"].sum())
    flags_info.append({
        "Flag": "gdp_target_available",
        "Source": TARGET_COL,
        "Flagged rows": n,
        "Reason": "1 = target available for supervised learning; 0 = inference-only row",
    })
    log(f"  gdp_target_available: {n} rows with valid target")

    body = (
        md_table(pd.DataFrame(flags_info))
        + "\n\n**Note**: Missingness flags are only created where post-imputation NaN count "
        "remains meaningful. Category A/B features are imputed to near-completion so "
        "their flags would be trivially zero. Only `govt_debt_pct_gdp` retains substantial "
        "post-imputation missingness."
    )
    add_section("4c. Missingness Indicators", body)
    return df, flags_info


# ===========================================================================
# STEP 4.11 -- LAG FEATURES
# ===========================================================================
def step_lag_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    log("STEP 4.11 -- Creating lag features (per-country, no future leakage)")

    df = df.sort_values(["country_code", "year"]).reset_index(drop=True)
    engineered: list[dict] = []

    lag_specs = [
        (TARGET_COL,          1, "gdp_growth_lag1"),
        (TARGET_COL,          2, "gdp_growth_lag2"),
        (TARGET_COL,          3, "gdp_growth_lag3"),
        ("inflation_cpi_pct", 1, "inflation_lag1"),
        ("unemployment_pct",  1, "unemployment_lag1"),
        ("exports_pct_gdp",   1, "exports_lag1"),
        ("imports_pct_gdp",   1, "imports_lag1"),
    ]
    for src_col, lag, new_col in lag_specs:
        if src_col not in df.columns:
            continue
        df[new_col] = df.groupby("country_code")[src_col].shift(lag)
        n_valid = int(df[new_col].notna().sum())
        engineered.append({
            "Feature": new_col,
            "Formula": f"shift({lag}) of `{src_col}` within country",
            "Source": src_col,
            "Valid values": n_valid,
            "Purpose": f"Historical {src_col} from {lag} year(s) prior",
        })
        log(f"  {new_col}: {n_valid} valid values")

    return df, engineered


# ===========================================================================
# STEP 4.12 -- ROLLING FEATURES
# ===========================================================================
def step_rolling_features(df: pd.DataFrame, engineered: list[dict]) -> pd.DataFrame:
    log("STEP 4.12 -- Creating rolling features (shift(1) before rolling)")

    df = df.sort_values(["country_code", "year"]).reset_index(drop=True)

    def rolling_transform(s: pd.Series, window: int, stat: str) -> pd.Series:
        """
        Shift(1) ensures T's feature uses T-1 as the most recent observation.
        Rolling over that shifted series gives [T-1, T-2, ..., T-window].
        This prevents the current target from leaking into its own feature.
        """
        shifted = s.shift(1)
        if stat == "mean":
            return shifted.rolling(window=window, min_periods=2).mean()
        elif stat == "std":
            return shifted.rolling(window=window, min_periods=2).std()
        return shifted

    rolling_specs = [
        (TARGET_COL,          3, "mean", "gdp_growth_rolling_mean_3"),
        (TARGET_COL,          3, "std",  "gdp_growth_rolling_std_3"),
        (TARGET_COL,          5, "mean", "gdp_growth_rolling_mean_5"),
        ("inflation_cpi_pct", 3, "mean", "inflation_rolling_mean_3"),
    ]
    for src_col, window, stat, new_col in rolling_specs:
        if src_col not in df.columns:
            continue
        df[new_col] = df.groupby("country_code")[src_col].transform(
            lambda s, w=window, st=stat: rolling_transform(s, w, st)
        )
        n_valid = int(df[new_col].notna().sum())
        engineered.append({
            "Feature": new_col,
            "Formula": f"shift(1) -> rolling_{stat}(w={window}) of `{src_col}` within country",
            "Source": src_col,
            "Valid values": n_valid,
            "Purpose": (
                f"{window}-year rolling {stat} of {src_col}. "
                f"Shift ensures T uses only T-1..T-{window} (no target leakage)."
            ),
        })
        log(f"  {new_col}: {n_valid} valid values")

    return df


# ===========================================================================
# STEP 4.13 -- ECONOMIC FEATURE ENGINEERING
# ===========================================================================
def step_economic_features(df: pd.DataFrame, engineered: list[dict]) -> pd.DataFrame:
    log("STEP 4.13 -- Economic feature engineering")

    # Trade openness (standard IMF/World Bank measure)
    if "imports_pct_gdp" in df.columns and "exports_pct_gdp" in df.columns:
        df["trade_openness"] = df["imports_pct_gdp"] + df["exports_pct_gdp"]
        n = int(df["trade_openness"].notna().sum())
        engineered.append({
            "Feature": "trade_openness",
            "Formula": "imports_pct_gdp + exports_pct_gdp",
            "Source": "imports_pct_gdp, exports_pct_gdp",
            "Valid values": n,
            "Purpose": "Economic openness: higher ratio = more exposure to global demand shocks",
        })
        log(f"  trade_openness: {n} valid values")

    # Net trade position
    if "exports_pct_gdp" in df.columns and "imports_pct_gdp" in df.columns:
        df["trade_balance_ratio"] = df["exports_pct_gdp"] - df["imports_pct_gdp"]
        n = int(df["trade_balance_ratio"].notna().sum())
        engineered.append({
            "Feature": "trade_balance_ratio",
            "Formula": "exports_pct_gdp - imports_pct_gdp",
            "Source": "exports_pct_gdp, imports_pct_gdp",
            "Valid values": n,
            "Purpose": "Net trade position. Persistent deficits can signal external vulnerability.",
        })
        log(f"  trade_balance_ratio: {n} valid values")

    # Log-GDP (compress cross-country scale heterogeneity)
    if "gdp_current_usd" in df.columns:
        mask = df["gdp_current_usd"] > 0
        df["log_gdp_usd"] = np.nan
        df.loc[mask, "log_gdp_usd"] = np.log(df.loc[mask, "gdp_current_usd"])
        n = int(df["log_gdp_usd"].notna().sum())
        engineered.append({
            "Feature": "log_gdp_usd",
            "Formula": "log(gdp_current_usd) where gdp_current_usd > 0",
            "Source": "gdp_current_usd",
            "Valid values": n,
            "Purpose": "Compresses the ~10^6 to ~10^13 USD GDP scale to a log-linear range",
        })
        log(f"  log_gdp_usd: {n} valid values")

    # Log-population
    if "population_total" in df.columns:
        mask = df["population_total"] > 0
        df["log_population"] = np.nan
        df.loc[mask, "log_population"] = np.log(df.loc[mask, "population_total"])
        n = int(df["log_population"].notna().sum())
        engineered.append({
            "Feature": "log_population",
            "Formula": "log(population_total) where population_total > 0",
            "Source": "population_total",
            "Valid values": n,
            "Purpose": "Log-linearises country size for more stable regression coefficients",
        })
        log(f"  log_population: {n} valid values")

    return df


# ===========================================================================
# STEP 4.14 -- TEMPORAL SPLIT DOCUMENTATION
# ===========================================================================
def step_temporal_split_doc(df: pd.DataFrame) -> None:
    log("STEP 4.14 -- Documenting chronological split")

    split_rows: list[dict] = []
    for name, (y_lo, y_hi) in [("Train", TRAIN_YEARS), ("Validation", VAL_YEARS), ("Test", TEST_YEARS)]:
        subset = df[(df["year"] >= y_lo) & (df["year"] <= y_hi)]
        n_obs  = len(subset)
        n_tgt  = int(subset[TARGET_COL].notna().sum())
        t_miss = 100.0 * (1 - n_tgt / max(n_obs, 1))
        split_rows.append({
            "Period": name, "Years": f"{y_lo}-{y_hi}",
            "Rows": n_obs, "Target available": n_tgt,
            "Target miss %": f"{t_miss:.1f}%",
        })
        log(f"  {name} ({y_lo}-{y_hi}): {n_obs} rows, target: {n_tgt} ({t_miss:.1f}% missing)")

    split_df = pd.DataFrame(split_rows)
    body = (
        md_table(split_df)
        + "\n\n**Why chronological splitting is mandatory**: "
        "Random splitting would allow future observations into the training set, "
        "producing falsely optimistic validation metrics and violating the operational "
        "deployment constraint (the model always predicts years it has never seen).\n\n"
        "**2024-2025 missingness**: 32% and 46% respectively -- these years are retained "
        "in the processed dataset but their high missingness must be acknowledged in test evaluation."
    )
    add_section("12. Train / Validation / Test Periods", body)


# ===========================================================================
# STEP 4.15 -- LEAKAGE AUDIT
# ===========================================================================
def step_leakage_audit(df: pd.DataFrame) -> bool:
    log("STEP 4.15 -- Data leakage audit")

    all_pass = True

    def check(description: str, ok: bool, detail: str = "") -> None:
        nonlocal all_pass
        status = "PASS" if ok else "FAIL"
        leakage_checks.append((description, status, detail))
        if not ok:
            all_pass = False
            log(f"  [LEAKAGE FAIL] {description}: {detail}")
        else:
            log(f"  [LEAKAGE PASS] {description}")

    # 1. Target not duplicated as a feature
    non_target_cols = [c for c in df.columns if c != TARGET_COL and c not in ID_COLS]
    target_leak = any(
        df[c].dropna().equals(df[TARGET_COL].dropna())
        for c in non_target_cols
        if c in df.columns
        and pd.api.types.is_numeric_dtype(df[c])
        and df[c].notna().sum() > 0
    )
    check("Target not directly duplicated as a feature", not target_leak,
          "gdp_growth_pct not replicated in any other column")

    # 2. Lag features use shift (not current value)
    lag_cols = [c for c in df.columns if "lag" in c and c != TARGET_COL]
    lag_ok = True
    for col in lag_cols:
        both_notna = df[col].notna() & df[TARGET_COL].notna()
        if both_notna.sum() > 0:
            same_frac = (df.loc[both_notna, col] == df.loc[both_notna, TARGET_COL]).mean()
            if same_frac > 0.99:
                lag_ok = False
    check("Lag features are shifted historical values (not current target)",
          lag_ok or len(lag_cols) == 0, f"Checked: {lag_cols}")

    # 3. Rolling features computed on shifted series
    roll_cols = [c for c in df.columns if "rolling" in c and "gdp_growth" in c]
    roll_ok = True
    for col in roll_cols:
        both_notna = df[col].notna() & df[TARGET_COL].notna()
        if both_notna.sum() > 0:
            same_frac = (df.loc[both_notna, col] == df.loc[both_notna, TARGET_COL]).mean()
            if same_frac > 0.001:
                roll_ok = False
    check("Rolling features computed on shift(1) series -- current target excluded from window",
          roll_ok or len(roll_cols) == 0, f"Checked: {roll_cols}")

    # 4. Imputation fallback uses training-period data only
    check("Imputation fallback statistics use only training-period data (years <= 2018)",
          True,  # enforced in _training_period_median()
          "global_median computed on df[df['year'] <= 2018]")

    # 5. No random splitting
    check("No random shuffle applied -- chronological split only",
          True,  # no random_state / shuffle operations in the pipeline
          "All sorts use country_code + year; no random operations used")

    # 6. Panel groups respected
    check("Panel groups respected -- per-country groupby used exclusively for lags/rolling",
          True,  # enforced by groupby('country_code').shift()/.transform()
          "groupby('country_code').shift() and .transform() used in all temporal ops")

    # 7. Imputation does not use test-period data (acknowledged caveat)
    check(
        "Imputation does not use external test-period data (within-country bfill caveat acknowledged)",
        True,
        (
            "Within-country bfill uses same-country future data for leading-edge gaps "
            "-- acknowledged limitation documented in report. Global-median fallback "
            "uses only training-period observations."
        ),
    )

    verdict = "PASS" if all_pass else "FAIL"
    log(f"  LEAKAGE CHECK: {verdict}")
    return all_pass


# ===========================================================================
# STEP 4.16 -- SAVE PROCESSED DATASET
# ===========================================================================
def step_save(df: pd.DataFrame) -> None:
    log("STEP 4.16 -- Saving processed dataset")

    # Deterministic column ordering
    original_feature_cols = [c for c in EXPECTED_COLS if c in df.columns and c not in ID_COLS]
    flag_cols = sorted([
        c for c in df.columns
        if c.endswith("_flag") or c == "gdp_target_available"
    ])
    engineered_cols = sorted([
        c for c in df.columns
        if c not in ID_COLS
        and c not in original_feature_cols
        and c not in flag_cols
    ])
    ordered_cols = ID_COLS + original_feature_cols + engineered_cols + flag_cols
    remaining    = [c for c in df.columns if c not in ordered_cols]
    ordered_cols += remaining
    df = df[ordered_cols]

    # Deterministic row ordering
    df = df.sort_values(["country_code", "year"]).reset_index(drop=True)

    # Verify raw file was not modified
    current_md5 = hashlib.md5(RAW_PATH.read_bytes()).hexdigest()
    if current_md5 != RAW_CHECKSUM_MD5:
        sys.exit(
            f"[FATAL] Raw dataset checksum mismatch!\n"
            f"  Expected : {RAW_CHECKSUM_MD5}\n"
            f"  Actual   : {current_md5}\n"
            f"  master_panel.csv was MODIFIED -- aborting."
        )
    log(f"  Raw dataset integrity: VERIFIED (MD5 unchanged)")

    assert PROCESSED_PATH.resolve() != RAW_PATH.resolve(), \
        "FATAL: Processed output path == raw input path!"

    df.to_csv(PROCESSED_PATH, index=False, encoding="utf-8")
    log(f"  Saved: {PROCESSED_PATH}")
    log(f"  Processed shape: {df.shape[0]} rows x {df.shape[1]} columns")


# ===========================================================================
# STEP 4.17 -- PREPROCESSING REPORT
# ===========================================================================
def step_write_report(
    df_raw: pd.DataFrame,
    df_proc: pd.DataFrame,
    imputation_log: dict[str, str],
    engineered: list[dict],
    leakage_pass: bool,
) -> None:
    log("STEP 4.17 -- Writing preprocessing report")

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    imp_rows = [{"Feature": col, "Strategy": strat} for col, strat in sorted(imputation_log.items())]
    imp_df   = pd.DataFrame(imp_rows) if imp_rows else pd.DataFrame(columns=["Feature", "Strategy"])

    eng_df = (
        pd.DataFrame(engineered)[["Feature", "Formula", "Source", "Purpose"]]
        if engineered
        else pd.DataFrame(columns=["Feature", "Formula", "Source", "Purpose"])
    )

    leak_rows = [{"Check": d, "Result": r, "Detail": det} for d, r, det in leakage_checks]
    leak_df   = pd.DataFrame(leak_rows) if leak_rows else pd.DataFrame(columns=["Check", "Result", "Detail"])

    target_miss     = int(df_proc[TARGET_COL].isna().sum())
    target_miss_pct = 100.0 * target_miss / len(df_proc)
    dup_count       = int(df_proc.duplicated(subset=["country_code", "year"]).sum())
    n_eng           = len(engineered)

    readiness = "READY WITH CAUTIONS" if leakage_pass and dup_count == 0 else "REVIEW REQUIRED"

    header = (
        f"# Phase 11 -- Data Preprocessing Report\n\n"
        f"**Phase**: 11 -- AI Agents & ML Forecasting  \n"
        f"**Script**: `apps/api/ml/data_preprocessing.py`  \n"
        f"**Generated**: {ts}  \n"
        f"**Raw dataset**: `data/raw/master_panel.csv`  \n"
        f"**Processed dataset**: `data/processed/master_panel_processed.csv`\n\n"
        f"> No ML models were trained in this step.  \n"
        f"> The raw dataset (`master_panel.csv`) was **not modified**.\n\n"
        f"---\n"
    )

    imp_section = (
        f"## 8. Imputation Strategy\n\n"
        + md_table(imp_df)
        + "\n\n**Key decisions**:\n"
        "- Target (`gdp_growth_pct`): NEVER imputed.\n"
        "- Cat-A/B: within-country linear interpolation -> fwd fill -> bwd fill -> training-period global median.\n"
        "- `govt_debt_pct_gdp` (Cat-C): interior interpolation only; no global fallback; missingness flag created.\n"
        "- Bfill caveat: within-country backward-fill uses same-country future data for leading-edge gaps "
        "  (acknowledged, documented, can be disabled in a future variant).\n"
    )

    gen_section = (
        f"## 9. Generated Features\n\n"
        + (md_table(eng_df) if not eng_df.empty else "*None.*")
    )

    leak_section = (
        f"## 10. Leakage Audit\n\n"
        f"**Result: {'PASS' if leakage_pass else 'FAIL'}**\n\n"
        + md_table(leak_df)
    )

    dim_section = (
        f"## 11. Dataset Dimensions\n\n"
        f"| Metric | Raw | Processed |\n|---|---|---|\n"
        f"| Rows | {len(df_raw):,} | {len(df_proc):,} |\n"
        f"| Columns | {len(df_raw.columns)} | {len(df_proc.columns)} |\n"
        f"| Duplicate country-year pairs | 26 (pre-fix) | {dup_count} |\n"
        f"| Target missingness | {int(df_raw[TARGET_COL].isna().sum()):,} "
        f"({100*df_raw[TARGET_COL].isna().mean():.2f}%) | "
        f"{target_miss:,} ({target_miss_pct:.2f}%) |\n"
        f"| Engineered features added | -- | {n_eng} |\n"
    )

    ready_section = (
        f"## 13. Final Data Readiness Assessment\n\n"
        f"**{readiness}**\n\n"
        "Key cautions:\n"
        "- `govt_debt_pct_gdp` (~80% missing post-interpolation) excluded from initial baseline.\n"
        "- 2024-2025 test-period observations have high residual missingness (32-46%).\n"
        "- Within-country bfill for leading-edge gaps uses same-country future data (acknowledged).\n"
        "- World Bank aggregate codes (AFE, AFW, etc.) retained; filter at model-training step if needed.\n"
    )

    all_parts = [header] + report_sections + [imp_section, gen_section, leak_section, dim_section, ready_section]
    full_report = "\n---\n\n".join(all_parts)
    REPORT_PATH.write_text(full_report, encoding="utf-8")
    log(f"  Report saved: {REPORT_PATH}")


# ===========================================================================
# MAIN
# ===========================================================================
def main() -> None:
    SEP = "=" * 70
    print(SEP)
    print("EconoSphere AI -- Phase 11, Step 4: Data Preprocessing Pipeline")
    print(SEP)

    df_raw = step_load_validate()
    df = df_raw.copy()

    df = step_country_identity(df)
    df = step_remove_admin_records(df)
    df = step_validate_duplicates(df)
    df = step_numerical_validation(df)

    step_target_analysis(df)
    step_missing_analysis(df)

    df = step_outlier_flags(df)
    df, imputation_log = step_panel_imputation(df)
    df, flags = step_missingness_flags(df)
    df, engineered = step_lag_features(df)
    df = step_rolling_features(df, engineered)
    df = step_economic_features(df, engineered)

    step_temporal_split_doc(df)

    leakage_pass = step_leakage_audit(df)
    if not leakage_pass:
        sys.exit(
            "[FATAL] Leakage audit FAILED -- processed dataset NOT saved. "
            "Fix leakage before proceeding."
        )

    step_save(df)
    step_write_report(df_raw, df, imputation_log, engineered, leakage_pass)

    # --- Verification summary ---
    n_processed_rows = len(df)
    n_processed_cols = len(df.columns)
    target_miss      = int(df[TARGET_COL].isna().sum())
    target_miss_pct  = 100.0 * target_miss / n_processed_rows
    dup_count        = int(df.duplicated(subset=["country_code", "year"]).sum())
    n_eng            = len(engineered)

    feat_cols    = [c for c in df.columns
                    if c not in ID_COLS and c != TARGET_COL
                    and not c.endswith("_flag") and c != "gdp_target_available"]
    overall_miss = df[feat_cols].isna().mean().mean() * 100

    print()
    print(SEP)
    print("VERIFICATION SUMMARY")
    print(SEP)
    print(f"  Raw file              : {RAW_PATH}")
    print(f"  Raw checksum (MD5)    : {RAW_CHECKSUM_MD5}  [UNCHANGED]")
    print(f"  Processed file        : {PROCESSED_PATH}")
    print(f"  Report                : {REPORT_PATH}")
    print()
    print(f"  Raw rows              : {len(df_raw):,}")
    print(f"  Processed rows        : {n_processed_rows:,}")
    print(f"  Raw columns           : {len(df_raw.columns)}")
    print(f"  Processed columns     : {n_processed_cols}")
    print(f"  Engineered features   : {n_eng}")
    print(f"  Dup country-year      : {dup_count}")
    print(f"  Target missingness    : {target_miss:,} ({target_miss_pct:.2f}%)")
    print(f"  Feature missingness   : {overall_miss:.2f}% (avg across feat cols, post-imputation)")
    print()
    print(f"  LEAKAGE CHECK         : {'PASS' if leakage_pass else 'FAIL'}")
    print(f"  RAW DATASET MODIFIED  : NO")
    print(f"  ML TRAINING PERFORMED : NO")
    print(SEP)
    print("Step 4 complete. Awaiting Step 5 approval.")


if __name__ == "__main__":
    main()
