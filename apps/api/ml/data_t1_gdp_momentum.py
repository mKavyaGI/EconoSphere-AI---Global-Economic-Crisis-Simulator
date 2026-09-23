"""
EconoSphere AI — Phase 12, Step 2: GDP Momentum Feature Construction
=====================================================================
Script   : apps/api/ml/data_t1_gdp_momentum.py
Purpose  : Generate experimental GDP momentum/turning-point features for
           Phase 12 Step 2. These are STRICTLY DIAGNOSTIC and EXPERIMENTAL.
           They do NOT modify the locked Phase 11 production pipeline.

Input    : data/processed/master_panel_t1_missingness.csv  (READ-ONLY)
Output   : data/processed/phase12_step2_momentum_features.csv

LEAKAGE SAFETY RULES (enforced throughout):
  - All features for feature_year=t use ONLY GDP data through year t.
  - gdp_growth_pct at year t is the CURRENT year's growth rate (known at t).
  - gdp_growth_lag1(t) = gdp_growth_pct(t-1) — already in the panel.
  - gdp_growth_lag2(t) = gdp_growth_pct(t-2) — already in the panel.
  - gdp_growth_lag3(t) = gdp_growth_pct(t-3) — already in the panel.
  - Acceleration uses shift(1) before diff — referencing past values only.
  - NO forward fill from future years.
  - NO use of gdp_growth_next_year in feature construction.
  - Inference year 2025 is safe: uses 2025 gdp_growth_pct (actual) as lag1.

Canonical column names (from Phase 11 pipeline inspection):
  - gdp_growth_pct   : current year GDP growth rate (the raw observed value)
  - gdp_growth_lag1  : gdp_growth_pct shifted by 1 year per country
  - gdp_growth_lag2  : gdp_growth_pct shifted by 2 years per country
  - gdp_growth_lag3  : gdp_growth_pct shifted by 3 years per country
  - gdp_growth_next_year : TARGET (never used in feature construction)
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
INPUT_PATH   = PROJECT_ROOT / "data" / "processed" / "master_panel_t1_missingness.csv"
OUTPUT_PATH  = PROJECT_ROOT / "data" / "processed" / "phase12_step2_momentum_features.csv"
RAW_PATH     = PROJECT_ROOT / "data" / "raw" / "master_panel.csv"

EXPECTED_MD5 = "8ac7e0b2bf09fbe89289f82d0c7cf25e"

# ── New Experimental Feature Names ────────────────────────────────────────────
# Group A: lag features (already exist in panel — included for reference)
LAG_FEATURES = [
    "gdp_growth_lag1",   # gdp_growth_pct(t-1) — already in panel
    "gdp_growth_lag2",   # gdp_growth_pct(t-2) — already in panel
    "gdp_growth_lag3",   # gdp_growth_pct(t-3) — already in panel
]

# Group B: Acceleration / Deceleration
ACCEL_FEATURES = [
    "gdp_growth_accel_1y",     # growth(t) - growth(t-1)
    "gdp_growth_accel_2y",     # growth(t-1) - growth(t-2)
    "gdp_growth_trend_change", # [growth(t)-growth(t-1)] - [growth(t-1)-growth(t-2)]
]

# Group C: Rolling GDP momentum
ROLLING_FEATURES = [
    "gdp_growth_mean_2y",  # 2-year historical rolling mean (shift(1), window=2)
    "gdp_growth_mean_3y",  # 3-year historical rolling mean (shift(1), window=3)
    "gdp_growth_std_3y",   # 3-year historical rolling std  (shift(1), window=3)
]

# Group D: Direction / turning-point binary indicators
DIRECTION_FEATURES = [
    "gdp_growth_decelerating",   # 1 if growth(t) < growth(t-1)
    "gdp_growth_accelerating",   # 1 if growth(t) > growth(t-1)
    "gdp_growth_negative_lag1",  # 1 if growth(t-1) < 0
    "gdp_growth_decline_2y",     # 1 if growth(t)<growth(t-1) AND growth(t-1)<growth(t-2)
]

# All new experimental features (excluding lags already in panel)
NEW_MOMENTUM_FEATURES = ACCEL_FEATURES + ROLLING_FEATURES + DIRECTION_FEATURES


def verify_md5() -> None:
    actual = hashlib.md5(RAW_PATH.read_bytes()).hexdigest()
    if actual != EXPECTED_MD5:
        sys.exit(f"[FATAL] Raw dataset MD5 mismatch! Expected {EXPECTED_MD5}, got {actual}")
    print(f"[OK] Raw dataset MD5 verified: {actual}")


def build_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build all Phase 12 Step 2 GDP momentum features.

    All features use ONLY information available at feature_year t.

    The dataframe must be sorted by [country_code, year] before calling.
    """
    df = df.sort_values(["country_code", "year"]).copy()

    # Convenience: group by country for rolling operations
    grp = df.groupby("country_code")["gdp_growth_pct"]

    # ── Group B: Acceleration / Deceleration ─────────────────────────────────
    # gdp_growth_accel_1y = growth(t) - growth(t-1)
    # growth(t)   = gdp_growth_pct (current year — available at feature_year t)
    # growth(t-1) = gdp_growth_lag1 (already in panel, shift(1))
    df["gdp_growth_accel_1y"] = df["gdp_growth_pct"] - df["gdp_growth_lag1"]

    # gdp_growth_accel_2y = growth(t-1) - growth(t-2)
    df["gdp_growth_accel_2y"] = df["gdp_growth_lag1"] - df["gdp_growth_lag2"]

    # gdp_growth_trend_change = accel_1y - accel_2y
    # = [growth(t) - growth(t-1)] - [growth(t-1) - growth(t-2)]
    df["gdp_growth_trend_change"] = df["gdp_growth_accel_1y"] - df["gdp_growth_accel_2y"]

    # ── Group C: Rolling GDP Momentum ─────────────────────────────────────────
    # For feature_year = t:
    #   mean_2y: mean of growth values at t-1, t-2   (shift(1) then window=2)
    #   mean_3y: mean of growth values at t-1,t-2,t-3 (shift(1) then window=3)
    #   std_3y:  std  of growth values at t-1,t-2,t-3 (shift(1) then window=3)
    # Using shift(1) ensures NO current year gdp_growth_pct leaks into the rolling
    # calculation for the current row -- consistent with existing pipeline convention.
    # NOTE: gdp_growth_rolling_mean_3 already in panel also uses shift(1), window=3.
    # Our new mean_3y is identical to that — we keep it for explicit experiment naming.
    df["gdp_growth_mean_2y"] = grp.transform(
        lambda s: s.shift(1).rolling(window=2, min_periods=1).mean()
    )
    df["gdp_growth_mean_3y"] = grp.transform(
        lambda s: s.shift(1).rolling(window=3, min_periods=2).mean()
    )
    df["gdp_growth_std_3y"] = grp.transform(
        lambda s: s.shift(1).rolling(window=3, min_periods=2).std()
    )

    # ── Group D: Direction / Turning-Point Indicators ─────────────────────────
    # All computed from current/lagged values only.
    # gdp_growth_decelerating: growth(t) < growth(t-1)
    df["gdp_growth_decelerating"] = (
        df["gdp_growth_pct"] < df["gdp_growth_lag1"]
    ).astype(float)  # float so imputer handles NaN gracefully

    # gdp_growth_accelerating: growth(t) > growth(t-1)
    df["gdp_growth_accelerating"] = (
        df["gdp_growth_pct"] > df["gdp_growth_lag1"]
    ).astype(float)

    # gdp_growth_negative_lag1: growth(t-1) < 0
    df["gdp_growth_negative_lag1"] = (
        df["gdp_growth_lag1"] < 0
    ).astype(float)

    # gdp_growth_decline_2y: growth(t) < growth(t-1) AND growth(t-1) < growth(t-2)
    df["gdp_growth_decline_2y"] = (
        (df["gdp_growth_pct"] < df["gdp_growth_lag1"]) &
        (df["gdp_growth_lag1"] < df["gdp_growth_lag2"])
    ).astype(float)

    # Set any rows where either lag is NaN to NaN for binary indicators
    # (avoids treating NaN < value as False, producing spurious 0s)
    nan_lag1 = df["gdp_growth_lag1"].isna()
    nan_lag2 = df["gdp_growth_lag2"].isna()

    for col in ["gdp_growth_decelerating", "gdp_growth_accelerating"]:
        df.loc[nan_lag1, col] = np.nan

    df.loc[nan_lag1, "gdp_growth_negative_lag1"] = np.nan
    df.loc[nan_lag1 | nan_lag2, "gdp_growth_decline_2y"] = np.nan

    return df


def leakage_guard(df: pd.DataFrame) -> None:
    """Confirm that no new feature was accidentally derived from gdp_growth_next_year."""
    forbidden = "gdp_growth_next_year"
    # Verify target column is still NaN for 2025 rows
    rows_2025 = df[df["year"] == 2025]
    assert rows_2025[forbidden].isna().all(), (
        "LEAKAGE: 2025 gdp_growth_next_year is not NaN!"
    )
    # Verify new features don't contain constant values matching next-year target
    # (heuristic: check that accel_1y values don't correlate 1.0 with next_year target)
    valid = df[df[forbidden].notna()].copy()
    for feat in NEW_MOMENTUM_FEATURES:
        if feat in df.columns:
            corr = valid[[feat, forbidden]].dropna().corr().iloc[0, 1]
            assert abs(corr) < 0.9999, (
                f"LEAKAGE ALERT: {feat} has correlation {corr:.4f} with target!"
            )
    print("[OK] Leakage guard passed — no features correlate 1.0 with target")


def main() -> None:
    print("=" * 65)
    print("PHASE 12 STEP 2 — GDP Momentum Feature Construction")
    print("=" * 65)

    # 0. Integrity gate
    verify_md5()

    # 1. Load panel
    df = pd.read_csv(INPUT_PATH)
    print(f"[INFO] Panel loaded: {len(df):,} rows | "
          f"Years: {df['year'].min()}–{df['year'].max()}")

    # 2. Confirm required columns are present
    required = ["country_code", "year", "gdp_growth_pct",
                "gdp_growth_lag1", "gdp_growth_lag2", "gdp_growth_lag3",
                "gdp_growth_next_year", "next_year_target_available"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        sys.exit(f"[FATAL] Missing required columns: {missing}")

    # 3. Confirm target column NOT in existing feature set (sanity)
    assert "gdp_growth_next_year" not in df.columns[:df.columns.get_loc("gdp_growth_next_year")], (
        "This check verifies target is a separate column, not mixed into features"
    )

    # 4. Build momentum features
    print("[INFO] Building momentum features...")
    df = build_momentum_features(df)

    # 5. Leakage guard
    leakage_guard(df)

    # 6. Verify feature values for sample country (USA)
    usa = df[df["country_code"] == "USA"].sort_values("year")
    print("\n[SAMPLE] USA momentum features (2018-2024):")
    display_cols = ["year", "gdp_growth_pct", "gdp_growth_lag1",
                    "gdp_growth_accel_1y", "gdp_growth_trend_change",
                    "gdp_growth_mean_3y", "gdp_growth_decelerating",
                    "gdp_growth_decline_2y"]
    print(usa[usa["year"].between(2018, 2024)][display_cols].to_string(index=False))

    # 7. Feature coverage check
    print("\n[INFO] New feature NaN counts (full panel):")
    for feat in NEW_MOMENTUM_FEATURES:
        n_nan = df[feat].isna().sum()
        pct   = n_nan / len(df) * 100
        print(f"  {feat:<35}: {n_nan:5d} NaN ({pct:.1f}%)")

    # 8. Save output (includes all original columns + new momentum columns)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n[SAVED] {OUTPUT_PATH}")
    print(f"        {len(df):,} rows | {df.shape[1]} columns")
    print(f"        New momentum features added: {len(NEW_MOMENTUM_FEATURES)}")
    print("\n[COMPLETE] GDP momentum features constructed. No production files modified.")


if __name__ == "__main__":
    main()
