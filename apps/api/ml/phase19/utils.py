"""
EconoSphere AI — Phase 19
utils.py : Data loading, hash verification, governance helpers.

GOVERNANCE: This module never writes to or reads from production model artifacts.
It only reads the production dataset (read-only) and hashes production artifacts
to verify they have not been mutated.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
PHASE19_DIR  = Path(__file__).resolve().parent
PROJECT_ROOT = PHASE19_DIR.parents[3]

DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "master_panel_t1_missingness.csv"
RAW_PATH     = PROJECT_ROOT / "data" / "raw" / "master_panel.csv"

PROD_MODEL_PATH    = PROJECT_ROOT / "models" / "phase11" / "best_t1_gdp_growth_model.joblib"
PROD_MANIFEST_PATH = PROJECT_ROOT / "models" / "phase11" / "phase11_production_release_manifest.json"
PROD_DATASET_PATH  = DATASET_PATH   # same file, checked separately

PHASE19_HASHES_DIR = PHASE19_DIR / "hashes"

# ── World Bank aggregate codes (same exclusion list as Phase 11/12) ────────────
WB_AGGREGATES = {
    "AFE","AFW","ARB","CEB","CSS","EAP","EAR","EAS","ECA","ECS","EMU","EUU",
    "FCS","HIC","HPC","IBD","IBT","IDA","IDB","IDX","INX","LAC","LCN","LDC",
    "LIC","LMC","LMY","LTE","MEA","MIC","MNA","NAC","OED","OSS","PRE","PSS",
    "PST","SAS","SSA","SSF","SST","TEA","TEC","TLA","TMN","TSA","TSS","UMC","WLD",
}

# ── Phase 11 locked feature set (31 features) ─────────────────────────────────
BASE_FEATURES = [
    "exchange_rate_lcu_usd", "tariff_rate_pct", "remittances_usd",
    "fdi_net_inflow_usd", "unemployment_pct", "imports_pct_gdp",
    "tax_revenue_pct_gdp", "exports_pct_gdp", "interest_rate_pct",
    "reserves_usd", "current_account_pct_gdp", "inflation_cpi_pct",
    "population_total", "gdp_current_usd", "gdp_growth_lag1",
    "gdp_growth_lag2", "gdp_growth_lag3", "inflation_lag1",
    "unemployment_lag1", "exports_lag1", "imports_lag1",
    "gdp_growth_rolling_mean_3", "gdp_growth_rolling_std_3",
    "gdp_growth_rolling_mean_5", "inflation_rolling_mean_3",
    "trade_openness", "trade_balance_ratio", "log_gdp_usd",
    "log_population",
]
ROLLING_EXTRA  = ["gdp_growth_rolling_std_5", "inflation_rolling_std_3"]
LOCKED_FEATURES = BASE_FEATURES + ROLLING_EXTRA  # 31 total

TARGET_COL      = "gdp_growth_next_year"
TARGET_AVAIL_COL = "next_year_target_available"
YEAR_COL        = "year"
COUNTRY_COL     = "country_code"

FORBIDDEN_IN_FEATURES = [
    "gdp_growth_next_year",
    "next_year_target_available",
    "gdp_growth_pct",
    "growth_regime",
    "total_missing_feature_ratio",
    "total_missing_feature_count",
]

# ── Logging ────────────────────────────────────────────────────────────────────
LOG_PATH = PHASE19_DIR / "logs" / "phase19_run.log"

def get_logger(name: str = "phase19") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s"))
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(levelname)-7s  %(message)s"))
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger


# ── Hash helpers ───────────────────────────────────────────────────────────────
def file_hashes(path: Path) -> dict:
    """Return MD5, SHA256, and file size for a given path."""
    if not path.exists():
        return {"md5": "MISSING", "sha256": "MISSING", "size_bytes": 0}
    data = path.read_bytes()
    return {
        "md5":        hashlib.md5(data).hexdigest(),
        "sha256":     hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }


def record_hashes(label: str) -> dict:
    """
    Compute and return hashes of the three frozen production artifacts.
    Also write them to the phase19/hashes/ directory with the given label
    ('before' or 'after').
    """
    artifacts = {
        "best_t1_gdp_growth_model.joblib":         PROD_MODEL_PATH,
        "phase11_production_release_manifest.json": PROD_MANIFEST_PATH,
        "master_panel_t1_missingness.csv":          PROD_DATASET_PATH,
    }
    result: dict = {}
    for name, path in artifacts.items():
        result[name] = file_hashes(path)

    PHASE19_HASHES_DIR.mkdir(parents=True, exist_ok=True)
    out = PHASE19_HASHES_DIR / f"{label}_production_hashes.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def verify_hash_immutability(before: dict, after: dict) -> tuple[bool, list[str]]:
    """
    Compare before/after hash dictionaries.
    Returns (all_ok, list_of_failures).
    """
    failures: list[str] = []
    for artifact, b_info in before.items():
        a_info = after.get(artifact, {})
        if b_info.get("md5") != a_info.get("md5"):
            failures.append(
                f"MD5 MISMATCH for {artifact}: "
                f"before={b_info.get('md5')} after={a_info.get('md5')}"
            )
        if b_info.get("sha256") != a_info.get("sha256"):
            failures.append(
                f"SHA256 MISMATCH for {artifact}: "
                f"before={b_info.get('sha256')} after={a_info.get('sha256')}"
            )
    return len(failures) == 0, failures


# ── Data loading ───────────────────────────────────────────────────────────────
def load_dataset() -> pd.DataFrame:
    """
    Load master_panel_t1_missingness.csv, exclude World Bank aggregate rows,
    and return a clean DataFrame.  Read-only — never modifies the file.
    """
    df = pd.read_csv(DATASET_PATH, low_memory=False)
    df = df[~df[COUNTRY_COL].isin(WB_AGGREGATES)].copy()
    return df


def get_valid_origins(df: pd.DataFrame, min_train_rows: int = 50) -> list[dict]:
    """
    Dynamically determine valid forecast origins from the dataset.

    A forecast origin is valid when:
      1.  Feature year Y has rows where next_year_target_available == 1
          (meaning gdp_growth_next_year for Y+1 is real, not NaN).
      2.  There are at least min_train_rows training observations
          (rows from years strictly before Y with valid targets).

    Returns a list of dicts, one per valid origin, each containing:
        feature_year  : Y
        target_year   : Y + 1
        n_eval        : number of evaluation rows at this origin
        n_train       : number of training rows available
    """
    all_years = sorted(df[YEAR_COL].unique())
    results = []
    skipped = []

    for year in all_years:
        target_year = year + 1

        # Evaluation rows: feature_year == year AND target available
        eval_mask = (df[YEAR_COL] == year) & (df[TARGET_AVAIL_COL] == 1)
        n_eval = int(eval_mask.sum())
        if n_eval == 0:
            skipped.append({
                "feature_year": year,
                "target_year":  target_year,
                "reason":       "no evaluation rows with valid targets",
            })
            continue

        # Verify the target column is not NaN for those rows (extra guard)
        eval_rows = df[eval_mask]
        if eval_rows[TARGET_COL].isna().all():
            skipped.append({
                "feature_year": year,
                "target_year":  target_year,
                "reason":       "all gdp_growth_next_year values are NaN",
            })
            continue

        # Training rows: strictly before feature_year, with valid targets
        train_mask = (df[YEAR_COL] < year) & (df[TARGET_AVAIL_COL] == 1)
        n_train = int(train_mask.sum())
        if n_train < min_train_rows:
            skipped.append({
                "feature_year": year,
                "target_year":  target_year,
                "reason":       f"insufficient training rows ({n_train} < {min_train_rows})",
            })
            continue

        results.append({
            "feature_year": year,
            "target_year":  target_year,
            "n_eval":       n_eval,
            "n_train":      n_train,
        })

    return results, skipped


def get_train_eval_split(
    df: pd.DataFrame,
    feature_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (train_df, eval_df) for a given feature_year (forecast origin).

    TRAIN : rows where year < feature_year  AND  next_year_target_available == 1
    EVAL  : rows where year == feature_year AND  next_year_target_available == 1

    This strictly enforces: no future information leaks into training.
    feature_year row itself is the forecast row; its target (Y+1) is the
    unknown future in a live setting, though we have it for back-testing.
    """
    train_mask = (df[YEAR_COL] < feature_year) & (df[TARGET_AVAIL_COL] == 1)
    eval_mask  = (df[YEAR_COL] == feature_year) & (df[TARGET_AVAIL_COL] == 1)
    return df[train_mask].copy(), df[eval_mask].copy()


def check_no_target_leakage(X: pd.DataFrame) -> list[str]:
    """
    Check that none of the forbidden target-related columns appear in X.
    Returns a list of violations (empty means clean).
    """
    violations = [c for c in FORBIDDEN_IN_FEATURES if c in X.columns]
    return violations


def check_feature_years_not_in_future(
    X: pd.DataFrame,
    feature_year: int,
) -> bool:
    """
    For the eval rows, verify that all features come from year <= feature_year.
    Returns True if no future data is detected.
    (Structural check — the splitting logic enforces this; this is an audit.)
    """
    if YEAR_COL in X.columns:
        return bool((X[YEAR_COL] <= feature_year).all())
    return True  # year column was already dropped from features
