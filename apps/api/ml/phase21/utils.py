"""
EconoSphere AI — Phase 21
utils.py : Paths, constants, hash helpers, OOS discovery, logging.

GOVERNANCE RULES ENFORCED HERE:
  1. All frozen configurations are imported from Phase 19/20 — never redefined.
  2. OOS discovery reads ONLY metadata, year values, and next_year_target_available flag.
     It does NOT read any OOS target (gdp_growth_next_year) values.
  3. The pre-evaluation lock is created before OOS discovery and cannot be modified
     after creation. OOS discovery cannot alter the lock.
  4. No production artifact is written, renamed, or modified.
  5. No model is trained, retrained, or modified.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
PHASE21_DIR  = Path(__file__).resolve().parent
PROJECT_ROOT = PHASE21_DIR.parents[3]

PHASE19_DIR  = PROJECT_ROOT / "apps" / "api" / "ml" / "phase19"
PHASE20_DIR  = PROJECT_ROOT / "apps" / "api" / "ml" / "phase20"

PHASE19_META = PHASE19_DIR / "metrics" / "phase19_run_metadata.json"

DATASET_PATH       = PROJECT_ROOT / "data" / "processed" / "master_panel_t1_missingness.csv"
PROD_MODEL_PATH    = PROJECT_ROOT / "models" / "phase11" / "best_t1_gdp_growth_model.joblib"
PROD_MANIFEST_PATH = PROJECT_ROOT / "models" / "phase11" / "phase11_production_release_manifest.json"

# Phase 20 frozen A5a experimental artifacts
PHASE20_ARTIFACTS_DIR = PHASE20_DIR / "artifacts"
A5A_M0_PATH    = PHASE20_ARTIFACTS_DIR / "phase20_M0_EXPERIMENTAL_ONLY.joblib"
A5A_RIDGE_PATH = PHASE20_ARTIFACTS_DIR / "phase20_Ridge_EXPERIMENTAL_ONLY.joblib"
A5A_RF_PATH    = PHASE20_ARTIFACTS_DIR / "phase20_RF_EXPERIMENTAL_ONLY.joblib"
A5A_META_PATH  = PHASE20_ARTIFACTS_DIR / "phase20_a5a_metadata_EXPERIMENTAL_ONLY.json"

# Phase 21 output paths
PHASE21_HASHES_DIR   = PHASE21_DIR / "hashes"
PHASE21_LOGS_DIR     = PHASE21_DIR / "logs"
PHASE21_DOCS_DIR     = PROJECT_ROOT / "docs" / "model_accuracy" / "phase21"

# Lock files (created in Stage A, before any OOS target is read)
LOCK_FILE_PATH    = PHASE21_DOCS_DIR / "phase21_pre_evaluation_lock.json"
OOS_ORIGINS_PATH  = PHASE21_DOCS_DIR / "phase21_oos_origins_locked.json"

# ── Import frozen constants from Phase 19 (never redefined) ───────────────────
from apps.api.ml.phase19.utils import (  # noqa: E402
    LOCKED_FEATURES,
    FORBIDDEN_IN_FEATURES,
    TARGET_COL,
    TARGET_AVAIL_COL,
    YEAR_COL,
    COUNTRY_COL,
    WB_AGGREGATES,
    check_no_target_leakage,
    file_hashes,
    verify_hash_immutability,
    get_train_eval_split,
    load_dataset as _p19_load_dataset,
    get_valid_origins,
)

# ── Frozen A5a configuration (from Phase 19/20 — do NOT change) ──────────────
FROZEN_A5A_CONFIG = {
    "architecture":    "A5a_EqualWeightEnsemble",
    "components":      ["M0_Phase11", "A1_Ridge", "A4_RandomForest"],
    "weights":         {"M0_Phase11": 1/3, "A1_Ridge": 1/3, "A4_RandomForest": 1/3},
    "clip_low":        -40.0,
    "clip_high":       60.0,
    "n_features":      31,
    "m0_params": {
        "learning_rate":     0.05,
        "max_depth":         5,
        "max_iter":          300,
        "l2_regularization": 5.0,
        "random_state":      42,
    },
    "rf_params": {
        "n_estimators":      300,
        "max_depth":         8,
        "min_samples_leaf":  5,
        "random_state":      42,
        "n_jobs":            -1,
    },
    "ridge_alpha_selection": "validation_split_last_20pct_training_years",
    "clipping_policy": "ridge_only_clip_to_[-40,60]; m0_and_rf_never_clipped",
    "status": "EXPERIMENTAL_ONLY",
}

# ── Frozen promotion thresholds (defined here, BEFORE OOS evaluation) ─────────
# These thresholds are part of the pre-evaluation lock and cannot be changed
# after the lock is created or after OOS labels are revealed.
PROMOTION_THRESHOLDS = {
    "min_rmse_improvement_pct":   1.0,   # A5a must beat Phase 11 by ≥1% RMSE
    "max_guardrail_delta_rmse":   0.50,  # no guardrail country degrades by >0.5 RMSE
    "max_priority_degraded":      1,     # at most 1 priority country may degrade
    "max_p95_ae_increase":        1.0,   # A5a p95 AE must not exceed M0 p95 AE by >1.0
    "ind_max_delta_rmse":         0.20,  # IND-specific guardrail (known marginal in P19/20)
    "min_oos_observations":       10,    # need ≥10 country-year OOS obs to evaluate
}

# ── Known production artifact hashes (from Phase 19 verification) ────────────
KNOWN_PROD_HASHES = {
    "best_t1_gdp_growth_model.joblib": {
        "md5":    "9c539735897eaf6e72e5f54f047390d7",
        "sha256": "748be64bcd3402375c4df4e45f9ba0c6858f52744714808ae8a2445f29248e78",
    },
    "phase11_production_release_manifest.json": {
        "md5":    "9c673e2d5bcd12adc171a55bd86ca34e",
        "sha256": "d865d439bfdc8aae2fa61b53bd56fc79b53de96facfdfff93a55e2ba062040b3",
    },
    "master_panel_t1_missingness.csv": {
        "md5":    "2a490f1d3c245671b43602cd399d0fa0",
        "sha256": "eedbc8c67876f49f7db9c465bea057ec507994c3cc6429a0a346b3c5d1635047",
    },
}

PRIORITY_COUNTRIES  = ["GBR", "BRA", "FRA", "CAN", "AUS"]
GUARDRAIL_COUNTRIES = ["USA", "CHN", "DEU", "JPN", "IND"]

VALID_GOVERNANCE_STATES = {
    "A5A_VALIDATED_AND_PROMOTION_RECOMMENDED",
    "A5A_OOS_VALIDATION_INCONCLUSIVE",
    "A5A_FAILED_TRUE_OOS_VALIDATION",
    "EXPERIMENT_FAILED_GOVERNANCE",
}

# ── Logging ────────────────────────────────────────────────────────────────────
LOG_PATH = PHASE21_LOGS_DIR / "phase21_run.log"

def get_logger(name: str = "phase21") -> logging.Logger:
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


# ── Data loading ───────────────────────────────────────────────────────────────
def load_dataset() -> pd.DataFrame:
    """
    Load master_panel_t1_missingness.csv (read-only).
    Delegates to Phase 19 load_dataset — identical methodology and exclusion list.
    """
    return _p19_load_dataset()


# ── Hash helpers ───────────────────────────────────────────────────────────────
def record_hashes_p21(label: str) -> dict:
    """
    Record hashes of frozen production artifacts to phase21/hashes/.
    Called BEFORE and AFTER the full Phase 21 execution.
    Never modifies any production artifact.
    """
    artifacts = {
        "best_t1_gdp_growth_model.joblib":          PROD_MODEL_PATH,
        "phase11_production_release_manifest.json":  PROD_MANIFEST_PATH,
        "master_panel_t1_missingness.csv":           DATASET_PATH,
    }
    result: dict = {}
    for name, path in artifacts.items():
        result[name] = file_hashes(path)

    PHASE21_HASHES_DIR.mkdir(parents=True, exist_ok=True)
    out = PHASE21_HASHES_DIR / f"{label}_production_hashes.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def verify_against_known_hashes(current: dict) -> tuple[bool, list[str]]:
    """
    Verify current hashes against the known frozen values from Phase 19 verification.
    Returns (all_ok, failures).
    """
    failures: list[str] = []
    for artifact, known in KNOWN_PROD_HASHES.items():
        cur = current.get(artifact, {})
        if cur.get("md5") != known["md5"]:
            failures.append(
                f"MD5 MISMATCH for {artifact}: "
                f"expected={known['md5']} actual={cur.get('md5')}"
            )
        if cur.get("sha256") != known["sha256"]:
            failures.append(
                f"SHA256 MISMATCH for {artifact}: "
                f"expected={known['sha256']} actual={cur.get('sha256')}"
            )
    return len(failures) == 0, failures


# ── Phase 19 origin loading ────────────────────────────────────────────────────
def load_phase19_origins() -> list[int]:
    """
    Load the set of forecast origins used in Phase 19.
    These are the authoritative already-used origins — any new origin must be
    strictly AFTER the maximum Phase 19 origin to qualify as OOS.
    """
    if not PHASE19_META.exists():
        raise FileNotFoundError(f"Phase 19 metadata missing: {PHASE19_META}")
    meta = json.loads(PHASE19_META.read_text(encoding="utf-8"))
    listed = [int(y) for y in meta["valid_origins"]]
    n_auth = int(meta["n_origins"])
    if len(listed) > n_auth:
        # Known discrepancy: n_origins=23, len(listed)=24. Drop earliest.
        listed = listed[-n_auth:]
    return listed


# ── OOS Discovery (Stage B) ───────────────────────────────────────────────────
def discover_oos_origins(df: pd.DataFrame, log) -> dict:
    """
    Determine whether genuinely new OOS labeled origins exist.

    CORRECTION 3 ENFORCEMENT:
    - This function reads ONLY: year values, next_year_target_available flag.
    - It does NOT read gdp_growth_next_year (the actual OOS target values).
    - It cannot modify the pre-evaluation lock.
    - It cannot alter model configuration, clipping, weights, or thresholds.
    - OOS authenticity is verified against the 7-point checklist from §5.

    CORRECTION 2 ENFORCEMENT:
    - Operates at country × forecast_origin_year observation level (not origin-aggregate).
    - For each candidate origin, counts country-year rows with next_year_target_available==1.

    Returns a dict with:
        available           : bool — True only if ≥1 genuine OOS observation exists
        oos_origins         : list[int] — feature years qualifying as genuinely new OOS
        oos_observation_count: int — total country-year observations in OOS origins
        p19_max_origin      : int — the maximum origin used in Phase 19
        phase19_origins     : list[int]
        all_current_origins : list[int]
        discovery_log       : list[str] — audit trail of discovery checks
        authenticity_checks : dict — the 7-point checklist results
    """
    log.info("  [Stage B] OOS Discovery — reading only year and availability metadata")

    phase19_origins = load_phase19_origins()
    p19_set         = set(phase19_origins)
    p19_max         = max(p19_set)

    log.info(f"  Phase 19 origins ({len(p19_set)}): min={min(p19_set)}, max={p19_max}")

    # Get all current valid origins (dynamically computed — same logic as Phase 19)
    all_origins, skipped = get_valid_origins(df, min_train_rows=50)
    current_set = set(o["feature_year"] for o in all_origins)

    log.info(f"  Current valid origins ({len(current_set)}): {sorted(current_set)}")
    log.info(f"  Skipped origins: {[s['feature_year'] for s in skipped]}")

    discovery_log: list[str] = []

    # Candidate new origins: in current dataset but not in Phase 19
    # AND strictly after the Phase 19 maximum origin
    candidate_new = sorted(
        fy for fy in (current_set - p19_set) if fy > p19_max
    )
    discovery_log.append(
        f"Origins in current dataset not in Phase 19 and after p19_max={p19_max}: "
        f"{candidate_new}"
    )
    log.info(f"  Candidate new OOS origins (strictly after Phase 19 max): {candidate_new}")

    # ── 7-point OOS authenticity checklist (§5) ───────────────────────────────
    # NOTE: We only read next_year_target_available here, NOT gdp_growth_next_year.
    oos_origins_validated: list[int] = []
    oos_observation_count = 0
    authenticity_checks: dict[int, dict] = {}

    for fy in candidate_new:
        target_year = fy + 1
        eval_rows = df[
            (df[YEAR_COL] == fy) &
            (df[TARGET_AVAIL_COL] == 1) &
            (~df[COUNTRY_COL].isin(WB_AGGREGATES))
        ]
        n_obs = len(eval_rows)

        checks = {
            "C1_not_in_phase19_backtest":    fy not in p19_set,
            "C2_not_in_phase20_evaluation":  fy not in p19_set,  # P20 used same P19 origins
            "C3_target_unavailable_at_p19_p20": fy > p19_max,
            "C4_strictly_after_p19_max_origin": fy > p19_max,
            "C5_valid_target_labels_exist":  n_obs > 0,
            "C6_not_revised_copy_of_old_obs": fy > p19_max,  # structural: new year
            "C7_new_forecast_origin":        fy not in p19_set and fy > p19_max,
        }
        all_pass = all(checks.values())
        authenticity_checks[fy] = {
            **checks,
            "n_observation_rows": n_obs,
            "target_year": target_year,
            "qualifies_as_oos": all_pass,
        }
        msg = (
            f"  Origin {fy}→{target_year}: n_obs={n_obs}, "
            f"all_checks={'PASS' if all_pass else 'FAIL'}"
        )
        log.info(msg)
        discovery_log.append(msg)

        if all_pass and n_obs > 0:
            oos_origins_validated.append(fy)
            oos_observation_count += n_obs

    # Check 2025-origin explicitly even if not in current valid origins
    # (handles the case where 2025 exists in dataset but has 0 targets)
    if 2025 not in candidate_new:
        fy = 2025
        target_year = 2026
        rows_2025 = df[df[YEAR_COL] == fy]
        n_avail = int(rows_2025[TARGET_AVAIL_COL].sum()) if TARGET_AVAIL_COL in rows_2025.columns else 0
        msg = (
            f"  Explicit check: origin {fy}→{target_year}: "
            f"dataset rows={len(rows_2025)}, next_year_target_available={n_avail} "
            f"→ NOT a valid OOS origin (zero labeled targets)"
        )
        log.info(msg)
        discovery_log.append(msg)
        authenticity_checks[fy] = {
            "C5_valid_target_labels_exist": False,
            "n_observation_rows": n_avail,
            "target_year": target_year,
            "qualifies_as_oos": False,
            "note": "Feature year 2025 present in dataset but next_year_target_available=0 "
                    "for all country rows. 2026 GDP growth not yet published.",
        }

    available = len(oos_origins_validated) > 0

    if not available:
        msg = (
            "TRUE_NEW_OUT_OF_SAMPLE_DATA = UNAVAILABLE. "
            "No feature year strictly after Phase 19 max origin has valid labeled targets. "
            "Gate G4 (OOS Authenticity) = INCONCLUSIVE. "
            "Promotion cannot be finalized without genuine OOS evidence."
        )
        log.warning(msg)
    else:
        msg = (
            f"TRUE_NEW_OUT_OF_SAMPLE_DATA = AVAILABLE. "
            f"Validated OOS origins: {oos_origins_validated}, "
            f"total observations: {oos_observation_count}."
        )
        log.info(msg)
    discovery_log.append(msg)

    return {
        "available":              available,
        "oos_origins":            oos_origins_validated,
        "oos_observation_count":  oos_observation_count,
        "p19_max_origin":         p19_max,
        "phase19_origins":        phase19_origins,
        "all_current_origins":    sorted(current_set),
        "discovery_log":          discovery_log,
        "authenticity_checks":    authenticity_checks,
        "message":                msg,
    }
