"""
EconoSphere AI — Phase 21
lock.py : Pre-evaluation lock creation and OOS origin freezing.

STAGE A — PRE-OOS LOCK:
  The lock captures the complete frozen state BEFORE any OOS target value
  is read. The lock:
    - Contains model configuration, feature list, ensemble weights, clipping,
      evaluation metrics, promotion thresholds, and evaluation rules.
    - Does NOT contain any metric derived from new OOS labels.
    - Is hashed after creation. Any modification invalidates the hash.
    - Cannot be altered by OOS discovery (Stage B).

STAGE C — OOS ORIGIN FREEZE:
  Once OOS discovery determines which origins are eligible, their identities
  are frozen in phase21_oos_origins_locked.json before any evaluation occurs.
  The origin list cannot be modified after Stage D begins.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from .utils import (
    A5A_META_PATH,
    A5A_M0_PATH,
    A5A_RF_PATH,
    A5A_RIDGE_PATH,
    FROZEN_A5A_CONFIG,
    KNOWN_PROD_HASHES,
    LOCK_FILE_PATH,
    LOCKED_FEATURES,
    OOS_ORIGINS_PATH,
    PHASE21_DOCS_DIR,
    PROD_MANIFEST_PATH,
    PROD_MODEL_PATH,
    PROMOTION_THRESHOLDS,
    file_hashes,
    get_logger,
    record_hashes_p21,
)

log = get_logger()


def _hash_dict(d: dict) -> str:
    """SHA256 of a canonical JSON serialisation of a dict."""
    canonical = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _git_commit() -> str:
    """Return current git commit hash, or 'UNAVAILABLE' if not in a git repo."""
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "UNAVAILABLE"


def create_pre_evaluation_lock() -> dict:
    """
    Stage A — PRE-OOS LOCK.

    Creates phase21_pre_evaluation_lock.json containing the complete frozen
    state of the experiment BEFORE any OOS target value is read.

    CORRECTION 3: This lock does NOT contain any OOS-derived metric.
    OOS discovery (Stage B) reads availability metadata only and cannot
    modify this lock after it is created and hashed.

    Returns the lock dict (also written to disk).
    """
    log.info("  [Stage A] Creating pre-evaluation lock...")

    PHASE21_DOCS_DIR.mkdir(parents=True, exist_ok=True)

    # Hash current production artifacts
    prod_hashes = {
        "best_t1_gdp_growth_model.joblib":         file_hashes(PROD_MODEL_PATH),
        "phase11_production_release_manifest.json": file_hashes(PROD_MANIFEST_PATH),
    }

    # Hash experimental A5a artifacts
    experimental_hashes = {
        "phase20_M0_EXPERIMENTAL_ONLY.joblib":    file_hashes(A5A_M0_PATH),
        "phase20_Ridge_EXPERIMENTAL_ONLY.joblib": file_hashes(A5A_RIDGE_PATH),
        "phase20_RF_EXPERIMENTAL_ONLY.joblib":    file_hashes(A5A_RF_PATH),
    }

    # Load manifest for manifest-driven feature list verification
    manifest = json.loads(PROD_MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest_features = manifest.get("feature_names", [])

    lock = {
        "phase": 21,
        "lock_purpose": "PRE_OOS_EVALUATION_LOCK",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),

        # ── Model configuration (frozen, do not change) ───────────────────
        "production_model": {
            "name": "Phase 11 M0",
            "artifact": "models/phase11/best_t1_gdp_growth_model.joblib",
            "manifest": "models/phase11/phase11_production_release_manifest.json",
            "hashes": prod_hashes,
        },
        "candidate_model": {
            "name": "Phase 20 A5a",
            "architecture": "Equal-weight ensemble: (M0 + Ridge + RandomForest) / 3",
            "artifacts": {
                k: str(v) for k, v in {
                    "M0":    A5A_M0_PATH,
                    "Ridge": A5A_RIDGE_PATH,
                    "RF":    A5A_RF_PATH,
                }.items()
            },
            "artifact_hashes": experimental_hashes,
            "config": FROZEN_A5A_CONFIG,
        },

        # ── Feature set (frozen from Phase 19) ────────────────────────────
        "feature_list": LOCKED_FEATURES,
        "feature_count": len(LOCKED_FEATURES),
        "manifest_feature_count": len(manifest_features),
        "feature_lists_match": sorted(LOCKED_FEATURES) == sorted(manifest_features),

        # ── Preprocessing (frozen from Phase 19) ──────────────────────────
        "preprocessing": {
            "imputer": "SimpleImputer(strategy='median')",
            "scaler_for_ridge": "StandardScaler() (inside Ridge pipeline)",
            "m0_no_scaler": True,
            "rf_no_scaler": True,
        },

        # ── Ensemble weights (frozen) ─────────────────────────────────────
        "ensemble_weights": FROZEN_A5A_CONFIG["weights"],
        "ensemble_formula": "A5a = (1/3)*M0 + (1/3)*Ridge + (1/3)*RandomForest",

        # ── Clipping (frozen from Phase 19) ───────────────────────────────
        "clipping": {
            "ridge_clip_low":  FROZEN_A5A_CONFIG["clip_low"],
            "ridge_clip_high": FROZEN_A5A_CONFIG["clip_high"],
            "m0_clipped": False,
            "rf_clipped":  False,
            "policy": FROZEN_A5A_CONFIG["clipping_policy"],
        },

        # ── OOS origin selection rule ─────────────────────────────────────
        "oos_origin_selection_rule": (
            "A forecast origin qualifies as genuinely new OOS if and only if: "
            "(1) it was not used in Phase 19 backtest, "
            "(2) it was not evaluated in Phase 20, "
            "(3) its target was unavailable during Phase 19/20 architecture selection, "
            "(4) it occurs strictly after the maximum Phase 19 evaluation origin, "
            "(5) it has valid next_year_target_available==1 rows (country-year level), "
            "(6) it is not a revised copy of an already-used historical target, "
            "(7) it represents a new forecast origin, not a re-downloaded old observation."
        ),

        # ── Evaluation metrics (frozen before any OOS label is read) ──────
        "evaluation_metrics": [
            "RMSE", "MAE", "median_AE", "p90_AE", "p95_AE", "max_AE",
            "mean_error (bias)", "per_country", "per_origin"
        ],

        # ── Promotion thresholds (frozen before OOS evaluation) ───────────
        # CORRECTION 3: These are set here and CANNOT be changed after seeing OOS results.
        "promotion_thresholds": PROMOTION_THRESHOLDS,

        # ── Statistical test (frozen) ─────────────────────────────────────
        "statistical_test": "Paired Wilcoxon signed-rank test on absolute error differences",
        "null_hypothesis": "Median of (|error_M0| - |error_A5a|) = 0",
        "alternative_hypothesis": "A5a absolute errors are smaller than M0",

        # ── Training methodology (frozen) ─────────────────────────────────
        "training_methodology": {
            "design": "T+1 expanding-window chronological backtest",
            "train_split": "year < feature_year, next_year_target_available==1",
            "eval_split":  "year == feature_year, next_year_target_available==1",
            "ridge_alpha_selection": "validation split (last 20% of training years only)",
            "no_random_splits": True,
            "no_test_set_tuning": True,
            "min_train_rows": 50,
        },

        # ── Governance rules applied ──────────────────────────────────────
        "governance": {
            "no_model_development": True,
            "no_production_mutation": True,
            "no_fabricated_data": True,
            "no_post_hoc_tuning": True,
            "no_automatic_promotion": True,
            "lock_cannot_be_modified_after_creation": True,
            "oos_discovery_cannot_alter_lock": True,
        },

        # ── Software environment ──────────────────────────────────────────
        "environment": {
            "python_version": sys.version,
            "platform": platform.platform(),
        },
    }

    # Compute hash of the lock itself
    lock["lock_hash_sha256"] = _hash_dict(
        {k: v for k, v in lock.items() if k != "lock_hash_sha256"}
    )

    LOCK_FILE_PATH.write_text(json.dumps(lock, indent=2, default=str), encoding="utf-8")
    log.info(f"  Pre-evaluation lock written: {LOCK_FILE_PATH}")
    log.info(f"  Lock SHA256: {lock['lock_hash_sha256'][:16]}...")
    log.info("  [Stage A] Lock complete. OOS labels have NOT been read.")
    return lock


def freeze_oos_origins(discovery_result: dict) -> dict:
    """
    Stage C — FREEZE OOS ORIGINS.

    After OOS discovery determines which origins are eligible, freeze them
    before any evaluation begins. Once this file is written, the origin
    list cannot be modified.

    If no OOS origins are available, writes an explicit record of that fact.

    CORRECTION 4: Does NOT fabricate any metric. If unavailable, the frozen
    record clearly states OOS_EVALUATION_EXECUTED=FALSE.

    Returns the frozen origins dict (also written to disk).
    """
    log.info("  [Stage C] Freezing OOS origins...")

    PHASE21_DOCS_DIR.mkdir(parents=True, exist_ok=True)

    available = discovery_result["available"]

    frozen = {
        "phase": 21,
        "stage": "C_FREEZE_OOS_ORIGINS",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "oos_available": available,
        "oos_origins": discovery_result["oos_origins"],
        "oos_observation_count": discovery_result["oos_observation_count"],
        "p19_max_origin": discovery_result["p19_max_origin"],
        "phase19_origins": discovery_result["phase19_origins"],
        "all_current_origins": discovery_result["all_current_origins"],
        "authenticity_checks": discovery_result["authenticity_checks"],
        "discovery_log": discovery_result["discovery_log"],

        # Explicit flags — CORRECTION 4
        "OOS_EVALUATION_EXECUTED":    available,
        "OOS_DATA_AVAILABLE":         available,
        "OOS_PREDICTIONS_GENERATED":  available,

        "governance_note": (
            "OOS_EVALUATION_EXECUTED=FALSE: No genuinely new labeled OOS data available. "
            "This record explicitly marks the absence. An empty prediction CSV should NOT "
            "be interpreted as a successful zero-prediction run."
        ) if not available else (
            "OOS origins are locked. Evaluation proceeds with these origins only. "
            "No origin may be added or removed after this file is written."
        ),
    }

    # Hash the frozen origins
    frozen["frozen_hash_sha256"] = _hash_dict(
        {k: v for k, v in frozen.items() if k != "frozen_hash_sha256"}
    )

    OOS_ORIGINS_PATH.write_text(json.dumps(frozen, indent=2, default=str), encoding="utf-8")
    log.info(f"  OOS origins frozen: {OOS_ORIGINS_PATH}")
    log.info(f"  OOS available: {available}, origins: {discovery_result['oos_origins']}")
    return frozen
