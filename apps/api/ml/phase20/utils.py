"""
EconoSphere AI — Phase 20
utils.py : Paths, constants, Phase 19 origin verification, OOS discovery.

CORRECTION 1 (User-approved): Phase 19 origins are loaded dynamically from
    the Phase 19 metadata JSON and verified against a freshly computed origin
    list — never hard-coded.

CORRECTION 2 (User-approved): Phase 19 RMSE constants are REFERENCE ONLY.
    The audit re-runs the methodology and verifies within tolerance.

CORRECTION 4 (User-approved): OOS observations are identified early and
    frozen; never peeked before all design decisions are finalized.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
PHASE20_DIR  = Path(__file__).resolve().parent
PROJECT_ROOT = PHASE20_DIR.parents[3]

PHASE19_DIR  = PROJECT_ROOT / "apps" / "api" / "ml" / "phase19"
PHASE19_META = PHASE19_DIR / "metrics" / "phase19_run_metadata.json"
PHASE19_CSV  = PHASE19_DIR / "metrics" / "phase19_master_metrics.csv"

DATASET_PATH       = PROJECT_ROOT / "data" / "processed" / "master_panel_t1_missingness.csv"
PROD_MODEL_PATH    = PROJECT_ROOT / "models" / "phase11" / "best_t1_gdp_growth_model.joblib"
PROD_MANIFEST_PATH = PROJECT_ROOT / "models" / "phase11" / "phase11_production_release_manifest.json"

PHASE20_HASHES_DIR = PHASE20_DIR / "hashes"
PHASE20_LOGS_DIR   = PHASE20_DIR / "logs"

# Output paths
METRICS_DIR     = PHASE20_DIR / "metrics"
PREDICTIONS_DIR = PHASE20_DIR / "predictions"
ARTIFACTS_DIR   = PHASE20_DIR / "artifacts"

# ── Reference constants from Phase 19 (REPORTING USE ONLY — not pass/fail) ──
# Per Correction 2: these are NOT used as promotion criteria.
# The audit re-runs and reproduces, then compares against these as a sanity check.
PHASE19_REF_M0_RMSE  = 5.4883   # Phase 19 reported value
PHASE19_REF_A5A_RMSE = 5.3175   # Phase 19 reported value
PHASE19_REF_DELTA    = -0.1708  # Phase 19 reported A5a improvement
PHASE19_REF_WINS     = 18       # Phase 19 A5a origin wins
PHASE19_REF_N_ORIGINS = 23      # Phase 19 n_origins (from n_origins field, NOT len(list))

# Reproduction tolerance (Correction 2): reproduced value must be within this
# tolerance of the Phase 19 reference. If exceeded, REPRODUCTION_MISMATCH is flagged.
REPRODUCTION_TOLERANCE = 0.05   # ±0.05 RMSE absolute

# ── World Bank aggregate exclusion (same as Phase 19) ────────────────────────
WB_AGGREGATES = {
    "AFE","AFW","ARB","CEB","CSS","EAP","EAR","EAS","ECA","ECS","EMU","EUU",
    "FCS","HIC","HPC","IBD","IBT","IDA","IDB","IDX","INX","LAC","LCN","LDC",
    "LIC","LMC","LMY","LTE","MEA","MIC","MNA","NAC","OED","OSS","PRE","PSS",
    "PST","SAS","SSA","SSF","SST","TEA","TEC","TLA","TMN","TSA","TSS","UMC","WLD",
}

# ── Import locked constants from Phase 19 (never redefined here) ─────────────
from apps.api.ml.phase19.utils import (
    LOCKED_FEATURES,
    FORBIDDEN_IN_FEATURES,
    TARGET_COL,
    TARGET_AVAIL_COL,
    YEAR_COL,
    COUNTRY_COL,
    check_no_target_leakage,
    check_feature_years_not_in_future,
    file_hashes,
    verify_hash_immutability,
    get_train_eval_split,
    load_dataset as _p19_load_dataset,
)

# Re-export for Phase 20 callers
def load_dataset() -> pd.DataFrame:
    """Delegate to Phase 19 load_dataset — identical methodology."""
    return _p19_load_dataset()


# ── Logging ────────────────────────────────────────────────────────────────────
LOG_PATH = PHASE20_LOGS_DIR / "phase20_run.log"

def get_logger(name: str = "phase20") -> logging.Logger:
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


# ── Phase 19 Origin Verification (Correction 1) ───────────────────────────────

def load_phase19_metadata() -> dict:
    """Load Phase 19 run metadata JSON."""
    if not PHASE19_META.exists():
        raise FileNotFoundError(f"Phase 19 metadata missing: {PHASE19_META}")
    return json.loads(PHASE19_META.read_text(encoding="utf-8"))


def get_phase19_origins_from_metadata() -> list[int]:
    """
    Load the Phase 19 valid origins from the metadata JSON.
    Uses the 'valid_origins' field but cross-checks against 'n_origins'.

    IMPORTANT: The Phase 19 metadata has a known discrepancy where
    len(valid_origins) = 24 but n_origins = 23. This function resolves it by
    returning the authoritative set determined by n_origins, dropping the
    earliest listed origin if len > n_origins.
    """
    meta = load_phase19_metadata()
    listed  = [int(y) for y in meta["valid_origins"]]
    n_auth  = int(meta["n_origins"])

    if len(listed) == n_auth:
        return listed

    # Discrepancy: n_origins=23, len(listed)=24.
    # Per the Phase 19 logs, the first successfully evaluated origin was 2002.
    # The extra entry in the metadata list is 2001 which was skipped in practice.
    # Resolution: keep only the last n_auth entries (i.e. drop the earliest).
    resolved = listed[-n_auth:]
    return resolved


def get_valid_origins_p20(df: pd.DataFrame, min_train_rows: int = 50) -> tuple[list[dict], list[dict]]:
    """
    Dynamically determine valid forecast origins — identical logic to Phase 19.
    Imported here rather than rewriting.
    """
    from apps.api.ml.phase19.utils import get_valid_origins
    return get_valid_origins(df, min_train_rows=min_train_rows)


def verify_origin_sets(
    actual_origins: list[dict],
    log,
) -> dict:
    """
    Compare the dynamically computed origins against the Phase 19 metadata.

    Returns:
        {
          "match": bool,
          "phase19_set": set[int],
          "phase20_set": set[int],
          "in_p20_not_p19": list[int],  # new origins (potential OOS)
          "in_p19_not_p20": list[int],  # origins that disappeared
          "phase19_origins": list[int],
        }
    """
    p19_origins = get_phase19_origins_from_metadata()
    p19_set = set(p19_origins)
    p20_set = set(int(o["feature_year"]) for o in actual_origins)

    in_p20_not_p19 = sorted(p20_set - p19_set)
    in_p19_not_p20 = sorted(p19_set - p20_set)
    match = (p19_set == p20_set)

    log.info(f"  Phase 19 origins ({len(p19_set)}): {sorted(p19_set)}")
    log.info(f"  Phase 20 origins ({len(p20_set)}): {sorted(p20_set)}")
    if match:
        log.info("  ✓ Origin sets match exactly.")
    else:
        log.warning(f"  Origin set mismatch!")
        log.warning(f"    In P20 not P19 (potential new OOS): {in_p20_not_p19}")
        log.warning(f"    In P19 not P20 (disappeared):       {in_p19_not_p20}")

    return {
        "match":           match,
        "phase19_set":     p19_set,
        "phase20_set":     p20_set,
        "in_p20_not_p19":  in_p20_not_p19,
        "in_p19_not_p20":  in_p19_not_p20,
        "phase19_origins": p19_origins,
    }


# ── Out-of-Sample Discovery (Correction 4) ───────────────────────────────────

def identify_oos_origins(origin_comparison: dict) -> dict:
    """
    Identify genuinely new out-of-sample origins — those in Phase 20 but
    not in Phase 19.

    CORRECTION 4 ENFORCEMENT:
    - OOS origins are identified here and returned as a frozen list.
    - They must NOT be evaluated until all design decisions are finalized.
    - The calling orchestrator must pass OOS origins ONLY to the final
      evaluation function, after all clipping/ablation/country analysis
      has been completed on Phase 19 origins.

    Returns:
        {
          "available": bool,
          "oos_origins": list[int],   # frozen — do not evaluate early
          "message": str,
        }
    """
    new_origins = origin_comparison["in_p20_not_p19"]
    p19_max = max(origin_comparison["phase19_set"]) if origin_comparison["phase19_set"] else 0
    true_oos = [o for o in new_origins if o > p19_max]

    if not true_oos:
        return {
            "available":    False,
            "oos_origins":  [],
            "message":      "TRUE_NEW_OUT_OF_SAMPLE_DATA = UNAVAILABLE. "
                            "All Phase 20 origins were already used in Phase 19 (or skipped due to insufficient data). "
                            "Gate 11 = INCONCLUSIVE. Promotion cannot be final.",
        }
    return {
        "available":    True,
        "oos_origins":  true_oos,
        "message":      f"TRUE_NEW_OUT_OF_SAMPLE_DATA = AVAILABLE. "
                        f"New labeled origins: {true_oos}. "
                        f"These will be evaluated as a final exam AFTER all design "
                        f"decisions are frozen.",
    }


# ── Hash helpers for Phase 20 ─────────────────────────────────────────────────

def record_hashes_p20(label: str) -> dict:
    """
    Record hashes of frozen production artifacts to phase20/hashes/.
    Identical logic to Phase 19 record_hashes(), but writes to PHASE20_HASHES_DIR.
    """
    artifacts = {
        "best_t1_gdp_growth_model.joblib":          PROD_MODEL_PATH,
        "phase11_production_release_manifest.json":  PROD_MANIFEST_PATH,
        "master_panel_t1_missingness.csv":           DATASET_PATH,
    }
    result: dict = {}
    for name, path in artifacts.items():
        result[name] = file_hashes(path)

    PHASE20_HASHES_DIR.mkdir(parents=True, exist_ok=True)
    out = PHASE20_HASHES_DIR / f"{label}_production_hashes.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


# ── Phase 19 baseline metrics loading (Correction 2) ─────────────────────────

def load_phase19_master_metrics() -> "pd.DataFrame":
    """Load Phase 19 master metrics CSV for reference comparison."""
    import pandas as pd
    if not PHASE19_CSV.exists():
        raise FileNotFoundError(f"Phase 19 metrics CSV missing: {PHASE19_CSV}")
    return pd.read_csv(PHASE19_CSV)


def verify_reproduction(
    reproduced_m0_rmse: float,
    reproduced_a5a_rmse: float,
    log,
) -> dict:
    """
    Compare reproduced values against Phase 19 reference values.
    Per Correction 2: reference values are NOT the pass/fail criterion.
    The reproduction check just confirms methodological consistency.
    """
    delta_m0  = abs(reproduced_m0_rmse  - PHASE19_REF_M0_RMSE)
    delta_a5a = abs(reproduced_a5a_rmse - PHASE19_REF_A5A_RMSE)
    m0_ok  = delta_m0  <= REPRODUCTION_TOLERANCE
    a5a_ok = delta_a5a <= REPRODUCTION_TOLERANCE

    status = "REPRODUCTION_PASS" if (m0_ok and a5a_ok) else "REPRODUCTION_MISMATCH"

    log.info(f"  Phase 19 ref M0  RMSE: {PHASE19_REF_M0_RMSE:.4f} | Reproduced: {reproduced_m0_rmse:.4f} | Δ={delta_m0:.4f} | {'✓' if m0_ok else '✗'}")
    log.info(f"  Phase 19 ref A5a RMSE: {PHASE19_REF_A5A_RMSE:.4f} | Reproduced: {reproduced_a5a_rmse:.4f} | Δ={delta_a5a:.4f} | {'✓' if a5a_ok else '✗'}")
    log.info(f"  Reproduction status: {status}")

    return {
        "status":              status,
        "ref_m0_rmse":         PHASE19_REF_M0_RMSE,
        "ref_a5a_rmse":        PHASE19_REF_A5A_RMSE,
        "reproduced_m0_rmse":  reproduced_m0_rmse,
        "reproduced_a5a_rmse": reproduced_a5a_rmse,
        "delta_m0":            delta_m0,
        "delta_a5a":           delta_a5a,
        "tolerance":           REPRODUCTION_TOLERANCE,
        "m0_within_tolerance": m0_ok,
        "a5a_within_tolerance": a5a_ok,
    }
