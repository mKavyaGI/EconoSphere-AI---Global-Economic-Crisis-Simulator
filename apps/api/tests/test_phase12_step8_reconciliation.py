import pytest
import json
import pandas as pd
from pathlib import Path
import hashlib

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
MODELS_DIR_12 = PROJECT_ROOT / "models" / "phase12"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

def test_01_json_exists():
    assert (MODELS_DIR_12 / "step8_reconciliation_metadata.json").exists()

def test_02_csv_exists():
    assert (PROCESSED_DIR / "phase12_step8_reconciliation.csv").exists()

def test_03_immutability_status_pass():
    with open(MODELS_DIR_12 / "step8_reconciliation_metadata.json") as f:
        meta = json.load(f)
        assert meta["artifact_immutability_status"] == "PASS"

def test_04_raw_dataset_md5():
    with open(MODELS_DIR_12 / "step8_reconciliation_metadata.json") as f:
        meta = json.load(f)
        assert meta["raw_dataset_md5"] == "8ac7e0b2bf09fbe89289f82d0c7cf25e"

def test_05_reconciliation_classification():
    with open(MODELS_DIR_12 / "step8_reconciliation_metadata.json") as f:
        meta = json.load(f)
        assert meta["reconciliation_classification"] == "RECONCILED — ARTIFACT/PIPELINE VERSION MISMATCH"

def test_06_production_status():
    with open(MODELS_DIR_12 / "step8_reconciliation_metadata.json") as f:
        meta = json.load(f)
        assert meta["production_status"] == "NOT CLEARED — PROVENANCE RECONCILIATION REQUIRED"

def test_07_candidate_b_status():
    with open(MODELS_DIR_12 / "step8_reconciliation_metadata.json") as f:
        meta = json.load(f)
        assert meta["candidate_b_status"] == "EXPERIMENTAL — NOT PRODUCTION"

def test_08_documented_config_extraction():
    with open(MODELS_DIR_12 / "step8_reconciliation_metadata.json") as f:
        meta = json.load(f)
        assert meta["documented_feature_count"] == 31
        assert "calibration_q90" in meta["documented_phase11_config"]

def test_09_physical_artifact_extraction():
    with open(MODELS_DIR_12 / "step8_reconciliation_metadata.json") as f:
        meta = json.load(f)
        assert meta["physical_feature_count"] == 29
        assert meta["physical_joblib_config"]["max_depth"] == 6
        assert meta["physical_joblib_config"]["l2_regularization"] == 1.0

def test_10_csv_gates_presence():
    df = pd.read_csv(PROCESSED_DIR / "phase12_step8_reconciliation.csv")
    checks = df["check"].tolist()
    assert "Raw Dataset Integrity" in checks
    assert "Feature Schema Mismatch" in checks
    assert "Hyperparameter Mismatch" in checks
    assert "Serialization Provenance" in checks
    assert "Metric Provenance" in checks
    assert "Candidate B Comparison Validity" in checks
    assert "Leakage and Temporal Integrity" in checks
    assert "Artifact Immutability" in checks

def test_11_target_exclusion():
    with open(MODELS_DIR_12 / "step8_reconciliation_metadata.json") as f:
        meta = json.load(f)
        assert "Target correctly excluded" in meta["leakage_findings"]

def test_12_chronological_bounds():
    with open(MODELS_DIR_12 / "step8_reconciliation_metadata.json") as f:
        meta = json.load(f)
        assert "Strict chronological splits observed" in meta["temporal_findings"]
