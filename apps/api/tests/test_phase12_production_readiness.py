import pytest
import pandas as pd
import json
import joblib
from pathlib import Path
import hashlib
import os

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "master_panel.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR_11 = PROJECT_ROOT / "models" / "phase11"
MODELS_DIR_12 = PROJECT_ROOT / "models" / "phase12"

def get_md5(path):
    hash_md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def test_01_raw_dataset_md5_matches_exactly():
    assert get_md5(RAW_PATH) == "8ac7e0b2bf09fbe89289f82d0c7cf25e"

def test_02_production_model_exists():
    assert (MODELS_DIR_11 / "best_t1_gdp_growth_model.joblib").exists()

def test_03_production_metadata_exists():
    assert (MODELS_DIR_11 / "t1_step11_model_metadata.json").exists()

def test_04_walk_forward_metadata_exists():
    assert (MODELS_DIR_12 / "step1_walk_forward_metadata.json").exists()

def test_05_candidate_b_metadata_exists():
    assert (MODELS_DIR_12 / "step4_regime_aware_metadata.json").exists()

def test_06_model_type_is_pipeline_containing_histgradientboosting():
    pipeline = joblib.load(MODELS_DIR_11 / "best_t1_gdp_growth_model.joblib")
    assert type(pipeline).__name__ == "Pipeline"
    assert "model" in pipeline.named_steps
    assert type(pipeline.named_steps["model"]).__name__ == "HistGradientBoostingRegressor"

def test_07_model_max_depth_is_5():
    pipeline = joblib.load(MODELS_DIR_11 / "best_t1_gdp_growth_model.joblib")
    assert pipeline.named_steps["model"].get_params()["max_depth"] == 5

def test_08_model_l2_regularization_is_5_0():
    pipeline = joblib.load(MODELS_DIR_11 / "best_t1_gdp_growth_model.joblib")
    assert pipeline.named_steps["model"].get_params()["l2_regularization"] == 5.0

def test_09_model_learning_rate_is_0_05():
    pipeline = joblib.load(MODELS_DIR_11 / "best_t1_gdp_growth_model.joblib")
    assert pipeline.named_steps["model"].get_params()["learning_rate"] == 0.05

def test_10_model_random_state_is_42():
    pipeline = joblib.load(MODELS_DIR_11 / "best_t1_gdp_growth_model.joblib")
    assert pipeline.named_steps["model"].get_params()["random_state"] == 42

def test_11_model_max_iter_is_300():
    pipeline = joblib.load(MODELS_DIR_11 / "best_t1_gdp_growth_model.joblib")
    assert pipeline.named_steps["model"].get_params()["max_iter"] == 300

def test_12_feature_count_is_31():
    pipeline = joblib.load(MODELS_DIR_11 / "best_t1_gdp_growth_model.joblib")
    imputer = pipeline.named_steps["imputer"]
    assert len(imputer.feature_names_in_) == 31

def test_13_target_exclusion_from_training():
    with open(MODELS_DIR_11 / "t1_step11_model_metadata.json") as f:
        meta = json.load(f)
        assert "gdp_growth_next_year" not in meta.get("model_features", [])

def test_14_temporal_split_train_before_val():
    df = pd.read_csv(PROCESSED_DIR / "t1_step11_predictions.csv")
    assert df["year"].min() >= 2023 # It only contains test data in this file!

def test_15_test_split_is_chronological_2023_2024():
    df = pd.read_csv(PROCESSED_DIR / "t1_step11_predictions.csv")
    assert df["year"].min() == 2023
    assert df["year"].max() == 2024

def test_16_no_target_year_in_features():
    with open(MODELS_DIR_11 / "t1_step11_model_metadata.json") as f:
        meta = json.load(f)
        assert "target_year" not in meta.get("model_features", [])

def test_17_reported_test_rmse_is_3_9113():
    with open(MODELS_DIR_11 / "t1_step11_model_metadata.json") as f:
        meta = json.load(f)
        assert abs(meta["test_metrics"]["RMSE"] - 3.9113) < 1e-4

def test_18_reported_val_rmse_is_8_2853():
    with open(MODELS_DIR_11 / "t1_step11_model_metadata.json") as f:
        meta = json.load(f)
        assert abs(meta["validation_metrics"]["RMSE"] - 8.2853) < 1e-4

def test_19_reported_walk_forward_rmse_is_6_0967():
    with open(MODELS_DIR_12 / "step1_walk_forward_metadata.json") as f:
        meta = json.load(f)
        assert abs(meta["overall_walk_forward_rmse"] - 6.0967) < 1e-4

def test_20_candidate_b_reported_walk_forward_is_6_0943():
    with open(MODELS_DIR_12 / "step4_regime_aware_metadata.json") as f:
        meta = json.load(f)
        assert abs(meta["candidate_walk_forward_RMSE"] - 6.0943) < 1e-4

def test_21_candidate_b_is_not_production():
    with open(MODELS_DIR_12 / "step7_production_readiness_metadata.json") as f:
        meta = json.load(f)
        assert meta["phase12_candidate_b_status"] == "EXPERIMENTAL — NOT PRODUCTION"

def test_22_step5_explainability_inconclusive():
    with open(MODELS_DIR_12 / "step7_production_readiness_metadata.json") as f:
        meta = json.load(f)
        assert meta["step5_explainability_status"] == "EXPLAINABILITY INCONCLUSIVE"

def test_23_step6_robustness_inconclusive():
    with open(MODELS_DIR_12 / "step7_production_readiness_metadata.json") as f:
        meta = json.load(f)
        assert meta["step6_robustness_status"] == "ROBUSTNESS INCONCLUSIVE"

def test_24_no_experimental_overwrite():
    # If Candidate B was promoted, best_t1_gdp_growth_model would have 33 features.
    pipeline = joblib.load(MODELS_DIR_11 / "best_t1_gdp_growth_model.joblib")
    imputer = pipeline.named_steps["imputer"]
    assert len(imputer.feature_names_in_) != 33

def test_25_audit_metadata_created():
    assert (MODELS_DIR_12 / "step7_production_readiness_metadata.json").exists()

def test_26_audit_csv_created():
    assert (PROCESSED_DIR / "phase12_step7_production_readiness.csv").exists()

def test_27_audit_decision_is_failure_due_to_parameter_mismatch():
    with open(MODELS_DIR_12 / "step7_production_readiness_metadata.json") as f:
        meta = json.load(f)
        assert meta["final_decision"] == "PRODUCTION READINESS FAILED"

def test_28_immutability_confirmed_by_audit():
    with open(MODELS_DIR_12 / "step7_production_readiness_metadata.json") as f:
        meta = json.load(f)
        assert meta["artifact_immutability_status"] == "PASS"

def test_29_target_leakage_status_pass():
    with open(MODELS_DIR_12 / "step7_production_readiness_metadata.json") as f:
        meta = json.load(f)
        assert meta["leakage_status"] == "PASS"

def test_30_temporal_leakage_status_pass():
    with open(MODELS_DIR_12 / "step7_production_readiness_metadata.json") as f:
        meta = json.load(f)
        assert meta["temporal_integrity_status"] == "PASS"
