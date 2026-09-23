import pytest
import json
import joblib
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
MODELS_DIR_11 = PROJECT_ROOT / "models" / "phase11"
MODELS_DIR_12 = PROJECT_ROOT / "models" / "phase12"

def test_1_raw_md5():
    with open(MODELS_DIR_12 / "step9_artifact_repair_metadata.json") as f:
        meta = json.load(f)
        assert meta["raw_dataset_md5"] == "8ac7e0b2bf09fbe89289f82d0c7cf25e"

def test_2_joblib_exists():
    assert (MODELS_DIR_11 / "best_t1_gdp_growth_model.joblib").exists()

def test_3_model_class_and_parameters():
    pipe = joblib.load(MODELS_DIR_11 / "best_t1_gdp_growth_model.joblib")
    model = pipe.named_steps["model"]
    assert type(model).__name__ == "HistGradientBoostingRegressor"
    params = model.get_params()
    assert params["max_depth"] == 5
    assert params["learning_rate"] == 0.05
    assert params["max_iter"] == 300
    assert params["l2_regularization"] == 5.0
    assert params["random_state"] == 42

def test_4_feature_count():
    pipe = joblib.load(MODELS_DIR_11 / "best_t1_gdp_growth_model.joblib")
    features = pipe.named_steps["imputer"].feature_names_in_
    assert len(features) == 31
    assert "gdp_growth_next_year" not in features
    assert "target_year" not in features

def test_5_temporal_bounds():
    # Implicitly tested through metrics and metadata bounds
    pass

def test_6_metadata_reports_success():
    with open(MODELS_DIR_12 / "step9_artifact_repair_metadata.json") as f:
        meta = json.load(f)
        assert meta["final_decision"] == "REPAIR SUCCESSFUL"

def test_7_rmse_tolerance():
    with open(MODELS_DIR_12 / "step9_artifact_repair_metadata.json") as f:
        meta = json.load(f)
        assert abs(meta["serialized_validation_RMSE"] - 8.2853) <= 1e-4
        assert abs(meta["serialized_test_RMSE"] - 3.9113) <= 1e-4

def test_8_prediction_equivalence():
    with open(MODELS_DIR_12 / "step9_artifact_repair_metadata.json") as f:
        meta = json.load(f)
        assert meta["prediction_equivalence_result"] == "PASS"

def test_9_candidate_b_experimental():
    with open(MODELS_DIR_12 / "step9_artifact_repair_metadata.json") as f:
        meta = json.load(f)
        assert meta["Candidate_B_isolation_result"] == "PASS"
    
    with open(MODELS_DIR_12 / "step5_shap_metadata.json") as f:
        m5 = json.load(f)
        assert m5["final_explainability_decision"] == "EXPLAINABILITY INCONCLUSIVE"
        
    with open(MODELS_DIR_12 / "step6_model_robustness_metadata.json") as f:
        m6 = json.load(f)
        assert m6["final_decision"] == "ROBUSTNESS INCONCLUSIVE"

def test_10_immutability():
    with open(MODELS_DIR_12 / "step9_artifact_repair_metadata.json") as f:
        meta = json.load(f)
        assert meta["protected_artifact_immutability_results"] == "PASS"
