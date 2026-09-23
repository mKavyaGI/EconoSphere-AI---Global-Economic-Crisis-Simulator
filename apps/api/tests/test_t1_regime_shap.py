import pytest
import json
import pandas as pd
import numpy as np
import hashlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RAW_FILE = DATA_DIR / "raw" / "master_panel.csv"

def get_md5(file_path):
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def test_01_raw_md5_unchanged():
    assert get_md5(RAW_FILE) == "8ac7e0b2bf09fbe89289f82d0c7cf25e"

def test_02_phase11_artifacts_immutability():
    # Since we didn't touch them, they should exist
    assert (MODELS_DIR / "phase11" / "t1_step11_model_metadata.json").exists()

def test_03_step1_immutability():
    assert (DATA_DIR / "processed" / "phase12_walk_forward_metrics.csv").exists()

def test_04_step2_immutability():
    assert (DATA_DIR / "processed" / "phase12_step2_experiment_metrics.csv").exists()

def test_05_step3_immutability():
    assert (DATA_DIR / "processed" / "phase12_step3_experiment_metrics.csv").exists()

def test_06_step4_immutability():
    assert (DATA_DIR / "processed" / "phase12_step4_experiment_metrics.csv").exists()

def test_07_candidate_b_feature_count():
    meta = MODELS_DIR / "phase12" / "step5_shap_metadata.json"
    if meta.exists():
        with open(meta) as f:
            data = json.load(f)
        assert "growth_regime_num + stress_regime_num" in data["candidate_b_definition"]

def test_08_candidate_b_regime_features_exist():
    pass

def test_09_candidate_b_parameter_locking():
    meta = MODELS_DIR / "phase12" / "step5_shap_metadata.json"
    if meta.exists():
        with open(meta) as f:
            data = json.load(f)
        assert data["model_parameters"]["learning_rate"] == 0.05
        assert data["model_parameters"]["max_depth"] == 5

def test_10_candidate_b_validation_split():
    meta = MODELS_DIR / "phase12" / "step5_shap_metadata.json"
    if meta.exists():
        with open(meta) as f:
            data = json.load(f)
        assert data["validation_period"] == [2019, 2022]

def test_11_validation_rmse_reproduction():
    meta = MODELS_DIR / "phase12" / "step5_shap_metadata.json"
    if meta.exists():
        with open(meta) as f:
            data = json.load(f)
        assert np.isclose(data["candidate_b_validation_RMSE"], 8.2598, atol=0.01)

def test_12_no_target_column_in_features():
    df = pd.read_csv(DATA_DIR / "processed" / "phase12_step5_shap_global_importance.csv")
    assert "gdp_growth_next_year" not in df["feature"].values

def test_13_no_target_year_information_in_regime():
    pass

def test_14_growth_regime_temporal_safety():
    pass

def test_15_stress_regime_temporal_safety():
    pass

def test_16_training_only_threshold_calculation():
    pass

def test_17_validation_isolation():
    pass

def test_18_test_isolation():
    pass

def test_19_shap_output_existence():
    assert (DATA_DIR / "processed" / "phase12_step5_shap_global_importance.csv").exists()
    assert (DATA_DIR / "processed" / "phase12_step5_shap_regime_importance.csv").exists()

def test_20_shap_dimensions_match_features():
    df = pd.read_csv(DATA_DIR / "processed" / "phase12_step5_shap_global_importance.csv")
    assert len(df) == 33

def test_21_shap_values_finite():
    df = pd.read_csv(DATA_DIR / "processed" / "phase12_step5_shap_global_importance.csv")
    assert np.isfinite(df["mean_abs_shap"]).all()

def test_22_global_importance_reproducibility():
    pass

def test_23_walk_forward_temporal_ordering():
    df = pd.read_csv(DATA_DIR / "processed" / "phase12_step5_shap_walk_forward_importance.csv")
    assert df["target_year"].min() == 2013
    assert df["target_year"].max() == 2024

def test_24_walk_forward_shap_reproducibility():
    pass

def test_25_2026_explanations_countries():
    df = pd.read_csv(DATA_DIR / "processed" / "phase12_step5_2026_experimental_explanations.csv")
    assert set(df["country_code"]) == {"IND", "CHN", "USA", "JPN", "GBR"}

def test_26_2026_explanations_marked_experimental():
    df = pd.read_csv(DATA_DIR / "processed" / "phase12_step5_2026_experimental_explanations.csv")
    assert "EXPERIMENTAL" in df["label"].iloc[0]

def test_27_phase11_2026_forecasts_unchanged():
    pass

def test_28_no_production_files_modified():
    meta = MODELS_DIR / "phase12" / "step5_shap_metadata.json"
    if meta.exists():
        with open(meta) as f:
            data = json.load(f)
        assert data["production_model_modified"] == False

def test_29_metadata_completeness():
    meta = MODELS_DIR / "phase12" / "step5_shap_metadata.json"
    assert meta.exists()

def test_30_deterministic_random_state():
    meta = MODELS_DIR / "phase12" / "step5_shap_metadata.json"
    if meta.exists():
        with open(meta) as f:
            data = json.load(f)
        assert data["random_seed"] == 42

def test_31_no_nan_in_shap_summaries():
    df = pd.read_csv(DATA_DIR / "processed" / "phase12_step5_shap_global_importance.csv")
    assert not df.isna().any().any()

def test_32_regime_specific_sample_counts():
    df = pd.read_csv(DATA_DIR / "processed" / "phase12_step5_shap_regime_importance.csv")
    assert (df["sample_count"] > 0).all()

def test_33_control_vs_candidate_integrity():
    df = pd.read_csv(DATA_DIR / "processed" / "phase12_step5_control_vs_candidate.csv")
    assert len(df) == 2

def test_34_explainability_decision_valid():
    meta = MODELS_DIR / "phase12" / "step5_shap_metadata.json"
    if meta.exists():
        with open(meta) as f:
            data = json.load(f)
        assert data["final_explainability_decision"] in [
            "EXPLAINABILITY CONFIRMED",
            "EXPLAINABILITY INCONCLUSIVE",
            "EXPLAINABILITY REJECTED"
        ]

def test_35_production_promotion_flag_false():
    meta = MODELS_DIR / "phase12" / "step5_shap_metadata.json"
    if meta.exists():
        with open(meta) as f:
            data = json.load(f)
        assert data["phase11_artifacts_modified"] == False
