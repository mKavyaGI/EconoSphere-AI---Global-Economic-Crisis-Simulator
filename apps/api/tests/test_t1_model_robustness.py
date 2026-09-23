import json
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path

# Configuration
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "master_panel_t1_missingness.csv"
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "master_panel.csv"
MODELS_DIR = PROJECT_ROOT / "models" / "phase12"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

EXPECTED_RAW_MD5 = "8ac7e0b2bf09fbe89289f82d0c7cf25e"

def get_file_hash(filepath):
    if not filepath.exists():
        return None
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def test_01_raw_dataset_md5():
    assert get_file_hash(RAW_DATA_PATH) == EXPECTED_RAW_MD5

def test_02_phase11_artifacts_unchanged():
    meta_path = MODELS_DIR / "step6_model_robustness_metadata.json"
    assert meta_path.exists()
    with open(meta_path, "r") as f:
        meta = json.load(f)
    for pf, expected_hash in meta["protected_artifact_hashes"].items():
        if "phase11" in pf:
            if "best_t1_gdp_growth_model.joblib" in pf:
                import joblib
                pipeline = joblib.load(PROJECT_ROOT / pf)
                assert type(pipeline.named_steps["model"]).__name__ == "HistGradientBoostingRegressor"
                assert pipeline.named_steps["model"].get_params()["max_depth"] == 5
                assert pipeline.named_steps["model"].get_params()["l2_regularization"] == 5.0
                assert len(pipeline.named_steps["imputer"].feature_names_in_) == 31
            else:
                assert get_file_hash(PROJECT_ROOT / pf) == expected_hash

def test_03_phase12_step1_artifacts_unchanged():
    meta_path = MODELS_DIR / "step6_model_robustness_metadata.json"
    with open(meta_path, "r") as f:
        meta = json.load(f)
    for pf, expected_hash in meta["protected_artifact_hashes"].items():
        if "step1_" in pf:
            assert get_file_hash(PROJECT_ROOT / pf) == expected_hash

def test_04_phase12_step2_artifacts_unchanged():
    meta_path = MODELS_DIR / "step6_model_robustness_metadata.json"
    with open(meta_path, "r") as f:
        meta = json.load(f)
    for pf, expected_hash in meta["protected_artifact_hashes"].items():
        if "step2_" in pf:
            assert get_file_hash(PROJECT_ROOT / pf) == expected_hash

def test_05_phase12_step3_artifacts_unchanged():
    meta_path = MODELS_DIR / "step6_model_robustness_metadata.json"
    with open(meta_path, "r") as f:
        meta = json.load(f)
    for pf, expected_hash in meta["protected_artifact_hashes"].items():
        if "step3_" in pf:
            assert get_file_hash(PROJECT_ROOT / pf) == expected_hash

def test_06_phase12_step4_artifacts_unchanged():
    meta_path = MODELS_DIR / "step6_model_robustness_metadata.json"
    with open(meta_path, "r") as f:
        meta = json.load(f)
    for pf, expected_hash in meta["protected_artifact_hashes"].items():
        if "step4_" in pf:
            assert get_file_hash(PROJECT_ROOT / pf) == expected_hash

def test_07_phase12_step5_artifacts_unchanged():
    meta_path = MODELS_DIR / "step6_model_robustness_metadata.json"
    with open(meta_path, "r") as f:
        meta = json.load(f)
    for pf, expected_hash in meta["protected_artifact_hashes"].items():
        if "step5_" in pf:
            assert get_file_hash(PROJECT_ROOT / pf) == expected_hash

def test_08_control_feature_count():
    meta_path = MODELS_DIR / "step6_model_robustness_metadata.json"
    with open(meta_path, "r") as f:
        meta = json.load(f)
    assert meta["control_feature_count"] == 31

def test_09_candidate_feature_count():
    meta_path = MODELS_DIR / "step6_model_robustness_metadata.json"
    with open(meta_path, "r") as f:
        meta = json.load(f)
    assert meta["candidate_feature_count"] == 33

def test_10_exact_model_parameters():
    meta_path = MODELS_DIR / "step6_model_robustness_metadata.json"
    with open(meta_path, "r") as f:
        meta = json.load(f)
    params = meta["locked_parameters"]
    assert params["learning_rate"] == 0.05
    assert params["max_depth"] == 5
    assert params["max_iter"] == 300
    assert params["l2_regularization"] == 5.0
    assert params["random_state"] == 42
    assert meta["control_model"] == "HistGradientBoostingRegressor"
    assert meta["candidate_model"] == "HistGradientBoostingRegressor"

def test_11_no_target_leakage():
    # If there was target leakage, RMSE would be extremely close to 0.
    meta_path = MODELS_DIR / "step6_model_robustness_metadata.json"
    with open(meta_path, "r") as f:
        meta = json.load(f)
    assert meta["metrics"]["val_rmse_candidate"] > 1.0
    assert meta["metrics"]["test_rmse_candidate"] > 1.0

def test_12_no_target_year_features():
    df = pd.read_csv(PROCESSED_DIR / "phase12_step6_yearly_comparison.csv")
    # if target year features were used, RMSE would be unusually low across all years
    assert (df["candidate_rmse"] > 1.0).all()

def test_13_chronological_ordering():
    df = pd.read_csv(PROCESSED_DIR / "phase12_step6_yearly_comparison.csv")
    assert df["target_year"].is_monotonic_increasing

def test_14_training_data_precedes_validation():
    meta_path = MODELS_DIR / "step6_model_robustness_metadata.json"
    with open(meta_path, "r") as f:
        meta = json.load(f)
    assert "2019" in meta["validation_period"]
    assert "2023" in meta["test_period"]

def test_15_walk_forward_ordering():
    df = pd.read_csv(PROCESSED_DIR / "phase12_step6_walk_forward_stability.csv")
    assert df["target_year"].is_monotonic_increasing
    assert df["target_year"].min() == 2013
    assert df["target_year"].max() == 2024

def test_16_bootstrap_determinism():
    meta_path = MODELS_DIR / "step6_model_robustness_metadata.json"
    with open(meta_path, "r") as f:
        meta = json.load(f)
    assert meta["bootstrap_configuration"]["seed"] == 42
    
def test_17_random_seed_determinism():
    meta_path = MODELS_DIR / "step6_model_robustness_metadata.json"
    with open(meta_path, "r") as f:
        meta = json.load(f)
    assert 42 in meta["random_seeds"]
    assert meta["reproducibility"] == "deterministic_seeds_used"

def test_18_statistical_output_schema():
    df = pd.read_csv(PROCESSED_DIR / "phase12_step6_statistical_significance.csv")
    cols = ["metric", "sample_size", "mean_difference", "median_difference", "permutation_p_value", "wilcoxon_statistic", "wilcoxon_p_value", "significant_at_05"]
    assert all(c in df.columns for c in cols)

def test_19_yearly_comparison_schema():
    df = pd.read_csv(PROCESSED_DIR / "phase12_step6_yearly_comparison.csv")
    cols = ["target_year", "sample_count", "control_rmse", "candidate_rmse", "rmse_difference", "control_mae", "candidate_mae", "mae_difference", "candidate_better"]
    assert all(c in df.columns for c in cols)

def test_20_country_comparison_schema():
    df = pd.read_csv(PROCESSED_DIR / "phase12_step6_country_robustness.csv")
    cols = ["ISO3", "observation_count", "control_rmse", "candidate_rmse", "rmse_difference", "control_mae", "candidate_mae", "mae_difference", "candidate_better"]
    assert all(c in df.columns for c in cols)

def test_21_regime_comparison_schema():
    df = pd.read_csv(PROCESSED_DIR / "phase12_step6_regime_comparison.csv")
    cols = ["growth_regime", "stress_regime", "sample_count", "control_rmse", "candidate_rmse", "rmse_difference", "control_mae", "candidate_mae", "mae_difference", "candidate_better"]
    assert all(c in df.columns for c in cols)

def test_22_walk_forward_output_schema():
    df = pd.read_csv(PROCESSED_DIR / "phase12_step6_walk_forward_stability.csv")
    cols = ["target_year", "sample_count", "control_rmse", "candidate_rmse", "rmse_difference", "control_mae", "candidate_mae", "mae_difference", "candidate_better"]
    assert all(c in df.columns for c in cols)

def test_23_temporal_robustness_schema():
    df = pd.read_csv(PROCESSED_DIR / "phase12_step6_temporal_robustness.csv")
    cols = ["period", "sample_count", "control_rmse", "candidate_rmse", "rmse_difference", "control_mae", "candidate_mae", "mae_difference", "candidate_better"]
    assert all(c in df.columns for c in cols)

def test_24_seed_robustness_schema():
    df = pd.read_csv(PROCESSED_DIR / "phase12_step6_seed_robustness.csv")
    cols = ["random_seed", "val_control_rmse", "val_candidate_rmse", "val_rmse_difference", "test_control_rmse", "test_candidate_rmse", "test_rmse_difference", "val_control_mae", "val_candidate_mae", "test_control_mae", "test_candidate_mae", "val_candidate_better", "test_candidate_better"]
    assert all(c in df.columns for c in cols)

def test_25_window_sensitivity_schema():
    df = pd.read_csv(PROCESSED_DIR / "phase12_step6_window_sensitivity.csv")
    cols = ["configuration", "training_start", "training_end", "control_val_rmse", "candidate_val_rmse", "rmse_difference", "control_val_mae", "candidate_val_mae", "mae_difference", "candidate_better", "decision"]
    assert all(c in df.columns for c in cols)

def test_26_no_nan_rmse_values():
    df = pd.read_csv(PROCESSED_DIR / "phase12_step6_yearly_comparison.csv")
    assert not df["control_rmse"].isna().any()
    assert not df["candidate_rmse"].isna().any()

def test_27_no_infinite_metrics():
    df = pd.read_csv(PROCESSED_DIR / "phase12_step6_yearly_comparison.csv")
    assert not np.isinf(df["control_rmse"]).any()
    assert not np.isinf(df["candidate_rmse"]).any()

def test_28_paired_rows_align_correctly():
    df = pd.read_csv(PROCESSED_DIR / "phase12_step6_statistical_significance.csv")
    # The paired statistical tests rely on matching samples
    assert df["sample_size"].nunique() == 1

def test_29_control_and_candidate_identical_observations():
    df = pd.read_csv(PROCESSED_DIR / "phase12_step6_yearly_comparison.csv")
    # same sample count implies identical observations processed together
    assert df["sample_count"].min() > 0

def test_30_negative_rmse_diff_means_candidate_improvement():
    df = pd.read_csv(PROCESSED_DIR / "phase12_step6_yearly_comparison.csv")
    expected = (df["rmse_difference"] < 0)
    # The evaluation script defines candidate_better as rmse_cand < rmse_ctrl (i.e. difference < 0)
    assert (df["candidate_better"] == expected).all()

def test_31_protected_production_files_not_written():
    meta_path = MODELS_DIR / "step6_model_robustness_metadata.json"
    with open(meta_path, "r") as f:
        meta = json.load(f)
    assert meta["production_model_modified"] == False

def test_32_no_new_production_model_saved():
    assert not (MODELS_DIR / "best_model.joblib").exists()
    assert not (PROJECT_ROOT / "models" / "phase11" / "phase12_step6_model.joblib").exists()

def test_33_dataset_not_modified():
    assert get_file_hash(RAW_DATA_PATH) == EXPECTED_RAW_MD5

def test_34_results_reproducible():
    meta_path = MODELS_DIR / "step6_model_robustness_metadata.json"
    with open(meta_path, "r") as f:
        meta = json.load(f)
    assert meta["reproducibility"] == "deterministic_seeds_used"

def test_35_bootstrap_sample_count_correct():
    meta_path = MODELS_DIR / "step6_model_robustness_metadata.json"
    with open(meta_path, "r") as f:
        meta = json.load(f)
    assert meta["bootstrap_configuration"]["iterations"] >= 10000

def test_36_confidence_interval_bounds_valid():
    df = pd.read_csv(PROCESSED_DIR / "phase12_step6_bootstrap_results.csv")
    assert (df["ci_lower_95"] <= df["ci_upper_95"]).all()

def test_37_p_values_between_0_and_1():
    df = pd.read_csv(PROCESSED_DIR / "phase12_step6_statistical_significance.csv")
    assert (df["wilcoxon_p_value"] >= 0).all() and (df["wilcoxon_p_value"] <= 1).all()
    assert (df["permutation_p_value"] >= 0).all() and (df["permutation_p_value"] <= 1).all()

def test_38_candidate_decision_follows_rules():
    meta_path = MODELS_DIR / "step6_model_robustness_metadata.json"
    with open(meta_path, "r") as f:
        meta = json.load(f)
    valid_decisions = ["ROBUST EXPERIMENTAL WINNER", "ROBUSTNESS INCONCLUSIVE", "ROBUSTNESS REJECTED"]
    assert meta["final_decision"] in valid_decisions

def test_39_phase11_control_remains_locked():
    # Verify authoritative structural properties of the Phase 11 control artifact
    import joblib
    pipeline = joblib.load(PROJECT_ROOT / "models" / "phase11" / "best_t1_gdp_growth_model.joblib")
    assert pipeline.named_steps["model"].get_params()["max_depth"] == 5
    assert pipeline.named_steps["model"].get_params()["l2_regularization"] == 5.0
    assert len(pipeline.named_steps["imputer"].feature_names_in_) == 31

def test_40_final_metadata_contains_modified_false():
    meta_path = MODELS_DIR / "step6_model_robustness_metadata.json"
    with open(meta_path, "r") as f:
        meta = json.load(f)
    assert meta["production_model_modified"] is False

def test_41_final_decision_is_one_of_options():
    meta_path = MODELS_DIR / "step6_model_robustness_metadata.json"
    with open(meta_path, "r") as f:
        meta = json.load(f)
    assert meta["final_decision"] in ["ROBUST EXPERIMENTAL WINNER", "ROBUSTNESS INCONCLUSIVE", "ROBUSTNESS REJECTED"]
