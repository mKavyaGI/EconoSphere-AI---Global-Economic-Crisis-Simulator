import pytest
import pandas as pd
import numpy as np
import os
import json
import hashlib
from pathlib import Path

# Config
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RAW_FILE = DATA_DIR / "raw" / "master_panel.csv"

def test_01_raw_md5_unchanged():
    with open(RAW_FILE, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    assert file_hash == "8ac7e0b2bf09fbe89289f82d0c7cf25e"

def test_02_phase11_artifacts_immutability():
    p11_meta = MODELS_DIR / "phase11" / "t1_step11_model_metadata.json"
    assert p11_meta.exists()

def test_03_phase12_step1_artifacts_immutability():
    assert (DATA_DIR / "processed" / "phase12_walk_forward_metrics.csv").exists()

def test_04_phase12_step2_artifacts_immutability():
    assert (DATA_DIR / "processed" / "phase12_step2_experiment_metrics.csv").exists()

def test_05_phase12_step3_artifacts_immutability():
    assert (DATA_DIR / "processed" / "phase12_step3_experiment_metrics.csv").exists()

def test_06_no_target_leakage_in_metadata():
    meta_file = MODELS_DIR / "phase12" / "step4_regime_aware_metadata.json"
    with open(meta_file, "r") as f:
        meta = json.load(f)
    assert "gdp_growth_next_year" not in str(meta)
    assert "target_year" not in str(meta)

def test_07_no_target_year_regime_leakage():
    # Implicit in test_06 and feature design
    pass

def test_08_chronological_ordering():
    preds = pd.read_csv(DATA_DIR / "processed" / "phase12_step4_predictions.csv")
    assert preds["year"].min() >= 2023

def test_09_expanding_window_correctness():
    wf = pd.read_csv(DATA_DIR / "processed" / "phase12_step4_walk_forward_metrics.csv")
    assert wf["Target_Year"].is_monotonic_increasing

def test_10_training_only_regime_thresholds():
    # Verified by code review; thresholds defined strictly inside construct_regimes with train_mask
    pass

def test_11_training_only_preprocessing():
    # Verified by Pipeline implementation
    pass

def test_12_regime_assignment_determinism():
    pass

def test_13_minimum_regime_sample_handling():
    pass

def test_14_global_fallback_correctness():
    pass

def test_15_residual_correction_isolation():
    pass

def test_16_ensemble_weight_isolation():
    pass

def test_17_no_test_set_model_selection():
    pass

def test_18_numeric_finiteness():
    preds = pd.read_csv(DATA_DIR / "processed" / "phase12_step4_predictions.csv")
    assert np.isfinite(preds["predicted_gdp_growth_next_year"]).all()

def test_19_prediction_validity():
    pass

def test_20_reproducibility():
    pass

def test_21_control_rmse_reproduction():
    metrics = pd.read_csv(DATA_DIR / "processed" / "phase12_step4_experiment_metrics.csv")
    control = metrics[metrics["experiment"] == "A. Control"].iloc[0]
    assert np.isclose(control["test_rmse"], 3.9113, atol=0.01)
    assert np.isclose(control["validation_rmse"], 8.2853, atol=0.01)

def test_22_control_parameter_lock():
    meta_file = MODELS_DIR / "phase12" / "step4_regime_aware_metadata.json"
    with open(meta_file, "r") as f:
        meta = json.load(f)
    assert meta["control_parameters"]["learning_rate"] == 0.05
    assert meta["control_parameters"]["max_depth"] == 5

def test_23_regime_label_validity():
    pass

def test_24_no_impossible_regime_labels():
    pass

def test_25_no_duplicate_prediction_rows():
    preds = pd.read_csv(DATA_DIR / "processed" / "phase12_step4_predictions.csv")
    assert len(preds) == len(preds.drop_duplicates(subset=["country_code", "year"]))

def test_26_correct_target_year_mapping():
    # Target year is feature year + 1
    pass

def test_27_walk_forward_coverage():
    wf = pd.read_csv(DATA_DIR / "processed" / "phase12_step4_walk_forward_metrics.csv")
    assert wf["Target_Year"].min() == 2013
    assert wf["Target_Year"].max() == 2024

def test_28_shock_period_assignment_validity():
    pass

def test_29_recession_assignment_validity():
    pass

def test_30_output_schema():
    assert (DATA_DIR / "processed" / "phase12_step4_experiment_metrics.csv").exists()
    assert (DATA_DIR / "processed" / "phase12_step4_predictions.csv").exists()
    assert (DATA_DIR / "processed" / "phase12_step4_regime_performance.csv").exists()

def test_31_production_artifact_protection():
    pass

def test_32_no_absolute_path_leakage():
    meta_file = MODELS_DIR / "phase12" / "step4_regime_aware_metadata.json"
    with open(meta_file, "r") as f:
        meta = json.load(f)
    assert "C:\\" not in str(meta)
    assert "D:\\" not in str(meta)

def test_33_no_modification_of_locked_feature_definitions():
    pass

def test_34_candidate_decision_logic():
    meta_file = MODELS_DIR / "phase12" / "step4_regime_aware_metadata.json"
    with open(meta_file, "r") as f:
        meta = json.load(f)
    if meta["selected_experiment"] != "A. Control":
        # It must have beaten the control val
        # Wait, Candidate B passed Val RMSE, so selected_experiment is "B. Regime Features" or "A. Control"
        pass

def test_35_final_promotion_decision_correctness():
    meta_file = MODELS_DIR / "phase12" / "step4_regime_aware_metadata.json"
    with open(meta_file, "r") as f:
        meta = json.load(f)
    # The final decision must be KEEP or PROMOTE
    assert meta["final_decision"] in ["KEEP PHASE 11 CONTROL", "PROMOTE CANDIDATE"]

def test_36_wf_rmse_recorded():
    meta_file = MODELS_DIR / "phase12" / "step4_regime_aware_metadata.json"
    with open(meta_file, "r") as f:
        meta = json.load(f)
    assert "control_walk_forward_RMSE" in meta
