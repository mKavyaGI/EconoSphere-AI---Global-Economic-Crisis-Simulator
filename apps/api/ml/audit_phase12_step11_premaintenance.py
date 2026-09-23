import json
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime, timezone

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "master_panel.csv"
MODELS_DIR_11 = PROJECT_ROOT / "models" / "phase11"
MODELS_DIR_12 = PROJECT_ROOT / "models" / "phase12"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

PROTECTED_PATHS = [
    RAW_PATH,
    MODELS_DIR_11 / "best_t1_gdp_growth_model.joblib",
    MODELS_DIR_11 / "t1_step11_model_metadata.json",
    PROCESSED_DIR / "t1_step11_predictions.csv",
    PROCESSED_DIR / "t1_step11_2026_forecasts.csv",
    MODELS_DIR_12 / "step1_walk_forward_metadata.json",
    MODELS_DIR_12 / "step2_gdp_momentum_metadata.json",
    MODELS_DIR_12 / "step3_forward_indicators_metadata.json",
    MODELS_DIR_12 / "step4_regime_aware_metadata.json",
    MODELS_DIR_12 / "step5_shap_metadata.json",
    MODELS_DIR_12 / "step6_model_robustness_metadata.json",
    MODELS_DIR_12 / "step7_production_readiness_metadata.json",
    MODELS_DIR_12 / "step8_reconciliation_metadata.json",
    MODELS_DIR_12 / "step9_artifact_repair_metadata.json",
    MODELS_DIR_12 / "step10_test_reconciliation_metadata.json",
]

def hash_file(path: Path) -> str:
    if not path.exists():
        return "MISSING"
    hash_md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def main():
    print("STEP 1: PRE-MAINTENANCE SNAPSHOT")
    
    # 1. git status
    git_status = subprocess.check_output(["git", "status"], text=True)
    
    # 2-4. Hashes
    hashes = {str(p.relative_to(PROJECT_ROOT)): hash_file(p) for p in PROTECTED_PATHS}
    
    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_status": git_status,
        "protected_artifact_hashes": hashes,
        "test_suite_result_pre_maintenance": "FAIL (6 failures expected)",
        "failing_tests": [
            "apps/api/tests/test_phase12_production_readiness.py::test_07_model_max_depth_discrepancy_reported",
            "apps/api/tests/test_phase12_production_readiness.py::test_08_model_l2_regularization_discrepancy_reported",
            "apps/api/tests/test_phase12_production_readiness.py::test_12_feature_count_discrepancy_reported",
            "apps/api/tests/test_phase12_production_readiness.py::test_24_no_experimental_overwrite",
            "apps/api/tests/test_t1_model_robustness.py::test_02_phase11_artifacts_unchanged",
            "apps/api/tests/test_t1_model_robustness.py::test_39_phase11_control_remains_locked"
        ]
    }
    
    out_path = MODELS_DIR_12 / "phase12_step11_premaintenance_metadata.json"
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved snapshot to {out_path}")

if __name__ == "__main__":
    main()
