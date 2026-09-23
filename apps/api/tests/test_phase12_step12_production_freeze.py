import os
import json
import hashlib
import joblib
import pytest

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
MANIFEST_PATH = os.path.join(BASE_DIR, 'models/phase11/phase11_production_release_manifest.json')
MODEL_PATH = os.path.join(BASE_DIR, 'models/phase11/best_t1_gdp_growth_model.joblib')
DATA_RAW_PATH = os.path.join(BASE_DIR, 'data/raw/master_panel.csv')

def calculate_hashes(file_path):
    if not os.path.exists(file_path):
        return None, None
    md5_hash = hashlib.md5()
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            md5_hash.update(byte_block)
            sha256_hash.update(byte_block)
    return md5_hash.hexdigest(), sha256_hash.hexdigest()

@pytest.fixture(scope="module")
def manifest():
    assert os.path.exists(MANIFEST_PATH), "Release manifest does not exist."
    with open(MANIFEST_PATH, 'r') as f:
        return json.load(f)

@pytest.fixture(scope="module")
def model():
    assert os.path.exists(MODEL_PATH), "Model file does not exist."
    return joblib.load(MODEL_PATH)

def test_manifest_exists(manifest):
    assert manifest is not None

def test_release_id(manifest):
    assert manifest.get('release_id') == "ECONOSPHERE-PHASE11-PROD-2026-08-21"

def test_production_status(manifest):
    assert manifest.get('production_status') == "PHASE 11 PRODUCTION FROZEN"

def test_model_file_exists():
    assert os.path.exists(MODEL_PATH)

def test_model_hash_matches(manifest):
    md5, sha256 = calculate_hashes(MODEL_PATH)
    assert manifest.get('model_md5') == md5
    assert manifest.get('model_sha256') == sha256

def test_dataset_exists():
    assert os.path.exists(DATA_RAW_PATH)

def test_dataset_md5(manifest):
    md5, sha256 = calculate_hashes(DATA_RAW_PATH)
    assert md5 == "8ac7e0b2bf09fbe89289f82d0c7cf25e"
    assert manifest.get('dataset_md5') == md5

def test_dataset_sha256(manifest):
    md5, sha256 = calculate_hashes(DATA_RAW_PATH)
    assert manifest.get('dataset_sha256') == sha256

def test_model_class(model):
    model_class = model.__class__.__name__
    if model_class == "Pipeline":
        model_class = model.steps[-1][1].__class__.__name__
    assert model_class == "HistGradientBoostingRegressor"

def test_feature_count(model):
    if hasattr(model, "feature_names_in_"):
        features = list(model.feature_names_in_)
    else:
        features = list(model.steps[-1][1].feature_names_in_)
    assert len(features) == 31

def test_target_absent(model):
    if hasattr(model, "feature_names_in_"):
        features = list(model.feature_names_in_)
    else:
        features = list(model.steps[-1][1].feature_names_in_)
    assert 'target' not in features
    assert 't1_gdp_growth' not in features

def test_target_year_absent(model):
    if hasattr(model, "feature_names_in_"):
        features = list(model.feature_names_in_)
    else:
        features = list(model.steps[-1][1].feature_names_in_)
    assert 'target_year' not in features

def test_hyperparameters(model, manifest):
    if model.__class__.__name__ == "Pipeline":
        params = model.steps[-1][1].get_params()
    else:
        params = model.get_params()
    assert params.get('max_depth') == 5
    assert params.get('l2_regularization') == 5.0
    assert params.get('learning_rate') == 0.05
    assert params.get('max_iter') == 300
    assert params.get('random_state') == 42
    
    man_params = manifest.get('hyperparameters', {})
    assert man_params.get('max_depth') == 5
    assert man_params.get('l2_regularization') == 5.0
    assert man_params.get('learning_rate') == 0.05
    assert man_params.get('max_iter') == 300
    assert man_params.get('random_state') == 42

def test_validation_rmse(manifest):
    assert manifest.get('validation_rmse') == 8.2853

def test_test_rmse(manifest):
    assert manifest.get('test_rmse') == 3.9113

def test_candidate_b_experimental(manifest):
    assert manifest.get('candidate_b_status') == "EXPERIMENTAL — NOT PRODUCTION"

def test_train_period(manifest):
    assert manifest.get('train_period') == "<= 2018"

def test_validation_period(manifest):
    assert manifest.get('validation_period') == "2019-2022"

def test_test_period(manifest):
    assert manifest.get('test_period') == "2023-2024"

def test_release_fingerprint_exists(manifest):
    assert 'release_fingerprint' in manifest
    assert len(manifest.get('release_fingerprint')) == 64  # sha256 length

def test_release_fingerprint_deterministic(manifest, model):
    md5, sha256 = calculate_hashes(MODEL_PATH)
    data_md5, data_sha256 = calculate_hashes(DATA_RAW_PATH)
    
    # Needs to match logic in the freeze script
    if hasattr(model, "feature_names_in_"):
        features = list(model.feature_names_in_)
    else:
        features = list(model.steps[-1][1].feature_names_in_)
    
    schema_hash = hashlib.sha256("".join(features).encode('utf-8')).hexdigest()
    
    expected_params = {
        "max_depth": 5,
        "l2_regularization": 5.0,
        "learning_rate": 0.05,
        "max_iter": 300,
        "random_state": 42
    }
    config_hash = hashlib.sha256(json.dumps(expected_params, sort_keys=True).encode('utf-8')).hexdigest()
    
    combined = f"{sha256}_{data_sha256}_{schema_hash}_{config_hash}_8.2853_3.9113"
    expected_fingerprint = hashlib.sha256(combined.encode('utf-8')).hexdigest()
    
    assert manifest.get('release_fingerprint') == expected_fingerprint

def test_no_candidate_b_features(model):
    if hasattr(model, "feature_names_in_"):
        features = list(model.feature_names_in_)
    else:
        features = list(model.steps[-1][1].feature_names_in_)
    assert 'growth_regime_num' not in features
    assert 'stress_regime_num' not in features

def test_manifest_contains_required_fields(manifest):
    required_fields = [
        "release_id", "release_timestamp", "model_path", "model_md5", "model_sha256",
        "dataset_path", "dataset_md5", "dataset_sha256", "model_class", "feature_count",
        "feature_names", "hyperparameters", "validation_rmse", "test_rmse",
        "train_period", "validation_period", "test_period", "target_column",
        "candidate_b_status", "test_count", "test_passed", "test_failed",
        "production_status", "release_fingerprint", "source_audit_phase"
    ]
    for field in required_fields:
        assert field in manifest

def test_freeze_script_does_not_modify(manifest):
    # This implicitly verified because model and raw hashes matched above
    # and they still match their known exact historical state.
    assert manifest.get('dataset_md5') == "8ac7e0b2bf09fbe89289f82d0c7cf25e"

def test_release_manifest_hashes_correspond(manifest):
    model_md5, model_sha256 = calculate_hashes(MODEL_PATH)
    assert manifest.get('model_md5') == model_md5
    assert manifest.get('model_sha256') == model_sha256
    
    data_md5, data_sha256 = calculate_hashes(DATA_RAW_PATH)
    assert manifest.get('dataset_md5') == data_md5
    assert manifest.get('dataset_sha256') == data_sha256
