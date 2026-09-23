import os
import json
import hashlib
import joblib
import pandas as pd
from typing import Dict, Any, List, Optional
import structlog

logger = structlog.get_logger()

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../'))
MANIFEST_PATH = os.path.join(BASE_DIR, 'models/phase11/phase11_production_release_manifest.json')
MODEL_PATH = os.path.join(BASE_DIR, 'models/phase11/best_t1_gdp_growth_model.joblib')

class FrozenModelVerificationError(Exception):
    """Raised when the frozen model violates integrity constraints."""
    pass

class Phase11ProductionModel:
    _instance = None
    _model = None
    _manifest = None
    _feature_names = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        logger.info("Initializing Phase 11 Production Model Loader...")
        
        # 1. Load and verify manifest
        if not os.path.exists(MANIFEST_PATH):
            raise FrozenModelVerificationError(f"Release manifest not found at {MANIFEST_PATH}")
            
        with open(MANIFEST_PATH, 'r') as f:
            self._manifest = json.load(f)
            
        expected_release_id = "ECONOSPHERE-PHASE11-PROD-2026-08-21"
        if self._manifest.get("release_id") != expected_release_id:
            raise FrozenModelVerificationError(f"Invalid release ID. Expected {expected_release_id}")
            
        if self._manifest.get("production_status") != "PHASE 11 PRODUCTION FROZEN":
            raise FrozenModelVerificationError("Production status is not frozen.")
            
        # 2. Hash verification
        if not os.path.exists(MODEL_PATH):
            raise FrozenModelVerificationError(f"Model file not found at {MODEL_PATH}")
            
        md5_hash = hashlib.md5()
        sha256_hash = hashlib.sha256()
        with open(MODEL_PATH, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                md5_hash.update(byte_block)
                sha256_hash.update(byte_block)
                
        actual_md5 = md5_hash.hexdigest()
        actual_sha256 = sha256_hash.hexdigest()
        
        if actual_md5 != self._manifest.get("model_md5"):
            raise FrozenModelVerificationError(f"Model MD5 mismatch. Actual: {actual_md5}, Expected: {self._manifest.get('model_md5')}")
            
        if actual_sha256 != self._manifest.get("model_sha256"):
            raise FrozenModelVerificationError(f"Model SHA-256 mismatch. Actual: {actual_sha256}, Expected: {self._manifest.get('model_sha256')}")

        # 3. Load model
        loaded_model = joblib.load(MODEL_PATH)
        
        # 4. Verify model configuration
        model_class = loaded_model.__class__.__name__
        if model_class == "Pipeline":
            regressor = loaded_model.steps[-1][1]
            regressor_class = regressor.__class__.__name__
        else:
            regressor = loaded_model
            regressor_class = model_class
            
        if regressor_class != "HistGradientBoostingRegressor":
            raise FrozenModelVerificationError(f"Unexpected model class: {regressor_class}")

        params = regressor.get_params()
        expected_params = self._manifest.get("hyperparameters", {})
        for k, v in expected_params.items():
            if params.get(k) != v:
                raise FrozenModelVerificationError(f"Hyperparameter mismatch for {k}. Expected {v}, got {params.get(k)}")

        # Extract features
        if hasattr(loaded_model, "feature_names_in_"):
            actual_features = list(loaded_model.feature_names_in_)
        elif hasattr(regressor, "feature_names_in_"):
            actual_features = list(regressor.feature_names_in_)
        else:
            raise FrozenModelVerificationError("Could not extract feature names from model.")

        if len(actual_features) != 31:
            raise FrozenModelVerificationError(f"Expected 31 features, found {len(actual_features)}")
            
        expected_features = self._manifest.get("feature_names", [])
        if actual_features != expected_features:
            raise FrozenModelVerificationError("Feature schema mismatch between model and manifest.")
            
        # Ensure target and experimental features are absent
        banned = ['gdp_growth_next_year', 'target', 't1_gdp_growth', 'target_year', 'growth_regime_num', 'stress_regime_num']
        for b in banned:
            if b in actual_features:
                raise FrozenModelVerificationError(f"Banned feature {b} found in production model schema.")

        self._model = loaded_model
        self._feature_names = actual_features
        logger.info("Phase 11 Production Model successfully loaded and verified.")

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "model_version": self._manifest.get("release_id"),
            "status": self._manifest.get("production_status"),
            "model_class": self._manifest.get("model_class"),
            "feature_count": self._manifest.get("feature_count"),
            "validation_rmse": self._manifest.get("validation_rmse"),
            "test_rmse": self._manifest.get("test_rmse"),
            "release_fingerprint": self._manifest.get("release_fingerprint")
        }
        
    def predict(self, input_features: Dict[str, float]) -> float:
        """
        Perform inference.
        input_features must be a dictionary mapped exactly to the 31 features.
        """
        # Construct DataFrame in the exact expected feature order
        row = []
        for fn in self._feature_names:
            if fn not in input_features:
                raise ValueError(f"Missing required feature: {fn}")
            row.append(input_features[fn])
            
        df = pd.DataFrame([row], columns=self._feature_names)
        
        # Inference
        prediction = self._model.predict(df)[0]
        return float(prediction)

# Singleton instance
production_model = Phase11ProductionModel()
