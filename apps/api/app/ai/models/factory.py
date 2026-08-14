"""
Dependency Injection factory for AI models.
Allows seamless hot-swapping between XGBoost, Transformers, GNNs, LightGBM, and RL without modifying endpoints.
"""
from app.ai.interfaces import AbstractModel
from app.ai.models.wrappers import (
    XGBoostModel,
    LightGBMModel,
    TransformerModel,
    GNNModel,
    EconometricVARModel,
    ReinforcementLearningModel
)

class ModelFactory:
    """
    Factory for producing concrete ML inference wrapper instances by name or architecture class.
    """
    _model_registry_map = {
        "XGBOOST": XGBoostModel,
        "LIGHTGBM": LightGBMModel,
        "TRANSFORMER": TransformerModel,
        "GNN": GNNModel,
        "VAR": EconometricVARModel,
        "RL": ReinforcementLearningModel,
        "DEFAULT": XGBoostModel
    }

    @classmethod
    def get_model(cls, model_type: str = "XGBOOST") -> AbstractModel:
        key = model_type.upper().strip()
        model_cls = cls._model_registry_map.get(key, cls._model_registry_map["DEFAULT"])
        return model_cls()
