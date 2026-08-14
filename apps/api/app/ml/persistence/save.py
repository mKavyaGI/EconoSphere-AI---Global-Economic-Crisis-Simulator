"""Reusable model artifact serialization and structured metadata persistence utilities."""

import json
import pickle
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Union

try:
    import joblib
except ImportError:
    joblib = None


class ModelSaveError(Exception):
    """Exception raised when model serialization or metadata disk persistence fails."""
    pass


def save_model(
    model: Any,
    file_path: Union[str, Path],
    metadata: Union[dict[str, Any], None] = None,
    serialization_method: str = "joblib",
) -> Path:
    """Serializes a machine learning model to disk alongside an accompanying structured JSON metadata artifact.

    Args:
        model: Model object instance to serialize and store.
        file_path: Destination filesystem file path for model storage.
        metadata: Optional dictionary of descriptive metrics, hyperparameter configs, or provenance flags.
        serialization_method: Target archiving library to use ('joblib' or 'pickle').

    Returns:
        The validated Path object pointing to the written model artifact.

    Raises:
        ModelSaveError: If serialization encounters errors or an unsupported serialization method is requested.
    """
    target_path = Path(file_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    valid_methods = {"joblib", "pickle"}
    if serialization_method not in valid_methods:
        raise ModelSaveError(
            f"Unsupported serialization method '{serialization_method}'. "
            f"Expected one of {valid_methods}"
        )

    try:
        if serialization_method == "joblib":
            if joblib is None:
                raise ModelSaveError("joblib package is not available in current Python environment.")
            joblib.dump(model, target_path)
        elif serialization_method == "pickle":
            with open(target_path, "wb") as fp:
                pickle.dump(model, fp)
    except Exception as err:
        raise ModelSaveError(f"Failed to serialize model to {target_path}: {err}") from err

    # Construct and persist accompanying validation metadata artifact
    meta_path = Path(f"{target_path}.metadata.json")
    meta_content: dict[str, Any] = {
        "serialization_method": serialization_method,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "model_type": type(model).__name__,
        "model_module": type(model).__module__,
        "custom_metadata": metadata if metadata is not None else {},
    }

    try:
        with open(meta_path, "w", encoding="utf-8") as fp:
            json.dump(meta_content, fp, indent=2, sort_keys=True)
    except Exception as err:
        raise ModelSaveError(f"Failed to write companion metadata artifact to {meta_path}: {err}") from err

    return target_path
