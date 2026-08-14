"""Reusable model deserialization, structural loading, and metadata version validation utilities."""

import json
import pickle
import sys
from pathlib import Path
from typing import Any, Union

try:
    import joblib
except ImportError:
    joblib = None


class ModelLoadError(Exception):
    """Exception raised during model artifact deserialization or metadata version mismatch."""
    pass


def load_model(
    file_path: Union[str, Path],
    expected_method: Union[str, None] = None,
    expected_model_type: Union[str, None] = None,
    validate_python_version: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """Loads a serialized machine learning model from disk and verifies compatibility against stored metadata.

    Args:
        file_path: Path to the previously serialized model file artifact.
        expected_method: Optional serialization format constraint ('joblib' or 'pickle') to enforce.
        expected_model_type: Optional class name string (e.g., 'ForecastingModel') to validate against metadata.
        validate_python_version: If True, raises errors if the saving Python major/minor version differs from current runtime.

    Returns:
        A tuple containing (deserialized_model_instance, metadata_dictionary).

    Raises:
        ModelLoadError: If files are unreadable, serialization libraries are missing, or compatibility checks fail.
    """
    target_path = Path(file_path)
    if not target_path.exists():
        raise ModelLoadError(f"Target model file not found at path: {target_path}")

    # Attempt to load and validate companion metadata if available
    meta_path = Path(f"{target_path}.metadata.json")
    metadata: dict[str, Any] = {}
    serialization_method = expected_method if expected_method is not None else "joblib"

    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as fp:
                metadata = json.load(fp)
                if expected_method is None and "serialization_method" in metadata:
                    serialization_method = str(metadata["serialization_method"])
        except Exception as err:
            raise ModelLoadError(f"Failed to read companion metadata file {meta_path}: {err}") from err

        # Perform runtime validation checks against persisted metadata
        if expected_model_type is not None:
            saved_type = metadata.get("model_type", "")
            if saved_type and saved_type != expected_model_type:
                raise ModelLoadError(
                    f"Model type mismatch: artifact contains '{saved_type}', but expected '{expected_model_type}'."
                )

        if validate_python_version:
            saved_py = metadata.get("python_version", "").split()[0]
            current_py = f"{sys.version_info.major}.{sys.version_info.minor}"
            if saved_py and not saved_py.startswith(current_py):
                raise ModelLoadError(
                    f"Python runtime version mismatch: model saved with {saved_py}, "
                    f"but current execution runtime is {sys.version}."
                )

    # Deserialize and load model from disk
    try:
        if serialization_method == "joblib":
            if joblib is None:
                raise ModelLoadError("joblib library unavailable in current runtime environment.")
            model = joblib.load(target_path)
        elif serialization_method == "pickle":
            with open(target_path, "rb") as fp:
                model = pickle.load(fp)
        else:
            raise ModelLoadError(f"Unsupported serialization method specified: '{serialization_method}'")
    except Exception as err:
        raise ModelLoadError(f"Failed to deserialize model artifact from {target_path}: {err}") from err

    return model, metadata
