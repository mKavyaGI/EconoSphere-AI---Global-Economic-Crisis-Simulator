"""Reusable evaluation metrics implemented entirely with NumPy without external dependencies.

Includes MAE, RMSE, MAPE, R², and SMAPE calculations designed specifically for evaluating
predictive quality in time-series forecasting and regression models.
"""

from typing import Any
import numpy as np


def _to_numpy(data: Any) -> np.ndarray:
    """Converts diverse array-like data input types into numerical NumPy ndarrays.

    Args:
        data: Array, list, tuple, or pandas Series/DataFrame representing numerical values.

    Returns:
        A formatted 1D or 2D numpy array of float64 type.
    """
    if not isinstance(data, np.ndarray):
        arr = np.array(data, dtype=np.float64)
    else:
        arr = data.astype(np.float64, copy=False)
    return arr


def mae(y_true: Any, y_pred: Any) -> float:
    """Calculates Mean Absolute Error (MAE).

    Args:
        y_true: Array-like ground truth target observations.
        y_pred: Array-like model prediction estimations.

    Returns:
        The non-negative floating-point MAE score.
    """
    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred)
    return float(np.mean(np.abs(yt - yp)))


def rmse(y_true: Any, y_pred: Any) -> float:
    """Calculates Root Mean Squared Error (RMSE).

    Args:
        y_true: Array-like ground truth target observations.
        y_pred: Array-like model prediction estimations.

    Returns:
        The non-negative floating-point RMSE score.
    """
    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred)
    return float(np.sqrt(np.mean(np.square(yt - yp))))


def mape(y_true: Any, y_pred: Any, epsilon: float = 1e-8) -> float:
    """Calculates Mean Absolute Percentage Error (MAPE).

    Args:
        y_true: Array-like ground truth target observations.
        y_pred: Array-like model prediction estimations.
        epsilon: Small constant added to the denominator to prevent division by zero.

    Returns:
        The MAPE evaluation score expressed as a numerical percentage (0.0 to 100.0+).
    """
    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred)
    denominator = np.maximum(np.abs(yt), epsilon)
    return float(np.mean(np.abs(yt - yp) / denominator) * 100.0)


def r2(y_true: Any, y_pred: Any, epsilon: float = 1e-8) -> float:
    """Calculates Coefficient of Determination ($R^2$ score).

    Args:
        y_true: Array-like ground truth target observations.
        y_pred: Array-like model prediction estimations.
        epsilon: Small threshold to protect against division by zero on invariant targets.

    Returns:
        The scalar R² value (with 1.0 being a perfect regression match).
    """
    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred)
    ss_res = np.sum(np.square(yt - yp))
    ss_tot = np.sum(np.square(yt - np.mean(yt)))
    if ss_tot < epsilon:
        return 0.0 if ss_res >= epsilon else 1.0
    return float(1.0 - (ss_res / ss_tot))


def smape(y_true: Any, y_pred: Any, epsilon: float = 1e-8) -> float:
    """Calculates Symmetric Mean Absolute Percentage Error (SMAPE).

    Args:
        y_true: Array-like ground truth target observations.
        y_pred: Array-like model prediction estimations.
        epsilon: Small numerical stabilizer applied to the denominator.

    Returns:
        The SMAPE evaluation score expressed as a percentage bounded between 0.0% and 200.0%.
    """
    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred)
    numerator = np.abs(yt - yp)
    denominator = (np.abs(yt) + np.abs(yp)) / 2.0
    denominator = np.maximum(denominator, epsilon)
    return float(np.mean(numerator / denominator) * 100.0)


def compute_all_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Computes a standardized dictionary of all available regression metrics.

    Args:
        y_true: Ground truth target array.
        y_pred: Model predicted output array.

    Returns:
        A dictionary mapping metric shorthand names to computed float scores.
    """
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "R2": r2(y_true, y_pred),
        "SMAPE": smape(y_true, y_pred),
    }
