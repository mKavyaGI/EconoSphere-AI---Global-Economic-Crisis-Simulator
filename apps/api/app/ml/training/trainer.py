"""Generic implementation of model training lifecycle workflows and evaluation execution."""

from pathlib import Path
from typing import Any, Union
from dataclasses import dataclass, field

from app.ml.base import BaseModel, BaseTrainer
from app.ml.metrics import compute_all_metrics, mae, rmse, mape, r2, smape


@dataclass
class TrainerConfig:
    """Structured configuration parameters governing generic training execution."""
    batch_size: int = 32
    max_epochs: int = 100
    learning_rate: float = 0.001
    random_seed: int = 42
    evaluate_on_train: bool = True
    custom_options: dict[str, Any] = field(default_factory=dict)


class Trainer(BaseTrainer):
    """Generic model trainer responsible for orchestrating dataset preparation, training, and evaluation.

    Encapsulates abstract workflow sequencing without coupling to specific numerical ML algorithm internals.
    """

    def __init__(self, model: BaseModel, config: Union[TrainerConfig, None] = None, **kwargs: Any) -> None:
        """Initializes the generic Trainer with a model and runtime configuration.

        Args:
            model: Concrete BaseModel instance to train and evaluate.
            config: Optional structured TrainerConfig object.
            **kwargs: Arbitrary keyword options merged into the trainer configuration.
        """
        super().__init__(model, **kwargs)
        self.trainer_config = config if config is not None else TrainerConfig(**kwargs)
        self.X_train: Any = None
        self.y_train: Any = None
        self.X_val: Any = None
        self.y_val: Any = None
        self.is_prepared: bool = False
        self.is_trained: bool = False

    def prepare(
        self,
        X_train: Any,
        y_train: Any,
        X_val: Any = None,
        y_val: Any = None,
        **kwargs: Any
    ) -> None:
        """Registers training and validation feature matrices and targets for optimization.

        Args:
            X_train: Feature matrix for initial model optimization.
            y_train: Target observations corresponding to X_train.
            X_val: Optional validation feature matrix for convergence evaluation.
            y_val: Optional validation targets corresponding to X_val.
            **kwargs: Extra data preparation or formatting options.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.is_prepared = True

    def train(self, *args: Any, **kwargs: Any) -> Any:
        """Executes model optimization against previously prepared training datasets.

        Args:
            *args: Optional positional training arguments passed to model fit.
            **kwargs: Optional runtime overrides passed to model fit.

        Returns:
            Training execution summary or return status from the model's fit method.

        Raises:
            RuntimeError: If train() is invoked prior to execute prepare().
        """
        if not self.is_prepared or self.X_train is None:
            raise RuntimeError("Trainer has not been prepared with data. Call prepare(X_train, y_train) first.")

        fit_kwargs = {**self.trainer_config.custom_options, **kwargs}
        result = self.model.fit(self.X_train, self.y_train, *args, **fit_kwargs)
        self.is_trained = True
        return result

    def evaluate(self, X: Any = None, y: Any = None, metrics: Union[list[str], None] = None, **kwargs: Any) -> dict[str, float]:
        """Evaluates predictive accuracy on designated test or validation observations using numerical metrics.

        Args:
            X: Test feature matrix. If None, falls back to the prepared validation set (X_val).
            y: Test target values. If None, falls back to the prepared validation set (y_val).
            metrics: Explicit list of evaluation metrics to calculate (e.g., ['RMSE', 'MAPE']).
            **kwargs: Extra prediction runtime options.

        Returns:
            A dictionary mapping metric names to evaluated scalar scores.

        Raises:
            RuntimeError: If evaluate() is called before the model is trained or if evaluation data is missing.
        """
        if not self.is_trained:
            raise RuntimeError("Model has not been trained yet. Call train() prior to evaluation.")

        eval_X = X if X is not None else self.X_val
        eval_y = y if y is not None else self.y_val

        if eval_X is None or eval_y is None:
            raise RuntimeError("No evaluation feature inputs or targets provided to evaluate().")

        predictions = self.predict(eval_X, **kwargs)

        if metrics is None:
            return compute_all_metrics(eval_y, predictions)

        metric_map = {
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape,
            "R2": r2,
            "SMAPE": smape,
        }

        results: dict[str, float] = {}
        for m in metrics:
            m_upper = m.upper()
            if m_upper in metric_map:
                results[m_upper] = metric_map[m_upper](eval_y, predictions)
            else:
                raise ValueError(f"Unsupported evaluation metric required: '{m}'. Supported metrics: {list(metric_map.keys())}")

        return results

    def predict(self, X: Any, **kwargs: Any) -> Any:
        """Runs model inference on feature observations to emit predictive estimates.

        Args:
            X: Input features for inference.
            **kwargs: Prediction parameters forwarded to model predict.

        Returns:
            Predicted target outputs from the underlying model.

        Raises:
            RuntimeError: If inference is attempted before training execution completes.
        """
        if not self.is_trained:
            raise RuntimeError("Model has not been trained yet. Call train() prior to calling predict().")
        return self.model.predict(X, **kwargs)

    def save(self, path: Union[str, Path]) -> None:
        """Persists the managed trained model instance to the specified filesystem path.

        Args:
            path: Target filesystem path for model persistence.
        """
        self.model.save(path)

    def load(self, path: Union[str, Path]) -> None:
        """Loads a previously persisted model state from disk into the managed instance.

        Args:
            path: Source filesystem path of the saved model.
        """
        self.model.load(path)
        self.is_trained = True
