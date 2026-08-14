"""Core architectural interfaces and abstract base classes for the ML framework.

Every machine learning model, dataset handler, and trainer in EconoSphere AI must inherit
from these primitives to ensure reliable and standardized training workflows.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, TypeVar, Union

T = TypeVar("T", bound="BaseModel")


class BaseDataset(ABC):
    """Abstract base class representing an EconoSphere AI machine learning dataset."""

    @abstractmethod
    def load_data(self) -> Any:
        """Loads and returns the underlying data structure from disk or memory.

        Returns:
            The raw or parsed dataset representation (e.g., pandas DataFrame).
        """
        pass

    @abstractmethod
    def get_features(self) -> Any:
        """Extracts and returns the feature inputs matrix (X).

        Returns:
            The feature matrix or DataFrame ready for preprocessing or modeling.
        """
        pass

    @abstractmethod
    def get_target(self) -> Any:
        """Extracts and returns the target variables (y) for supervised learning.

        Returns:
            The target array, series, or DataFrame.
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Returns the total sample count in the dataset.

        Returns:
            An integer count of rows or observations.
        """
        pass


class BaseModel(ABC):
    """Abstract base class for all predictive and forecasting machine learning models."""

    def __init__(self, **params: Any) -> None:
        """Initializes the model instance with arbitrary configuration hyperparameters.

        Args:
            **params: Hyperparameters and initialization configuration keyword arguments.
        """
        self.params = params

    @abstractmethod
    def fit(self, X: Any, y: Any = None, **kwargs: Any) -> "BaseModel":
        """Fits the model to the training features and targets.

        Args:
            X: Input feature observations.
            y: Target observation outputs corresponding to X.
            **kwargs: Additional algorithm-specific training parameters.

        Returns:
            Self (the trained model instance).
        """
        pass

    @abstractmethod
    def predict(self, X: Any, **kwargs: Any) -> Any:
        """Generates target value inferences for the provided feature inputs.

        Args:
            X: Input feature observations for inference.
            **kwargs: Inference runtime configuration options.

        Returns:
            Predicted output arrays or DataFrames.
        """
        pass

    def get_params(self) -> dict[str, Any]:
        """Retrieves the parameter configuration of the model.

        Returns:
            A dictionary mapping hyperparameter names to their assigned values.
        """
        return self.params.copy()

    def set_params(self, **params: Any) -> "BaseModel":
        """Updates the hyperparameter configuration of the model instance.

        Args:
            **params: Keyword arguments representing parameter values to update.

        Returns:
            Self with updated parameters.
        """
        self.params.update(params)
        return self

    @abstractmethod
    def save(self, path: Union[str, Path]) -> None:
        """Persists the trained model state and parameters to disk.

        Args:
            path: Destination filesystem file path.
        """
        pass

    @abstractmethod
    def load(self, path: Union[str, Path]) -> None:
        """Loads a persisted model state from disk into this instance.

        Args:
            path: Source filesystem file path containing the saved state.
        """
        pass


class BaseTrainer(ABC):
    """Abstract base class defining model orchestration, training, and evaluation lifecycle."""

    def __init__(self, model: BaseModel, **kwargs: Any) -> None:
        """Initializes the trainer with a model instance and execution configurations.

        Args:
            model: An instance of BaseModel to be trained and evaluated.
            **kwargs: Configuration parameters governing training lifecycle and optimization.
        """
        self.model = model
        self.config = kwargs

    @abstractmethod
    def prepare(self, *args: Any, **kwargs: Any) -> None:
        """Prepares training, validation, and test datasets prior to training execution.

        Args:
            *args: Positional dataset arguments or DataFrames.
            **kwargs: Keyword arguments for data transformations and splits.
        """
        pass

    @abstractmethod
    def train(self, *args: Any, **kwargs: Any) -> Any:
        """Executes the complete optimization and training routine for the model.

        Args:
            *args: Positional training arguments.
            **kwargs: Runtime overrides and callback parameters.

        Returns:
            A training history or loss convergence summary object.
        """
        pass

    @abstractmethod
    def evaluate(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        """Evaluates model prediction quality against test or validation datasets.

        Args:
            *args: Positional evaluation datasets.
            **kwargs: Metric selection or calculation overrides.

        Returns:
            A dictionary mapping metric identifiers (e.g., 'RMSE', 'MAPE') to float values.
        """
        pass

    @abstractmethod
    def predict(self, X: Any, **kwargs: Any) -> Any:
        """Executes prediction inference using the managed model instance.

        Args:
            X: Feature inputs to evaluate.
            **kwargs: Additional prediction runtime arguments.

        Returns:
            Model prediction output.
        """
        pass

    @abstractmethod
    def save(self, path: Union[str, Path]) -> None:
        """Saves the managed model alongside training checkpoints or configurations to disk.

        Args:
            path: Destination file path for saving.
        """
        pass

    @abstractmethod
    def load(self, path: Union[str, Path]) -> None:
        """Restores a saved model and trainer state from disk.

        Args:
            path: Source file path of the saved trainer state.
        """
        pass
