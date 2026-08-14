"""Model and trainer registry system for automatic component discovery and instantiation."""

from typing import Any, Callable, Type, Union
from app.ml.base import BaseModel, BaseTrainer


class ModelRegistryError(Exception):
    """Exception raised when an operation on the ModelRegistry encounters an error."""
    pass


class ModelRegistry:
    """Central registry enabling automatic registration and discoverability of ML models and trainers.

    Allows forecasting modules and pipelines to instantiate models dynamically by string identifiers
    without creating hardcoded importing dependencies across subsystems.
    """

    def __init__(self) -> None:
        """Initializes internal storage dictionaries for registered components."""
        self._models: dict[str, Type[BaseModel]] = {}
        self._trainers: dict[str, Type[BaseTrainer]] = {}

    def register(
        self,
        name: str,
        target: Union[Type[BaseModel], Type[BaseTrainer], Any],
        target_type: str = "model"
    ) -> Any:
        """Registers a model or trainer class under a unique identifier.

        Args:
            name: Unique string name representing the component.
            target: The class type or constructor to register.
            target_type: Either "model" or "trainer" indicating the component namespace.

        Returns:
            The registered target unchanged, enabling functional or decorator chaining.

        Raises:
            ModelRegistryError: If a component with the same name already exists in the namespace.
        """
        if target_type == "trainer":
            if name in self._trainers:
                raise ModelRegistryError(f"Trainer '{name}' is already registered in the registry.")
            self._trainers[name] = target
        else:
            if name in self._models:
                raise ModelRegistryError(f"Model '{name}' is already registered in the registry.")
            self._models[name] = target
        return target

    def register_model(self, name: str) -> Callable[[Type[BaseModel]], Type[BaseModel]]:
        """Decorator factory for registering a model class in the registry.

        Args:
            name: Unique identifier for the model.

        Returns:
            A class decorator that registers the targeted model class.
        """
        def decorator(cls: Type[BaseModel]) -> Type[BaseModel]:
            self.register(name, cls, target_type="model")
            return cls
        return decorator

    def register_trainer(self, name: str) -> Callable[[Type[BaseTrainer]], Type[BaseTrainer]]:
        """Decorator factory for registering a trainer class in the registry.

        Args:
            name: Unique identifier for the trainer.

        Returns:
            A class decorator that registers the targeted trainer class.
        """
        def decorator(cls: Type[BaseTrainer]) -> Type[BaseTrainer]:
            self.register(name, cls, target_type="trainer")
            return cls
        return decorator

    def get(self, name: str, target_type: str = "model") -> Union[Type[BaseModel], Type[BaseTrainer], Any]:
        """Retrieves a registered component class by its registration name.

        Args:
            name: The registration string identifier.
            target_type: Either "model" or "trainer".

        Returns:
            The registered class type.

        Raises:
            ModelRegistryError: If the requested name does not exist in the specified namespace.
        """
        storage = self._trainers if target_type == "trainer" else self._models
        if name not in storage:
            raise ModelRegistryError(
                f"Component '{name}' not found in registry (type: {target_type}). "
                f"Available entries: {list(storage.keys())}"
            )
        return storage[name]

    def list_models(self) -> list[str]:
        """Lists all model identifiers currently registered.

        Returns:
            A list of registered model names sorted alphabetically.
        """
        return sorted(list(self._models.keys()))

    def list_trainers(self) -> list[str]:
        """Lists all trainer identifiers currently registered.

        Returns:
            A list of registered trainer names sorted alphabetically.
        """
        return sorted(list(self._trainers.keys()))


# Global model registry instance for shared across the application lifecycle
default_registry = ModelRegistry()
