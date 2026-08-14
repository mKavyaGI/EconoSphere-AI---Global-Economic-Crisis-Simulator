"""Experiment tracking system for structured logging and local persistence of ML training runs.

Designed to eliminate heavy third-party tracking server dependencies (like MLFlow) by saving
reproducible audit records directly to structured local JSON storage.
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Union
from pydantic import BaseModel as PydanticModel, Field


class Experiment(PydanticModel):
    """Structured immutable or historical record representing a unique ML model training trial."""

    experiment_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model_name: str
    dataset_name: str
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    training_time_seconds: float = 0.0
    random_seed: Union[int, None] = None
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    notes: str = ""

    def save(self, directory: Union[str, Path]) -> Path:
        """Serializes and writes this experiment record as an independent JSON artifact on disk.

        Args:
            directory: Directory folder path where experimental run JSON files are stored.

        Returns:
            The complete filesystem file Path of the created experiment artifact.
        """
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        filename = f"exp_{self.experiment_id}.json"
        file_path = dir_path / filename

        data = self.model_dump()
        with open(file_path, "w", encoding="utf-8") as fp:
            json.dump(data, fp, indent=2, sort_keys=True)

        return file_path

    @classmethod
    def load_from_file(cls, file_path: Union[str, Path]) -> "Experiment":
        """Restores a saved experiment instance directly from a JSON artifact file.

        Args:
            file_path: Source path to the JSON experiment file.

        Returns:
            An instantiated Experiment data object.

        Raises:
            FileNotFoundError: If the specified record file path does not exist.
        """
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"Experiment record file not found at path: {p}")
        with open(p, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        return cls.model_validate(data)


class ExperimentTracker:
    """Local storage manager and analytical querying registry for ML training experiments."""

    def __init__(self, storage_directory: Union[str, Path] = "notebooks/experiments") -> None:
        """Initializes the ExperimentTracker with a dedicated filesystem storage path.

        Args:
            storage_directory: Folder path where training trial JSON logs are retained.
        """
        self.storage_dir = Path(storage_directory)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def log_experiment(
        self,
        model_name: str,
        dataset_name: str,
        hyperparameters: dict[str, Any],
        metrics: dict[str, float],
        training_time_seconds: float = 0.0,
        random_seed: Union[int, None] = None,
        notes: str = "",
    ) -> Experiment:
        """Constructs, validates, and persists a training trial record to storage.

        Args:
            model_name: Identifier of the trained machine learning architecture.
            dataset_name: Designation or filename of the underlying training dataset.
            hyperparameters: Parameter configuration map used during optimization.
            metrics: Dictionary of evaluated predictive evaluation metric scores.
            training_time_seconds: Recorded training processing duration in seconds.
            random_seed: Random seed parameter ensuring trial reproducibility.
            notes: Optional descriptive notes or analytical observations.

        Returns:
            The recorded and persisted Experiment model instance.
        """
        exp = Experiment(
            model_name=model_name,
            dataset_name=dataset_name,
            hyperparameters=hyperparameters,
            metrics=metrics,
            training_time_seconds=training_time_seconds,
            random_seed=random_seed,
            notes=notes,
        )
        exp.save(self.storage_dir)
        return exp

    def list_experiments(self, model_name: Union[str, None] = None) -> list[Experiment]:
        """Scans the storage directory and returns historical experiment runs.

        Args:
            model_name: Optional filter to return exclusively runs matching this model identifier.

        Returns:
            A list of historical Experiment instances ordered by timestamp.
        """
        if not self.storage_dir.exists():
            return []
        experiments: list[Experiment] = []
        for p in self.storage_dir.glob("exp_*.json"):
            try:
                exp = Experiment.load_from_file(p)
                if model_name is None or exp.model_name == model_name:
                    experiments.append(exp)
            except Exception:
                continue

        experiments.sort(key=lambda x: x.timestamp, reverse=True)
        return experiments

    def get_best_experiment(self, metric_name: str, minimize: bool = True, model_name: Union[str, None] = None) -> Union[Experiment, None]:
        """Identifies and returns the historically optimal experiment run evaluated against a specified metric.

        Args:
            metric_name: Target metric identifier (e.g., 'RMSE', 'R2', 'MAPE').
            minimize: If True, searches for the minimal metric score (e.g., RMSE); if False, maximizes (e.g., R2).
            model_name: Optional model identifier filter.

        Returns:
            The optimal historical Experiment instance, or None if no matching records exist.
        """
        runs = self.list_experiments(model_name=model_name)
        valid_runs = [r for r in runs if metric_name in r.metrics]
        if not valid_runs:
            return None

        best = sorted(
            valid_runs,
            key=lambda x: x.metrics[metric_name],
            reverse=not minimize
        )[0]
        return best
