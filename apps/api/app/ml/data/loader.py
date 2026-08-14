"""Generic dataset loading infrastructure supporting CSV, Parquet, and DataFrame inputs."""

from pathlib import Path
from typing import Any, Union
import pandas as pd
from app.ml.base import BaseDataset


class DatasetLoadError(Exception):
    """Exception raised when a dataset fails to load from the designated source."""
    pass


class DatasetLoader(BaseDataset):
    """Generic dataset loader responsible for loading, validating, and extracting tabular data.

    Supports loading directly from CSV files, Parquet files, or in-memory pandas DataFrames,
    while fulfilling the BaseDataset architectural interface.
    """

    def __init__(
        self,
        source: Union[str, Path, pd.DataFrame],
        target_column: Union[str, list[str], None] = None,
        feature_columns: Union[list[str], None] = None,
        **kwargs: Any
    ) -> None:
        """Initializes the generic dataset loader.

        Args:
            source: A filepath to a CSV/Parquet file or an active pandas DataFrame.
            target_column: Optional column name(s) designating target variable(s).
            feature_columns: Optional explicit list of feature column names to extract.
            **kwargs: Additional keyword options forwarded to pandas read_csv or read_parquet.
        """
        self.source = source
        self.target_column = target_column
        self.feature_columns = feature_columns
        self.kwargs = kwargs
        self._dataframe: Union[pd.DataFrame, None] = None

    def load_data(self) -> pd.DataFrame:
        """Loads and caches the dataset into an internal pandas DataFrame.

        Returns:
            The complete dataset as a pandas DataFrame.

        Raises:
            DatasetLoadError: If the source format is unrecognized or file reading fails.
        """
        if isinstance(self.source, pd.DataFrame):
            self._dataframe = self.source.copy()
            return self._dataframe

        file_path = Path(self.source)
        if not file_path.exists():
            raise DatasetLoadError(f"Dataset source file does not exist at path: {file_path}")

        try:
            suffix = file_path.suffix.lower()
            if suffix == ".csv":
                self._dataframe = pd.read_csv(file_path, **self.kwargs)
            elif suffix in [".parquet", ".pq"]:
                self._dataframe = pd.read_parquet(file_path, **self.kwargs)
            else:
                raise DatasetLoadError(
                    f"Unsupported dataset format '{suffix}' for file: {file_path}. "
                    "Supported extensions are .csv and .parquet"
                )
            return self._dataframe
        except Exception as err:
            raise DatasetLoadError(f"Failed to load dataset from {file_path}: {err}") from err

    def _ensure_loaded(self) -> pd.DataFrame:
        """Ensures the dataframe has been loaded prior to performing operations."""
        if self._dataframe is None:
            return self.load_data()
        return self._dataframe

    def get_features(self) -> pd.DataFrame:
        """Extracts and returns the feature matrix from the loaded dataset.

        Returns:
            A pandas DataFrame comprising solely the feature columns.
        """
        df = self._ensure_loaded()
        if self.feature_columns is not None:
            return df[self.feature_columns].copy()
        if self.target_column is not None:
            targets = (
                [self.target_column]
                if isinstance(self.target_column, str)
                else self.target_column
            )
            feature_cols = [c for c in df.columns if c not in targets]
            return df[feature_cols].copy()
        return df.copy()

    def get_target(self) -> pd.DataFrame:
        """Extracts and returns the target observations from the loaded dataset.

        Returns:
            A pandas DataFrame containing only the target variable(s).

        Raises:
            ValueError: If no target_column was defined during initialization.
        """
        if self.target_column is None:
            raise ValueError("No target_column specified for this dataset instance.")
        df = self._ensure_loaded()
        if isinstance(self.target_column, str):
            return df[[self.target_column]].copy()
        return df[self.target_column].copy()

    def __len__(self) -> int:
        """Returns the total number of data rows in the loaded dataset.

        Returns:
            The integer number of samples.
        """
        df = self._ensure_loaded()
        return len(df)
