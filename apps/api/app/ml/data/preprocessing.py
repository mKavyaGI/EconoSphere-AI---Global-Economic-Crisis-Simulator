"""Reusable, framework-agnostic data preprocessing, transformation, and feature selection utilities."""

from typing import Any, Union
import numpy as np
import pandas as pd


class MissingValueImputer:
    """Configurable imputer for handling missing observational values in tabular datasets."""

    def __init__(
        self,
        strategy: str = "mean",
        fill_value: Any = None,
        columns: Union[list[str], None] = None,
    ) -> None:
        """Initializes the missing value imputer.

        Args:
            strategy: Imputation approach ('mean', 'median', 'mode', 'constant', 'drop', 'forward').
            fill_value: Default constant replacement value when strategy is 'constant'.
            columns: Specific column subsets to transform. If None, transforms all columns.

        Raises:
            ValueError: If an unaccepted strategy string is specified.
        """
        valid_strategies = {"mean", "median", "mode", "constant", "drop", "forward"}
        if strategy not in valid_strategies:
            raise ValueError(f"Unsupported imputation strategy: {strategy}. Expected one of {valid_strategies}")
        self.strategy = strategy
        self.fill_value = fill_value
        self.columns = columns
        self._statistics: dict[str, Any] = {}

    def fit(self, df: pd.DataFrame) -> "MissingValueImputer":
        """Calculates and stores imputation statistics across target training columns.

        Args:
            df: Training dataset containing potential missing values.

        Returns:
            Self with populated replacement statistics.
        """
        target_cols = self.columns if self.columns is not None else df.columns
        for col in target_cols:
            if col not in df.columns:
                continue
            series = df[col]
            if self.strategy == "mean" and pd.api.types.is_numeric_dtype(series):
                self._statistics[col] = series.mean()
            elif self.strategy == "median" and pd.api.types.is_numeric_dtype(series):
                self._statistics[col] = series.median()
            elif self.strategy == "mode":
                mode_val = series.mode()
                if not mode_val.empty:
                    self._statistics[col] = mode_val.iloc[0]
            elif self.strategy == "constant":
                self._statistics[col] = self.fill_value
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies missing value imputation onto a provided dataset.

        Args:
            df: Target DataFrame to transform.

        Returns:
            A transformed DataFrame with imputed values.
        """
        result = df.copy()
        if self.strategy == "drop":
            subset = self.columns if self.columns is not None else df.columns
            return result.dropna(subset=subset).reset_index(drop=True)
        if self.strategy == "forward":
            subset = self.columns if self.columns is not None else df.columns
            result[subset] = result[subset].ffill().bfill()
            return result

        target_cols = self.columns if self.columns is not None else df.columns
        for col in target_cols:
            if col in result.columns and col in self._statistics:
                result[col] = result[col].fillna(self._statistics[col])
        return result

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fits numerical statistics and transforms the dataframe in a single execution step.

        Args:
            df: DataFrame to compute metrics from and impute.

        Returns:
            The transformed pandas DataFrame.
        """
        return self.fit(df).transform(df)


class StandardScaler:
    """Standardizes numerical dataset features to zero mean and unit variance using NumPy."""

    def __init__(self, columns: Union[list[str], None] = None, epsilon: float = 1e-8) -> None:
        """Initializes the Standard Scaler.

        Args:
            columns: Specific feature column names to scale. Defaults to all numerical columns.
            epsilon: Small scalar added to feature standard deviation to prevent zero division.
        """
        self.columns = columns
        self.epsilon = epsilon
        self._means: dict[str, float] = {}
        self._stds: dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> "StandardScaler":
        """Calculates feature means and standard deviations from training data.

        Args:
            df: Training dataset.

        Returns:
            Self with computed parameter distributions.
        """
        target_cols = self.columns if self.columns is not None else df.select_dtypes(include=[np.number]).columns
        for col in target_cols:
            if col in df.columns:
                arr = df[col].to_numpy(dtype=np.float64, na_value=np.nan)
                self._means[col] = float(np.nanmean(arr))
                std_val = float(np.nanstd(arr))
                self._stds[col] = std_val if std_val > 0 else 1.0
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies Z-score standard normalization to target features in the dataset.

        Args:
            df: Input DataFrame to standardize.

        Returns:
            Transformed DataFrame with zero mean and unit variance across scaled columns.
        """
        result = df.copy()
        for col, mean_val in self._means.items():
            if col in result.columns:
                std_val = self._stds[col]
                result[col] = (result[col] - mean_val) / (std_val + self.epsilon)
        return result

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fits standardization parameters and transforms data sequentially.

        Args:
            df: Input training dataset.

        Returns:
            The standard-scaled DataFrame.
        """
        return self.fit(df).transform(df)


class MinMaxScaler:
    """Transforms numerical dataset features to bounded minimum and maximum value ranges."""

    def __init__(
        self,
        feature_range: tuple[float, float] = (0.0, 1.0),
        columns: Union[list[str], None] = None,
        epsilon: float = 1e-8,
    ) -> None:
        """Initializes the MinMax Scaler.

        Args:
            feature_range: Tuple indicating the lower and upper bounds of transformed values.
            columns: Explicit numerical feature columns to bound. Defaults to all numerical columns.
            epsilon: Small float stabilizer to protect against identical min/max bounds.
        """
        self.feature_range = feature_range
        self.columns = columns
        self.epsilon = epsilon
        self._mins: dict[str, float] = {}
        self._maxs: dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> "MinMaxScaler":
        """Records absolute empirical minimums and maximums across targeted training features.

        Args:
            df: Training dataset.

        Returns:
            Self with estimated range boundaries.
        """
        target_cols = self.columns if self.columns is not None else df.select_dtypes(include=[np.number]).columns
        for col in target_cols:
            if col in df.columns:
                arr = df[col].to_numpy(dtype=np.float64, na_value=np.nan)
                self._mins[col] = float(np.nanmin(arr))
                self._maxs[col] = float(np.nanmax(arr))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rescales targeted numerical columns to fall within the specified boundary bounds.

        Args:
            df: DataFrame to transform.

        Returns:
            Rescaled pandas DataFrame.
        """
        result = df.copy()
        min_out, max_out = self.feature_range
        for col, min_in in self._mins.items():
            if col in result.columns:
                max_in = self._maxs[col]
                range_in = np.maximum(max_in - min_in, self.epsilon)
                normalized = (result[col] - min_in) / range_in
                result[col] = normalized * (max_out - min_out) + min_out
        return result

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates minimum and maximum parameters and immediately rescales the dataset.

        Args:
            df: Training dataframe to compute boundaries from and rescale.

        Returns:
            Rescaled pandas DataFrame.
        """
        return self.fit(df).transform(df)


class CategoricalEncoder:
    """Converts qualitative string categorical columns into numerical structural mappings."""

    def __init__(self, method: str = "onehot", columns: Union[list[str], None] = None) -> None:
        """Initializes the categorical encoder.

        Args:
            method: Encoding technique to employ ('onehot' or 'label').
            columns: Specific column subset to encode. If None, auto-detects object/category dtypes.

        Raises:
            ValueError: If an unrecognizably formatted method argument is passed.
        """
        if method not in {"onehot", "label"}:
            raise ValueError(f"Unsupported categorical encoding method: {method}")
        self.method = method
        self.columns = columns
        self._label_maps: dict[str, dict[Any, int]] = {}
        self._categories: dict[str, list[str]] = {}

    def fit(self, df: pd.DataFrame) -> "CategoricalEncoder":
        """Identifies unique categorical vocabularies and prepares structural mapping indexes.

        Args:
            df: Training dataset containing categorical features.

        Returns:
            Self with recorded category vocabularies.
        """
        target_cols = self.columns if self.columns is not None else df.select_dtypes(include=["object", "category"]).columns
        for col in target_cols:
            if col not in df.columns:
                continue
            unique_vals = df[col].dropna().unique()
            if self.method == "label":
                self._label_maps[col] = {val: idx for idx, val in enumerate(sorted(unique_vals))}
            elif self.method == "onehot":
                self._categories[col] = [str(val) for val in sorted(unique_vals)]
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Translates categorical string observations into numerical label or one-hot structures.

        Args:
            df: DataFrame containing target categorical observations.

        Returns:
            Transformed DataFrame containing numeric categorical representations.
        """
        result = df.copy()
        if self.method == "label":
            for col, mapping in self._label_maps.items():
                if col in result.columns:
                    result[col] = result[col].map(mapping).fillna(-1).astype(int)
        elif self.method == "onehot":
            for col, categories in self._categories.items():
                if col in result.columns:
                    for cat in categories:
                        new_col = f"{col}_{cat}"
                        result[new_col] = (result[col].astype(str) == cat).astype(int)
                    result.drop(columns=[col], inplace=True)
        return result

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Learns categorical mappings and translates features in a unified operation.

        Args:
            df: Input dataframe to fit and transform.

        Returns:
            The numerically encoded pandas DataFrame.
        """
        return self.fit(df).transform(df)


def parse_datetime(
    df: pd.DataFrame,
    column: str,
    extract_features: bool = True,
    drop_original: bool = False,
) -> pd.DataFrame:
    """Parses text or integer timestamps into datetime format and optionally extracts temporal signals.

    Args:
        df: Input DataFrame containing timestamp observations.
        column: Name of the timestamp or date column.
        extract_features: If True, derives numeric year, month, day, and dayofweek feature columns.
        drop_original: If True, removes the original datetime column after extraction.

    Returns:
        Transformed DataFrame containing parsed datetime structures and extracted temporal signals.

    Raises:
        ValueError: If the specified column is absent from the dataset.
    """
    if column not in df.columns:
        raise ValueError(f"Date column '{column}' not found in DataFrame.")
    result = df.copy()
    result[column] = pd.to_datetime(result[column])
    if extract_features:
        result[f"{column}_year"] = result[column].dt.year
        result[f"{column}_month"] = result[column].dt.month
        result[f"{column}_day"] = result[column].dt.day
        result[f"{column}_dayofweek"] = result[column].dt.dayofweek
    if drop_original and extract_features:
        result.drop(columns=[column], inplace=True)
    return result


def select_features(
    df: pd.DataFrame,
    include_columns: Union[list[str], None] = None,
    exclude_columns: Union[list[str], None] = None,
    numeric_only: bool = False,
    variance_threshold: float = 0.0,
) -> pd.DataFrame:
    """Filters dataset columns using explicit selection rules, data type constraints, or variance thresholds.

    Args:
        df: Input DataFrame containing candidate feature columns.
        include_columns: Explicit list of feature column names to preserve.
        exclude_columns: List of specific column names to omit.
        numeric_only: If True, discards all non-numerical features.
        variance_threshold: Drops numerical columns displaying empirical variance below this threshold.

    Returns:
        A pruned pandas DataFrame composed solely of features satisfying selection constraints.
    """
    result = df.copy()
    if include_columns is not None:
        valid_cols = [c for c in include_columns if c in result.columns]
        result = result[valid_cols]
    if exclude_columns is not None:
        drop_cols = [c for c in exclude_columns if c in result.columns]
        result = result.drop(columns=drop_cols)
    if numeric_only:
        result = result.select_dtypes(include=[np.number])
    if variance_threshold > 0.0:
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            var = float(np.nanvar(result[col].to_numpy()))
            if var <= variance_threshold:
                result.drop(columns=[col], inplace=True)
    return result
