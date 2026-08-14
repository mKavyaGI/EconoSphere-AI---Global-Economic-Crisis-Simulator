"""Reusable data splitting routines supporting random, time-series, and walk-forward evaluations."""

from typing import Iterator, Union
import numpy as np
import pandas as pd


def random_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: Union[int, None] = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Randomly partitions a dataset into training, validation, and test subsets.

    Args:
        df: Input pandas DataFrame to partition.
        train_ratio: Proportion of samples assigned to the training split.
        val_ratio: Proportion of samples assigned to the validation split.
        test_ratio: Proportion of samples assigned to the testing split.
        random_seed: Random state seed for reproducible shuffling.

    Returns:
        A tuple containing (train_df, val_df, test_df) as independent DataFrames.

    Raises:
        ValueError: If split ratio totals deviate significantly from 1.0.
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0 (received sum: {total_ratio})")

    n = len(df)
    indices = np.arange(n)
    if random_seed is not None:
        np.random.seed(random_seed)
    np.random.shuffle(indices)

    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    return (
        df.iloc[train_idx].copy().reset_index(drop=True),
        df.iloc[val_idx].copy().reset_index(drop=True),
        df.iloc[test_idx].copy().reset_index(drop=True),
    )


def time_series_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    date_column: Union[str, None] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits sequential time-series data chronologically into train, validation, and test sets.

    Args:
        df: Input pandas DataFrame containing historical observation rows.
        train_ratio: Chronological fraction assigned to initial training.
        val_ratio: Subsequent fraction assigned to validation.
        test_ratio: Final chronological fraction assigned to evaluation testing.
        date_column: Optional timestamp column name to sort by prior to splitting.

    Returns:
        A tuple containing chronological splits (train_df, val_df, test_df).

    Raises:
        ValueError: If split ratios fail to equal 1.0 or date_column is missing.
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(f"Time-series split ratios must sum to 1.0 (received: {total_ratio})")

    if date_column is not None:
        if date_column not in df.columns:
            raise ValueError(f"Date column '{date_column}' not present in DataFrame columns.")
        data = df.sort_values(by=date_column, ascending=True).reset_index(drop=True)
    else:
        data = df.copy().reset_index(drop=True)

    n = len(data)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    return (
        data.iloc[:train_end].copy().reset_index(drop=True),
        data.iloc[train_end:val_end].copy().reset_index(drop=True),
        data.iloc[val_end:].copy().reset_index(drop=True),
    )


def walk_forward_split(
    df: pd.DataFrame,
    n_splits: int = 5,
    train_window: Union[int, None] = None,
    test_size: int = 1,
    date_column: Union[str, None] = None,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Generates expanding or rolling window walk-forward sequential evaluation splits.

    Ideal for backtesting time-series forecasting algorithms under operational conditions.

    Args:
        df: Time-ordered historical DataFrame.
        n_splits: Number of successive sequential training/testing fold pairs to emit.
        train_window: Fixed rolling sample size for training. If None, uses expanding window.
        test_size: Number of observation rows included in each forward test evaluation fold.
        date_column: Optional timestamp column to ensure correct chronological sorting.

    Returns:
        A list of tuples, where each element is a pair of (train_df, test_df) DataFrames.

    Raises:
        ValueError: If the dataset is too small to accommodate the required splits and windows.
    """
    if date_column is not None:
        data = df.sort_values(by=date_column, ascending=True).reset_index(drop=True)
    else:
        data = df.copy().reset_index(drop=True)

    total_rows = len(data)
    required_test_samples = n_splits * test_size
    min_train = train_window if train_window is not None else 1

    if total_rows < min_train + required_test_samples:
        raise ValueError(
            f"Dataset length ({total_rows}) insufficient for {n_splits} walk-forward splits "
            f"with train_window={min_train} and test_size={test_size}."
        )

    splits: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    # Work backwards from the end of the dataset to ensure latest observations are evaluated
    start_test_idx = total_rows - required_test_samples

    for idx in range(n_splits):
        current_test_start = start_test_idx + idx * test_size
        current_test_end = current_test_start + test_size

        if train_window is not None:
            train_start = current_test_start - train_window
        else:
            train_start = 0

        train_fold = data.iloc[train_start:current_test_start].copy().reset_index(drop=True)
        test_fold = data.iloc[current_test_start:current_test_end].copy().reset_index(drop=True)
        splits.append((train_fold, test_fold))

    return splits
