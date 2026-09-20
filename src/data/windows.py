from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.types import TimeSeriesDataset


@dataclass
class ForecastWindow:
    history: np.ndarray
    target: np.ndarray

    history_timestamps: pd.DatetimeIndex
    target_timestamps: pd.DatetimeIndex

    columns: list[str]

    context_length: int
    prediction_length: int
    window_id: int = 0
    cutoff_index: int | None = None


def make_window(
    dataset: TimeSeriesDataset,
    context_length: int,
    prediction_length: int,
    target_end: int,
    columns: list[str] | None = None,
    window_id: int = 0,
) -> ForecastWindow:
    """
    Validations
    """
    if context_length <= 0:
        raise ValueError("context_length must be > 0.")
    if prediction_length <= 0:
        raise ValueError("prediction_length must be > 0.")

    columns = columns or list(dataset.values.columns)
    missing_columns = [column for column in columns if column not in dataset.values.columns]
    if missing_columns:
        raise ValueError(f"Unknown columns: {missing_columns}")

    target_start = target_end - prediction_length
    history_start = target_start - context_length
    if history_start < 0:
        raise ValueError("Not enough observations for this forecasting window.")
    if target_end > dataset.n_observations:
        raise ValueError("target_end exceeds dataset length.")

    """
    Making window
    """
    values = dataset.values[columns]
    history = values.iloc[history_start:target_start].to_numpy(dtype=np.float32)
    target = values.iloc[target_start:target_end].to_numpy(dtype=np.float32)

    history_timestamps = dataset.timestamps[history_start:target_start]
    target_timestamps = dataset.timestamps[target_start:target_end]

    return ForecastWindow(
        history=history,
        target=target,
        history_timestamps=history_timestamps,
        target_timestamps=target_timestamps,
        columns=columns,
        context_length=context_length,
        prediction_length=prediction_length,
        window_id=window_id,
        cutoff_index=target_start,
    )


def make_last_window(
    dataset: TimeSeriesDataset,
    context_length: int,
    prediction_length: int,
    columns: list[str] | None = None,
) -> ForecastWindow:
    return make_window(
        dataset=dataset,
        context_length=context_length,
        prediction_length=prediction_length,
        target_end=dataset.n_observations,
        columns=columns,
    )


def make_rolling_windows(
    dataset: TimeSeriesDataset,
    context_length: int,
    prediction_length: int,
    n_windows: int,
    stride: int | None = None,
    columns: list[str] | None = None,
) -> list[ForecastWindow]:
    if n_windows <= 0:
        raise ValueError("n_windows must be > 0.")

    stride = stride or prediction_length
    if stride <= 0:
        raise ValueError("stride must be > 0.")

    required_length = context_length + prediction_length
    max_windows = 1 + (dataset.n_observations - required_length) // stride
    if n_windows > max_windows:
        raise ValueError(f"Requested {n_windows} windows, but only {max_windows} are available.")

    first_target_end = dataset.n_observations - (n_windows - 1) * stride
    return [
        make_window(
            dataset=dataset,
            context_length=context_length,
            prediction_length=prediction_length,
            target_end=first_target_end + window_id * stride,
            columns=columns,
            window_id=window_id,
        )
        for window_id in range(n_windows)
    ]
