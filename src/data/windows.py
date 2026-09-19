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
    

def make_last_window(dataset: TimeSeriesDataset, context_length: int, prediction_length: int) -> ForecastWindow:
    required_length = context_length + prediction_length
    if dataset.n_observations < required_length:
        raise ValueError(f"{dataset.name} contains {dataset.n_observations} observations, but {required_length} are required.")
    
    history_start = -required_length
    target_start = -prediction_length
    
    history = dataset.values.iloc[history_start : target_start].to_numpy(dtype=np.float32)
    target = dataset.values.iloc[target_start:].to_numpy(dtype=np.float32)
    
    history_timestamps = dataset.timestamps[history_start : target_start]
    target_timestamps = dataset.timestamps[target_start:]
    
    return ForecastWindow(
        history=history, target=target,
        history_timestamps=history_timestamps, target_timestamps=target_timestamps,
        columns=list(dataset.values.columns),
        context_length=context_length, prediction_length=prediction_length)