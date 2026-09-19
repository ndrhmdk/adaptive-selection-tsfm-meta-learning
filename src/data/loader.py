from pathlib import Path
import pandas as pd

from src.data.config import PROJECT_ROOT, load_dataset_config
from src.data.types import TimeSeriesDataset

def infer_frequency(timestamps: pd.DatetimeIndex) -> str | None:
    if len(timestamps) < 3:
        return None
    frequency = pd.infer_freq(timestamps)
    return frequency

def load_dataset(dataset_name: str) -> TimeSeriesDataset:
    configs = load_dataset_config()
    if dataset_name not in configs:
        available = ', '.join(configs.keys())
        raise ValueError(f"Unknown dataset '{dataset_name}'. Available datasets: {available}")

    config = configs[dataset_name]
    path = PROJECT_ROOT / Path(config['path'])
    if not path.exists():
        raise FileNotFoundError(f"Dataset '{dataset_name}' was  configured but the file does not exists:\n{path}")
    
    df = pd.read_csv(path)
    timestamps_column = config['timestamp_column']
    if timestamps_column not in df.columns:
        raise ValueError(f"Timestamp column '{timestamps_column}' not found in {dataset_name}.")
    
    df[timestamps_column] = pd.to_datetime(df[timestamps_column], errors='raise')
    df = df.sort_values(timestamps_column)
    if df[timestamps_column].duplicated().any():
        duplicates = df[timestamps_column].duplicated().sum()
        raise ValueError(f"'{dataset_name}' contains {duplicates} duplicated timestamps.")
    
    timestamps = pd.DatetimeIndex(df[timestamps_column])
    
    configured_targets = config.get('target_columns')
    if configured_targets:
        target_columns = configured_targets
    else:
        target_columns = [column for column in df.columns if column != timestamps_column]
        
    missing_columns = [column for column in target_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing target columns in '{dataset_name}': {missing_columns}")
    
    values = df[target_columns].copy()
    for column in values.columns:
        values[column] = pd.to_numeric(values[column], errors='coerce')
    inferred_frequency = infer_frequency(timestamps)
    
    return TimeSeriesDataset(
        name=dataset_name,
        domain=config["domain"],
        timestamps=timestamps,
        values=values,
        frequency=inferred_frequency,
        target_columns=target_columns,
        primary_target=config.get("primary_target"))