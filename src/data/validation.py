import numpy as np

from src.data.types import TimeSeriesDataset


def validate_dataset(dataset: TimeSeriesDataset) -> dict:
    values = dataset.values
    total_values = values.size

    missing_values = int(values.isna().sum().sum())
    missing_rate = missing_values / total_values if total_values > 0 else 0.0

    infinite_values = int(np.isinf(values.to_numpy(dtype=float)).sum())
    constant_columns = [
        column for column in values.columns if values[column].nunique(dropna=True) <= 1
    ]

    report = {
        "dataset": dataset.name,
        "domain": dataset.domain,
        "n_observations": dataset.n_observations,
        "n_variates": dataset.n_variates,
        "start_time": str(dataset.start_time),
        "end_time": str(dataset.end_time),
        "frequency": dataset.frequency,
        "missing_values": missing_values,
        "missing_rate": missing_rate,
        "infinite_values": infinite_values,
        "constant_columns": constant_columns,
    }
    return report
