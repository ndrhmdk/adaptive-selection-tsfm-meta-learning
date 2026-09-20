import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.loader import load_dataset
from src.data.windows import make_rolling_windows
from src.evaluation.metrics import evaluate_forecast
from src.models.base import BaseForecaster

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_pilot_benchmark(
    forecaster: BaseForecaster,
    dataset_name: str,
    context_length: int,
    prediction_lengths: list[int],
    n_windows: int,
    columns: list[str],
    results_dir: str | Path = "results/pilot",
) -> pd.DataFrame:
    dataset = load_dataset(dataset_name)
    if dataset.seasonal_period is None:
        raise ValueError(f"{dataset_name} must define seasonal_period.")

    metric_rows = []
    forecast_rows = []

    for prediction_length in prediction_lengths:
        windows = make_rolling_windows(
            dataset=dataset,
            context_length=context_length,
            prediction_length=prediction_length,
            n_windows=n_windows,
            stride=prediction_length,
            columns=columns,
        )
        print(
            f"\n\033[1;34m- \033[0m\033[1m{forecaster.__class__.__name__}\033[0m  \033[90m•\033[0m  Dataset: \033[36m{dataset_name}\033[0m  \033[90m•\033[0m  Horizon: \033[33m{prediction_length}\033[0m  \033[90m•\033[0m  Windows: \033[32m{len(windows):,}\033[0m"
        )

        for window in windows:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()

            forecast = forecaster.predict(
                history=window.history, prediction_length=prediction_length
            )

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            inference_time = time.perf_counter() - start

            if forecast.predictions.shape != window.target.shape:
                raise ValueError(
                    f"Prediction shape {forecast.predictions.shape} does not match target shape {window.target.shape}."
                )

            if not np.isfinite(forecast.predictions).all():
                raise ValueError("Forecast contains NaN or infinite values.")

            metrics = evaluate_forecast(
                y_true=window.target,
                y_pred=forecast.predictions,
                history=window.history,
                seasonal_period=dataset.seasonal_period,
            )
            metric_rows.append(
                {
                    "dataset": dataset.name,
                    "model": forecast.model_name,
                    "target": "|".join(columns),
                    "window_id": window.window_id,
                    "cutoff_timestamp": window.history_timestamps[-1],
                    "context_length": context_length,
                    "prediction_length": prediction_length,
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "smape": metrics["smape"],
                    "mase": metrics["mase"],
                    "inference_time_seconds": inference_time,
                }
            )

            for variable_index, variable in enumerate(window.columns):
                for step in range(prediction_length):
                    row = {
                        "dataset": dataset.name,
                        "model": forecast.model_name,
                        "variable": variable,
                        "window_id": window.window_id,
                        "prediction_length": prediction_length,
                        "timestamp": window.target_timestamps[step],
                        "horizon_step": step + 1,
                        "actual": float(window.target[step, variable_index]),
                        "prediction": float(forecast.predictions[step, variable_index]),
                    }

                    if forecast.lower is not None:
                        row["lower"] = float(forecast.lower[step, variable_index])
                    if forecast.upper is not None:
                        row["upper"] = float(forecast.upper[step, variable_index])

                    forecast_rows.append(row)

            print(
                f"Window {window.window_id:02d} - MASE: {metrics['mase']:.4f} - {inference_time:.2f}s"
            )

    metrics_df = pd.DataFrame(metric_rows)
    forecasts_df = pd.DataFrame(forecast_rows)
    model_name = metrics_df["model"].iloc[0]

    output_dir = Path(results_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    metrics_dir = output_dir / "metrics"
    forecasts_dir = output_dir / "forecasts"

    metrics_dir.mkdir(parents=True, exist_ok=True)
    forecasts_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = metrics_dir / f"{model_name}_{dataset_name}.parquet"
    forecasts_path = forecasts_dir / f"{model_name}_{dataset_name}.parquet"

    metrics_df.to_parquet(metrics_path, index=False)
    forecasts_df.to_parquet(forecasts_path, index=False)

    print(f"\nSaved metrics: {metrics_path.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"Saved forecasts: {forecasts_path.relative_to(PROJECT_ROOT).as_posix()}")

    return metrics_df
