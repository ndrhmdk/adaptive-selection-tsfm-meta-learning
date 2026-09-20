import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.loader import load_dataset
from src.data.windows import make_last_window
from src.evaluation.metrics import evaluate_forecast, mase_per_series
from src.models.base import BaseForecaster

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_single_forecast(
    forecaster: BaseForecaster,
    dataset_name: str,
    context_length: int,
    prediction_length: int,
    save_results: bool = True,
    results_dir: str | Path | None = None,
):
    """
    Run one forecasting experiment using the
    final context + horizon window of a dataset.

    When saving, results_dir defaults to the project's results directory.
    Relative results_dir paths are resolved from the project root.
    Repeated runs with the same model, dataset, and horizon overwrite files.

    Returns
    -------
    dict
        dataset
        window
        forecast
        metrics
        prediction_df
        saved_paths (empty when save_results=False)
    """
    print()

    # load dataset
    dataset = load_dataset(dataset_name)

    # construct evaluation window
    window = make_last_window(
        dataset=dataset, context_length=context_length, prediction_length=prediction_length
    )
    print(f"Dataset: {dataset_name}")
    print(f"History shape: {window.history.shape}")
    print(f"Target shape: {window.target.shape}")

    # inference
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    forecast = forecaster.predict(history=window.history, prediction_length=prediction_length)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inference_time = time.perf_counter() - start_time

    print()

    # validation
    if forecast.predictions.shape != window.target.shape:
        raise ValueError("Prediction shape does not match target shape.")
    else:
        print("Prediction shape matches target shape.")

    if not np.isfinite(forecast.predictions).all():
        raise ValueError("Forecast contains NaN or infinite values.")
    else:
        print("Forecast does not contain NaN or infinite values.")

    print()

    # metrics
    if dataset.seasonal_period is None:
        raise ValueError("seasonal_period must be configured for MASE.")
    else:
        print("seasonal_period has been configured for MASE.")
    metrics = evaluate_forecast(
        y_true=window.target,
        y_pred=forecast.predictions,
        history=window.history,
        seasonal_period=dataset.seasonal_period,
    )
    series_mase = mase_per_series(
        y_true=window.target,
        y_pred=forecast.predictions,
        history=window.history,
        seasonal_period=dataset.seasonal_period,
    )
    metrics.update(
        {
            "dataset": dataset.name,
            "model": forecast.model_name,
            "context_length": context_length,
            "prediction_length": prediction_length,
            "inference_time_seconds": inference_time,
        }
    )
    metrics["mase_per_variable"] = {
        column: float(score) for column, score in zip(window.columns, series_mase)
    }

    # build prediction dataframe
    rows = []
    for variable_index, variable in enumerate(window.columns):
        for step in range(prediction_length):
            row = {
                "dataset": dataset.name,
                "model": forecast.model_name,
                "variable": variable,
                "timestamp": (window.target_timestamps[step]),
                "horizon_step": step + 1,
                "actual": float(
                    window.target[
                        step,
                        variable_index,
                    ]
                ),
                "prediction": float(forecast.predictions[step, variable_index]),
            }

            if forecast.lower is not None:
                row["lower"] = float(forecast.lower[step, variable_index])
            if forecast.upper is not None:
                row["upper"] = float(forecast.upper[step, variable_index])

            rows.append(row)
    prediction_df = pd.DataFrame(rows)

    # save results
    saved_paths = {}
    if save_results:
        output_dir = Path(results_dir) if results_dir is not None else PROJECT_ROOT / "results"
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir
        output_dir = output_dir.resolve()
        forecast_dir = output_dir / "forecasts"
        metrics_dir = output_dir / "metrics"
        forecast_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)

        stem = f"{forecast.model_name}_{dataset.name}_h{prediction_length}"
        forecast_path = forecast_dir / f"{stem}.parquet"
        metrics_path = metrics_dir / f"{stem}.json"
        prediction_df.to_parquet(forecast_path, index=False)

        pd.DataFrame([metrics]).to_json(metrics_path, orient="records", indent=4)
        print(
            f"Saved:\n- Forecast: {forecast_path.relative_to(PROJECT_ROOT).as_posix()}\n- Metrics: {metrics_path.relative_to(PROJECT_ROOT).as_posix()}"
        )

    return {
        "dataset": dataset,
        "window": window,
        "forecast": forecast,
        "metrics": metrics,
        "prediction_df": prediction_df,
        "saved_paths": saved_paths,
    }
