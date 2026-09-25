import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.loader import load_dataset
from src.data.windows import make_rolling_windows
from src.evaluation.metrics import evaluate_forecast
from src.models.base import BaseForecaster

BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAG = "\033[95m"
RED = "\033[91m"
RESET = "\033[0m"


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_dataset_benchmark(
    forecaster: BaseForecaster,
    dataset_name: str,
    targets: list[str],
    context_length: int,
    prediction_lengths: list[int],
    n_windows: int,
    save_forecasts: bool = False,
    results_dir: str | Path = "results/benchmark",
) -> pd.DataFrame:
    dataset = load_dataset(dataset_name)
    if dataset.seasonal_period is None:
        raise ValueError(
            f"{BOLD}{RED}{dataset_name}{RESET} must define {BOLD}seasonal_period{RESET}."
        )

    metric_rows = []
    forecast_rows = []

    for target in targets:
        for prediction_length in prediction_lengths:
            windows = make_rolling_windows(
                dataset=dataset,
                context_length=context_length,
                prediction_length=prediction_length,
                n_windows=n_windows,
                stride=prediction_length,
                columns=[target],
            )

            header_text = (
                f"{forecaster.__class__.__name__} | {dataset_name} | {target} "
                f"| H={prediction_length} | {len(windows)} windows"
            )
            dashes = "-" * max(10, 60 - len(header_text))
            print(f"\n{BOLD}{YELLOW}{header_text}{RESET} {dashes}")
            # print(f"\n{forecaster.__class__.__name__}: {dataset_name} - {target} - H={prediction_length} - {len(windows)} windows")

            for window in windows:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.perf_counter()

                forecast = forecaster.predict(
                    history=window.history, prediction_length=prediction_length
                )

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                runtime = time.perf_counter() - start

                if forecast.predictions.shape != window.target.shape:
                    raise ValueError(
                        f"Prediction shape {BOLD}{CYAN}{forecast.predictions.shape}{RESET} "
                        f"does not match target shape {BOLD}{CYAN}{window.target.shape}{RESET}."
                    )

                if not np.isfinite(forecast.predictions).all():
                    raise ValueError(
                        f"{BOLD}{YELLOW}{forecast.model_name}{RESET} produced non-finite values."
                    )

                metrics = evaluate_forecast(
                    y_true=window.target,
                    y_pred=forecast.predictions,
                    history=window.history,
                    seasonal_period=dataset.seasonal_period,
                )
                metric_rows.append(
                    {
                        "dataset": dataset.name,
                        "domain": dataset.domain,
                        "target": target,
                        "model": forecast.model_name,
                        "window_id": window.window_id,
                        "cutoff_timestamp": str(window.history_timestamps[-1]),
                        "context_length": context_length,
                        "prediction_length": prediction_length,
                        "mae": metrics["mae"],
                        "rmse": metrics["rmse"],
                        "smape": metrics["smape"],
                        "mase": metrics["mase"],
                        "inference_time_seconds": runtime,
                    }
                )

                if save_forecasts:
                    for step in range(prediction_length):
                        row = {
                            "dataset": dataset.name,
                            "target": target,
                            "model": forecast.model_name,
                            "window_id": window.window_id,
                            "prediction_length": prediction_length,
                            "timestamp": str(window.target_timestamps[step]),
                            "horizon_step": step + 1,
                            "actual": float(window.target[step, 0]),
                            "prediction": float(forecast.predictions[step, 0]),
                        }

                        if forecast.lower is not None:
                            row["lower"] = float(forecast.lower[step, 0])
                        if forecast.upper is not None:
                            row["upper"] = float(forecast.upper[step, 0])

                        forecast_rows.append(row)

                print(
                    f"  + Window {CYAN}{window.window_id:02d}{RESET} | "
                    f"Target: {MAG}{target}{RESET} | "
                    f"MASE: {GREEN}{metrics['mase']:.4f}{RESET} | "
                    f"Time: {CYAN}{runtime:.2f}s{RESET}"
                )

    metrics_df = pd.DataFrame(metric_rows)

    output_dir = Path(results_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    model_name = metrics_df["model"].iloc[0]

    metrics_path = metrics_dir / f"{model_name}_{dataset_name}.parquet"
    metrics_df.to_parquet(metrics_path, index=False)

    if save_forecasts:
        forecasts_dir = output_dir / "forecasts"
        forecasts_dir.mkdir(parents=True, exist_ok=True)
        forecasts_path = forecasts_dir / f"{model_name}_{dataset_name}.parquet"

        pd.DataFrame(forecast_rows).to_parquet(forecasts_path, index=False)
        print(f"  + Forecasts saved: {GREEN}{forecasts_path}{RESET}")

    print(f"  + Metrics saved:   {GREEN}{metrics_path}{RESET}")
    return metrics_df
