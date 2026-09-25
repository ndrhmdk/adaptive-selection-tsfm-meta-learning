import argparse
from pathlib import Path

import yaml

from src.data.loader import load_dataset
from src.evaluation.benchmark import run_dataset_benchmark
from src.models.base import BaseForecaster

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_CONFIG = PROJECT_ROOT / "configs" / "benchmark.yaml"
TARGET_CONFIG = PROJECT_ROOT / "configs" / "benchmark_targets.yaml"


def create_forecaster(
    model_name: str,
    seasonal_period: int | None = None,
) -> BaseForecaster:
    if model_name == "chronos2":
        from src.models.chronos import Chronos2Forecaster

        return Chronos2Forecaster()

    if model_name == "moirai2":
        from src.models.moirai import Moirai2Forecaster

        return Moirai2Forecaster()

    if model_name == "timesfm3":
        from src.models.timesfm import TimesFM3Forecaster

        return TimesFM3Forecaster()

    if model_name == "seasonal_naive":
        from src.models.baselines import SeasonalNaiveForecaster

        if seasonal_period is None:
            raise ValueError("seasonal_period is required for Seasonal Naive.")

        return SeasonalNaiveForecaster(seasonal_period=seasonal_period)

    raise ValueError(f"Unknown model: {model_name}")


def load_configs() -> tuple[dict, dict]:
    with open(BENCHMARK_CONFIG, "r", encoding="utf-8") as file:
        benchmark = yaml.safe_load(file)["benchmark"]

    with open(TARGET_CONFIG, "r", encoding="utf-8") as file:
        targets = yaml.safe_load(file)["targets"]

    return benchmark, targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        choices=[
            "chronos2",
            "moirai2",
            "timesfm3",
            "seasonal_naive",
        ],
    )

    parser.add_argument(
        "--dataset",
        default="all",
    )

    parser.add_argument(
        "--windows",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--save-forecasts",
        action="store_true",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    benchmark, targets = load_configs()

    if args.dataset == "all":
        datasets = benchmark["datasets"]
    else:
        if args.dataset not in benchmark["datasets"]:
            raise ValueError(f"{args.dataset} is not part of the benchmark.")

        datasets = [args.dataset]

    n_windows = args.windows or benchmark["windows_per_target"]

    shared_forecaster = None

    if args.model != "seasonal_naive":
        shared_forecaster = create_forecaster(args.model)

    for dataset_name in datasets:
        dataset = load_dataset(dataset_name)

        if args.model == "seasonal_naive":
            forecaster = create_forecaster(
                model_name=args.model,
                seasonal_period=dataset.seasonal_period,
            )
        else:
            forecaster = shared_forecaster

        run_dataset_benchmark(
            forecaster=forecaster,
            dataset_name=dataset_name,
            targets=targets[dataset_name],
            context_length=benchmark["context_length"],
            prediction_lengths=benchmark["prediction_lengths"],
            n_windows=n_windows,
            save_forecasts=args.save_forecasts,
        )


if __name__ == "__main__":
    main()
