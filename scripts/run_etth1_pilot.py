import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.pilot import run_pilot_benchmark
from src.models.base import BaseForecaster


def create_forecaster(model_name: str) -> BaseForecaster:
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

        return SeasonalNaiveForecaster(seasonal_period=24)
    raise ValueError(f"Unknown model: {model_name}")


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
        "--windows",
        type=int,
        default=20,
    )

    return parser.parse_args()


def main():
    args = parse_args()
    forecaster = create_forecaster(args.model)
    metrics = run_pilot_benchmark(
        forecaster=forecaster,
        dataset_name="ETTh1",
        context_length=512,
        prediction_lengths=[24, 96, 192],
        n_windows=args.windows,
        columns=["OT"],
    )

    print("\nMean MASE by horizon:")
    print(metrics.groupby("prediction_length")["mase"].mean().round(4))


if __name__ == "__main__":
    main()
