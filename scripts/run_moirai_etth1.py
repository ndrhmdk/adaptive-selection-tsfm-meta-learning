import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.experiment import run_single_forecast
from src.models.moirai import Moirai2Forecaster


def main():
    model = Moirai2Forecaster()
    result = run_single_forecast(
        forecaster=model, dataset_name="ETTh1", context_length=512, prediction_length=96
    )

    print("\nResults:")

    for key, value in result["metrics"].items():
        if key != "mase_per_variable":
            print(f"{key:25}: {value}")


if __name__ == "__main__":
    main()
