from pathlib import Path

import yaml

from src.data.loader import load_dataset
from src.data.validation import validate_dataset

BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAG = "\033[95m"
RESET = "\033[0m"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_CONFIG = PROJECT_ROOT / "configs" / "benchmark.yaml"
TARGET_CONFIG = PROJECT_ROOT / "configs" / "benchmark_targets.yaml"


def main():
    with open(BENCHMARK_CONFIG, "r", encoding="utf-8") as file:
        benchmark = yaml.safe_load(file)["benchmark"]
        print(f"Successfully loaded {BOLD}{YELLOW}{BENCHMARK_CONFIG.relative_to(PROJECT_ROOT).as_posix()}")

    with open(TARGET_CONFIG, "r", encoding="utf-8") as file:
        targets = yaml.safe_load(file)["targets"]
        print(f"Successfully loaded {BOLD}{YELLOW}{TARGET_CONFIG.relative_to(PROJECT_ROOT).as_posix()}")

    context_length = benchmark["context_length"]
    max_horizon = max(benchmark["prediction_lengths"])
    required_length = context_length + max_horizon

    for dataset_name in benchmark["datasets"]:
        dataset = load_dataset(dataset_name)
        report = validate_dataset(dataset)

        dashes = "-" * 50
        print(f"----- {BOLD}{YELLOW}{dataset_name:12}{RESET}{dashes}")
        
        print(f"  + Observations:    {CYAN}{dataset.n_observations}{RESET}")
        print(f"  + Variates:        {CYAN}{dataset.n_variates}{RESET}")
        print(f"  + Frequency:       {CYAN}{dataset.frequency}{RESET}")
        print(f"  + Seasonal period: {CYAN}{dataset.seasonal_period}{RESET}")
        print(f"  + Missing rate:    {CYAN}{report['missing_rate']:.4f}{RESET}")
        print(f"  + Targets:         {MAG}{targets[dataset_name]}{RESET}")
        
        if dataset.n_observations < required_length:
            raise ValueError(f"  *** {BOLD}{YELLOW}{dataset_name}{RESET} has only {BOLD}{CYAN}{dataset.n_observations}{RESET} observations. At least {BOLD}{CYAN}{required_length}{RESET} are required.")
        
        for target in targets[dataset_name]:
            if target not in dataset.values.columns:
                raise ValueError(f"Target {BOLD}{MAG}{target}{RESET} does not exist in {BOLD}{YELLOW}{dataset_name}")
            
        print()
        
    print("All benchmark datasets passed validation.")


if __name__ == "__main__":
    main()