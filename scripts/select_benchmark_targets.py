from pathlib import Path
import yaml

from src.data.loader import load_dataset
from src.data.targets import select_benchmark_targets

BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RESET = "\033[0m"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_CONFIG = PROJECT_ROOT / "configs" / "benchmark.yaml"
OUTPUT_PATH = PROJECT_ROOT / "configs" / "benchmark_targets.yaml"


def main():
    with open(BENCHMARK_CONFIG, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)["benchmark"]
        
    datasets = config['datasets']
    n_targets = config["target_per_dataset"]
    seed = config["target_selection_seed"]
    
    selections = {}
    for dataset_name in datasets:
        dataset = load_dataset(dataset_name)
        targets = select_benchmark_targets(
            dataset=dataset,
            n_targets=n_targets,
            seed=seed
        )
        selections[dataset_name] = targets
        shape_str = str(dataset.values.shape)
            
        print(
            f"{CYAN}{dataset_name:<15}{RESET} "
            f"shape: {YELLOW}{shape_str:<15}{RESET} - "
            f"targets: {GREEN}{targets}{RESET}")
        
    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        yaml.safe_dump(
            {"targets": selections},
            file,
            sort_keys=False
        )
        
    print(f"Saved in: {YELLOW}{OUTPUT_PATH.relative_to(PROJECT_ROOT).as_posix()}")
    
if __name__ == "__main__":
    main()