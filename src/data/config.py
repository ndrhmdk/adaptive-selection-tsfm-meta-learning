from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_dataset_config(config_path: str | Path = "configs/datasets.yaml") -> dict:
    config_path = PROJECT_ROOT / config_path
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config["datasets"]
