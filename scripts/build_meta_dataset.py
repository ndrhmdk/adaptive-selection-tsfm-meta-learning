from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.loader import load_dataset
from src.data.windows import make_rolling_windows
from src.features.extractor import extract_window_features

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BENCHMARK_CONFIG = PROJECT_ROOT / "configs" / "benchmark.yaml"
TARGET_CONFIG = PROJECT_ROOT / "configs" / "benchmark_targets.yaml"

with open(BENCHMARK_CONFIG, "r", encoding="utf-8") as f:
    benchmark = yaml.safe_load(f)["benchmark"]

with open(TARGET_CONFIG, "r", encoding="utf-8") as f:
    targets = yaml.safe_load(f)["targets"]

TSFM_MODELS = ["chronos2", "moirai2", "timesfm3"]
ALL_MODELS = TSFM_MODELS + ["seasonal_naive"]


def build_features(benchmark: dict, targets: dict) -> pd.DataFrame:
    rows = []
    for dataset_name in benchmark["datasets"]:
        dataset = load_dataset(dataset_name)

        for target in targets[dataset_name]:
            for prediction_length in benchmark["prediction_lengths"]:
                windows = make_rolling_windows(
                    dataset=dataset,
                    context_length=benchmark["context_length"],
                    prediction_length=prediction_length,
                    n_windows=benchmark["windows_per_target"],
                    stride=prediction_length,
                    columns=[target],
                )
                for window in windows:
                    features = extract_window_features(
                        window=window, seasonal_period=dataset.seasonal_period
                    )
                    rows.append(
                        {
                            "dataset": dataset.name,
                            "domain": dataset.domain,
                            "target": target,
                            "window_id": window.window_id,
                            **features,
                        }
                    )
    return pd.DataFrame(rows)


def load_metrics(benchmark: dict) -> pd.DataFrame:
    metrics_dir = PROJECT_ROOT / "results" / "benchmark" / "metrics"

    frames = []
    for dataset_name in benchmark["datasets"]:
        for model in ALL_MODELS:
            path = metrics_dir / f"{model}_{dataset_name}.parquet"
            if not path.exists():
                raise FileNotFoundError(f"Missing benchmark result: {path}")
            frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True)


def build_meta_dataset() -> pd.DataFrame:
    features = build_features(benchmark=benchmark, targets=targets)
    metrics = load_metrics(benchmark=benchmark)
    key = ["dataset", "domain", "target", "window_id", "prediction_length"]

    mase = metrics.pivot(index=key, columns="model", values="mase").reset_index()
    mase = mase.rename(
        columns={
            "chronos2": "chronos2_mase",
            "moirai2": "moirai2_mase",
            "timesfm3": "timesfm3_mase",
            "seasonal_naive": "seasonal_naive_mase",
        }
    )

    meta = features.merge(mase, on=key, how="inner", validate="one_to_one")
    tsfm_column = ["chronos2_mase", "moirai2_mase", "timesfm3_mase"]

    model_names = np.array(TSFM_MODELS)
    scores = meta[tsfm_column].to_numpy()

    order = np.argsort(scores, axis=1)
    best = np.take_along_axis(scores, order[:, :1], axis=1).ravel()
    second_best = np.take_along_axis(scores, order[:, 1:2], axis=1).ravel()

    meta["best_tsfm"] = model_names[order[:, 0]]
    meta["oracle_mase"] = best
    meta["winner_margin"] = second_best - best
    meta["winner_margin_relative"] = meta["winner_margin"] / np.maximum(best, 1e-8)

    return meta


def main():
    meta = build_meta_dataset()

    output_dir = PROJECT_ROOT / "results" / "meta_dataset"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "meta_dataset.parquet"
    meta.to_parquet(output_path, index=False)

    print(f"Shape: {meta.shape}")

    print("\nRows by dataset:")
    print(meta["dataset"].value_counts().sort_index())

    print("\nBest TSFM:")
    print(meta["best_tsfm"].value_counts())

    print("\nMean TSFM MASE")
    print(meta[["chronos2_mase", "moirai2_mase", "timesfm3_mase"]].mean().sort_values())

    print(f"\nSaved to {output_path.relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
