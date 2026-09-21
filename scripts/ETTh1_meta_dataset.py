from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loader import load_dataset
from src.data.windows import make_rolling_windows
from src.features.extractor import extract_window_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TSFM_MODELS = ["chronos2", "moirai2", "timesfm3",]
PREDICTION_LENGTHS = [24, 96, 192,]


def build_features() -> pd.DataFrame:
    dataset = load_dataset("ETTh1")
    rows = []
    for prediction_length in PREDICTION_LENGTHS:
        windows = make_rolling_windows(
            dataset=dataset,
            context_length=512,
            prediction_length=prediction_length,
            n_windows=20,
            stride=prediction_length,
            columns=["OT"])
        for window in windows:
            features = extract_window_features(
                window=window,
                seasonal_period=dataset.seasonal_period,
            )
            rows.append(
                {
                    "dataset": dataset.name,
                    "target": "OT",
                    "window_id": window.window_id,
                    **features,
                }
            )
    return pd.DataFrame(rows)


def load_metrics() -> pd.DataFrame:
    metrics_dir = PROJECT_ROOT / "results" / "pilot" / "metrics"
    frames = []
    for model in TSFM_MODELS + ["seasonal_naive"]:
        path = metrics_dir / f"{model}_ETTh1.parquet"
        frame = pd.read_parquet(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def build_meta_dataset() -> pd.DataFrame:
    features = build_features()
    metrics = load_metrics()
    key = [
        "dataset",
        "target",
        "window_id",
        "prediction_length",
    ]
    mase = metrics.pivot(
        index=key,
        columns="model",
        values="mase",
    ).reset_index()
    mase = mase.rename(
        columns={
            "chronos2": "chronos2_mase",
            "moirai2": "moirai2_mase",
            "timesfm3": "timesfm3_mase",
            "seasonal_naive": "seasonal_naive_mase",
        }
    )
    meta = features.merge(
        mase,
        on=key,
        how="inner",
        validate="one_to_one",
    )
    tsfm_columns = [
        "chronos2_mase",
        "moirai2_mase",
        "timesfm3_mase",
    ]
    model_names = np.array(
        [
            "chronos2",
            "moirai2",
            "timesfm3",
        ]
    )

    scores = meta[tsfm_columns].to_numpy()
    order = np.argsort(scores, axis=1)
    meta["best_tsfm"] = model_names[order[:, 0]]
    best = np.take_along_axis(
        scores,
        order[:, :1],
        axis=1,
    ).ravel()
    second_best = np.take_along_axis(
        scores,
        order[:, 1:2],
        axis=1,
    ).ravel()
    meta["winner_margin"] = second_best - bes
    meta["winner_margin_relative"] = (
        meta["winner_margin"] / np.maximum(best, 1e-8))

    return meta


def main():
    meta = build_meta_dataset()
    output_dir = PROJECT_ROOT / "results" / "meta_dataset"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "ETTh1.parquet"
    meta.to_parquet(output_path, index=False)
    print(meta.head())
    print()
    print(f"Shape: {meta.shape}")
    print()
    print("Best TSFM counts:")
    print(meta["best_tsfm"].value_counts())
    print()
    print(f"Saved: {output_path.relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()