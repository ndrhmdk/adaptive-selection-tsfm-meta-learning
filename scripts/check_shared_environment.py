"""Validate the shared stack; --forecast exercises all checkpoints in one process."""

import argparse
import gc
import importlib
import io
import sys
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecast", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    print(f"Python: {sys.executable}", flush=True)
    for module in (
        "chronos",
        "uni2ts.model.moirai2",
        "timesfm3",
        "timesfm",
        "datasets",
        "statsmodels.api",
        "ipykernel",
        "src.features.extractor",
        "src.models.chronos",
        "src.models.moirai",
        "src.models.timesfm",
    ):
        importlib.import_module(module)
        print(f"OK: {module}", flush=True)

    import numpy as np
    import pandas as pd
    import shap
    import torch
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier

    from src.data.loader import load_dataset
    from src.data.windows import make_last_window
    from src.evaluation.metrics import evaluate_forecast

    for package in ("torch", "chronos-forecasting", "uni2ts", "timesfm", "shap"):
        print(f"{package}: {version(package)}", flush=True)
    print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
    # Synthetic data only verifies future router libraries, not router quality.
    x, y = make_classification(
        n_samples=60,
        n_features=6,
        n_informative=4,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42,
    )
    for cls in (RandomForestClassifier, XGBClassifier):
        model = cls(n_estimators=5, max_depth=3, random_state=42, n_jobs=1).fit(x, y)
        assert model.predict(x[:3]).shape == (3,)
        explanations = shap.TreeExplainer(model).shap_values(x[:3])
        assert np.isfinite(np.asarray(explanations)).all()
        print(f"OK: {cls.__name__} training, prediction, SHAP", flush=True)
    del model

    dataset = load_dataset("ETTh1")
    window = make_last_window(dataset, context_length=512, prediction_length=24)
    metrics = evaluate_forecast(
        window.target, window.target, window.history, dataset.seasonal_period
    )
    assert all(np.isfinite(value) and value == 0 for value in metrics.values())
    buffer = io.BytesIO()
    dataset.values.head().to_parquet(buffer, index=False)
    buffer.seek(0)
    pd.testing.assert_frame_equal(pd.read_parquet(buffer), dataset.values.head())
    figure = Figure()
    FigureCanvasAgg(figure)
    figure.subplots().plot(window.target[:, 0])
    figure.savefig(io.BytesIO(), format="png")
    print("OK: data loading, metrics, Parquet and plotting", flush=True)

    if args.forecast:
        from src.models.chronos import Chronos2Forecaster
        from src.models.moirai import Moirai2Forecaster
        from src.models.timesfm import TimesFM3Forecaster

        for cls in (Chronos2Forecaster, Moirai2Forecaster, TimesFM3Forecaster):
            model = cls(device=args.device)
            for variates in (1, window.history.shape[1]):
                result = model.predict(window.history[:, :variates], 24)
                for name in ("predictions", "lower", "upper"):
                    values = getattr(result, name)
                    assert values.shape == (24, variates), (cls.__name__, name, values.shape)
                    assert np.isfinite(values).all(), (cls.__name__, name)
                print(f"OK: {cls.__name__} forecast (24, {variates})", flush=True)
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print("Shared environment checks passed.", flush=True)


if __name__ == "__main__":
    main()
