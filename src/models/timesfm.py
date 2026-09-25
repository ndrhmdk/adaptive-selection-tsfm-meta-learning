import numpy as np
import torch
from timesfm3 import ModelConfig, TimesFM3Evaluator

from src.models.base import BaseForecaster, ForecastResult


class TimesFM3Forecaster(BaseForecaster):
    def __init__(
        self,
        checkpoint: str = "google/timesfm-3.0-pytorch",
        device: str | None = None,
        batch_size: int = 16,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.checkpoint = checkpoint

        print(f"Loading TimesFM 3.0 from {checkpoint} on {device}...")
        config = ModelConfig(
            checkpoint_path=checkpoint, per_core_batch_size=batch_size, device=device
        )
        self.evaluator = TimesFM3Evaluator(config)
        print("TimesFM 3.0 loaded.")

    def predict(self, history: np.ndarray, prediction_length: int) -> ForecastResult:
        history = np.asarray(history, dtype=np.float32)
        if history.ndim != 2:
            raise ValueError("history must have shape (context_length, n_variates)")
        if len(history) == 0:
            raise ValueError("hisotry cannot be empty")
        if not np.isfinite(history).all():
            raise ValueError("TimesFM history currently rrequires finite values.")
        # context_length = history.shape[0]
        n_variates = history.shape[1]

        """
        The convention: (time, variates)
        TimesFM 3.0 expects: (batch, variates, time)
        """
        context = history.T
        outputs = list(
            self.evaluator.predict_batch(
                contexts=[context],
                horizon=prediction_length,
                return_quantiles=True,
                use_symmetric_averaging=False,
            )
        )
        if len(outputs) != 1:
            raise ValueError("Expected exactly one TimesFM output")
        output = outputs[0]
        point = np.asarray(output.forecast, dtype=np.float32)
        quantiles = np.asarray(output.quantiles, dtype=np.float32)

        """
        Multivariate
            point: (variables, horizon)
            quantiles: (variables, horizon, 9)
        """
        if n_variates == 1:
            if point.ndim == 1:
                point = point[np.newaxis, :]
            if quantiles.ndim == 2:
                quantiles = quantiles[
                    np.newaxis,
                    ...,
                ]

        expected_point = (n_variates, prediction_length)
        if point.shape != expected_point:
            raise ValueError(
                f"Unexpected TimesFM point forecast shape.\n- Expectedd: {expected_point}\n- Received: {point.shape}"
            )

        expected_quantiles = (n_variates, prediction_length, 9)
        if quantiles.shape != expected_quantiles:
            raise ValueError(
                f"Unexpected TimesFM point quantile shape.\n- Expectedd: {expected_quantiles}\n- Received: {quantiles.shape}"
            )

        """
        Converts to project convention
            (future_time, variables)
        """
        predictions = point.T
        lower = quantiles[:, :, 0].T
        upper = quantiles[:, :, 8].T

        return ForecastResult(
            predictions=predictions, lower=lower, upper=upper, model_name="timesfm3"
        )
