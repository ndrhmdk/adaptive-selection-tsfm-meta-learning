import numpy as np
import torch

from chronos import Chronos2Pipeline
from src.models.base import BaseForecaster, ForecastResult


class Chronos2Forecaster(BaseForecaster):
    def __init__(self, checkpoint: str = "amazon/chronos-2", device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.checkpoint = checkpoint
        print(f"Loading Chronos-2 from {checkpoint} on {device}")

        self.pipeline = Chronos2Pipeline.from_pretrained(checkpoint, device_map=device)
        print("Chronos-2 loaded")

    def predict(self, history: np.ndarray, prediction_length: int) -> ForecastResult:
        if history.ndim != 2:
            raise ValueError("history must have shape (context_length, n_variates)")
        if len(history) == 0:
            raise ValueError("history cannot be empty.")

        """
        The convention: (time, variates)
        Chronos-2 expects: (batch, variates, time)
        """
        chronos_input = torch.tensor(history.T, dtype=torch.float32).unsqueeze(0)
        quantiles, point_forecasts = self.pipeline.predict_quantiles(
            inputs=chronos_input,
            prediction_length=prediction_length,
            quantile_levels=[0.1, 0.5, 0.9],
        )

        """
        Chronos-2 returns a list because each item can have different dimensionality.
        We only supplied one forecasting task.
        """
        point = point_forecasts[0].cpu().numpy()
        q = quantiles[0].cpu().numpy()

        """
        Chronos output:
            point: (n_variates, horizon)
            quantiles: (n_variates, horizon, n_quantiles)
        """
        predictions = point.T

        lower = q[:, :, 0].T
        upper = q[:, :, 2].T

        return ForecastResult(
            predictions=predictions, lower=lower, upper=upper, model_name="chronos2"
        )
