import numpy as np
from src.models.base import BaseForecaster, ForecastResult


class SeasonalNaiveForecaster(BaseForecaster):
    def __init__(self, seasonal_period: int):
        if seasonal_period <= 0:
            raise ValueError("seasonal_period must be > 0.")
        self.seasonal_period = seasonal_period

    def predict(self, history: np.ndarray, prediction_length: int) -> ForecastResult:
        history = np.asarray(history, dtype=np.float32)
        if history.ndim != 2:
            raise ValueError("history must have shape (context_length, n_variates)")
        if len(history) < self.seasonal_period:
            raise ValueError("history is shorter than seasonal_period")

        pattern = history[-self.seasonal_period :]
        repetitions = (prediction_length + self.seasonal_period - 1) // self.seasonal_period
        predictions = np.tile(pattern, (repetitions, 1))[:prediction_length]

        return ForecastResult(predictions=predictions, model_name="seasonal_naive")
