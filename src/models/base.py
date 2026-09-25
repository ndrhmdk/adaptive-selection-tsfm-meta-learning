from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class ForecastResult:
    """
    Standard output returned by every forecasting model.

    Shapes
    ------
    predictions: (prediction_length, n_variates)
    lower: Optional lower prediction interval.
    upper: Optional upper prediction interval.
    """

    predictions: np.ndarray

    lower: np.ndarray | None = None
    upper: np.ndarray | None = None

    model_name: str | None = None


class BaseForecaster(ABC):
    @abstractmethod
    def predict(self, history: np.ndarray, prediction_length: int) -> ForecastResult:
        """
        Parameters
        ----------
        history:
            Array of shape: (context_length, n_variates)
        prediction_length:
            Number of future time steps to predict

        Returns
        -------
        ForecastResult
            Prediction in shape: (prediction_length, n_variates)
        """
        raise NotImplementedError
