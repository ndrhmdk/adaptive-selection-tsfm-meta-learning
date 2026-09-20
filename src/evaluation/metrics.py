import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_percentage_error,
)

EPS = 1e-8


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    numerator = np.abs(y_true - y_pred)
    denominator = np.abs(y_true) + np.abs(y_pred)
    result = 200 * numerator / np.maximum(denominator, EPS)
    return float(np.mean(result))


def mase_per_series(
    y_true: np.ndarray, y_pred: np.ndarray, history: np.ndarray, seasonal_period: int
) -> np.ndarray:
    """
    Returns one MASE value per variate.

    Shapes
    ------
    history: (context, n_variates)
    y_true: (horizon, n_variates)
    y_pred: (horizon, n_variates)
    """
    if seasonal_period <= 0:
        raise ValueError("seasonal_period must be > 0.")
    if len(history) < seasonal_period:
        raise ValueError("History is too short for the selected seasonal period.")

    naive_errors = np.abs(history[seasonal_period:] - history[:-seasonal_period])
    scale = np.mean(naive_errors, axis=0)
    forecast_error = np.mean(np.abs(y_true - y_pred), axis=0)
    scale = np.where(scale < EPS, np.nan, scale)

    return forecast_error / scale


def mase(
    y_true: np.ndarray, y_pred: np.ndarray, history: np.ndarray, seasonal_period: int
) -> float:
    scores = mase_per_series(
        y_true=y_true, y_pred=y_pred, history=history, seasonal_period=seasonal_period
    )
    return float(np.nanmean(scores))


def evaluate_forecast(
    y_true: np.ndarray, y_pred: np.ndarray, history: np.ndarray, seasonal_period: int
) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "smape": smape(y_true, y_pred),
        "mase": mase(
            y_true=y_true, y_pred=y_pred, history=history, seasonal_period=seasonal_period
        ),
    }
