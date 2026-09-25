import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from statsmodels.tsa.seasonal import STL

from src.data.windows import ForecastWindow

EPS = 1e-8


def _acf(values: np.ndarray, lag: int) -> float:
    if lag <= 0 or len(values) <= lag:
        return np.nan

    x = values[:-lag]
    y = values[lag:]

    if np.std(x) < EPS or np.std(y) < EPS:
        return 0.0

    return float(np.corrcoef(x, y)[0, 1])


def _spectral_features(values: np.ndarray) -> tuple[float, float]:
    centered = values - np.mean(values)
    power = np.abs(np.fft.rfft(centered)) ** 2
    if len(power) <= 1:
        return 0.0, np.nan

    power = power[1:]
    total_power = np.sum(power)
    if total_power < EPS:
        return 0.0, np.nan

    probabilities = power / total_power
    entropy = -np.sum(probabilities * np.log(probabilities + EPS))
    entropy /= np.log(len(probabilities))

    dominant_frequency_index = np.argmax(power) + 1
    dominant_period = len(values) / dominant_frequency_index

    return float(entropy), float(dominant_period)


def _stl_strenth(values: np.ndarray, seasonal_period: int) -> tuple[float, float]:
    if seasonal_period < 2:
        return np.nan, np.nan
    if len(values) < seasonal_period * 2:
        return np.nan, np.nan

    result = STL(values, period=seasonal_period, robust=True).fit()

    remainder_variance = np.var(result.resid)

    trend_denominator = np.var(result.trend + result.resid)
    seasonal_denominator = np.var(result.seasonal + result.resid)

    trend_strength = 1 - remainder_variance / max(trend_denominator, EPS)
    seasonal_strength = 1 - remainder_variance / max(seasonal_denominator, EPS)

    return float(max(0.0, trend_strength)), float(max(0.0, seasonal_strength))


def _linear_trend(values: np.ndarray) -> float:
    std = np.std(values)
    if std < EPS:
        return 0.0

    normalized = (values - np.mean(values)) / std
    time = np.linspace(-1, 1, len(values))

    slope = np.polyfit(time, normalized, 1)[0]
    return float(slope)


def _outlier_rate(values: np.ndarray) -> float:
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad < EPS:
        return 0.0

    modified_z = 0.6745 * np.abs(values - median) / mad
    return float(np.mean(modified_z > 3.5))


def _prepare_series(values: np.ndarray) -> tuple[np.ndarray, float]:
    series = pd.Series(values, dtype=float)
    missing_rate = float(series.isna().mean())
    if series.isna().all():
        raise ValueError("Cannot extract features from an entirely missing series.")

    series = series.interpolate(limit_direction="both")
    series = series.fillna(series.mean())

    return series.to_numpy(dtype=float), missing_rate


def extract_window_features(window: ForecastWindow, seasonal_period: int) -> dict:
    if window.history.ndim != 2:
        raise ValueError("window.history must have shape (time, variable).")
    if window.history.shape[1] != 1:
        raise ValueError("Primary meta-feature extraction expects one target series.")

    values, missing_rate = _prepare_series(window.history[:, 0])

    mean = float(np.mean(values))
    std = float(np.std(values))
    cv = std / max(abs(mean), EPS)

    spectral_entropy, dominant_period = _spectral_features(values)
    trend_strength, seasonal_strength = _stl_strenth(values, seasonal_period)

    timestamps_ns = window.history_timestamps.asi8

    if len(timestamps_ns) > 1:
        sampling_interval_seconds = float(np.median(np.diff(timestamps_ns)) / 1e9)
    else:
        sampling_interval_seconds = np.nan

    return {
        "context_length": window.context_length,
        "prediction_length": window.prediction_length,
        "seasonal_period": seasonal_period,
        "horizon_ratio": window.prediction_length / seasonal_period,
        "sampling_interval_seconds": sampling_interval_seconds,
        "mean": mean,
        "std": std,
        "cv": float(cv),
        "skewness": float(skew(values, bias=False)),
        "kurtosis": float(kurtosis(values, bias=False)),
        "acf1": _acf(values, 1),
        "seasonal_acf": _acf(values, seasonal_period),
        "linear_trend": _linear_trend(values),
        "trend_strength": trend_strength,
        "seasonal_strength": seasonal_strength,
        "spectral_entropy": spectral_entropy,
        "dominant_period": dominant_period,
        "missing_rate": missing_rate,
        "outlier_rate": _outlier_rate(values),
    }
