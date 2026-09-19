import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.loader import load_dataset
from src.data.windows import make_last_window
from src.evaluation.metrics import evaluate_forecast, mase_per_series
from src.models.chronos import Chronos2Forecaster

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTEXT_LENGTH = 512
PREDICTION_LENGTH = 96 

### **Load Dataset**
dataset = load_dataset("ETTh1")
print(f"Dataset: {dataset.name}")
print(f"Observations: {dataset.n_observations}")
print(f"Variates: {dataset.n_variates}")
print(f"Frequency: {dataset.frequency}")

print()

### **Construct Holdout Window**
window = make_last_window(
    dataset=dataset,
    context_length=CONTEXT_LENGTH,
    prediction_length=PREDICTION_LENGTH)
print(f"History shape: {window.history.shape}")
print(f"Target shape: {window.history.shape}")

print()

### **Load Model**
forecaster = Chronos2Forecaster()

### **Run Inference**
if torch.cuda.is_available():
    torch.cuda.synchronize()
start_time = time.perf_counter()

forecast = forecaster.predict(
    history=window.history,
    prediction_length=PREDICTION_LENGTH)

if torch.cuda.is_available():
    torch.cuda.synchronize()
inference_time = time.perf_counter() - start_time
print(f"Prediction shape: {forecast.predictions.shape}")

print()

### **Validate Prediction**
if forecast.predictions.shape != window.target.shape:
    raise ValueError("Prediction shape does not match target shape.")
else:
    print("Prediction shape matches target shape.")

if not np.isfinite(forecast.predictions).all():
    raise ValueError("Forecast contains NaN or infinite values.")
else:
    print("Forecast does not contain NaN or infinite values.")
    
print() 

### **Metrics**
if dataset.seasonal_period is None:
    raise ValueError("seasonal_period must be configured for MASE.")
else:
    print("seasonal_period has been configured for MASE.")
metrics = evaluate_forecast(
    y_true=window.target,
    y_pred=forecast.predictions,
    history=window.history,
    seasonal_period=dataset.seasonal_period)
per_series_mase = mase_per_series(
    y_true=window.target,
    y_pred=forecast.predictions,
    history=window.history,
    seasonal_period=dataset.seasonal_period)
metrics["inference_time_seconds"] = inference_time
metrics["context_length"] = CONTEXT_LENGTH
metrics["prediction_length"] = PREDICTION_LENGTH
metrics["dataset"] = dataset.name
metrics["model"] = forecast.model_name
metrics["device"] = str(forecaster.device)
print("--- Results ----------------------------")

for key, value in metrics.items():
    formatted_value = f"{value:.4f}" if isinstance(value, float) else value
    print(f"{key:25}: {formatted_value}")

print()

print("MASE by Variables:")
for column, score in zip(window.columns, per_series_mase):
    print(f"{column:10}: {score:.4f}")
### **Save Predictions**
prediction_dir = PROJECT_ROOT / "results" / "forecasts"
metrics_dir = PROJECT_ROOT / "results" / "metrics"
prediction_dir.mkdir(parents=True, exist_ok=True)
metrics_dir.mkdir(parents=True, exist_ok=True)
rows = []
for variable_index, variable in enumerate(window.columns):
    for step in range(PREDICTION_LENGTH):
        rows.append({
            "dataset": dataset.name,
            "model": forecast.model_name,
            "variable": variable,
            "timestamp": window.target_timestamps[step],
            "horizon_step": step + 1,
            "actual": float(window.target[step, variable_index,]),
            "prediction": float(forecast.predictions[step, variable_index]),
            "lower_10": float(forecast.lower[step, variable_index]),
            "upper_90": float(forecast.upper[step, variable_index])
        })
prediction_df = pd.DataFrame(rows)
prediction_df.head()
prediction_path = prediction_dir / "chronos2_ETTh1_h96.parquet"
prediction_df.to_parquet(prediction_path, index=False)

print()

### **Save Metrics**
metrics["mase_per_variables"] = {
    column: float(score)
    for column, score in zip(window.columns, per_series_mase)}
metrics_path = metrics_dir / "chronos2_ETTh1_h96.json"
with open(metrics_path, "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4)
print(f"Prediction saved: {prediction_path.relative_to(PROJECT_ROOT).as_posix()}")
print(f"Metrics saved: {metrics_path.relative_to(PROJECT_ROOT).as_posix()}")