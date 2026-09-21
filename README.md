# Adaptive TSFM selection

One Python 3.11 environment supports Chronos-2, Moirai-2.0, TimesFM-3.0
and the proposed Random Forest/XGBoost meta-learner with SHAP.

```powershell
uv sync --locked
uv run --locked python scripts/check_shared_environment.py
```

Select `.venv/Scripts/python.exe` for every notebook. See
[environment setup and compatibility checks](environments/README.md) for GPU
forecast validation and guidance on future models.
