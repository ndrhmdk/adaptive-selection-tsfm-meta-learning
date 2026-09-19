# Research Decision Log

## 2026-09-19 — Protocol v1.0 frozen

Primary TSFMs:
- Chronos-2
- Moirai-2.0-R-small
- TimesFM 3.0

TimesFM fallback:
- TimesFM 2.5 only if TimesFM 3.0 causes a blocking compatibility issue.

Primary metric:
- MASE

Secondary metrics:
- MAE
- sMAPE
- RMSE

Context length:
- 512

Forecast horizons:
- 24
- 96
- 192

Pilot datasets:
- ETTh1
- Weather
- Electricity

Initial full datasets:
- ETTh1
- Weather
- Electricity
- Traffic
- Exchange Rate
- Solar

Primary meta-learner:
- XGBoost

Comparison meta-learner:
- Random Forest

Primary generalization experiment:
- Leave-One-Dataset-Out

Seed:
- 2026