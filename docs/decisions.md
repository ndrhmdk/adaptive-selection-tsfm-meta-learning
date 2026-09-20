# Research Decision Log

## 2026-09-19 — **Protocol v1.0 frozen**

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

## 2026-09-20 — **Multivariate dimensionality constraint**

During Moirai-2 integration, the model's token budget was
identified as a constraint for very high-dimensional datasets.

ETTh1 can be evaluated jointly with all 7 variables.

However, high-dimensional datasets such as Electricity and Traffic
cannot necessarily be provided to every TSFM using all channels
simultaneously under the same context and horizon.

Decision:

- Continue the ETTh1 pilot using joint multivariate forecasting.
- Do not yet change the primary benchmark protocol.
- After TimesFM integration, define one common dimensionality policy
  supported by all three models.
- Candidate policies:
  1. per-series univariate forecasting;
  2. capped variable groups;
  3. both, with univariate as primary and multivariate as secondary.