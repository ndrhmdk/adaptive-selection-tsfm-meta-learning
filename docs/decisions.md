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

## 2026-09-20 — **Model-specific Python environments**

During Moirai-2 integration, Uni2TS introduced dependency
conflicts with the existing Chronos environment.

Decision:

Use isolated model-specific Python runtimes while keeping
one shared repository, dataset pipeline, evaluation framework,
and results directory.

Current environments:

- `.venv`
  - shared/core environment
  - Chronos-2

- `.venv-uni2ts`
  - Moirai-2
  - Uni2TS

- `.venv-timesfm`
  - TimesFM 3.0

This isolation prevents model-specific dependency requirements
from affecting other TSFM implementations.

All models continue to receive identical forecasting windows
and are evaluated by the same shared metric implementation.

## **TimesFM 3.0 license**

TimesFM 3.0 pretrained weights are used only for
non-commercial academic research as part of this graduation project.

Checkpoint:
`google/timesfm-3.0-pytorch`

## 2026-09-20 — **ETTh1 pilot results and router formulation**

The ETTh1 pilot evaluated:

- Chronos-2
- Moirai-2
- TimesFM-3
- Seasonal Naive

using:

- target: OT
- context length: 512
- horizons: 24, 96, 192
- 20 rolling windows per horizon

This produced 60 forecasting tasks per model.

### **TSFM winner counts**

| Model | Wins |
|---|---:|
| Chronos-2 | 23 |
| Moirai-2 | 21 |
| TimesFM-3 | 16 |

No TSFM dominated all forecasting tasks.

The best fixed TSFM was Chronos-2 with mean MASE 1.1294.

The oracle TSFM selector achieved mean MASE 1.0221.

Therefore, the estimated maximum routing opportunity on the ETTh1 pilot is approximately 9.5% relative MASE improvement over the best fixed TSFM.

A horizon-only selection rule achieved approximately 1.1122 MASE, suggesting that forecast horizon alone does not explain all model-performance variation.

### **Near-tie observation**

The median difference between the best and second-best TSFM was approximately 0.0584 MASE.

Approximately 22 of 60 tasks had less than 5% relative separation between the two best models.

Because exact winner classification treats near-ties and large mistakes equally, classification will no longer be the only primary formulation.

### **Updated meta-learning design**

Primary:
- predict expected MASE for each TSFM;
- select the model with the lowest predicted MASE.

Initial models:
- XGBRegressor
- RandomForestRegressor

Secondary:
- multiclass best-model classification.

The classification labels will still be retained in the meta-dataset for analysis.

### **Forecasting mode**

The primary benchmark will use per-series univariate forecasting so all TSFMs receive equivalent information.

Multivariate routing will be treated as a secondary experiment for datasets and models where equivalent multivariate inference can be performed reliably.

Seasonal Naive remains an evaluation baseline rather than a primary routing class.