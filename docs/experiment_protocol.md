# **Experimental Protocol**

## **Adaptive Selection of Time-Series Foundation Models Using Interpretable Meta-Learning**

**Protocol version:** 1.0
**Status:** Frozen for primary experiment
**Research mode:** Four-week intensive run of the original 16-week project methodology

---

# **1. Primary Objective**

The primary objective is to investigate whether measurable characteristics of a time series can be used to automatically select an appropriate pretrained Time-Series Foundation Model (TSFM) for a forecasting task.

The project does not attempt to identify one universally best forecasting model.

Instead, the project studies the mapping:

Time-series characteristics
→ Relative model performance
→ Recommended forecasting model

The proposed system will extract interpretable meta-features from the historical context of a forecasting instance and use a machine-learning meta-model to recommend one TSFM.

---

# **2. Research Questions**

## RQ1 — Model Heterogeneity

How does the forecasting performance of different Time-Series Foundation Models vary across datasets, forecasting horizons, frequencies, dimensionalities, and statistical characteristics?

## RQ2 — Time-Series Characteristics

Which characteristics of a forecasting task are associated with the relative performance of different TSFMs?

Characteristics of interest include:

* trend;
* seasonality;
* autocorrelation;
* spectral complexity;
* variability;
* distributional characteristics;
* dimensionality;
* frequency;
* forecast horizon.

## RQ3 — Adaptive Selection

Can an interpretable meta-learning router select a TSFM that produces lower forecasting error than always using one fixed TSFM?

## RQ4 — Generalization

Can the router generalize to a dataset that was not observed during meta-model training?

---

# **3. Optional Research Questions**

The following questions are explicitly considered extensions and are not required for completion of the primary experiment.

### RQ5 — Performance Prediction

Can the forecasting error of each candidate TSFM be predicted directly from time-series characteristics?

### RQ6 — Efficiency-Aware Routing

Can inference cost and forecasting accuracy be jointly considered when recommending a TSFM?

These questions will only be investigated after RQ1–RQ4 have been completed.

---

# **4. Candidate Foundation Models**

The primary routing pool contains exactly three pretrained forecasting models.

| Identifier | Model              | Role           |
| ---------- | ------------------ | -------------- |
| `chronos2` | Chronos-2          | Candidate TSFM |
| `moirai2`  | Moirai-2.0-R-small | Candidate TSFM |
| `timesfm3` | TimesFM 3.0        | Candidate TSFM |

These three models form the classes predicted by the meta-learning router.

## **Version Policy**

Model versions will be frozen before the full benchmark begins.

If TimesFM 3.0 produces a major installation or compatibility blocker during the pilot phase, TimesFM 2.5 will be used as the documented fallback.

Changing model versions after benchmark generation begins is not permitted unless the existing version contains a reproducibility-breaking bug.

---

# **5. Forecasting Baselines**

The minimum baseline is:

* Seasonal Naive.

Additional baselines may include:

* Naive;
* DLinear;
* PatchTST;
* iTransformer.

PatchTST and iTransformer are not classified as Time-Series Foundation Models in this project.

They are supervised deep-learning forecasting baselines.

The router's primary classification space remains:

Chronos-2
vs. Moirai-2.0
vs. TimesFM

---

# **6. Training Strategy**

The core TSFM comparison uses pretrained inference.

The following techniques are excluded from the primary experiment:

* full fine-tuning;
* LoRA fine-tuning;
* few-shot fine-tuning;
* per-dataset retraining of foundation models.

This restriction reduces computational cost and keeps model comparison consistent.

The project will avoid describing experiments as strictly "zero-shot" unless training-data leakage requirements have been verified for the relevant checkpoint and benchmark.

---

# **7. Dataset Strategy**

The project will use two stages.

## **Stage A — Development and Pilot**

The initial pipeline will be developed using:

1. ETTh1
2. Weather
3. Electricity

These datasets are used first because they provide different statistical characteristics while remaining easy to work with during pipeline development.

The pilot experiment will determine whether model performance varies enough to justify adaptive model routing.

## **Stage B — Main Experiment**

The initial target dataset collection is:

| Dataset       | Approximate domain   | Purpose                                 |
| ------------- | -------------------- | --------------------------------------- |
| ETTh1         | Energy               | structured multivariate time series     |
| Weather       | Nature / meteorology | many correlated environmental variables |
| Electricity   | Energy / demand      | high-dimensional consumption            |
| Traffic       | Transportation       | high-dimensional traffic behavior       |
| Exchange Rate | Finance / economics  | low-dimensional financial series        |
| Solar         | Renewable energy     | strong daily seasonality                |

The exact dataset configuration and preprocessing metadata must be stored with experimental results.

If time and computing resources allow, additional GIFT-Eval configurations from underrepresented domains will be added.

The full GIFT-Eval benchmark is not required for the first four-week run.

---

# **8. GIFT-Eval Compatibility**

When GIFT-Eval datasets are used, the official train/validation/test divisions will be respected.

Where practical, the experiment will follow the GIFT-Eval evaluation conventions so that results remain compatible with existing forecasting evaluation practices.

The complete 98-task GIFT-Eval benchmark is outside the initial sprint scope.

---

# **9. Forecasting Instance Definition**

A meta-learning observation corresponds to one forecasting instance rather than one entire dataset.

Each instance contains:

Historical context:

X(t-L+1), ..., X(t)

Forecast target:

X(t+1), ..., X(t+H)

where:

* L = context length;
* H = prediction horizon.

For every forecasting instance:

1. extract meta-features from historical information only;
2. forecast with every candidate TSFM;
3. calculate forecasting errors;
4. identify the lowest-error model;
5. store model performances and meta-features.

No information from the future prediction interval may be used when generating meta-features.

---

# **10. Context Length**

The default historical context length is:

$$
L = 512
$$

observations.

The same maximum context will initially be provided to all candidate models where technically supported.

If a dataset contains insufficient observations for a 512-step context, it will either:

1. use the maximum valid historical context; or
2. be excluded from that experimental configuration.

The decision must be recorded rather than silently changing context length.

---

# **11. Forecast Horizons**

The main experiment will initially evaluate:

$$
H \in \{24, 96, 192\}
$$

This provides short-, medium-, and longer-horizon forecasting tasks while keeping computational requirements manageable.

Forecast horizon is itself included as a meta-feature.

If a dataset cannot reasonably support one of these horizons, the unsupported configuration may be excluded and documented.

---

# **12. Window Generation**

Rolling forecasting windows will be generated from each dataset.

Default stride:

$$
stride = H
$$

where H is the prediction horizon.

This avoids excessive overlap between neighboring forecasting instances.

The initial pilot will use approximately:

20 windows per dataset-horizon combination.

The target main experiment will use approximately:

50–100 windows per dataset-horizon combination,

depending on computational cost and dataset length.

With six datasets, three horizons, and 50 windows per configuration, the target meta-dataset contains approximately:

$$
6 \times 3 \times 50 = 900
$$

forecasting instances.

Each instance is evaluated using all three TSFMs.

The experiment can later scale toward approximately 1,800 instances by increasing the number of windows to 100.

---

# **13. Meta-Features — Version 1**

The first meta-feature set will remain deliberately compact and interpretable.

## **Task / Temporal Features**

* sampling frequency;
* context length;
* forecast horizon;
* dominant period.

## **Dimensionality Features**

* number of variates;
* mean absolute cross-variable correlation;
* maximum absolute cross-variable correlation.

## **Trend / Seasonality Features**

* trend strength;
* seasonal strength;
* lag-1 autocorrelation;
* seasonal-lag autocorrelation;
* linear trend coefficient.

## **Distribution Features**

* mean;
* standard deviation;
* coefficient of variation;
* skewness;
* kurtosis.

## **Complexity Features**

* spectral entropy;
* residual variance.

## **Data-Quality Features**

* missing-value rate;
* outlier rate.

The feature set may be expanded after the baseline router has been evaluated.

Adding large feature libraries before completing the baseline experiment is not part of the initial protocol.

---

# **14. Multivariate Feature Aggregation**

For multivariate datasets, univariate statistical features will first be calculated independently for each variable.

They will then be summarized using statistics such as:

* mean;
* standard deviation;
* median;
* maximum.

For example:

`trend_strength_mean`

`trend_strength_std`

`acf1_mean`

`acf1_max`

Dataset-level structural features such as `n_variates` and inter-variable correlation are calculated separately.

This approach provides a fixed-dimensional meta-feature vector even when datasets contain different numbers of variables.

---

# **15. Forecast Evaluation Metrics**

## **Primary Metric**

MASE — Mean Absolute Scaled Error.

MASE is selected as the primary metric because it allows forecasting errors to be compared more meaningfully across time series with different numerical scales.

## **Secondary Metrics**

* MAE;
* sMAPE;
* RMSE.

Probabilistic metrics such as CRPS may be added later but are not required for the first complete experiment.

---

# **16. Ground-Truth Router Label**

For each forecasting instance, let:

$$
E_m
$$

denote the MASE obtained by candidate model \(m\).

The ground-truth model label is:

$$
y = \arg\min_m E_m
$$

Therefore:

$$
y \in
\{
Chronos2,
Moirai2,
TimesFM
\}
$$

The complete error vector must also be stored.

For example:

```text
chronos_mase = 0.82
moirai_mase  = 0.71
timesfm_mase = 0.76

best_model = moirai2
```

Keeping the complete error vector permits later regression-based routing experiments.

---

# **17. Meta-Learning Dataset Schema**

The final meta-dataset will contain approximately the following structure:

```text
instance_id
dataset
series_id
domain
frequency
window_start
context_length
prediction_length

n_variates

trend_strength_mean
trend_strength_std
seasonal_strength_mean
seasonal_strength_std

acf1_mean
seasonal_acf_mean

spectral_entropy_mean

mean
std
cv
skewness
kurtosis

mean_cross_correlation
max_cross_correlation

missing_rate
outlier_rate

chronos_mase
moirai_mase
timesfm_mase

chronos_mae
moirai_mae
timesfm_mae

best_model
```

This meta-dataset is considered one of the project's primary research outputs.

---

# **18. Baseline Model Selectors**

The learned router must be compared against the following selection strategies.

| Selector      | Description                                                     |
| ------------- | --------------------------------------------------------------- |
| Random        | Randomly selects one TSFM                                       |
| Fixed Chronos | Always selects Chronos-2                                        |
| Fixed Moirai  | Always selects Moirai-2                                         |
| Fixed TimesFM | Always selects TimesFM                                          |
| Single Best   | Always uses the model with the lowest training-set average MASE |
| Oracle        | Selects the actual lowest-error model after observing results   |
| Meta-Router   | Predicts the model from meta-features                           |

The Oracle represents an unreachable upper-bound selection strategy.

---

# **19. Meta-Learner**

The primary meta-learning model is:

XGBoost multiclass classifier.

The first comparison model is:

Random Forest classifier.

Input:

$$
X = \text{time-series meta-features}
$$

Output:

$$
P(model \mid X)
$$

Prediction:

$$
\hat{y} =
\arg\max_m P(m \mid X)
$$

No neural-network meta-model is required for the primary experiment.

---

# **20. Generalization Protocol**

Random splitting of forecasting windows is not accepted as the primary evaluation.

The primary evaluation uses:

Leave-One-Dataset-Out Cross-Validation.

For six datasets, six experiments are performed.

Example:

```text
Fold 1
Train:
Weather
Electricity
Traffic
Exchange
Solar

Test:
ETTh1
```

The process is repeated until every dataset has served as the unseen test dataset.

Windows from the held-out dataset must not appear during meta-model training.

This evaluates whether the meta-router learns relationships between time-series characteristics and model behavior rather than memorizing dataset identity.

---

# **21. Secondary Validation**

A random or temporally grouped within-dataset split may be performed as a secondary diagnostic experiment.

It must not replace Leave-One-Dataset-Out evaluation.

Results from random splitting must be clearly labelled because windows belonging to the same dataset are statistically related.

---

# **22. Router Evaluation Metrics**

The meta-router will be evaluated using both classification metrics and forecasting outcomes.

## Classification Metrics

* selection accuracy;
* macro F1;
* confusion matrix;
* top-2 accuracy.

## Forecasting Metrics

* MASE of selected forecasts;
* MAE of selected forecasts;
* improvement over Single Best;
* regret relative to Oracle.

---

# **23. Selection Regret**

For forecasting instance \(i\):

$$
Regret_i =
E_{\text{router},i}
-
E_{\text{oracle},i}
$$

Lower regret is better.

An ideal router has:

$$
Regret = 0
$$

---

# **24. Oracle Gap Closure**

The system will additionally report:

$$
OGC =
\frac{
E_{\text{single-best}} -
E_{\text{router}}
}{
E_{\text{single-best}} -
E_{\text{oracle}}
}
$$

Interpretation:

```text
OGC = 0
→ router provides no improvement over Single Best.

OGC = 1
→ router reaches Oracle performance.
```

Values between zero and one indicate the fraction of available adaptive-selection improvement captured by the router.

---

# **25. Interpretability**

XGBoost/Random Forest feature importance will be calculated.

The primary interpretability analysis will use:

SHAP.

Questions investigated include:

Does forecast horizon influence model selection?

Does stronger seasonality favor particular models?

Does increasing dimensionality change model preference?

Does high spectral entropy change model preference?

Does stronger autocorrelation change model selection?

Interpretability analysis must describe associations observed in the experimental data and should not automatically be interpreted as causal relationships.

---

# **26. Reproducibility**

Global random seed:

```text
2026
```

Where possible, the following seeds will be fixed:

```python
random.seed(2026)
numpy.random.seed(2026)
torch.manual_seed(2026)
```

CUDA deterministic behavior will be enabled where technically appropriate.

Every experimental result must record:

```text
model
checkpoint
dataset
dataset split
context length
prediction length
window identifier
random seed
device
precision
runtime
timestamp
```

---

# **27. Caching Policy**

Forecasts must be cached immediately after generation.

A completed foundation-model inference run must not need to be repeated simply to train a different meta-model.

Recommended structure:

```text
results/
    forecasts/
    metrics/
    features/
    meta_dataset/
```

Raw model predictions and aggregated metrics should be stored separately.

---

# **28. Pilot Success Criterion**

Before conducting the full benchmark, the pilot must answer:

> Is there sufficient variation in the winning model to justify adaptive routing?

The pilot consists of:

```text
Datasets:
ETTh1
Weather
Electricity

Models:
Chronos-2
Moirai-2
TimesFM

Horizons:
24
96
192

Windows:
approximately 20 per dataset-horizon
```

The routing experiment proceeds if model winners display meaningful variation across instances.

If one model dominates almost every forecasting instance, the project will first investigate:

* whether the datasets are sufficiently diverse;
* whether the evaluation metric is appropriate;
* whether additional domains should be included;
* whether the task should be expanded to heterogeneous forecasting-model routing.

This decision must be based on experimental evidence.

---

# **29. Scope Excluded From the Primary Four-Week Run**

The following are explicitly postponed until the primary router is completed:

```text
Fine-tuning TSFMs
Few-shot adaptation
Soft forecast ensembles
Neural meta-learners
Cost-aware routing
Large feature libraries
All GIFT-Eval datasets
Web application
Full PatchTST/iTransformer training study
Extensive hyperparameter optimization
```

These remain valid thesis extensions after the primary experiment is working.

---

# **30. Definition of a Successful Four-Week Run**

The four-week sprint is considered successful when the project contains:

1. a reproducible forecasting pipeline;
2. three working pretrained TSFM wrappers;
3. at least five to six heterogeneous datasets;
4. rolling forecasting-instance generation;
5. interpretable meta-feature extraction;
6. cached forecasts and errors;
7. a complete meta-learning dataset;
8. Random Forest/XGBoost routing;
9. Leave-One-Dataset-Out evaluation;
10. comparison against fixed, Single-Best, and Oracle selectors;
11. SHAP or feature-importance analysis;
12. initial research figures and tables;
13. a documented interpretation of whether adaptive selection improves forecasting performance.

The optional extensions are not required to declare the primary experiment complete.
