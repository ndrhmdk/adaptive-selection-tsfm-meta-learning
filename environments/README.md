# Shared research environment

Use `.venv` (Python 3.11) for Chronos-2, Moirai-2.0, TimesFM-3.0, notebooks,
data processing and the future meta-learner. `uv.lock` freezes the shared stack.

From the repository root in PowerShell:

```powershell
uv sync --locked
uv run --locked python scripts/check_shared_environment.py
uv run --locked python scripts/check_shared_environment.py --forecast --device cuda
```

The forecast check uses cached or downloadable checkpoints and runs all three
models sequentially in the same process, with one and seven variables. It releases
each model before loading the next to limit GPU memory use. Use `--device cpu`
if needed. Sharing an environment does not require keeping all models in GPU memory.

Run scripts with `uv run --locked python scripts/run_chronos_etth1.py` (or the
Moirai/TimesFM script). Select `.venv/Scripts/python.exe` for every notebook and
restart existing kernels once. Alternatively use `uv run --locked jupyter lab`.
Optional shell activation: `.\.venv\Scripts\Activate.ps1`.

## Compatibility choices

- Python 3.11 and PyTorch 2.4.1 with CUDA 12.4 on Windows/Linux.
- Uni2TS 2.0.0 requires PyTorch below 2.5, NumPy 1.26 and SciPy 1.11;
  its GluonTS dependency constrains pandas below 2.2.
- Chronos 2.3.2 uses Transformers 4.x below 4.50 in this stack.
- PyArrow stays below 21 for the legacy datasets 2.17 API used by Uni2TS.
- TimesFM uses the previous environment's upstream commit,
  `331c6d33cb1ac2611de3056d0ac7164aab6301eb`.
- uv resolves dependencies normally, without --no-deps or dependency overrides.

Upstream manifests: [Uni2TS](https://github.com/SalesforceAIResearch/uni2ts/blob/main/pyproject.toml)
and [pinned TimesFM](https://github.com/google-research/timesfm/blob/331c6d33cb1ac2611de3056d0ac7164aab6301eb/pyproject.toml).

## Proposal coverage and future models

The proposal specifies Chronos-2, Moirai-2.0 and TimesFM-2.5 or 3.0. The project
adapter uses 3.0. This TimesFM package also contains 2.5, but a project adapter and
checkpoint test are still required before using 2.5 in experiments.

Random Forest (scikit-learn), XGBoost 3.0.5 and SHAP 0.48 are installed for the proposed
interpretable router. The environment check fits both classifiers and computes
SHAP explanations on synthetic data. This verifies library compatibility, not
the accuracy of a future trained router. Windows/Linux use the official CPU-only
XGBoost build for the lightweight selector; the three TSFMs retain CUDA support.
XGBoost stays below 3.1 to retain the model format SHAP 0.48 understands.

The scientific Python stack supports data processing, features, metrics, plotting
and Parquet. Optional DLinear, PatchTST and iTransformer baselines still need an
implementation choice and model tests; having PyTorch does not establish
compatibility with every third-party implementation. Compatibility with unreleased
models cannot be guaranteed.

For future dependencies, use `uv add ...`, review lockfile changes, and rerun the
shared forecast check. Freeze checkpoint revisions separately for final experiments.

The old `.venv-uni2ts`, `.venv-timesfm` and `*.freeze.txt` files are historical
references. `shared-requirements.txt` lists common utilities only; use
`uv sync --locked` to reproduce the complete shared environment.

## Validation on this machine (2026-09-21)

`uv sync --locked` audited the installed environment; `uv pip check` reported
all 182 packages compatible. The shared check passed imports, Random Forest
and XGBoost training/prediction/SHAP, ETTh1 loading, metrics, Parquet and plotting.
With cached checkpoints and CUDA on the RTX 3050, all three model adapters
produced finite point and interval forecasts for a 512-step context and
24-step horizon, using one and seven variables in the same Python process.
These are smoke checks, not exhaustive horizon/dataset or accuracy validation.
