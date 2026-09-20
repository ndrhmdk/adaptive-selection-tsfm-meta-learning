# **Experimental Environments**

The TSFMs use isolated Python environments because their official
dependency stacks are not mutually compatible.

## **Core / Chronos**

Environment:
.venv

Used for:
- shared data pipeline
- Chronos-2
- analysis

## **Moirai-2**

Environment:
.venv-uni2ts

Used for:
- Salesforce Uni2TS
- Moirai-2.0-R-small

Reason:
Uni2TS dependencies conflict with the Chronos/core environment.

## **TimesFM 3.0**

Environment:
.venv-timesfm

Used for:
- TimesFM 3.0 PyTorch

Install the shared data, evaluation, Parquet export, plotting, and notebook
dependencies in this environment. Installing TimesFM alone does not provide
these project dependencies. From the repository root, run:

```powershell
uv pip install --python .venv-timesfm/Scripts/python.exe -r environments/shared-requirements.txt
uv pip check --python .venv-timesfm/Scripts/python.exe
```

Select `.venv-timesfm` as the notebook kernel and restart the kernel after
installing dependencies.

Check imports, ETTh1 loading, metric computation, plotting, and Parquet export:

```powershell
.venv-timesfm/Scripts/python.exe scripts/check_timesfm_environment.py
```

To also run a real TimesFM forecast and save/read back its outputs in a temporary
directory, add `--forecast`. This requires the model checkpoint to be cached or
downloadable.

All environments operate on the same:
- source code
- datasets
- forecasting windows
- metric definitions
- output files
