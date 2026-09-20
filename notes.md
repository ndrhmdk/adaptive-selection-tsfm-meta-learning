## **No. 1 - Before Running any Cells in Notebooks**

Put these cells on top
```py
%load_ext autoreload
%autoreload 2
```

```py
from pathlib import Path
import sys

working_directory = Path.cwd().resolve()
project_root = next(
    (path for path in (working_directory, *working_directory.parents)
     if (path / "pyproject.toml").is_file() and (path / "src").is_dir()),
    None,
)
```

## **No.2 - `uv` Cache**
Run these

```shell
uv cache prune
```

and 
```shell
# after you finished with the project
uv cache clean
```

## **No.3 - Standardize formatting automatically**
```shell
uvx ruff format src scripts
uvx ruff format src scripts --fix
```