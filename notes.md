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
if project_root is None:
    raise RuntimeError("Open this notebook from within the project directory.")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
```