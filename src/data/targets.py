import hashlib
import numpy as np
from src.data.types import TimeSeriesDataset

def _dataset_seed(name: str, seed: int) -> int:
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    offset = int.from_bytes(digest[:4], byteorder='little')
    return (seed + offset) % (2 ** 32)

def select_benchmark_targets(dataset: TimeSeriesDataset, n_targets: int, seed: int = 2026) -> list[str]:
    if n_targets <= 0:
        raise ValueError("n_targets must be positive.")
    columns = list(dataset.values.columns)
    if not columns: 
        raise ValueError(f"{dataset.name} contains no target columns.")
    if n_targets >= len(columns):
        return columns
    
    selected = []
    if dataset.primary_target and dataset.primary_target in columns:
        selected.append(dataset.primary_target)
        
    candidates = [
        column
        for column in columns
        if column not in selected]
    remaining = n_targets - len(selected)
    rng = np.random.default_rng(_dataset_seed(dataset.name, seed))
    sampled = rng.choice(candidates, size=remaining, replace=False).tolist()
    selected.extend(sampled)
    return selected
