import platform
import sys

import torch
import importlib.metadata


def get_environment_info() -> dict:
    info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "chronos_version": importlib.metadata.version("chronos-forecasting"),}

    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory
        info["gpu_memory_gb"] = total_memory / 1024**3

    return info