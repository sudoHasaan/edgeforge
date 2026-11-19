"""CPU latency measurement and environment info."""
import os
import platform

import torch

from .onnx_utils import time_fn


@torch.no_grad()
def torch_cpu_latency(model, image_size: int, batch_size=1, warmup=10, runs=100, threads: int | None = None) -> dict:
    prev = torch.get_num_threads()
    if threads:
        torch.set_num_threads(threads)
    try:
        model = model.eval().cpu()
        x = torch.randn(batch_size, 3, image_size, image_size)
        return time_fn(lambda: model(x), warmup, runs)
    finally:
        torch.set_num_threads(prev)


def cpu_name() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def environment_info() -> dict:
    import onnxruntime as ort
    return {
        "cpu": cpu_name(),
        "cpu_cores_logical": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "torch": torch.__version__,
        "onnxruntime": ort.__version__,
        "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
