"""Hardware-aware profiling and deployment constraints.

For every candidate we export it to ONNX and measure, on the CPU:
  * inference latency  - ONNX Runtime, batch 1, fixed thread count (mean of N timed runs)
  * storage size       - size of the exported .onnx file (fp32 weights + graph)
  * peak memory        - peak RSS of a fresh ONNX Runtime process running inference (where supported)
plus parameter count and MACs. Candidates violating the constraints are rejected before training.
"""
import json
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from ..common.benchmark import torch_cpu_latency
from ..common.onnx_utils import export_onnx, make_session, time_fn
from ..common.paths import ROOT
from ..common.utils import count_params, file_size_mb
from .generator import count_macs


@dataclass
class Constraints:
    max_latency_ms: float | None = 3.0     # ONNX Runtime CPU latency, batch 1
    max_size_mb: float | None = 1.0        # ONNX file size
    max_peak_mem_mb: float | None = None   # peak inference memory (None = record only)

    def check(self, hw: dict):
        """Return (ok, list of violated constraints)."""
        reasons = []
        if self.max_latency_ms is not None and hw["latency_ms"] > self.max_latency_ms:
            reasons.append(f"latency {hw['latency_ms']:.2f}ms > {self.max_latency_ms}ms")
        if self.max_size_mb is not None and hw["size_mb"] > self.max_size_mb:
            reasons.append(f"size {hw['size_mb']:.2f}MB > {self.max_size_mb}MB")
        if self.max_peak_mem_mb is not None and hw.get("inference_mem_mb") is not None \
                and hw["inference_mem_mb"] > self.max_peak_mem_mb:
            reasons.append(f"memory {hw['inference_mem_mb']:.1f}MB > {self.max_peak_mem_mb}MB")
        return not reasons, reasons

    def to_dict(self):
        return asdict(self)


def measure_peak_memory(onnx_file, resolution: int, threads: int = 1, runs: int = 20) -> dict | None:
    """Run src.nas.memprobe in a subprocess; returns None if unsupported or failed."""
    try:
        out = subprocess.run([sys.executable, "-m", "src.nas.memprobe", str(onnx_file), str(resolution),
                              str(runs), str(threads)], cwd=str(ROOT), capture_output=True, text=True, timeout=120)
        return json.loads(out.stdout.strip().splitlines()[-1])
    except Exception:
        return None


def profile_model(model, resolution: int, threads: int = 1, runs: int = 50, warmup: int = 10,
                  measure_memory: bool = True, torch_latency: bool = False, onnx_file=None) -> dict:
    """Export `model` to ONNX and measure its CPU deployment characteristics."""
    model = model.eval().cpu()
    tmp = None
    if onnx_file is None:
        tmp = tempfile.TemporaryDirectory()
        onnx_file = Path(tmp.name) / "candidate.onnx"
    try:
        export_onnx(model, onnx_file, resolution)
        sess = make_session(onnx_file, threads)
        import numpy as np
        x = np.random.randn(1, 3, resolution, resolution).astype(np.float32)
        name = sess.get_inputs()[0].name
        lat = time_fn(lambda: sess.run(None, {name: x}), warmup, runs)
        hw = {
            "params": count_params(model),
            "macs_M": round(count_macs(model, resolution) / 1e6, 3),
            "size_mb": round(file_size_mb(onnx_file), 4),
            "latency_ms": round(lat["mean_ms"], 4),
            "latency_p95_ms": round(lat["p95_ms"], 4),
            "threads": threads,
        }
        if torch_latency:
            hw["torch_latency_ms"] = round(torch_cpu_latency(model, resolution, 1, warmup, runs, threads)["mean_ms"], 4)
        if measure_memory:
            mem = measure_peak_memory(onnx_file, resolution, threads)
            hw.update(mem if mem else {"peak_rss_mb": None, "session_rss_mb": None, "inference_mem_mb": None})
        return hw
    finally:
        if tmp is not None:
            tmp.cleanup()
