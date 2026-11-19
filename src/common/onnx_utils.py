"""ONNX export and ONNX Runtime (CPU) inference.

Exported models embed their metadata (class names, normalisation, input size) as ONNX
`metadata_props`, so `OnnxPredictor` needs nothing except the .onnx file.
"""
import json
import time
from pathlib import Path

import numpy as np

from .inference import preprocess, topk_from_logits

META_KEYS = ("arch", "variant", "family", "dataset", "classes", "mean", "std", "image_size",
             "num_classes", "top1", "config")


def export_onnx(model, path, image_size: int, meta: dict | None = None, opset: int = 17) -> float:
    """Export `model` with a dynamic batch dimension, validate it and return the max abs
    difference between PyTorch and ONNX Runtime outputs on a random input."""
    import onnx
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = model.eval().cpu()
    dummy = torch.randn(2, 3, image_size, image_size)
    kwargs = dict(input_names=["input"], output_names=["logits"], opset_version=opset,
                  dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, do_constant_folding=True)
    try:
        torch.onnx.export(model, dummy, str(path), dynamo=False, **kwargs)
    except TypeError:  # older torch without the `dynamo` argument
        torch.onnx.export(model, dummy, str(path), **kwargs)

    m = onnx.load(str(path))
    onnx.checker.check_model(m)
    for k, v in (meta or {}).items():
        if k in META_KEYS:
            p = m.metadata_props.add()
            p.key, p.value = k, json.dumps(v)
    onnx.save(m, str(path))

    with torch.no_grad():
        ref = model(dummy).numpy()
    out = make_session(path).run(None, {"input": dummy.numpy()})[0]
    return float(np.abs(out - ref).max())


def make_session(path, threads: int | None = None):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads:
        so.intra_op_num_threads = threads
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


class OnnxPredictor:
    """CPU inference with ONNX Runtime."""

    def __init__(self, path, threads: int | None = None):
        self.path = Path(path)
        self.session = make_session(path, threads)
        md = self.session.get_modelmeta().custom_metadata_map
        self.meta = {k: json.loads(v) for k, v in md.items()}
        self.classes = self.meta.get("classes")
        self.image_size = self.meta.get("image_size", 32)
        self.mean, self.std = self.meta["mean"], self.meta["std"]
        self.input_name = self.session.get_inputs()[0].name

    def run(self, x: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.input_name: x.astype(np.float32, copy=False)})[0]

    def predict(self, images, topk=5):
        x = preprocess(images, self.image_size, self.mean, self.std)
        return topk_from_logits(self.run(x), self.classes, topk)

    def evaluate(self, images_nhwc: np.ndarray, labels: np.ndarray, batch_size=256) -> dict:
        """Top-1/top-5 accuracy (%) on uint8 images [N,H,W,3]."""
        c1 = c5 = 0
        for i in range(0, len(images_nhwc), batch_size):
            x = preprocess(images_nhwc[i:i + batch_size], self.image_size, self.mean, self.std)
            logits = self.run(x)
            y = labels[i:i + batch_size]
            top5 = np.argsort(-logits, axis=1)[:, :5]
            c1 += int((top5[:, 0] == y).sum())
            c5 += int((top5 == y[:, None]).any(axis=1).sum())
        n = len(labels)
        return {"top1": 100.0 * c1 / n, "top5": 100.0 * c5 / n}

    def latency(self, batch_size=1, warmup=10, runs=100) -> dict:
        x = np.random.randn(batch_size, 3, self.image_size, self.image_size).astype(np.float32)
        return time_fn(lambda: self.run(x), warmup, runs)


def time_fn(fn, warmup=10, runs=100) -> dict:
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts = np.asarray(ts)
    return {"mean_ms": float(ts.mean()), "p50_ms": float(np.percentile(ts, 50)),
            "p95_ms": float(np.percentile(ts, 95)), "std_ms": float(ts.std())}
