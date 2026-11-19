"""Unified PyTorch inference for every trained model (teacher, students, NAS).

Checkpoints carry their own metadata (architecture, classes, normalisation, input size),
so a single `Predictor` can load any of them without extra arguments.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .utils import load_checkpoint


def build_model_from_meta(meta: dict) -> torch.nn.Module:
    family = meta.get("family", "kd")
    if family == "kd":
        from ..kd.models import build_model
        return build_model(meta["arch"], meta["num_classes"], pretrained=False)
    if family == "nas":
        from ..nas.generator import build_network
        return build_network(meta["config"], meta["num_classes"])
    raise ValueError(f"Unknown model family '{family}'")


def load_model(path, device="cpu"):
    ck = load_checkpoint(path)
    model = build_model_from_meta(ck["meta"])
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ck["meta"]


def to_uint8_batch(images) -> np.ndarray:
    """Accept file paths, PIL images or a uint8 array [N,H,W,3]; return uint8 [N,H,W,3]."""
    if isinstance(images, np.ndarray):
        return images[None] if images.ndim == 3 else images
    if isinstance(images, (str, Path, Image.Image)):
        images = [images]
    out = []
    for im in images:
        if not isinstance(im, Image.Image):
            im = Image.open(im)
        out.append(np.asarray(im.convert("RGB")))
    return out  # list (images may differ in size)


def preprocess(images, image_size, mean, std) -> np.ndarray:
    """Resize + normalise to float32 NCHW. Shared by PyTorch and ONNX Runtime pipelines."""
    batch = to_uint8_batch(images)
    arrs = []
    for a in batch:
        if a.shape[0] != image_size or a.shape[1] != image_size:
            a = np.asarray(Image.fromarray(a).resize((image_size, image_size), Image.BILINEAR))
        arrs.append(a)
    x = np.stack(arrs).astype(np.float32) / 255.0
    x = (x - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2))


def topk_from_logits(logits: np.ndarray, classes, k=5):
    z = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    idx = np.argsort(-p, axis=1)[:, :k]
    return [[(classes[j], float(p[i, j])) for j in row] for i, row in enumerate(idx)]


class Predictor:
    """PyTorch predictor: `Predictor(ckpt).predict(images)` -> top-k (label, prob) per image."""

    def __init__(self, ckpt_path, device="cpu"):
        self.device = torch.device(device)
        self.model, self.meta = load_model(ckpt_path, self.device)
        self.classes = self.meta["classes"]
        self.image_size = self.meta.get("image_size", 32)
        self.mean, self.std = self.meta["mean"], self.meta["std"]

    @torch.no_grad()
    def logits(self, images) -> np.ndarray:
        x = torch.from_numpy(preprocess(images, self.image_size, self.mean, self.std)).to(self.device)
        return self.model(x).float().cpu().numpy()

    def predict(self, images, topk=5):
        return topk_from_logits(self.logits(images), self.classes, topk)

    @torch.no_grad()
    def evaluate(self, loader) -> dict:
        from .trainer import evaluate
        return evaluate(self.model, loader, self.device)
