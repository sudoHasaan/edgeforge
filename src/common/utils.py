"""Small helpers shared by the KD and NAS modules."""
import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def topk_correct(logits: torch.Tensor, target: torch.Tensor, ks=(1, 5)):
    """Number of correct predictions in the top-k, for each k."""
    maxk = min(max(ks), logits.size(1))
    pred = logits.topk(maxk, dim=1).indices
    hit = pred.eq(target.view(-1, 1))
    return [hit[:, : min(k, maxk)].any(dim=1).sum() for k in ks]


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def file_size_mb(path) -> float:
    return os.path.getsize(path) / (1024 ** 2)


def save_checkpoint(path, model: torch.nn.Module, meta: dict) -> None:
    """Save weights (fp32, contiguous, on CPU) together with metadata needed to rebuild the model."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().float().cpu().contiguous() if v.is_floating_point() else v.detach().cpu()
             for k, v in model.state_dict().items()}
    torch.save({"state_dict": state, "meta": meta}, path)


def load_checkpoint(path, map_location="cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)


def is_complete(path) -> bool:
    """True if a checkpoint exists and its training run finished."""
    path = Path(path)
    if not path.exists():
        return False
    return bool(load_checkpoint(path)["meta"].get("completed", False))


def write_csv(path, rows: list[dict]) -> None:
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    for r in rows[1:]:
        fields += [k for k in r if k not in fields]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def save_json(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path) as f:
        return json.load(f)
