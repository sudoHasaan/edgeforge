"""Generic training loop used by the teacher, baseline, KD and NAS pipelines.

Features: SGD + Nesterov, linear warm-up + cosine LR, mixed precision on GPU, gradient
clipping, best-checkpoint saving and resume-from-last (useful when Colab disconnects).
"""
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import load_checkpoint, save_checkpoint, topk_correct, write_csv


def autocast(device: torch.device):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda")


def _prep(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    return x.contiguous(memory_format=torch.channels_last) if device.type == "cuda" else x


def cosine_with_warmup(optimizer, total_steps: int, warmup_steps: int):
    def factor(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device) -> dict:
    model.eval()
    n, loss_sum = 0, torch.zeros((), device=device)
    c1 = torch.zeros((), device=device)
    c5 = torch.zeros((), device=device)
    for x, y in loader:
        x, y = _prep(x.to(device), device), y.to(device)
        with autocast(device):
            out = model(x)
        out = out.float()
        loss_sum += F.cross_entropy(out, y, reduction="sum")
        a1, a5 = topk_correct(out, y, (1, 5))
        c1 += a1
        c5 += a5
        n += y.size(0)
    return {"loss": loss_sum.item() / n, "top1": 100.0 * c1.item() / n, "top5": 100.0 * c5.item() / n}


def ce_step(model, x, y):
    out = model(x)
    return F.cross_entropy(out.float(), y), out


def fit(model: nn.Module, train_loader, val_loader, *, epochs: int, lr: float, weight_decay: float,
        device: torch.device, ckpt_path, history_path=None, loss_fn=None, extra_modules: nn.Module = None,
        meta: dict = None, resume: bool = True, grad_clip: float = 5.0, warmup_epochs: int = 1,
        verbose: bool = True, epoch_callback=None):
    """Train `model` and keep the checkpoint with the best validation top-1.

    loss_fn(model, x, y) -> (loss, logits). Defaults to cross-entropy.
    extra_modules: additional trainable modules (e.g. KD feature connectors) optimised jointly.
    ckpt_path=None disables checkpointing/resume (used for short NAS candidate runs).
    epoch_callback(epoch, row) is called after every epoch (e.g. Optuna pruning; may raise).
    Returns (history list, best top-1).
    """
    ckpt_path = Path(ckpt_path) if ckpt_path else None
    last_path = ckpt_path.with_name(ckpt_path.stem + "_last.pth") if ckpt_path else None
    loss_fn = loss_fn or ce_step
    meta = dict(meta or {})

    if device.type == "cuda":
        model.to(memory_format=torch.channels_last)
    params = list(model.parameters()) + (list(extra_modules.parameters()) if extra_modules else [])
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=True, weight_decay=weight_decay)
    steps = len(train_loader)
    scheduler = cosine_with_warmup(optimizer, epochs * steps, warmup_epochs * steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    start_epoch, best, history = 0, -1.0, []
    if resume and last_path is not None and last_path.exists():
        st = load_checkpoint(last_path, map_location=device)
        model.load_state_dict(st["model"])
        if extra_modules is not None and st.get("extra") is not None:
            extra_modules.load_state_dict(st["extra"])
        optimizer.load_state_dict(st["optimizer"])
        scheduler.load_state_dict(st["scheduler"])
        scaler.load_state_dict(st["scaler"])
        start_epoch, best, history = st["epoch"], st["best"], st["history"]
        if verbose:
            print(f"  resuming from epoch {start_epoch} (best top-1 so far {best:.2f}%)")

    for epoch in range(start_epoch, epochs):
        model.train()
        if extra_modules is not None:
            extra_modules.train()
        t0 = time.time()
        loss_sum = torch.zeros((), device=device)
        correct = torch.zeros((), device=device)
        seen = 0
        for x, y in train_loader:
            x, y = _prep(x.to(device), device), y.to(device)
            with autocast(device):
                loss, out = loss_fn(model, x, y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(params, grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            loss_sum += loss.detach().float() * y.size(0)
            correct += (out.detach().argmax(1) == y).sum()
            seen += y.size(0)

        val = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch + 1,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": round(loss_sum.item() / seen, 4),
            "train_top1": round(100.0 * correct.item() / seen, 2),
            "val_loss": round(val["loss"], 4),
            "val_top1": round(val["top1"], 2),
            "val_top5": round(val["top5"], 2),
            "epoch_time_s": round(time.time() - t0, 1),
        }
        history.append(row)
        if not math.isfinite(row["train_loss"]):
            raise RuntimeError("Training diverged (non-finite loss). Try a lower learning rate.")
        if val["top1"] > best:
            best = val["top1"]
            if ckpt_path is not None:
                save_checkpoint(ckpt_path, model, {**meta, "epoch": epoch + 1, "top1": val["top1"],
                                                   "top5": val["top5"], "completed": False})
        if last_path is not None:
            torch.save({
                "model": model.state_dict(),
                "extra": extra_modules.state_dict() if extra_modules is not None else None,
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(), "epoch": epoch + 1, "best": best, "history": history,
            }, last_path)
        if history_path:
            write_csv(history_path, history)
        if verbose:
            print(f"  ep {epoch + 1:3d}/{epochs} | loss {row['train_loss']:.3f} | train {row['train_top1']:5.2f}% "
                  f"| val {row['val_top1']:5.2f}% (top5 {row['val_top5']:5.2f}%) | best {best:5.2f}% "
                  f"| {row['epoch_time_s']}s", flush=True)
        if epoch_callback is not None:
            epoch_callback(epoch + 1, row)

    # Mark the best checkpoint as complete and drop the bulky resume file.
    if ckpt_path is not None and ckpt_path.exists():
        st = load_checkpoint(ckpt_path)
        st["meta"]["completed"] = True
        st["meta"]["epochs"] = epochs
        torch.save(st, ckpt_path)
    if last_path is not None and last_path.exists():
        last_path.unlink()
    return history, best
