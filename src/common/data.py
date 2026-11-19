"""CIFAR-10 / CIFAR-100 data pipeline.

The whole dataset (~150 MB as uint8) is kept on the training device and augmentation
(random crop with 4px padding + horizontal flip) runs as tensor ops. On Colab this removes
the CPU dataloader bottleneck and makes each epoch several times faster than a PIL pipeline.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torchvision import datasets

from .paths import DATA_DIR

CIFAR_STATS = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
}
NUM_CLASSES = {"cifar10": 10, "cifar100": 100}


def load_raw(name: str, train: bool, root=DATA_DIR):
    """Return (images uint8 [N,3,32,32], labels int64 [N], class names)."""
    cls = datasets.CIFAR10 if name == "cifar10" else datasets.CIFAR100
    ds = cls(str(root), train=train, download=True)
    x = torch.from_numpy(ds.data).permute(0, 3, 1, 2).contiguous()
    y = torch.tensor(ds.targets, dtype=torch.long)
    return x, y, list(ds.classes)


def random_crop_flip(x: torch.Tensor, pad: int = 4) -> torch.Tensor:
    """Per-sample random crop (zero padding) and horizontal flip on a batch [B,C,H,W]."""
    b, c, h, w = x.shape
    dev = x.device
    xp = F.pad(x, (pad, pad, pad, pad))
    i = torch.randint(0, 2 * pad + 1, (b,), device=dev)
    j = torch.randint(0, 2 * pad + 1, (b,), device=dev)
    rows = (i[:, None] + torch.arange(h, device=dev))[:, None, :, None]
    cols = (j[:, None] + torch.arange(w, device=dev))[:, None, None, :]
    bi = torch.arange(b, device=dev)[:, None, None, None]
    ci = torch.arange(c, device=dev)[None, :, None, None]
    out = xp[bi, ci, rows, cols]
    flip = torch.rand(b, device=dev) < 0.5
    return torch.where(flip[:, None, None, None], out.flip(3), out)


class TensorLoader:
    """Minimal DataLoader replacement over in-memory uint8 tensors."""

    def __init__(self, x, y, batch_size, mean, std, train, device, image_size=32):
        self.device = torch.device(device)
        self.x = x.to(self.device)
        self.y = y.to(self.device)
        self.batch_size = batch_size
        self.train = train
        self.image_size = image_size
        self.mean = torch.tensor(mean, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=self.device).view(1, 3, 1, 1)

    def __len__(self):
        n = len(self.x)
        return n // self.batch_size if self.train else -(-n // self.batch_size)

    def __iter__(self):
        n = len(self.x)
        idx = torch.randperm(n, device=self.device) if self.train else torch.arange(n, device=self.device)
        for k in range(len(self)):
            b = idx[k * self.batch_size:(k + 1) * self.batch_size]
            xb = self.x[b].float().div_(255.0)
            if self.train:
                xb = random_crop_flip(xb)
            xb = (xb - self.mean) / self.std
            if self.image_size != xb.shape[-1]:
                xb = F.interpolate(xb, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
            yield xb, self.y[b]


@dataclass
class DataBundle:
    train: TensorLoader
    val: TensorLoader
    test: TensorLoader
    num_classes: int
    classes: list
    mean: tuple
    std: tuple


def get_loaders(name="cifar100", batch_size=128, device="cpu", image_size=32,
                val_split=0, seed=0, eval_batch_size=512, root=DATA_DIR) -> DataBundle:
    """Build train / val / test loaders.

    val_split > 0 holds out that many training images as a validation set (used by NAS so the
    test set is never used for model selection). With val_split = 0 the test set doubles as val.
    """
    mean, std = CIFAR_STATS[name]
    xtr, ytr, classes = load_raw(name, True, root)
    xte, yte, _ = load_raw(name, False, root)
    if val_split > 0:
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(xtr), generator=g)
        vi, ti = perm[:val_split], perm[val_split:]
        xva, yva = xtr[vi], ytr[vi]
        xtr, ytr = xtr[ti], ytr[ti]
    else:
        xva, yva = xte, yte
    mk = lambda x, y, bs, tr: TensorLoader(x, y, bs, mean, std, tr, device, image_size)
    return DataBundle(
        train=mk(xtr, ytr, batch_size, True),
        val=mk(xva, yva, eval_batch_size, False),
        test=mk(xte, yte, eval_batch_size, False),
        num_classes=NUM_CLASSES[name], classes=classes, mean=mean, std=std,
    )
