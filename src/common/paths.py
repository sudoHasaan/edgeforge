"""Central place for repository paths so every script agrees on where things live."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"
ONNX_DIR = ROOT / "onnx"
RESULTS_DIR = ROOT / "results"

TEACHER_CKPT = CKPT_DIR / "teacher" / "resnet50.pth"
NAS_CKPT = CKPT_DIR / "nas" / "best_nas.pth"


def ckpt_path(model: str, variant: str) -> Path:
    """Checkpoint location for a model. variant: teacher | baseline | distilled | nas."""
    if variant == "teacher":
        return TEACHER_CKPT
    if variant == "nas":
        return NAS_CKPT
    return CKPT_DIR / variant / f"{model}.pth"


STUDENT_NAMES = ["mobilenetv2", "squeezenet", "efficientnet_lite0"]
MODEL_CHOICES = STUDENT_NAMES + ["resnet50", "nas", "all"]
VARIANT_CHOICES = ["baseline", "distilled", "all"]


def resolve_targets(models: list[str], variants: list[str]) -> list[tuple[str, str]]:
    """Expand CLI choices into (model, variant) pairs.
    'all' models = the three students; 'resnet50' -> teacher; 'nas' -> NAS model."""
    expanded = []
    for m in models:
        for x in (STUDENT_NAMES if m == "all" else [m]):
            if x not in expanded:
                expanded.append(x)
    models = expanded
    variants = ["baseline", "distilled"] if "all" in variants else variants
    out = []
    for m in models:
        if m == "resnet50":
            out.append((m, "teacher"))
        elif m == "nas":
            out.append((m, "nas"))
        else:
            out += [(m, v) for v in variants]
    return out


def onnx_path(model: str, variant: str) -> Path:
    if variant == "nas":
        return ONNX_DIR / "nas" / "best_nas.onnx"
    if variant == "teacher":
        return ONNX_DIR / "teacher" / "resnet50.onnx"
    return ONNX_DIR / variant / f"{model}.onnx"
