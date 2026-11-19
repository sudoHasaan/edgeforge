"""Train the ResNet-50 teacher on CIFAR-100.

    python scripts/train_teacher.py --epochs 30
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.data import get_loaders
from src.common.paths import RESULTS_DIR, TEACHER_CKPT
from src.common.trainer import fit
from src.common.utils import count_params, get_device, is_complete, set_seed
from src.kd.models import DEFAULT_LR, TEACHER, build_model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=DEFAULT_LR[TEACHER])
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--no-pretrained", action="store_true", help="train from scratch instead of ImageNet init")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(TEACHER_CKPT))
    ap.add_argument("--force", action="store_true", help="retrain even if a finished checkpoint exists")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    if is_complete(args.out) and not args.force:
        print(f"Teacher already trained: {args.out} (use --force to retrain)")
        return

    set_seed(args.seed)
    device = get_device()
    data = get_loaders("cifar100", args.batch_size, device)
    model = build_model(TEACHER, data.num_classes, pretrained=not args.no_pretrained).to(device)
    print(f"Teacher {TEACHER}: {count_params(model) / 1e6:.2f}M params | device {device}")

    meta = {"family": "kd", "arch": TEACHER, "variant": "teacher", "dataset": "cifar100",
            "num_classes": data.num_classes, "classes": data.classes, "mean": data.mean, "std": data.std,
            "image_size": 32, "pretrained": not args.no_pretrained, "lr": args.lr, "batch_size": args.batch_size}
    _, best = fit(model, data.train, data.test, epochs=args.epochs, lr=args.lr, weight_decay=args.wd,
                  device=device, ckpt_path=args.out, history_path=RESULTS_DIR / "history" / "teacher_resnet50.csv",
                  meta=meta, resume=not args.no_resume)
    print(f"Teacher best top-1: {best:.2f}% -> {args.out}")


if __name__ == "__main__":
    main()
