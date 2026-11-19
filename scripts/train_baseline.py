"""Baseline training of the student networks on CIFAR-100 (cross-entropy only, no distillation).

    python scripts/train_baseline.py --students all --epochs 30
    python scripts/train_baseline.py --students mobilenetv2 squeezenet
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.data import get_loaders
from src.common.paths import CKPT_DIR, RESULTS_DIR
from src.common.trainer import fit
from src.common.utils import count_params, get_device, is_complete, set_seed, write_csv
from src.kd.models import DEFAULT_LR, STUDENTS, build_model


def parse_students(names):
    return STUDENTS if names == ["all"] else names


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--students", nargs="+", default=["all"], choices=STUDENTS + ["all"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=None, help="override per-model default LR")
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=str(CKPT_DIR / "baseline"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    device = get_device()
    data = get_loaders("cifar100", args.batch_size, device)
    summary = []
    for name in parse_students(args.students):
        out = Path(args.out_dir) / f"{name}.pth"
        print(f"\n=== Baseline: {name} ===")
        if is_complete(out) and not args.force:
            print(f"  already trained: {out}")
            continue
        set_seed(args.seed)
        model = build_model(name, data.num_classes, pretrained=not args.no_pretrained).to(device)
        lr = args.lr or DEFAULT_LR[name]
        print(f"  {count_params(model) / 1e6:.2f}M params | lr {lr}")
        meta = {"family": "kd", "arch": name, "variant": "baseline", "dataset": "cifar100",
                "num_classes": data.num_classes, "classes": data.classes, "mean": data.mean, "std": data.std,
                "image_size": 32, "pretrained": not args.no_pretrained, "lr": lr, "batch_size": args.batch_size}
        _, best = fit(model, data.train, data.test, epochs=args.epochs, lr=lr, weight_decay=args.wd,
                      device=device, ckpt_path=out, history_path=RESULTS_DIR / "history" / f"baseline_{name}.csv",
                      meta=meta)
        summary.append({"student": name, "variant": "baseline", "best_top1": round(best, 2)})
        print(f"  {name} baseline best top-1: {best:.2f}%")
    if summary:
        write_csv(RESULTS_DIR / "baseline_summary.csv", summary)


if __name__ == "__main__":
    main()
