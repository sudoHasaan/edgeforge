"""Distil the trained ResNet-50 teacher into the student networks.

Loss = (1-alpha)*CE + alpha*T^2*KL(teacher_T || student_T) + beta*FeatureHint(mid, final)

    python scripts/train_kd.py --students all --epochs 30                  # full KD (logits + features)
    python scripts/train_kd.py --students mobilenetv2 --beta 0             # logit-only KD
    python scripts/train_kd.py --alpha 1.0 --beta 0 --out-dir checkpoints/ablation_soft_only
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.common.data import get_loaders
from src.common.inference import load_model
from src.common.paths import CKPT_DIR, RESULTS_DIR, TEACHER_CKPT
from src.common.trainer import evaluate, fit
from src.common.utils import count_params, get_device, is_complete, set_seed, write_csv
from src.kd.losses import DistillationLoss
from src.kd.models import DEFAULT_LR, STUDENTS, build_model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--students", nargs="+", default=["all"], choices=STUDENTS + ["all"])
    ap.add_argument("--teacher", default=str(TEACHER_CKPT))
    ap.add_argument("--alpha", type=float, default=0.9, help="weight of soft-target loss (1-alpha for hard labels)")
    ap.add_argument("--temperature", "-T", type=float, default=4.0)
    ap.add_argument("--beta", type=float, default=1.0, help="weight of feature distillation (0 disables)")
    ap.add_argument("--feat-layers", nargs="+", default=["mid", "final"], choices=["mid", "final"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=str(CKPT_DIR / "distilled"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    students = STUDENTS if args.students == ["all"] else args.students

    device = get_device()
    data = get_loaders("cifar100", args.batch_size, device)

    teacher, tmeta = load_model(args.teacher, device)
    teacher.requires_grad_(False)
    if device.type == "cuda":
        teacher.to(memory_format=torch.channels_last)
    t_acc = evaluate(teacher, data.test, device)
    print(f"Teacher {tmeta['arch']}: top-1 {t_acc['top1']:.2f}% | top-5 {t_acc['top5']:.2f}%")

    tag = Path(args.out_dir).name
    summary = []
    for name in students:
        out = Path(args.out_dir) / f"{name}.pth"
        print(f"\n=== Distillation: {tmeta['arch']} -> {name} ===")
        if is_complete(out) and not args.force:
            print(f"  already trained: {out}")
            continue
        set_seed(args.seed)
        student = build_model(name, data.num_classes, pretrained=not args.no_pretrained).to(device)
        criterion = DistillationLoss(student.feature_channels, teacher.feature_channels, alpha=args.alpha,
                                     temperature=args.temperature, beta=args.beta,
                                     feat_layers=args.feat_layers).to(device)
        if device.type == "cuda":
            criterion.to(memory_format=torch.channels_last)
        lr = args.lr or DEFAULT_LR[name]
        print(f"  {count_params(student) / 1e6:.2f}M params | lr {lr} | alpha {args.alpha} | "
              f"T {args.temperature} | beta {args.beta} | hint layers {criterion.feat_layers}")

        def kd_step(model, x, y):
            teacher.eval()
            with torch.no_grad():
                t_logits, t_feats = teacher(x, return_feats=True)
            s_logits, s_feats = model(x, return_feats=True)
            loss, _ = criterion(s_logits, s_feats, t_logits, t_feats, y)
            return loss, s_logits

        meta = {"family": "kd", "arch": name, "variant": "distilled", "dataset": "cifar100",
                "num_classes": data.num_classes, "classes": data.classes, "mean": data.mean, "std": data.std,
                "image_size": 32, "pretrained": not args.no_pretrained, "lr": lr, "batch_size": args.batch_size,
                "teacher": tmeta["arch"], "teacher_top1": t_acc["top1"], "alpha": args.alpha,
                "temperature": args.temperature, "beta": args.beta, "feat_layers": criterion.feat_layers}
        _, best = fit(student, data.train, data.test, epochs=args.epochs, lr=lr, weight_decay=args.wd,
                      device=device, ckpt_path=out, history_path=RESULTS_DIR / "history" / f"{tag}_{name}.csv",
                      loss_fn=kd_step, extra_modules=criterion, meta=meta)
        summary.append({"student": name, "variant": tag, "best_top1": round(best, 2), "alpha": args.alpha,
                        "temperature": args.temperature, "beta": args.beta})
        print(f"  {name} distilled best top-1: {best:.2f}%")
    if summary:
        write_csv(RESULTS_DIR / f"{tag}_summary.csv", summary)


if __name__ == "__main__":
    main()
