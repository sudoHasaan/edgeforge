"""Unified PyTorch inference for all trained models.

    # classify your own images with the distilled MobileNetV2
    python scripts/infer.py --model mobilenetv2 --variant distilled --image cat.jpg dog.png

    # evaluate every student (baseline + distilled) on the CIFAR-100 test set
    python scripts/infer.py --model all --variant all --eval

    # predict 8 random test images with every student
    python scripts/infer.py --model all --variant all --samples 8

    # the NAS model (Question 2) or any checkpoint file
    python scripts/infer.py --model nas --eval
    python scripts/infer.py --ckpt checkpoints/distilled/squeezenet.pth --image img.png
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.common.data import get_loaders, load_raw
from src.common.inference import Predictor
from src.common.paths import MODEL_CHOICES, VARIANT_CHOICES, ckpt_path, resolve_targets
from src.common.utils import get_device, write_csv


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", nargs="+", default=["all"], choices=MODEL_CHOICES)
    ap.add_argument("--variant", nargs="+", default=["distilled"], choices=VARIANT_CHOICES)
    ap.add_argument("--ckpt", help="explicit checkpoint path (overrides --model/--variant)")
    ap.add_argument("--image", nargs="+", help="image file(s) to classify")
    ap.add_argument("--eval", action="store_true", help="evaluate on the test set")
    ap.add_argument("--samples", type=int, default=0, help="predict N random test images")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--device", default=None)
    ap.add_argument("--save-csv", default=None, help="write --eval results to this CSV")
    args = ap.parse_args()
    device = get_device(args.device)

    targets = [(Path(args.ckpt).stem, "custom", Path(args.ckpt))] if args.ckpt else \
        [(m, v, ckpt_path(m, v)) for m, v in resolve_targets(args.model, args.variant)]

    rows, loader_cache, sample_cache = [], {}, {}
    for model_name, variant, path in targets:
        if not path.exists():
            print(f"[skip] {model_name}/{variant}: checkpoint not found at {path}")
            continue
        pred = Predictor(path, device)
        print(f"\n## {model_name} ({variant}) - {path}")

        if args.image:
            for img, res in zip(args.image, pred.predict(args.image, args.topk)):
                print(f"  {img}: " + ", ".join(f"{c} ({p:.1%})" for c, p in res))

        if args.samples:
            ds = pred.meta["dataset"]
            if ds not in sample_cache:
                x, y, classes = load_raw(ds, train=False)
                idx = np.random.default_rng(0).choice(len(x), args.samples, replace=False)
                sample_cache[ds] = (x[idx].permute(0, 2, 3, 1).numpy(), y[idx].numpy(), classes)
            imgs, labels, classes = sample_cache[ds]
            for res, lbl in zip(pred.predict(imgs, args.topk), labels):
                mark = "OK " if res[0][0] == classes[lbl] else "BAD"
                print(f"  [{mark}] true={classes[lbl]:<15} pred=" + ", ".join(f"{c} ({p:.1%})" for c, p in res))

        if args.eval:
            key = (pred.meta["dataset"], pred.image_size)
            if key not in loader_cache:
                loader_cache[key] = get_loaders(key[0], 256, device, image_size=key[1]).test
            r = pred.evaluate(loader_cache[key])
            print(f"  test top-1 {r['top1']:.2f}% | top-5 {r['top5']:.2f}%")
            rows.append({"model": model_name, "variant": variant, "top1": round(r["top1"], 2),
                         "top5": round(r["top5"], 2)})

    if rows:
        print("\nmodel                variant     top-1   top-5")
        for r in rows:
            print(f"{r['model']:<20} {r['variant']:<10} {r['top1']:6.2f}  {r['top5']:6.2f}")
        if args.save_csv:
            write_csv(args.save_csv, rows)


if __name__ == "__main__":
    main()
