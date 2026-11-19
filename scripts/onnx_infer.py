"""CPU inference with ONNX Runtime (PyTorch is not used for inference).

    python scripts/onnx_infer.py --model mobilenetv2 --variant distilled --image cat.jpg
    python scripts/onnx_infer.py --model all --variant all --eval          # test-set accuracy
    python scripts/onnx_infer.py --model all --variant all --latency       # CPU latency
    python scripts/onnx_infer.py --onnx onnx/nas/best_nas.onnx --image img.png
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import onnxruntime as ort

from src.common.onnx_utils import OnnxPredictor
from src.common.paths import DATA_DIR, MODEL_CHOICES, VARIANT_CHOICES, onnx_path, resolve_targets


def load_test_arrays(dataset: str):
    """CIFAR test set as uint8 [N,32,32,3] and int labels."""
    from torchvision import datasets
    cls = datasets.CIFAR10 if dataset == "cifar10" else datasets.CIFAR100
    ds = cls(str(DATA_DIR), train=False, download=True)
    return ds.data, np.asarray(ds.targets)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", nargs="+", default=["all"], choices=MODEL_CHOICES)
    ap.add_argument("--variant", nargs="+", default=["distilled"], choices=VARIANT_CHOICES)
    ap.add_argument("--onnx", help="explicit .onnx path")
    ap.add_argument("--image", nargs="+")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--samples", type=int, default=0, help="predict N random test images")
    ap.add_argument("--latency", action="store_true")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()
    print(f"ONNX Runtime {ort.__version__} | CPUExecutionProvider")

    targets = [(Path(args.onnx).stem, "custom", Path(args.onnx))] if args.onnx else \
        [(m, v, onnx_path(m, v)) for m, v in resolve_targets(args.model, args.variant)]

    cache = {}
    for name, variant, path in targets:
        if not path.exists():
            print(f"[skip] {name}/{variant}: {path} not found (run scripts/export_onnx.py first)")
            continue
        p = OnnxPredictor(path, args.threads)
        print(f"\n## {name} ({variant}) - {path}")
        if args.image:
            for img, res in zip(args.image, p.predict(args.image, args.topk)):
                print(f"  {img}: " + ", ".join(f"{c} ({pr:.1%})" for c, pr in res))
        ds = p.meta.get("dataset", "cifar100")
        if (args.eval or args.samples) and ds not in cache:
            cache[ds] = load_test_arrays(ds)
        if args.samples:
            x, y = cache[ds]
            idx = np.random.default_rng(0).choice(len(x), args.samples, replace=False)
            for res, lbl in zip(p.predict(x[idx], args.topk), y[idx]):
                mark = "OK " if res[0][0] == p.classes[lbl] else "BAD"
                print(f"  [{mark}] true={p.classes[lbl]:<15} pred=" + ", ".join(f"{c} ({pr:.1%})" for c, pr in res))
        if args.eval:
            r = p.evaluate(*cache[ds])
            print(f"  ORT test top-1 {r['top1']:.2f}% | top-5 {r['top5']:.2f}%")
        if args.latency:
            r = p.latency(batch_size=1)
            print(f"  ORT CPU latency (bs=1): mean {r['mean_ms']:.2f} ms | p95 {r['p95_ms']:.2f} ms")


if __name__ == "__main__":
    main()
