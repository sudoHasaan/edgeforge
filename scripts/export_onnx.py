"""Export trained models to ONNX (dynamic batch, opset 17) and verify against PyTorch.

    python scripts/export_onnx.py                              # all students, baseline + distilled
    python scripts/export_onnx.py --model mobilenetv2 --variant distilled
    python scripts/export_onnx.py --model resnet50             # teacher
    python scripts/export_onnx.py --model nas                  # NAS model (Question 2)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.inference import load_model
from src.common.onnx_utils import export_onnx
from src.common.paths import MODEL_CHOICES, VARIANT_CHOICES, ckpt_path, onnx_path, resolve_targets
from src.common.utils import file_size_mb


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", nargs="+", default=["all"], choices=MODEL_CHOICES)
    ap.add_argument("--variant", nargs="+", default=["all"], choices=VARIANT_CHOICES)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    for model_name, variant in resolve_targets(args.model, args.variant):
        src, dst = ckpt_path(model_name, variant), onnx_path(model_name, variant)
        if not src.exists():
            print(f"[skip] {model_name}/{variant}: no checkpoint at {src}")
            continue
        model, meta = load_model(src, "cpu")
        diff = export_onnx(model, dst, meta.get("image_size", 32), meta, args.opset)
        status = "OK" if diff < 1e-3 else "CHECK"
        print(f"[{status}] {model_name}/{variant} -> {dst} ({file_size_mb(dst):.2f} MB, "
              f"max |torch - ort| = {diff:.2e})")


if __name__ == "__main__":
    main()
