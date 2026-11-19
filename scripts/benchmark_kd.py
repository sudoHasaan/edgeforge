"""Benchmark teacher + all students (baseline and distilled).

For each model: test accuracy (PyTorch), parameters, checkpoint size, PyTorch CPU latency,
and - if exported - ONNX size, ONNX Runtime CPU latency and ONNX Runtime accuracy.

    python scripts/benchmark_kd.py                 # -> results/kd_benchmark.csv
    python scripts/benchmark_kd.py --runs 200 --threads 1
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.common.benchmark import environment_info, torch_cpu_latency
from src.common.data import get_loaders
from src.common.inference import load_model
from src.common.onnx_utils import OnnxPredictor
from src.common.paths import DATA_DIR, RESULTS_DIR, ckpt_path, onnx_path, resolve_targets
from src.common.trainer import evaluate
from src.common.utils import count_params, file_size_mb, get_device, save_json, write_csv


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=100, help="timed runs per latency measurement")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--threads", type=int, default=None, help="CPU threads (default: all)")
    ap.add_argument("--skip-ort-eval", action="store_true", help="skip ONNX Runtime accuracy check")
    ap.add_argument("--out", default=str(RESULTS_DIR / "kd_benchmark.csv"))
    args = ap.parse_args()

    device = get_device()
    test = get_loaders("cifar100", 256, device).test
    arrays = None
    env = environment_info()
    print(f"CPU: {env['cpu']} ({env['cpu_cores_logical']} logical cores) | torch {env['torch']} "
          f"| onnxruntime {env['onnxruntime']}")

    rows = []
    for model_name, variant in resolve_targets(["resnet50", "all"], ["all"]):
        ck = ckpt_path(model_name, variant)
        if not ck.exists():
            print(f"[skip] {model_name}/{variant}: no checkpoint")
            continue
        model, meta = load_model(ck, device)
        acc = evaluate(model, test, device)
        lat = torch_cpu_latency(model, 32, 1, args.warmup, args.runs, args.threads)
        row = {
            "model": model_name, "variant": variant,
            "params_M": round(count_params(model) / 1e6, 3),
            "ckpt_MB": round(file_size_mb(ck), 2),
            "top1": round(acc["top1"], 2), "top5": round(acc["top5"], 2),
            "torch_cpu_ms": round(lat["mean_ms"], 3), "torch_cpu_p95_ms": round(lat["p95_ms"], 3),
        }
        op = onnx_path(model_name, variant)
        if op.exists():
            sess = OnnxPredictor(op, args.threads)
            ol = sess.latency(1, args.warmup, args.runs)
            row.update({"onnx_MB": round(file_size_mb(op), 2), "ort_cpu_ms": round(ol["mean_ms"], 3),
                        "ort_cpu_p95_ms": round(ol["p95_ms"], 3),
                        "ort_speedup_x": round(lat["mean_ms"] / ol["mean_ms"], 2)})
            if not args.skip_ort_eval:
                if arrays is None:
                    from torchvision import datasets
                    ds = datasets.CIFAR100(str(DATA_DIR), train=False, download=True)
                    arrays = (ds.data, np.asarray(ds.targets))
                row["ort_top1"] = round(sess.evaluate(*arrays)["top1"], 2)
        rows.append(row)
        print(f"{model_name:<20} {variant:<10} top-1 {row['top1']:6.2f}% | {row['params_M']:7.3f}M params | "
              f"torch {row['torch_cpu_ms']:7.2f} ms | ort {row.get('ort_cpu_ms', float('nan')):7.2f} ms")

    # KD gain relative to baseline for each student
    base = {r["model"]: r["top1"] for r in rows if r["variant"] == "baseline"}
    for r in rows:
        if r["variant"] == "distilled" and r["model"] in base:
            r["kd_gain_pp"] = round(r["top1"] - base[r["model"]], 2)

    write_csv(args.out, rows)
    save_json(Path(args.out).with_name("kd_benchmark_env.json"), env)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
