"""Benchmark the NAS model and compare deployment cost of every exported ONNX model.

    python scripts/benchmark_nas.py

Outputs
    results/nas/nas_benchmark.csv        final NAS model: accuracy (PyTorch + ORT), size, latency, memory
    results/deployment_benchmark.csv     all ONNX models (teacher, students, NAS): size, CPU latency, memory
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.benchmark import environment_info, torch_cpu_latency
from src.common.data import get_loaders, load_raw_numpy
from src.common.inference import load_model
from src.common.onnx_utils import OnnxPredictor
from src.common.paths import NAS_CKPT, ONNX_DIR, RESULTS_DIR, onnx_path
from src.common.trainer import evaluate
from src.common.utils import count_params, file_size_mb, get_device, load_json, save_json, write_csv
from src.nas.generator import count_macs
from src.nas.hardware import Constraints, measure_peak_memory
from src.nas.search import NAS_DIR
from src.nas.search_space import describe


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--threads", type=int, default=1, help="CPU threads (same as the search by default)")
    args = ap.parse_args()
    env = environment_info()
    print(f"CPU: {env['cpu']} | threads {args.threads}")

    # ---- final NAS model ----
    if NAS_CKPT.exists():
        device = get_device()
        model, meta = load_model(NAS_CKPT, device)
        res = meta["image_size"]
        acc = evaluate(model, get_loaders("cifar10", 256, device, image_size=res).test, device)
        row = {"model": "nas", "arch": describe(meta["config"]), "dataset": "cifar10", "resolution": res,
               "top1": round(acc["top1"], 2), "top5": round(acc["top5"], 2),
               "params": count_params(model), "macs_M": round(count_macs(model, res) / 1e6, 3),
               "ckpt_MB": round(file_size_mb(NAS_CKPT), 3),
               "torch_cpu_ms": round(torch_cpu_latency(model, res, 1, args.warmup, args.runs, args.threads)["mean_ms"], 4)}
        op = onnx_path("nas", "nas")
        if op.exists():
            p = OnnxPredictor(op, args.threads)
            lat = p.latency(1, args.warmup, args.runs)
            x, y, _ = load_raw_numpy("cifar10", train=False)
            row.update({"onnx_MB": round(file_size_mb(op), 3), "ort_cpu_ms": round(lat["mean_ms"], 4),
                        "ort_cpu_p95_ms": round(lat["p95_ms"], 4), "ort_top1": round(p.evaluate(x, y)["top1"], 2)})
            mem = measure_peak_memory(op, res, args.threads) or {}
            row.update(mem)
            best = NAS_DIR / "best_architecture.json"
            if best.exists():
                c = Constraints(**load_json(best)["constraints"])
                ok, reasons = c.check({"latency_ms": row["ort_cpu_ms"], "size_mb": row["onnx_MB"],
                                       "inference_mem_mb": mem.get("inference_mem_mb")})
                row["meets_constraints"] = ok
                row["constraint_notes"] = "; ".join(reasons)
        else:
            print("NAS ONNX model not found - run: python scripts/export_onnx.py --model nas")
        write_csv(NAS_DIR / "nas_benchmark.csv", [row])
        print("NAS model:", row)
    else:
        print(f"No NAS checkpoint at {NAS_CKPT}")

    # ---- deployment comparison of every ONNX model ----
    rows = []
    for f in sorted(ONNX_DIR.rglob("*.onnx")):
        p = OnnxPredictor(f, args.threads)
        res = p.image_size
        lat = p.latency(1, args.warmup, args.runs)
        mem = measure_peak_memory(f, res, args.threads) or {}
        r = {"model": f.stem, "variant": f.parent.name, "dataset": p.meta.get("dataset"), "resolution": res,
             "onnx_MB": round(file_size_mb(f), 3), "ort_cpu_ms": round(lat["mean_ms"], 4),
             "ort_cpu_p95_ms": round(lat["p95_ms"], 4), "top1": p.meta.get("top1"), **mem}
        rows.append(r)
        print(f"{r['variant']:<10} {r['model']:<20} {r['onnx_MB']:8.3f} MB | {r['ort_cpu_ms']:8.3f} ms | "
              f"mem {r.get('inference_mem_mb')} MB")
    if rows:
        write_csv(RESULTS_DIR / "deployment_benchmark.csv", rows)
    save_json(RESULTS_DIR / "deployment_env.json", {**env, "benchmark_threads": args.threads})


if __name__ == "__main__":
    main()
