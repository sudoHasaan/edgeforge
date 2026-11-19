"""Hardware-aware Neural Architecture Search on CIFAR-10 (Optuna TPE).

    # 1. (optional) profile random architectures to pick constraint values for this CPU
    python scripts/nas_search.py --calibrate 30

    # 2. search, then automatically train the best architecture from scratch
    python scripts/nas_search.py --trials 40 --epochs 5 --max-latency-ms 3.0 --max-size-mb 1.0 --final-epochs 60

Outputs
    results/nas/search_history.csv / .json   every trial: config, hardware metrics, status, accuracy
    results/nas/best_architecture.json       selected architecture + its measurements
    results/nas/nas_study.db                 Optuna study (re-running resumes the search)
    checkpoints/nas/best_nas.pth             best architecture trained from scratch
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import optuna

from src.common.paths import NAS_CKPT
from src.common.utils import get_device, is_complete, load_checkpoint
from src.nas.hardware import Constraints
from src.nas.search import NAS_DIR, calibrate, run_search, train_best
from src.nas.search_space import describe


def none_if_neg(v):
    return None if v is not None and v <= 0 else v


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibrate", type=int, default=0, help="profile N random architectures and exit")
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=5, help="training epochs per candidate")
    ap.add_argument("--max-latency-ms", type=float, default=3.0, help="ONNX Runtime CPU latency limit (<=0: off)")
    ap.add_argument("--max-size-mb", type=float, default=1.0, help="ONNX model size limit (<=0: off)")
    ap.add_argument("--max-mem-mb", type=float, default=0, help="peak inference memory limit (<=0: record only)")
    ap.add_argument("--objective", choices=["reward", "accuracy"], default="reward")
    ap.add_argument("--threads", type=int, default=1, help="CPU threads for latency measurement")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--val-split", type=int, default=5000)
    ap.add_argument("--timeout", type=float, default=None, help="search time limit in seconds")
    ap.add_argument("--study-name", default="edgeforge_nas")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--final-epochs", type=int, default=60, help="epochs to train the best model (0: skip)")
    ap.add_argument("--final-batch-size", type=int, default=128)
    ap.add_argument("--final-lr", type=float, default=0.1)
    args = ap.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if args.calibrate:
        calibrate(args.calibrate, args.threads, args.seed)
        return

    device = get_device()
    constraints = Constraints(none_if_neg(args.max_latency_ms), none_if_neg(args.max_size_mb),
                              none_if_neg(args.max_mem_mb))
    print(f"Device {device} | constraints {constraints.to_dict()} | objective {args.objective}")
    study, best, rows = run_search(args.trials, args.epochs, constraints, args.objective, args.threads,
                                   args.batch_size, args.lr, 5e-4, args.val_split, args.seed, device,
                                   args.timeout, args.study_name)

    counts = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print(f"\nSearch finished: {len(rows)} trials -> {counts}")
    if best is None:
        print("No architecture satisfied the constraints. Relax them (see --calibrate) and run again.")
        return
    a = best.user_attrs
    print(f"Best trial {best.number}: {describe(a['config'])}")
    print(f"  val top-1 {a['val_top1']:.2f}% | {a['latency_ms']:.3f} ms | {a['size_mb']:.3f} MB | "
          f"{a['params'] / 1e3:.0f}K params")
    print(f"  saved {NAS_DIR / 'best_architecture.json'}")

    if args.final_epochs > 0:
        last = NAS_CKPT.with_name(NAS_CKPT.stem + "_last.pth")
        if NAS_CKPT.exists() and load_checkpoint(NAS_CKPT)["meta"].get("config") != a["config"]:
            print("\nBest architecture changed since the last final training - discarding the old model.")
            NAS_CKPT.unlink()
            last.unlink(missing_ok=True)
        if is_complete(NAS_CKPT):
            print(f"\nFinal model already trained: {NAS_CKPT} (run scripts/nas_train_best.py --force to retrain)")
            return
        print(f"\n=== Training best architecture from scratch ({args.final_epochs} epochs, full CIFAR-10) ===")
        _, acc = train_best(a["config"], args.final_epochs, args.final_batch_size, args.final_lr, seed=args.seed,
                            device=device, extra_meta={"search_trial": best.number, "search_val_top1": a["val_top1"]})
        print(f"Final NAS model test top-1: {acc:.2f}% -> {NAS_CKPT}")


if __name__ == "__main__":
    main()
