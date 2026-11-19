"""Train the best discovered architecture from scratch on the full CIFAR-10 training set.

(nas_search.py already does this automatically; use this script to retrain or train longer.)

    python scripts/nas_train_best.py --epochs 60
    python scripts/nas_train_best.py --config results/nas/best_architecture.json --epochs 100 --force
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.paths import NAS_CKPT
from src.common.utils import get_device, is_complete, load_json
from src.nas.search import NAS_DIR, train_best
from src.nas.search_space import describe


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(NAS_DIR / "best_architecture.json"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(NAS_CKPT))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if is_complete(args.out) and not args.force:
        print(f"Already trained: {args.out} (use --force to retrain)")
        return
    if args.force:
        Path(args.out).with_name(Path(args.out).stem + "_last.pth").unlink(missing_ok=True)
    best = load_json(args.config)
    cfg = best.get("config", best)
    print(f"Architecture: {describe(cfg)}")
    _, acc = train_best(cfg, args.epochs, args.batch_size, args.lr, seed=args.seed, device=get_device(),
                        ckpt_path=args.out, extra_meta={"search_trial": best.get("trial"),
                                                        "search_val_top1": best.get("val_top1")})
    print(f"Test top-1: {acc:.2f}% -> {args.out}")


if __name__ == "__main__":
    main()
