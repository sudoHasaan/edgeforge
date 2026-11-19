"""Hardware-aware NAS with Bayesian optimisation (Optuna TPE).

Per trial:
  1. TPE samples an architecture config from the search space.
  2. The generator builds it; the hardware profiler exports it to ONNX and measures CPU latency,
     model size and peak memory.
  3. If any deployment constraint is violated the trial is REJECTED (no training time spent).
  4. Otherwise the candidate is trained for a few epochs on CIFAR-10 (45k train) and scored
     on a held-out validation split (5k). Weak candidates are pruned early (median pruner).
  5. Objective (maximised):
        accuracy  -> val top-1
        reward    -> val top-1 * (latency / max_latency) ** w     (MnasNet-style, w = -0.07)
     The reward slightly favours faster models among those that satisfy the constraints.

The study is stored in SQLite, so an interrupted search resumes where it stopped.
"""
import math
import random
import time
from pathlib import Path

import optuna

from ..common.data import get_loaders
from ..common.paths import NAS_CKPT, RESULTS_DIR
from ..common.trainer import fit
from ..common.utils import save_json, set_seed, write_csv
from .generator import build_network
from .hardware import Constraints, profile_model
from .search_space import describe, flatten, optuna_config, random_config

NAS_DIR = RESULTS_DIR / "nas"


class LoaderCache:
    """One set of GPU-resident CIFAR-10 loaders per input resolution."""

    def __init__(self, device, batch_size, val_split, seed):
        self.device, self.batch_size, self.val_split, self.seed = device, batch_size, val_split, seed
        self._cache = {}

    def get(self, resolution):
        if resolution not in self._cache:
            self._cache[resolution] = get_loaders("cifar10", self.batch_size, self.device, image_size=resolution,
                                                  val_split=self.val_split, seed=self.seed)
        return self._cache[resolution]


def reward(acc: float, latency_ms: float, target_ms: float | None, w: float = -0.07) -> float:
    if not target_ms:
        return acc
    return acc * (latency_ms / target_ms) ** w


def trials_to_rows(study: optuna.Study) -> list[dict]:
    rows = []
    for t in study.trials:
        a = t.user_attrs
        if "config" not in a:
            continue
        row = {"trial": t.number, "status": a.get("status", t.state.name.lower()),
               "objective": round(t.value, 4) if t.value is not None else None,
               "val_top1": a.get("val_top1"), "epochs_trained": a.get("epochs_trained"),
               "reject_reason": a.get("reject_reason", ""), "arch": describe(a["config"])}
        row.update({k: a.get(k) for k in ("params", "macs_M", "size_mb", "latency_ms", "latency_p95_ms",
                                          "peak_rss_mb", "inference_mem_mb", "train_time_s")})
        row.update(flatten(a["config"]))
        rows.append(row)
    return rows


def save_history(study, out_dir=NAS_DIR):
    rows = trials_to_rows(study)
    write_csv(out_dir / "search_history.csv", rows)
    save_json(out_dir / "search_history.json",
              [{"trial": t.number, "state": t.state.name, "value": t.value, **t.user_attrs} for t in study.trials])
    return rows


def select_best(study: optuna.Study) -> optuna.trial.FrozenTrial | None:
    """Best trial among those that satisfied every constraint and finished training."""
    ok = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
          and t.user_attrs.get("status") == "feasible"]
    return max(ok, key=lambda t: t.value) if ok else None


def run_search(n_trials=40, epochs=5, constraints: Constraints = None, objective_type="reward", threads=1,
               batch_size=256, lr=0.1, wd=5e-4, val_split=5000, seed=42, device=None, timeout=None,
               study_name="edgeforge_nas", storage=None, out_dir=NAS_DIR):
    constraints = constraints or Constraints()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    storage = storage or f"sqlite:///{(out_dir / 'nas_study.db').as_posix()}"
    loaders = LoaderCache(device, batch_size, val_split, seed)

    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=10)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=max(1, epochs // 2))
    study = optuna.create_study(study_name=study_name, storage=storage, direction="maximize",
                                sampler=sampler, pruner=pruner, load_if_exists=True)
    study.set_user_attr("constraints", constraints.to_dict())
    study.set_user_attr("objective", objective_type)

    def objective(trial: optuna.Trial):
        cfg = optuna_config(trial)
        trial.set_user_attr("config", cfg)
        model = build_network(cfg, 10)
        hw = profile_model(model, cfg["resolution"], threads=threads)
        for k, v in hw.items():
            trial.set_user_attr(k, v)
        ok, reasons = constraints.check(hw)
        head = f"[trial {trial.number:3d}] {describe(cfg)}\n            {hw['params'] / 1e3:.0f}K params | " \
               f"{hw['size_mb']:.3f} MB | {hw['latency_ms']:.3f} ms | mem {hw.get('inference_mem_mb')} MB"
        if not ok:
            trial.set_user_attr("status", "rejected")
            trial.set_user_attr("reject_reason", "; ".join(reasons))
            print(f"{head}\n            REJECTED: {'; '.join(reasons)}", flush=True)
            raise optuna.TrialPruned()

        factor = reward(1.0, hw["latency_ms"], constraints.max_latency_ms) if objective_type == "reward" else 1.0
        data = loaders.get(cfg["resolution"])
        set_seed(seed + trial.number)
        model = build_network(cfg, 10).to(device)

        def on_epoch(epoch, row):
            trial.set_user_attr("epochs_trained", epoch)
            trial.report(row["val_top1"] * factor, epoch)
            if trial.should_prune():
                trial.set_user_attr("status", "pruned")
                trial.set_user_attr("val_top1", row["val_top1"])
                raise optuna.TrialPruned()

        t0 = time.time()
        _, best = fit(model, data.train, data.val, epochs=epochs, lr=lr, weight_decay=wd, device=device,
                      ckpt_path=None, verbose=False, epoch_callback=on_epoch)
        trial.set_user_attr("train_time_s", round(time.time() - t0, 1))
        trial.set_user_attr("val_top1", best)
        trial.set_user_attr("status", "feasible")
        value = best * factor
        print(f"{head}\n            val top-1 {best:.2f}% | objective {value:.2f} | {time.time() - t0:.0f}s", flush=True)
        return value

    def after_trial(study, trial):
        save_history(study, out_dir)
        b = select_best(study)
        if b is not None:
            save_best(b, study, out_dir)

    remaining = n_trials - len([t for t in study.trials if t.state.is_finished()])
    if remaining > 0:
        study.optimize(objective, n_trials=remaining, timeout=timeout, callbacks=[after_trial],
                       catch=(RuntimeError,))
    rows = save_history(study, out_dir)
    best = select_best(study)
    if best is not None:
        save_best(best, study, out_dir)
    return study, best, rows


def save_best(trial, study, out_dir=NAS_DIR):
    a = trial.user_attrs
    save_json(Path(out_dir) / "best_architecture.json", {
        "trial": trial.number, "objective": trial.value, "val_top1": a.get("val_top1"),
        "arch": describe(a["config"]), "config": a["config"],
        "hardware": {k: a.get(k) for k in ("params", "macs_M", "size_mb", "latency_ms", "latency_p95_ms",
                                           "peak_rss_mb", "inference_mem_mb", "threads")},
        "constraints": study.user_attrs.get("constraints"), "objective_type": study.user_attrs.get("objective"),
    })


def calibrate(n=30, threads=1, seed=0, out_dir=NAS_DIR):
    """Profile random architectures (no training) to choose sensible constraint values."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        cfg = random_config(rng)
        hw = profile_model(build_network(cfg, 10), cfg["resolution"], threads=threads, measure_memory=(i < 5))
        rows.append({"arch": describe(cfg), **hw})
        print(f"[{i + 1:2d}/{n}] {hw['latency_ms']:7.3f} ms | {hw['size_mb']:6.3f} MB | {describe(cfg)}", flush=True)
    write_csv(Path(out_dir) / "calibration.csv", rows)
    lat = sorted(r["latency_ms"] for r in rows)
    size = sorted(r["size_mb"] for r in rows)
    q = lambda v, p: v[min(len(v) - 1, int(math.floor(p * len(v))))]
    print("\nlatency ms  p25 {:.3f} | p50 {:.3f} | p75 {:.3f}".format(q(lat, .25), q(lat, .5), q(lat, .75)))
    print("size MB     p25 {:.3f} | p50 {:.3f} | p75 {:.3f}".format(q(size, .25), q(size, .5), q(size, .75)))
    print("Tip: constraints near the p50 values reject roughly half of the search space.")
    return rows


def train_best(config: dict, epochs=60, batch_size=128, lr=0.1, wd=5e-4, seed=42, device=None,
               ckpt_path=NAS_CKPT, extra_meta: dict | None = None, resume=True):
    """Train the selected architecture from scratch on the full CIFAR-10 training set."""
    set_seed(seed)
    data = get_loaders("cifar10", batch_size, device, image_size=config["resolution"])
    model = build_network(config, data.num_classes).to(device)
    meta = {"family": "nas", "arch": "nas", "variant": "nas", "dataset": "cifar10", "config": config,
            "num_classes": data.num_classes, "classes": data.classes, "mean": data.mean, "std": data.std,
            "image_size": config["resolution"], "lr": lr, "batch_size": batch_size, **(extra_meta or {})}
    history, best = fit(model, data.train, data.test, epochs=epochs, lr=lr, weight_decay=wd, device=device,
                        ckpt_path=ckpt_path, history_path=RESULTS_DIR / "history" / "nas_final.csv",
                        meta=meta, resume=resume)
    return history, best
