"""Generate report figures for the NAS experiments -> results/plots/.

    python scripts/make_plots_nas.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.common.paths import RESULTS_DIR
from src.common.plotting import INK_MUTED, SURFACE, apply_style, plt, save
from src.common.utils import load_json

NAS = RESULTS_DIR / "nas"
PLOTS = RESULTS_DIR / "plots"
STATUS_COLORS = {"feasible": "#2a78d6", "pruned": "#eda100", "rejected": "#b5b3aa"}
STATUS_MARKERS = {"feasible": "o", "pruned": "^", "rejected": "x"}


def load():
    hist = NAS / "search_history.csv"
    if not hist.exists():
        print("results/nas/search_history.csv not found - run scripts/nas_search.py first")
        return None, None
    best = load_json(NAS / "best_architecture.json") if (NAS / "best_architecture.json").exists() else None
    return pd.read_csv(hist), best


def constraint_lines(ax, best, x=None, y=None):
    c = (best or {}).get("constraints") or {}
    if x and c.get(x):
        ax.axvline(c[x], color=INK_MUTED, ls="--", lw=1.2)
        ax.annotate("limit", (c[x], 1), xycoords=("data", "axes fraction"), xytext=(4, -12),
                    textcoords="offset points", color=INK_MUTED, fontsize=8)
    if y and c.get(y):
        ax.axhline(c[y], color=INK_MUTED, ls="--", lw=1.2)


def search_progress(df, best):
    fig, ax = plt.subplots(figsize=(9, 4))
    for status in ("rejected", "pruned", "feasible"):
        d = df[df.status == status]
        if status == "rejected":
            ax.scatter(d.trial, [0] * len(d), marker="x", color=STATUS_COLORS[status], label=f"rejected by constraints ({len(d)})")
        else:
            ax.scatter(d.trial, d.val_top1, marker=STATUS_MARKERS[status], s=36, color=STATUS_COLORS[status],
                       edgecolor=SURFACE, linewidth=1, label=f"{status} ({len(d)})", zorder=3)
    feas = df[df.status == "feasible"].sort_values("trial")
    if len(feas):
        running = feas.val_top1.cummax()
        ax.step(feas.trial, running, where="post", color="#1f1f1d", lw=1.5, label="best so far")
    ax.set(xlabel="trial", ylabel="validation top-1 (%)", title="NAS search progress (CIFAR-10, short training)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9)
    save(fig, PLOTS / "nas_search_progress.png")


def acc_vs_latency(df, best):
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for status in ("pruned", "feasible"):
        d = df[df.status == status]
        ax.scatter(d.latency_ms, d.val_top1, marker=STATUS_MARKERS[status], s=40, color=STATUS_COLORS[status],
                   edgecolor=SURFACE, linewidth=1, label=status, zorder=3)
    if best:
        h = best["hardware"]
        ax.scatter([h["latency_ms"]], [best["val_top1"]], s=160, facecolor="none", edgecolor="#eb6834", lw=2,
                   zorder=4, label="selected")
        ax.annotate(f"trial {best['trial']}", (h["latency_ms"], best["val_top1"]), textcoords="offset points",
                    xytext=(8, 6), fontsize=9)
    constraint_lines(ax, best, x="max_latency_ms")
    ax.set(xlabel="ONNX Runtime CPU latency, batch 1 (ms)", ylabel="validation top-1 (%)",
           title="Trained candidates: accuracy vs measured CPU latency")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9)
    save(fig, PLOTS / "nas_accuracy_vs_latency.png")


def hardware_space(df, best):
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for status in ("rejected", "pruned", "feasible"):
        d = df[df.status == status]
        ax.scatter(d.latency_ms, d.size_mb, marker=STATUS_MARKERS[status], s=36, color=STATUS_COLORS[status],
                   label=status, zorder=3, **({} if status == "rejected" else {"edgecolor": SURFACE, "linewidth": 1}))
    if best:
        h = best["hardware"]
        ax.scatter([h["latency_ms"]], [h["size_mb"]], s=160, facecolor="none", edgecolor="#eb6834", lw=2,
                   zorder=4, label="selected")
    constraint_lines(ax, best, x="max_latency_ms", y="max_size_mb")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set(xlabel="CPU latency (ms, log)", ylabel="ONNX model size (MB, log)",
           title="All sampled architectures vs deployment constraints")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9)
    save(fig, PLOTS / "nas_hardware_space.png")


def final_training():
    f = RESULTS_DIR / "history" / "nas_final.csv"
    if not f.exists():
        return
    df = pd.read_csv(f)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(df.epoch, df.train_loss, color="#2a78d6", label="train loss")
    ax[0].plot(df.epoch, df.val_loss, color="#eb6834", label="test loss")
    ax[0].set(title="Best NAS architecture - loss", xlabel="epoch", ylabel="cross-entropy")
    ax[0].legend()
    ax[1].plot(df.epoch, df.train_top1, color="#2a78d6", label="train top-1")
    ax[1].plot(df.epoch, df.val_top1, color="#eb6834", label="test top-1")
    ax[1].set(title=f"Best NAS architecture - accuracy (best {df.val_top1.max():.2f}%)", xlabel="epoch",
              ylabel="accuracy (%)")
    ax[1].legend()
    save(fig, PLOTS / "nas_final_training.png")


def choice_effects(df):
    """Mean validation accuracy of trained candidates per value of each global decision."""
    d = df[df.status == "feasible"]
    if len(d) < 3:
        return
    cols = ["activation", "resolution", "num_stages", "s0_block"]
    fig, axes = plt.subplots(1, len(cols), figsize=(14, 3.4), sharey=True)
    for ax, c in zip(axes, cols):
        g = d.groupby(c).val_top1.agg(["mean", "count"]).reset_index()
        bars = ax.bar(g[c].astype(str), g["mean"], color="#2a78d6", edgecolor=SURFACE, linewidth=2)
        ax.bar_label(bars, labels=[f"{m:.1f}\nn={n}" for m, n in zip(g["mean"], g["count"])], fontsize=8, padding=2)
        ax.set_title(c.replace("s0_block", "first-stage block"))
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="x", labelrotation=30)
    axes[0].set_ylabel("mean val top-1 (%)")
    axes[0].set_ylim(0, d.val_top1.max() * 1.25)
    fig.suptitle("Effect of search decisions (trained candidates)", fontweight="bold", y=1.04)
    save(fig, PLOTS / "nas_choice_effects.png")


def main():
    apply_style()
    df, best = load()
    if df is None:
        return
    search_progress(df, best)
    acc_vs_latency(df, best)
    hardware_space(df, best)
    choice_effects(df)
    final_training()


if __name__ == "__main__":
    main()
