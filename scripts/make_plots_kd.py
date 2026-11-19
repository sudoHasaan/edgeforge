"""Generate report figures for the knowledge-distillation experiments.

Reads results/history/*.csv and results/kd_benchmark.csv, writes PNGs to results/plots/.

    python scripts/make_plots_kd.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.common.paths import RESULTS_DIR, STUDENT_NAMES
from src.common.plotting import (ENGINE_COLORS, INK_MUTED, MODEL_COLORS, MODEL_LABELS, SURFACE, VARIANT_COLORS, apply_style,
                                 plt, save)

HIST = RESULTS_DIR / "history"
PLOTS = RESULTS_DIR / "plots"


def read(path):
    return pd.read_csv(path) if path.exists() else None


def teacher_curve():
    df = read(HIST / "teacher_resnet50.csv")
    if df is None:
        return
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(df.epoch, df.train_loss, color=VARIANT_COLORS["baseline"], label="train loss")
    ax[0].plot(df.epoch, df.val_loss, color=VARIANT_COLORS["distilled"], label="test loss")
    ax[0].set(title="ResNet-50 teacher - loss", xlabel="epoch", ylabel="cross-entropy")
    ax[0].legend()
    ax[1].plot(df.epoch, df.train_top1, color=VARIANT_COLORS["baseline"], label="train top-1")
    ax[1].plot(df.epoch, df.val_top1, color=VARIANT_COLORS["distilled"], label="test top-1")
    best = df.val_top1.max()
    ax[1].annotate(f"best {best:.2f}%", (df.val_top1.idxmax() + 1, best), textcoords="offset points",
                   xytext=(-10, -18), ha="right", color=INK_MUTED)
    ax[1].set(title="ResNet-50 teacher - accuracy (CIFAR-100)", xlabel="epoch", ylabel="accuracy (%)")
    ax[1].legend()
    save(fig, PLOTS / "teacher_training.png")


def student_curves():
    teacher = read(HIST / "teacher_resnet50.csv")
    t_best = teacher.val_top1.max() if teacher is not None else None
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), sharey=True)
    any_data = False
    for ax, name in zip(axes, STUDENT_NAMES):
        for variant in ("baseline", "distilled"):
            df = read(HIST / f"{variant}_{name}.csv")
            if df is None:
                continue
            any_data = True
            ax.plot(df.epoch, df.val_top1, color=VARIANT_COLORS[variant], label=f"{variant} (best {df.val_top1.max():.1f}%)")
        if t_best is not None:
            ax.axhline(t_best, color=VARIANT_COLORS["teacher"], ls="--", lw=1.2, label=f"teacher {t_best:.1f}%")
        ax.set(title=MODEL_LABELS[name], xlabel="epoch")
        ax.legend(loc="lower right", fontsize=8)
    axes[0].set_ylabel("test top-1 accuracy (%)")
    fig.suptitle("Students on CIFAR-100: baseline vs knowledge distillation", fontweight="bold", y=1.03)
    if any_data:
        save(fig, PLOTS / "kd_training_curves.png")


def accuracy_bars(bench):
    stud = bench[bench.model.isin(STUDENT_NAMES)]
    if stud.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4.2))
    width = 0.36
    names = [n for n in STUDENT_NAMES if n in set(stud.model)]
    for i, variant in enumerate(("baseline", "distilled")):
        vals = [stud[(stud.model == n) & (stud.variant == variant)].top1 for n in names]
        vals = [v.iloc[0] if len(v) else 0 for v in vals]
        xs = [k + (i - 0.5) * width for k in range(len(names))]
        bars = ax.bar(xs, vals, width, color=VARIANT_COLORS[variant], label=variant,
                      edgecolor=SURFACE, linewidth=2)
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
    t = bench[bench.variant == "teacher"]
    if len(t):
        ax.axhline(t.top1.iloc[0], color=VARIANT_COLORS["teacher"], ls="--", lw=1.2,
                   label=f"ResNet-50 teacher ({t.top1.iloc[0]:.1f}%)")
    ax.set_xticks(range(len(names)), [MODEL_LABELS[n] for n in names])
    ax.set(ylabel="test top-1 accuracy (%)", title="CIFAR-100 accuracy: baseline vs distilled students")
    ax.set_ylim(0, max(100, bench.top1.max() + 8))
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", ncols=3, fontsize=9)
    save(fig, PLOTS / "kd_accuracy_comparison.png")


def accuracy_vs_latency(bench):
    col = "ort_cpu_ms" if "ort_cpu_ms" in bench and bench.ort_cpu_ms.notna().all() else "torch_cpu_ms"
    fig, ax = plt.subplots(figsize=(8, 4.8))
    markers = {"baseline": "o", "distilled": "^", "teacher": "s"}
    for _, r in bench.iterrows():
        ax.scatter(r[col], r.top1, s=70, color=MODEL_COLORS[r.model], marker=markers[r.variant],
                   edgecolor=SURFACE, linewidth=1.5, zorder=3)
    for name in STUDENT_NAMES:
        b = bench[(bench.model == name) & (bench.variant == "baseline")]
        d = bench[(bench.model == name) & (bench.variant == "distilled")]
        if len(b) and len(d):
            ax.annotate("", (d[col].iloc[0], d.top1.iloc[0]), (b[col].iloc[0], b.top1.iloc[0]),
                        arrowprops=dict(arrowstyle="->", color=MODEL_COLORS[name], lw=1.2))
        pts = bench[bench.model == name]
        if len(pts):
            r = pts.loc[pts.top1.idxmax()]
            ax.annotate(MODEL_LABELS[name], (r[col], r.top1), textcoords="offset points", xytext=(8, 4), fontsize=9)
    t = bench[bench.variant == "teacher"]
    if len(t):
        ax.annotate(MODEL_LABELS["resnet50"], (t[col].iloc[0], t.top1.iloc[0]), textcoords="offset points",
                    xytext=(-8, 6), ha="right", fontsize=9)
    for v, m in markers.items():
        ax.scatter([], [], marker=m, color=INK_MUTED, label=v)
    ax.set_xscale("log")
    engine = "ONNX Runtime" if col == "ort_cpu_ms" else "PyTorch"
    ax.set(xlabel=f"CPU latency, batch 1, {engine} (ms, log scale)", ylabel="test top-1 accuracy (%)",
           title="Accuracy vs CPU latency (arrows: baseline -> distilled)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9)
    save(fig, PLOTS / "kd_accuracy_vs_latency.png")


def size_and_latency(bench):
    # One row per architecture (baseline and distilled share the same architecture).
    arch = bench[bench.variant.isin(["teacher", "distilled"])].copy()
    if arch.empty:
        return
    labels = [MODEL_LABELS[m] for m in arch.model]
    colors = [MODEL_COLORS[m] for m in arch.model]
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    for ax, colname, title in [(axes[0], "params_M", "Parameters (millions)"),
                               (axes[1], "onnx_MB" if "onnx_MB" in arch else "ckpt_MB", "Model file size (MB)")]:
        bars = ax.barh(labels, arch[colname], color=colors, edgecolor=SURFACE, linewidth=2)
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
        ax.invert_yaxis()
    ax = axes[2]
    y = range(len(arch))
    h = 0.38
    b1 = ax.barh([k - h / 2 for k in y], arch.torch_cpu_ms, h, color=ENGINE_COLORS["torch"],
                 label="PyTorch", edgecolor=SURFACE, linewidth=2)
    ax.bar_label(b1, fmt="%.1f", padding=3, fontsize=8)
    if "ort_cpu_ms" in arch:
        b2 = ax.barh([k + h / 2 for k in y], arch.ort_cpu_ms, h, color=ENGINE_COLORS["ort"],
                     label="ONNX Runtime", edgecolor=SURFACE, linewidth=2)
        ax.bar_label(b2, fmt="%.1f", padding=3, fontsize=8)
    ax.set_yticks(list(y), labels)
    ax.invert_yaxis()
    ax.set_title("CPU latency, batch 1 (ms)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    save(fig, PLOTS / "kd_size_latency.png")


def main():
    apply_style()
    teacher_curve()
    student_curves()
    bench = read(RESULTS_DIR / "kd_benchmark.csv")
    if bench is not None:
        accuracy_bars(bench)
        accuracy_vs_latency(bench)
        size_and_latency(bench)
    else:
        print("results/kd_benchmark.csv not found - run scripts/benchmark_kd.py for the comparison plots")


if __name__ == "__main__":
    main()
