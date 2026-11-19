"""Shared matplotlib style for report figures (light theme, print-friendly)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#1f1f1d"
INK_MUTED = "#6b6a63"
GRID = "#e4e3dd"

# Fixed categorical order - colour follows the entity, never its rank.
VARIANT_COLORS = {"baseline": "#2a78d6", "distilled": "#eb6834", "teacher": "#6b6a63", "nas": "#1baf7a"}
MODEL_COLORS = {"mobilenetv2": "#2a78d6", "squeezenet": "#eb6834", "efficientnet_lite0": "#1baf7a",
                "resnet50": "#6b6a63"}
# Inference engines get neutral inks so they never read as one of the models.
ENGINE_COLORS = {"torch": "#b5b3aa", "ort": "#3d3c37"}
MODEL_LABELS = {"mobilenetv2": "MobileNetV2", "squeezenet": "SqueezeNet",
                "efficientnet_lite0": "EfficientNet-Lite0", "resnet50": "ResNet-50 (teacher)"}


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
        "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 10,
        "font.size": 10, "legend.frameon": False, "lines.linewidth": 2,
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
    })


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"saved {path}")
