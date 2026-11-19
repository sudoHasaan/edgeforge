# EdgeForge

Compressing and discovering lightweight CNNs for edge devices via **knowledge distillation** and **hardware-aware neural architecture search**, with CPU deployment through **ONNX Runtime**.

## Overview

| Module | What it does |
|---|---|
| **Knowledge Distillation** (`src/kd`) | Trains a ResNet-50 teacher on CIFAR-100 and distills it into MobileNetV2, SqueezeNet and EfficientNet-Lite0 using soft targets, temperature scaling, combined hard/soft loss and feature-based distillation. Includes non-distilled baselines for comparison. |
| **Hardware-Aware NAS** (`src/nas`) | Searches a CNN space (depth, width, kernel size, depthwise-separable blocks, activations, input resolution) on CIFAR-10, rejecting candidates that violate CPU latency, model size and memory constraints. |
| **Deployment** (`src/common`) | Unified PyTorch inference, ONNX export, ONNX Runtime CPU inference and automated benchmarking. |

## Repository Structure

```
edgeforge/
├── notebooks/          # Colab-ready notebooks (run these)
├── src/
│   ├── kd/             # teacher, students, KD losses, trainers
│   ├── nas/            # search space, generator, search, hardware profiler
│   └── common/         # data, ONNX export, inference, benchmarking
├── scripts/            # CLI entry points
├── checkpoints/        # teacher / baseline / distilled / nas weights
├── onnx/               # exported ONNX models
├── results/            # CSV/JSON results and plots
└── docs/               # technical report
```

## Quick Start (Google Colab)

Open a notebook directly in Colab:

| Notebook | Link |
|---|---|
| Knowledge Distillation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sudoHasaan/edgeforge/blob/main/notebooks/01_knowledge_distillation.ipynb) |
| Hardware-Aware NAS | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sudoHasaan/edgeforge/blob/main/notebooks/02_hardware_aware_nas.ipynb) |

> Set **Runtime → Change runtime type → GPU** before running.

## Local Installation

```bash
git clone https://github.com/sudoHasaan/edgeforge.git
cd edgeforge
pip install -r requirements.txt
```

## Execution Steps

### Part 1 - Knowledge Distillation (CIFAR-100)

Run from the repository root (the Colab notebook runs exactly these commands):

```bash
# 1. Teacher
python scripts/train_teacher.py --epochs 30

# 2. Baseline students (cross-entropy only)
python scripts/train_baseline.py --students all --epochs 30

# 3. Distilled students (soft targets + hard labels + feature hints)
python scripts/train_kd.py --students all --epochs 30 --alpha 0.9 --temperature 4 --beta 1.0

# 4. Unified PyTorch inference
python scripts/infer.py --model resnet50 all --variant all --eval
python scripts/infer.py --model mobilenetv2 --variant distilled --image path/to/image.jpg

# 5. ONNX export + ONNX Runtime CPU inference
python scripts/export_onnx.py --model resnet50 all --variant all
python scripts/onnx_infer.py --model all --variant all --eval --latency

# 6. Benchmark + figures
python scripts/benchmark_kd.py
python scripts/make_plots_kd.py
```

| Script | Purpose |
|---|---|
| `train_teacher.py` | ResNet-50 teacher (ImageNet init, CIFAR stem) |
| `train_baseline.py` | Students without distillation |
| `train_kd.py` | KD: `--alpha` (soft vs hard weight), `--temperature`, `--beta` (feature hints, 0 disables), `--feat-layers mid final` |
| `infer.py` | Unified PyTorch inference: `--image`, `--samples N`, `--eval`, any `--model`/`--variant` or `--ckpt` |
| `export_onnx.py` | ONNX export (opset 17, dynamic batch) with PyTorch-vs-ORT output check |
| `onnx_infer.py` | ONNX Runtime CPU inference: `--image`, `--samples N`, `--eval`, `--latency` |
| `benchmark_kd.py` | Accuracy, params, file size, PyTorch vs ORT CPU latency -> `results/kd_benchmark.csv` |
| `make_plots_kd.py` | Report figures -> `results/plots/` |

All training scripts skip models that are already trained and resume from the last epoch if interrupted.

**Outputs**

| Path | Contents |
|---|---|
| `checkpoints/teacher/resnet50.pth` | Teacher |
| `checkpoints/baseline/*.pth` | Baseline students |
| `checkpoints/distilled/*.pth` | Distilled students |
| `onnx/{teacher,baseline,distilled}/*.onnx` | ONNX models |
| `results/history/*.csv` | Per-epoch training logs |
| `results/kd_benchmark.csv` | Benchmark table |
| `results/plots/*.png` | Figures |

### Part 2 - Hardware-Aware NAS (CIFAR-10)

_Coming next._

## Results

_To be added after training runs._
