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

```bash
# 1. (optional) profile random architectures on this CPU to choose constraint values
python scripts/nas_search.py --calibrate 30

# 2. Optuna search under deployment constraints, then train the best architecture from scratch
python scripts/nas_search.py --trials 40 --epochs 5 --max-latency-ms 3.0 --max-size-mb 1.0 --final-epochs 60

# 3. ONNX export + inference (PyTorch and ONNX Runtime)
python scripts/export_onnx.py --model nas
python scripts/infer.py --model nas --eval
python scripts/onnx_infer.py --model nas --eval --latency

# 4. Benchmark + figures
python scripts/benchmark_nas.py
python scripts/make_plots_nas.py
```

**Search space** (`src/nas/search_space.py`)

| Dimension | Choices |
|---|---|
| Input resolution | 24, 28, 32 |
| Activation | ReLU, ReLU6, Hardswish, SiLU, LeakyReLU |
| Stem channels | 8, 16, 24, 32 |
| Stages (depth) | 3 or 4 stages, 1-3 blocks each (3-12 blocks) |
| Channels | per stage: {16,24,32}, {24,32,48,64}, {48,64,96,128}, {96,128,160,192} |
| Kernel size | 3, 5, 7 (per stage) |
| Block type | standard conv, depthwise separable, inverted residual (expansion 2/4/6) |

**How the search works** (`src/nas/search.py`, `src/nas/hardware.py`)

1. Optuna's TPE sampler proposes a config, and `src/nas/generator.py` builds the network.
2. The candidate is exported to ONNX and measured on the CPU: ONNX Runtime latency (batch 1, fixed threads), `.onnx` file size, and peak inference memory (measured in a separate process; Linux, macOS and Windows).
3. If the candidate breaks a constraint (latency, size, optional memory), it is **rejected without training**.
4. Feasible candidates are trained for a few epochs (45k train / 5k validation) and pruned early if they lag behind the median.
5. Objective: `reward = val_acc × (latency / limit)^-0.07` (MnasNet-style), or plain accuracy.
6. The best feasible trial is saved to `results/nas/best_architecture.json` and trained from scratch on all of CIFAR-10.

| Script | Purpose |
|---|---|
| `nas_search.py` | Calibration, search, automatic final training |
| `nas_train_best.py` | Retrain the selected architecture (e.g. longer) |
| `benchmark_nas.py` | NAS model metrics + deployment comparison of all ONNX models |
| `make_plots_nas.py` | Search and final-training figures |

**Outputs**

| Path | Contents |
|---|---|
| `results/nas/search_history.csv` / `.json` | Every trial: config, measured latency/size/memory, status, accuracy |
| `results/nas/best_architecture.json` | Selected architecture and its measurements |
| `results/nas/calibration.csv` | Random-architecture hardware profile |
| `checkpoints/nas/best_nas.pth` | Best architecture trained from scratch |
| `onnx/nas/best_nas.onnx` | ONNX export |
| `results/nas/nas_benchmark.csv`, `results/deployment_benchmark.csv` | Benchmarks |

## Results

_To be added after training runs._
