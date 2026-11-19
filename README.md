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
| Knowledge Distillation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<your-username>/edgeforge/blob/main/notebooks/01_knowledge_distillation.ipynb) |
| Hardware-Aware NAS | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<your-username>/edgeforge/blob/main/notebooks/02_hardware_aware_nas.ipynb) |

> Set **Runtime → Change runtime type → GPU** before running.

## Local Installation

```bash
git clone https://github.com/<your-username>/edgeforge.git
cd edgeforge
pip install -r requirements.txt
```

## Execution Steps

_To be added as modules are implemented._

## Results

_To be added after training runs._
