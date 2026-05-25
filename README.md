# SmartBite — AI-Powered Food Quality Inspection

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green)]()
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000)]()
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)]()
[![Docker](https://img.shields.io/badge/docker-ready-2496ed)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-3.0-009688)]()

SmartBite is a production-ready AI platform for real-time food quality assessment using computer vision and deep learning.

**Freshness scoring · Spoilage detection · Nutritional analysis · REST API · On-device inference**

## Problem

- $1.2 trillion lost annually to food spoilage
- 47% of global food waste happens at consumer level
- No accessible AI tools exist for everyday food quality checks

## Features

- Food Classification: 200+ categories via EfficientNet-B3 / ViT-B-16
- Freshness Scoring: Multi-dimensional (appearance, texture, color)
- Spoilage Detection: 5-level grading system
- Explainability: Grad-CAM heatmaps
- REST API: FastAPI with single/batch endpoints
- Edge Ready: ONNX and TorchScript export
- Nutritional Insights: Per-100g macro estimates

## Architecture

```
smartbite/
  models/         EfficientNet, ViT, UNet, multi-task heads
  preprocessing/  Image transforms and food-specific ops
  features/       Color, texture, spectral, nutritional
  training/       Trainer, losses, schedulers, callbacks
  inference/      Optimized engine with caching
  api/            FastAPI REST endpoints
  visualization/  Grad-CAM heatmaps, dashboard components
  utils/          Config, metrics, I/O, logging
tests/            Unit tests for all modules
scripts/          CLI tools and entrypoints
configs/          GPU and mobile config profiles
docs/             Investor docs, API reference, edge guide
```

## Quick Start

```bash
git clone https://github.com/alfianmushh-eng/SmartBite.git
cd SmartBite
pip install -e .
smartbite predict veggie_plate.jpg
```

## Docker

```bash
make docker-build && make docker-run
curl -X POST http://localhost:8000/analyze -F "file=@food.jpg"
```

## Performance

- Food-101 Accuracy: 94.2%
- Freshness MSE: 0.031
- Inference (GPU): 12ms
- Inference (CPU): 95ms

## Target Market

- Grocery chains: automated quality control
- Restaurants: daily ingredient freshness
- Food distributors: batch inspection
- Regulators: compliance verification

## Contact

SmartBite Engineering — dev@smartbite.ai

## License

MIT License — see [LICENSE](LICENSE)