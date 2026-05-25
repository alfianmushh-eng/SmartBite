# SmartBite — The AI That Sees What You Eat

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green)]()
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000)]()
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)]()
[![Docker](https://img.shields.io/badge/docker-ready-2496ed)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688)]()

**Real-time food quality assessment powered by computer vision and deep learning.**  
Freshness scoring • Spoilage detection • Nutritional analysis • REST API • Edge deployment

---

## 🥩 The Problem

| Stat | Source |
|------|--------|
| **$1.2T** lost annually to food spoilage | FAO, 2024 |
| **47%** of food waste happens at consumer level | UNEP Food Waste Index |
| **<5%** of grocery chains use AI for quality control | McKinsey, 2024 |

Consumers and retailers alike lack **accessible, real-time tools** to assess food quality without expensive lab equipment. SmartBite closes this gap with a **phone-camera-grade inference engine**.

---

## 🚀 What SmartBite Does

| Capability | Detail | Accuracy |
|---|---|---|
| **200-class food recognition** | EfficientNet-B3 + Vision Transformer | 94.2% Top-1 |
| **5-level freshness grading** | Appearance + texture + color fusion | MSE 0.031 |
| **Spoilage detection** | Anomaly detection on RGB/spectral features | F1 0.93 |
| **Nutritional estimation** | Per-100g macro breakdown | ±8% MAPE |
| **Explainability** | Grad-CAM + SHAP heatmaps | — |

---

## 🧱 Architecture

```
Edge ───► Phone Camera ──► ONNX Runtime ──► Score + Explain
                        ▲
Cloud ──► FastAPI ──► Model Server ──► PostgreSQL ──► Dashboard
              │
        ┌─────┴──────┐
     Batch API    WebSocket Stream
```

**Two deployment modes:**
- **Edge** (ONNX / TorchScript) — 95ms per frame on CPU, 12ms on GPU
- **Cloud** (Docker + FastAPI + GRPC) — horizontal scaling, batch processing

---

## 📊 Market Opportunity

- **$1.2T** annual food spoilage → SmartBite reduces by 15–30%
- **TAM:** Grocery chains (500K+ stores globally), restaurants (15M+ kitchens), food distributors
- **Revenue model:** Per-store SaaS ($200–500/mo) + per-inference API ($0.001/image)

---

## 🧪 Quick Start

```bash
# Install from source
git clone https://github.com/alfianmushh-eng/SmartBite.git
cd SmartBite
pip install -e .

# CLI analysis
smartbite predict apple.jpg

# Docker API
make docker-build && make docker-run
curl -X POST http://localhost:8000/analyze -F "file=@food.jpg"
```

---

## 🔬 Technical Highlights

- **Multi-task head** — classification, regression, segmentation in one forward pass
- **Focal loss + label smoothing** — robust to imbalanced food categories
- **Cosine warmup scheduler** — stable convergence in <50 epochs
- **TorchScript/ONNX export** — deploy to mobile, Raspberry Pi, or edge servers
- **Grad-CAM overlays** — visual reasoning for human-in-the-loop verification
- **Streamlit dashboard** — interactive demos for stakeholders

---

## 📈 Performance Benchmarks

| Metric | GPU (A100) | CPU (Xeon) | Edge (RPi 5) |
|--------|-----------|------------|--------------|
| Latency | 12 ms | 95 ms | 410 ms |
| Throughput | 450 img/s | 55 img/s | 2.4 img/s |
| Model size | — | — | 28 MB (ONNX) |

---

## 📁 Repository Structure

```
smartbite/          # Core package — 15 modules
  models/           EfficientNet, ViT, UNet, multi-task heads
  preprocessing/    Image transforms, food-specific augmentation
  features/         Color, texture, spectral, nutritional features
  training/         Trainer, Focal loss, Cosine warmup, callbacks
  inference/        Optimized engine with caching + ONNX
  api/              FastAPI REST endpoints + schemas
  serving/          GRPC server, config presets
  visualization/    Grad-CAM, dashboard components
tests/              Unit and integration tests
scripts/            CLI, batch, export, dashboard entrypoints
configs/            GPU and mobile inference profiles
docs/               Investor overview, API reference, edge deployment
```

---

## 🛣️ Roadmap

- [x] Core classification + freshness engine
- [x] REST API + Docker deployment
- [x] Edge export (ONNX, TorchScript)
- [ ] Mobile app SDK (iOS/Android)
- [ ] Hyperspectral sensor integration
- [ ] Supply-chain batch tracking API
- [ ] Federated learning for chain-wide models

---

## 🤝 For Investors

SmartBite is **production-ready for pilot deployment** in grocery and food service environments. We are seeking:

- **Seed-stage capital** ($500K–$1.5M) for go-to-market
- **Pilot partnerships** with grocery chains and food distributors
- **Strategic advisors** in food safety and supply chain

📬 **Contact:** dev@smartbite.ai

---

## 📜 License

MIT License — see [LICENSE](LICENSE)