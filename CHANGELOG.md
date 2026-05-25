# Changelog

## v0.1.0 (2026-05-27)

### Initial Release

**Core Features:**
- Food classification (210 categories) using EfficientNet-B3 and ViT-B-16
- Real-time freshness scoring with multi-dimensional quality assessment
- Spoilage detection with 5-level grading system
- Food region segmentation via UNet
- Grad-CAM explainability heatmaps

**Infrastructure:**
- FastAPI REST API with single/batch endpoints
- GPU-accelerated Docker deployment
- CLI tools for prediction and batch processing
- ONNX and TorchScript model export
- Inference caching with LRU eviction

**Developer Experience:**
- Comprehensive test suite
- Pre-commit hooks
- Makefile with common targets
- Optional dependency groups

**Documentation:**
- Investor overview with market analysis
- API reference with endpoint docs
- Edge deployment guide
- Example scripts for all workflows

