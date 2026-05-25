# Edge Deployment Guide

## TensorFlow Lite Export

```bash
python -c "
from smartbite.models.classifier import FoodClassifier
from smartbite.inference.optimize import optimize_onnx, optimize_torchscript
import torch

model = FoodClassifier(num_classes=50, pretrained=False)
example = torch.randn(1, 3, 192, 192)

# TorchScript export
optimize_torchscript(model, 'models/smartbite_mobile.pt', example)

# ONNX export
optimize_onnx(model, 'models/smartbite_mobile.onnx', example)
print('Edge models exported successfully')
"
```

## Raspberry Pi Setup

```bash
pip install smartbite
wget https://models.smartbite.ai/smartbite_mobile.pt
smartbite predict --checkpoint smartbite_mobile.pt --device cpu test.jpg
```

## Mobile SDK

- iOS: CoreML model available on request
- Android: TFLite model available on request
- Expected inference time: <100ms on modern devices
