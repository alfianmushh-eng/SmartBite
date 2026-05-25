#!/usr/bin/env python3
"""Model export utility for TorchScript and ONNX formats."""

import argparse
import torch
from smartbite.models.classifier import FoodClassifier
from smartbite.inference.optimize import optimize_torchscript, optimize_onnx


def main():
    parser = argparse.ArgumentParser(description="Export SmartBite model")
    parser.add_argument("--format", choices=["torchscript", "onnx"], default="torchscript")
    parser.add_argument("--output", default="models/smartbite_exported")
    parser.add_argument("--num-classes", type=int, default=210)
    args = parser.parse_args()

    model = FoodClassifier(num_classes=args.num_classes, pretrained=False)
    model.eval()
    example = torch.randn(1, 3, 224, 224)

    path = args.output
    if args.format == "torchscript":
        path = optimize_torchscript(model, f"{path}.pt", example)
    else:
        path = optimize_onnx(model, f"{path}.onnx", example)

    print(f"Model exported: {path}")


if __name__ == "__main__":
    main()