#!/usr/bin/env python3
"""Batch analysis CLI for SmartBite."""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
import cv2
import torch
from smartbite.inference.engine import InferenceEngine, InferenceConfig
from smartbite.models.classifier import FoodClassifier


def main():
    parser = argparse.ArgumentParser(description="SmartBite batch analyzer")
    parser.add_argument("input_dir", help="Directory containing food images")
    parser.add_argument("--checkpoint", help="Model checkpoint path")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--format", choices=["text", "csv"], default="text")
    args = parser.parse_args()

    model = FoodClassifier(num_classes=210)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    config = InferenceConfig(device=args.device)
    engine = InferenceEngine(model, config)

    image_exts = {".jpg", ".jpeg", ".png", ".webp"}
    images = sorted([p for p in Path(args.input_dir).iterdir() if p.suffix.lower() in image_exts])

    if not images:
        print(f"No images found in {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    results = []
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pred = engine.predict(img)
        results.append((img_path.name, pred))

    if args.format == "csv":
        print("filename,class,freshness,grade,spoilage,confidence")
        for name, pred in results:
            print(f"{name},{pred.food_class},{pred.freshness.overall:.2f},"
                  f"{pred.freshness.quality_grade.value},{pred.freshness.spoilage_level.name},"
                  f"{pred.freshness.confidence:.3f}")
    else:
        for name, pred in results:
            print(f"{name:30s} class={pred.food_class:>4s}  "
                  f"freshness={pred.freshness.overall:.2f}  "
                  f"grade={pred.freshness.quality_grade.value}  "
                  f"spoilage={pred.freshness.spoilage_level.name}")


if __name__ == "__main__":
    main()
