#!/usr/bin/env python3
"""CLI tool for food quality prediction."""

from __future__ import annotations
import argparse
import sys
import cv2
import torch
from smartbite.inference.engine import InferenceEngine, InferenceConfig
from smartbite.models.classifier import FoodClassifier


def main():
    parser = argparse.ArgumentParser(description="SmartBite food quality predictor")
    parser.add_argument("image", help="Path to food image")
    parser.add_argument("--checkpoint", help="Model checkpoint path")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    model = FoodClassifier(num_classes=210)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))

    config = InferenceConfig(device=args.device)
    engine = InferenceEngine(model, config)

    img = cv2.imread(args.image)
    if img is None:
        print(f"Error: could not load {args.image}", file=sys.stderr)
        sys.exit(1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = engine.predict(img)
    if args.json:
        import json
        print(json.dumps({
            "food_class": result.food_class,
            "freshness": {
                "overall": result.freshness.overall,
                "appearance": result.freshness.appearance,
                "texture": result.freshness.texture,
                "color": result.freshness.color,
                "grade": result.freshness.quality_grade.value,
                "spoilage": result.freshness.spoilage_level.name,
            },
            "confidence": result.freshness.confidence,
            "inference_ms": result.inference_time_ms,
        }, indent=2))
    else:
        print(f"Food class: {result.food_class}")
        print(f"Freshness:  {result.freshness.overall:.2f}")
        print(f"Grade:      {result.freshness.quality_grade.value}")
        print(f"Spoilage:   {result.freshness.spoilage_level.name}")
        print(f"Confidence: {result.freshness.confidence:.3f}")


if __name__ == "__main__":
    main()