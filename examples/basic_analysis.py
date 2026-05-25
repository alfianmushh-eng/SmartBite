"""Basic food quality analysis example."""

from smartbite.inference.engine import InferenceEngine, InferenceConfig
from smartbite.models.classifier import FoodClassifier
import cv2
import numpy as np


def main():
    dummy_img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)

    model = FoodClassifier(num_classes=210, pretrained=False)
    config = InferenceConfig(device="cpu")
    engine = InferenceEngine(model, config)

    result = engine.predict(dummy_img)
    print(f"Predicted food class: {result.food_class}")
    print(f"Freshness score: {result.freshness.overall:.2f}")
    print(f"Quality grade: {result.freshness.quality_grade.value}")
    print(f"Inference time: {result.inference_time_ms:.1f} ms")


if __name__ == "__main__":
    main()
