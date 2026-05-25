"""Batch food quality analysis for inventory."""

from smartbite.inference.engine import InferenceEngine, InferenceConfig
from smartbite.models.classifier import FoodClassifier
import numpy as np


def main():
    n_images = 10
    images = [np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8) for _ in range(n_images)]

    model = FoodClassifier(num_classes=210, pretrained=False)
    config = InferenceConfig(device="cpu")
    engine = InferenceEngine(model, config)

    results = [engine.predict(img) for img in images]
    avg_freshness = np.mean([r.freshness.overall for r in results])

    print(f"Processed {n_images} food items")
    print(f"Average freshness: {avg_freshness:.2f}")
    print(f"Spoiled items: {sum(1 for r in results if r.freshness.spoilage_level.name == 'ADVANCED')}")


if __name__ == "__main__":
    main()
