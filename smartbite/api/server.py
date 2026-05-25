from __future__ import annotations
from typing import Optional
import numpy as np
from smartbite.inference.engine import InferenceEngine, InferenceConfig


class FoodQualityAPI:
    def __init__(self, engine: InferenceEngine):
        self.engine = engine

    def analyze(self, image: np.ndarray) -> dict:
        result = self.engine.predict(image)
        return {
            "food_class": result.food_class,
            "freshness": {
                "overall": result.freshness.overall,
                "appearance": result.freshness.appearance,
                "texture": result.freshness.texture,
                "color": result.freshness.color,
                "spoilage_level": result.freshness.spoilage_level.name,
                "quality_grade": result.freshness.quality_grade.value,
                "confidence": result.freshness.confidence,
            },
            "inference_time_ms": result.inference_time_ms,
        }

    def health(self) -> dict:
        return {"status": "healthy", "model": "smartbite-v1", "backend": self.engine.config.backend.value}


def create_app(engine: Optional[InferenceEngine] = None) -> FoodQualityAPI:
    if engine is None:
        import torch.nn as nn
        dummy = nn.Identity()
        config = InferenceConfig(device="cpu")
        engine = InferenceEngine(dummy, config)
    return FoodQualityAPI(engine)
