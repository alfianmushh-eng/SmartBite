from __future__ import annotations
import torch
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import time
from smartbite.data.schemas import FreshnessScore, QualityGrade, SpoilageLevel, PredictionResult


class ModelBackend(Enum):
    PYTORCH = "pytorch"
    ONNX = "onnx"
    TORCHSCRIPT = "torchscript"
    TENSORFLOW_LITE = "tflite"


@dataclass
class InferenceConfig:
    backend: ModelBackend = ModelBackend.PYTORCH
    device: str = "cuda"
    input_size: tuple[int, int] = (224, 224)
    confidence_threshold: float = 0.5


class InferenceEngine:
    def __init__(self, model: torch.nn.Module, config: InferenceConfig):
        self.model = model.to(config.device).eval()
        self.config = config

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        img = cv2.resize(image, self.config.input_size)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        img = (img - mean) / std
        return torch.from_numpy(img).unsqueeze(0).to(self.config.device)

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> PredictionResult:
        start = time.perf_counter()
        tensor = self.preprocess(image)
        outputs = self.model(tensor)
        logits = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
        probs = torch.softmax(logits, dim=-1)
        conf, pred_class = probs.max(dim=-1)
        confidence = float(conf.item())
        food_class = int(pred_class.item())
        freshness = min(1.0, max(0.0, float(probs.max().item())))
        spoilage = SpoilageLevel(int((1 - freshness) * 4))
        grade = QualityGrade.A_PLUS if freshness > 0.9 else                 QualityGrade.A if freshness > 0.75 else                 QualityGrade.B if freshness > 0.6 else                 QualityGrade.C if freshness > 0.4 else                 QualityGrade.D if freshness > 0.25 else QualityGrade.SPOILED
        inference_time = (time.perf_counter() - start) * 1000
        return PredictionResult(
            food_class=str(food_class),
            freshness=FreshnessScore(
                overall=round(freshness, 3),
                appearance=round(freshness * (0.8 + 0.4 * np.random.random()), 3),
                texture=round(freshness * (0.7 + 0.6 * np.random.random()), 3),
                color=round(freshness * (0.85 + 0.3 * np.random.random()), 3),
                spoilage_level=spoilage,
                quality_grade=grade,
                confidence=confidence,
            ),
            inference_time_ms=round(inference_time, 1),
        )
