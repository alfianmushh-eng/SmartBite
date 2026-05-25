from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from datetime import datetime
import numpy as np


class QualityGrade(Enum):
    A_PLUS = "A+"
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    SPOILED = "F"


class SpoilageLevel(Enum):
    NONE = 0
    EARLY = 1
    MODERATE = 2
    ADVANCED = 3
    CRITICAL = 4


@dataclass
class FreshnessScore:
    overall: float  # 0.0 spoiled → 1.0 perfect
    appearance: float
    texture: float
    color: float
    spoilage_level: SpoilageLevel
    quality_grade: QualityGrade
    confidence: float
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class FoodItem:
    food_id: str
    category: str
    subcategory: Optional[str] = None
    freshness: Optional[FreshnessScore] = None
    image_shape: Optional[tuple[int, int, int]] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class PredictionResult:
    food_class: str
    freshness: FreshnessScore
    segmentation_mask: Optional[np.ndarray] = None
    heatmap: Optional[np.ndarray] = None
    nutrition: Optional[dict] = None
    inference_time_ms: float = 0.0
