from __future__ import annotations
import pytest
from smartbite.data.schemas import FreshnessScore, QualityGrade, SpoilageLevel


class TestDataSchemas:
    def test_freshness_score_defaults(self):
        score = FreshnessScore(overall=0.85, appearance=0.8, texture=0.7, color=0.9,
                               spoilage_level=SpoilageLevel.NONE,
                               quality_grade=QualityGrade.A, confidence=0.92)
        assert score.overall == 0.85
        assert score.quality_grade == QualityGrade.A
        assert score.spoilage_level == SpoilageLevel.NONE
