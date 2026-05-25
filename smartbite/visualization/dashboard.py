from __future__ import annotations
from smartbite.data.schemas import FreshnessScore, QualityGrade, SpoilageLevel


class FreshnessGauge:
    @staticmethod
    def render(score: FreshnessScore) -> str:
        pct = int(score.overall * 100)
        color = "green" if pct > 75 else "orange" if pct > 40 else "red"
        bar = "|" * (pct // 5) + "." * (20 - pct // 5)
        return f"[{bar}] {pct}% ({color})"


class QualityBadge:
    @staticmethod
    def render(grade: QualityGrade) -> str:
        colors = {QualityGrade.A_PLUS: "green", QualityGrade.A: "green",
                  QualityGrade.B: "blue", QualityGrade.C: "orange",
                  QualityGrade.D: "red", QualityGrade.SPOILED: "darkred"}
        return f"[{grade.value}] {colors.get(grade, 'gray')}"


class FoodCard:
    @staticmethod
    def render(food_class: str, freshness: FreshnessScore) -> str:
        gauge = FreshnessGauge.render(freshness)
        badge = QualityBadge.render(freshness.quality_grade)
        return f"Food: {food_class}  |  Grade: {badge}  |  {gauge}"


class ScoreTimeline:
    @staticmethod
    def render(scores: list[float]) -> str:
        lines = []
        for i, s in enumerate(scores):
            marker = "o" if s > 0.7 else "x" if s < 0.4 else "-"
            bar = "|" * int(s * 20)
            lines.append(f"#{i:3d}: {bar} {s:.2f} {marker}")
        return "
".join(lines)
