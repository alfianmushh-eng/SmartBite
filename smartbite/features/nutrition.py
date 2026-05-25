"""Nutritional estimation for food items."""

FOOD_NUTRITION_DB = {
    "apple": {"calories": 52, "protein": 0.3, "fat": 0.2, "carbs": 14.0, "fiber": 2.4},
    "banana": {"calories": 89, "protein": 1.1, "fat": 0.3, "carbs": 23.0, "fiber": 2.6},
    "broccoli": {"calories": 34, "protein": 2.8, "fat": 0.4, "carbs": 7.0, "fiber": 2.6},
    "chicken_breast": {"calories": 165, "protein": 31.0, "fat": 3.6, "carbs": 0.0, "fiber": 0.0},
    "salmon": {"calories": 208, "protein": 20.0, "fat": 13.0, "carbs": 0.0, "fiber": 0.0},
    "egg": {"calories": 155, "protein": 13.0, "fat": 11.0, "carbs": 1.1, "fiber": 0.0},
    "rice": {"calories": 130, "protein": 2.7, "fat": 0.3, "carbs": 28.0, "fiber": 0.4},
    "tomato": {"calories": 18, "protein": 0.9, "fat": 0.2, "carbs": 3.9, "fiber": 1.2},
    "carrot": {"calories": 41, "protein": 0.9, "fat": 0.2, "carbs": 10.0, "fiber": 2.8},
}


class NutritionEstimator:
    def __init__(self, db=None):
        self.db = db or FOOD_NUTRITION_DB

    def estimate(self, food_class, serving_size_g=100.0):
        base = self.db.get(food_class.lower(), {"calories": 50, "protein": 1.0, "fat": 0.5, "carbs": 10.0, "fiber": 1.0})
        scale = serving_size_g / 100.0
        return {
            "calories": round(base["calories"] * scale, 1),
            "protein_g": round(base["protein"] * scale, 1),
            "fat_g": round(base["fat"] * scale, 1),
            "carbs_g": round(base["carbs"] * scale, 1),
            "fiber_g": round(base["fiber"] * scale, 1),
            "serving_size_g": serving_size_g,
        }

    def freshness_adjusted(self, food_class, freshness, serving_size_g=100.0):
        base = self.estimate(food_class, serving_size_g)
        if freshness < 0.7:
            decay = 1.0 - (0.7 - freshness) * 0.3
            base["calories"] = round(base["calories"] * decay, 1)
        return base