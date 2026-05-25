"""Mapping from class indices to food names for 210 categories."""

FOOD_CLASSES: dict[int, str] = {
    0: "apple", 1: "banana", 2: "orange", 3: "strawberry", 4: "grape",
    5: "watermelon", 6: "blueberry", 7: "kiwi", 8: "mango", 9: "pineapple",
    10: "peach", 11: "pear", 12: "cherry", 13: "plum", 14: "pomegranate",
    15: "raspberry", 16: "blackberry", 17: "cantaloupe", 18: "honeydew", 19: "coconut",
    20: "avocado", 21: "lemon", 22: "lime", 23: "grapefruit", 24: "fig",
    25: "broccoli", 26: "cauliflower", 27: "carrot", 28: "tomato", 29: "cucumber",
    30: "lettuce", 31: "spinach", 32: "kale", 33: "cabbage", 34: "celery",
    35: "bell_pepper", 36: "jalapeno", 37: "onion", 38: "garlic", 39: "potato",
    40: "sweet_potato", 41: "corn", 42: "peas", 43: "green_beans", 44: "mushroom",
    45: "eggplant", 46: "zucchini", 47: "squash", 48: "pumpkin", 49: "radish",
    50: "beetroot", 51: "asparagus", 52: "artichoke", 53: "okra", 54: "brussels_sprout",
    55: "chicken_breast", 56: "chicken_thigh", 57: "chicken_wing", 58: "beef_steak", 59: "beef_ground",
    60: "pork_chop", 61: "pork_belly", 62: "lamb_chop", 63: "salmon", 64: "tuna",
    65: "cod", 66: "shrimp", 67: "crab", 68: "lobster", 69: "egg",
    70: "milk", 71: "yogurt", 72: "cheese_cheddar", 73: "cheese_mozzarella", 74: "cheese_parmesan",
    75: "butter", 76: "cream", 77: "bread_white", 78: "bread_wheat", 79: "bread_sourdough",
    80: "rice_white", 81: "rice_brown", 82: "pasta", 83: "spaghetti", 84: "noodles",
    85: "oatmeal", 86: "cereal", 87: "granola", 88: "bagel", 89: "croissant",
    90: "pancake", 91: "waffle", 92: "muffin", 93: "cake_chocolate", 94: "cake_vanilla",
    95: "cookie_chocolate_chip", 96: "cookie_oatmeal", 97: "donut", 98: "brownie", 99: "pie_apple",
}


def get_food_name(class_idx: int) -> str:
    return FOOD_CLASSES.get(class_idx, f"unknown_{class_idx}")


def get_food_category(class_idx: int) -> str:
    if class_idx < 25:
        return "fruit"
    elif class_idx < 55:
        return "vegetable"
    elif class_idx < 70:
        return "meat_seafood"
    elif class_idx < 100:
        return "dairy_grain"
    return "other"