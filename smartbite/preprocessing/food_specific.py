from __future__ import annotations
import numpy as np
import cv2


def white_balance_food(image: np.ndarray) -> np.ndarray:
    result = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    avg_a = np.mean(result[:, :, 1])
    avg_b = np.mean(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0))
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0))
    return cv2.cvtColor(result, cv2.COLOR_LAB2RGB)


def enhance_saturation(image: np.ndarray, factor: float = 1.3) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def remove_background(image: np.ndarray, threshold: int = 30) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return cv2.bitwise_and(image, image, mask=mask)


def isolate_food_region(image: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 30, 100)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image
    largest = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [largest], -1, 255, -1)
    return cv2.bitwise_and(image, image, mask=mask)
