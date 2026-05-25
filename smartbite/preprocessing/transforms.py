from __future__ import annotations
import numpy as np
import cv2
from typing import Callable


Transform = Callable[[np.ndarray], np.ndarray]


def resize_with_aspect(image: np.ndarray, target_size: int = 224) -> np.ndarray:
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def normalize_image(image: np.ndarray, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) -> np.ndarray:
    image = image.astype(np.float32) / 255.0
    for i in range(3):
        image[:, :, i] = (image[:, :, i] - mean[i]) / std[i]
    return image


def center_crop(image: np.ndarray, size: int = 224) -> np.ndarray:
    h, w = image.shape[:2]
    y1 = max(0, (h - size) // 2)
    x1 = max(0, (w - size) // 2)
    return image[y1:y1+size, x1:x1+size]


def random_horizontal_flip(image: np.ndarray, prob: float = 0.5) -> np.ndarray:
    return cv2.flip(image, 1) if np.random.random() < prob else image


def color_jitter(image: np.ndarray, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1) -> np.ndarray:
    img = image.astype(np.float32)
    if np.random.random() < 0.5:
        img *= 1 + np.random.uniform(-brightness, brightness)
    if np.random.random() < 0.5:
        mean = img.mean()
        img = mean + (img - mean) * (1 + np.random.uniform(-contrast, contrast))
    if np.random.random() < 0.5:
        hsv = cv2.cvtColor(img.clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] *= 1 + np.random.uniform(-saturation, saturation)
        hsv[:, :, 0] += np.random.uniform(-hue, hue) * 180
        img = cv2.cvtColor(hsv.clip(0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
    return img.clip(0, 255).astype(np.uint8)


def gaussian_blur(image: np.ndarray, max_kernel: int = 5) -> np.ndarray:
    k = np.random.choice([k for k in range(1, max_kernel + 1, 2)])
    return cv2.GaussianBlur(image, (k, k), 0)


def adjust_brightness(image: np.ndarray, factor: float = 1.2) -> np.ndarray:
    return np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def to_tensor(image: np.ndarray) -> np.ndarray:
    return np.transpose(image.astype(np.float32) / 255.0, (2, 0, 1))


def compose_transforms(*transforms: Transform) -> Transform:
    def apply(img: np.ndarray) -> np.ndarray:
        for t in transforms:
            img = t(img)
        return img
    return apply
