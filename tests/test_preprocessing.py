from __future__ import annotations
import numpy as np
import pytest
from smartbite.preprocessing.transforms import resize_with_aspect, normalize_image, center_crop


class TestPreprocessing:
    def setup_method(self):
        self.img = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)

    def test_resize_preserves_aspect(self):
        resized = resize_with_aspect(self.img, target_size=224)
        assert resized.shape[0] <= 224 or resized.shape[1] <= 224

    def test_normalize_output_range(self):
        normalized = normalize_image(self.img)
        assert normalized.dtype == np.float32
        assert normalized.min() >= -3.0 and normalized.max() <= 3.0

    def test_center_crop_size(self):
        cropped = center_crop(self.img, size=224)
        assert cropped.shape == (224, 224, 3)
