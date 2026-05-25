from __future__ import annotations
import numpy as np
import pytest
from smartbite.features.color import ColorHistogram, ColorMoments, DominantColors


class TestColorFeatures:
    def setup_method(self):
        self.img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

    def test_color_histogram_length(self):
        ch = ColorHistogram()
        features = ch.extract(self.img, bins=32)
        assert len(features) == 32 * 3

    def test_color_moments_length(self):
        cm = ColorMoments()
        features = cm.extract(self.img)
        assert len(features) == 9

    def test_dominant_colors_length(self):
        dc = DominantColors(n_colors=5)
        features = dc.extract(self.img)
        assert len(features) == 5 * 4
