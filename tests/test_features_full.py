from __future__ import annotations
import numpy as np
import pytest
from smartbite.features.color import ColorHistogram, ColorMoments, DominantColors
from smartbite.features.texture import GLCMFeatures, LBPFeatures, GaborFilterBank
from smartbite.features.spectral import SpectralSignature


class TestTextureFeatures:
    def setup_method(self):
        self.img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)

    def test_glcm_output(self):
        glcm = GLCMFeatures()
        features = glcm.extract(self.img)
        assert len(features) > 0
        assert not np.any(np.isnan(features))

    def test_lbp_output(self):
        lbp = LBPFeatures()
        features = lbp.extract(self.img)
        assert len(features) == 26

    def test_spectral_output(self):
        spec = SpectralSignature()
        features = spec.extract(self.img)
        assert len(features) == 10
