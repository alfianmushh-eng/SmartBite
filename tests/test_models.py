from __future__ import annotations
import torch
import pytest
from smartbite.models.classifier import FoodClassifier
from smartbite.models.segmentation import UNetSegmenter


class TestModels:
    def test_food_classifier_output_shape(self):
        model = FoodClassifier(num_classes=210, pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2, 210)

    def test_unet_output_shape(self):
        model = UNetSegmenter(in_channels=3, out_channels=1)
        x = torch.randn(1, 3, 128, 128)
        out = model(x)
        assert out.shape == (1, 1, 128, 128)
