from __future__ import annotations
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights


class FoodClassifier(nn.Module):
    def __init__(self, backbone: str = "efficientnet_b3", num_classes: int = 210,
                 dropout: float = 0.3, pretrained: bool = True):
        super().__init__()
        weights = EfficientNet_B3_Weights.DEFAULT if pretrained else None
        self.backbone = efficientnet_b3(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)


class EfficientNetClassifier:
    @staticmethod
    def create(num_classes: int = 210, pretrained: bool = True) -> FoodClassifier:
        return FoodClassifier(
            backbone="efficientnet_b3",
            num_classes=num_classes,
            pretrained=pretrained,
        )
