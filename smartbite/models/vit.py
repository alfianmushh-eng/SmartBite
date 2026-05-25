from __future__ import annotations
import torch
import torch.nn as nn
from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights


class ViTClassifier(nn.Module):
    def __init__(self, num_classes: int = 210, pretrained: bool = True):
        super().__init__()
        weights = ViT_B_16_Weights.DEFAULT if pretrained else None
        self.vit = vit_b_16(weights=weights)
        in_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(0.2),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit(x)


class FoodViT:
    @staticmethod
    def create(num_classes: int = 210) -> ViTClassifier:
        return ViTClassifier(num_classes=num_classes, pretrained=True)
