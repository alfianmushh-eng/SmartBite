from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class FreshnessHead(nn.Module):
    def __init__(self, in_features: int = 1536):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x)).squeeze(-1)


class QualityHead(nn.Module):
    def __init__(self, in_features: int = 1536, num_grades: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_grades),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self.net(x), dim=-1)


class MultiTaskHead(nn.Module):
    def __init__(self, in_features: int = 1536, num_classes: int = 210, num_grades: int = 6):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.class_head = nn.Linear(512, num_classes)
        self.freshness_head = nn.Sequential(nn.Linear(512, 64), nn.ReLU(), nn.Linear(64, 1))
        self.quality_head = nn.Linear(512, num_grades)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        shared = self.shared(x)
        return {
            "class_logits": self.class_head(shared),
            "freshness": torch.sigmoid(self.freshness_head(shared)).squeeze(-1),
            "quality": F.log_softmax(self.quality_head(shared), dim=-1),
        }
