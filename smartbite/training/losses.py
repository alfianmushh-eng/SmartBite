from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == "mean":
            return focal.mean()
        return focal.sum()


class CombinedLoss(nn.Module):
    def __init__(self, ce_weight: float = 1.0, focal_weight: float = 0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.focal = FocalLoss()
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.ce_weight * self.ce(logits, targets) + self.focal_weight * self.focal(logits, targets)


class FreshnessLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred.squeeze(), target.float())
