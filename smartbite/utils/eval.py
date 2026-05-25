from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
from smartbite.utils.metrics import ConfusionMatrix


@torch.no_grad()
def evaluate_model(model: nn.Module, loader, device: str = "cuda", num_classes: int = 210) -> dict:
    model.eval()
    cm = ConfusionMatrix(num_classes)
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1).cpu().numpy()
        targets = y.numpy()
        cm.update(preds, targets)
    return {
        "accuracy": cm.accuracy(),
        "mean_f1": cm.mean_f1(),
        "precision_macro": float(np.mean([cm.precision(i) for i in range(num_classes)])),
        "recall_macro": float(np.mean([cm.recall(i) for i in range(num_classes)])),
    }
