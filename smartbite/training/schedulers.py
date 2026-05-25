from __future__ import annotations
import torch
from torch.optim.lr_scheduler import _LRScheduler
import math


class CosineWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int = 500,
                 total_steps: int = 5000, min_lr: float = 1e-6, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            scale = step / max(self.warmup_steps, 1)
            return [base_lr * scale for base_lr in self.base_lrs]
        progress = (step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return [self.min_lr + (base_lr - self.min_lr) * cosine for base_lr in self.base_lrs]
