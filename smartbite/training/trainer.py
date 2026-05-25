from __future__ import annotations
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional
from tqdm import tqdm


def train_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                optimizer: torch.optim.Optimizer, scheduler: Optional[object],
                device: str = "cuda", scaler: Optional[torch.cuda.amp.GradScaler] = None,
                grad_clip: float = 0.0) -> dict:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in tqdm(loader, desc="train", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        if scaler:
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        total_loss += loss.item()
        _, preds = logits.max(1)
        correct += preds.eq(y).sum().item()
        total += y.size(0)
        if scheduler and hasattr(scheduler, "step"):
            scheduler.step()
    return {"loss": total_loss / max(len(loader), 1), "acc": correct / max(total, 1)}


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, criterion: nn.Module,
             device: str = "cuda") -> dict:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in tqdm(loader, desc="val", leave=False):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item()
        _, preds = logits.max(1)
        correct += preds.eq(y).sum().item()
        total += y.size(0)
    return {"loss": total_loss / max(len(loader), 1), "acc": correct / max(total, 1)}


class Trainer:
    def __init__(self, model: nn.Module, criterion: nn.Module, optimizer: torch.optim.Optimizer,
                 device: str = "cuda", callbacks: Optional[list] = None):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.callbacks = callbacks or []
        self.scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None

    def fit(self, train_loader: DataLoader, val_loader: DataLoader,
            epochs: int, grad_clip: float = 0.0) -> dict:
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        for epoch in range(1, epochs + 1):
            train_metrics = train_epoch(self.model, train_loader, self.criterion,
                                        self.optimizer, None, self.device,
                                        self.scaler, grad_clip)
            val_metrics = validate(self.model, val_loader, self.criterion, self.device)
            history["train_loss"].append(train_metrics["loss"])
            history["train_acc"].append(train_metrics["acc"])
            history["val_loss"].append(val_metrics["loss"])
            history["val_acc"].append(val_metrics["acc"])
            combined = {"epoch": epoch, **train_metrics, **val_metrics}
            print(f"Epoch {epoch:3d} | train_loss: {train_metrics['loss']:.4f} "
                  f"train_acc: {train_metrics['acc']:.4f} | val_loss: {val_metrics['loss']:.4f} "
                  f"val_acc: {val_metrics['acc']:.4f}")
            for cb in self.callbacks:
                cb.step(self.model, epoch, combined)
        return history
