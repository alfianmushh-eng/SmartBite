from __future__ import annotations
from pathlib import Path
import torch
import json
from datetime import datetime


class ModelCheckpoint:
    def __init__(self, dirpath: str, monitor: str = "val_loss", mode: str = "min", save_best_only: bool = True):
        self.dirpath = Path(dirpath)
        self.dirpath.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best_value = float("inf") if mode == "min" else float("-inf")

    def step(self, model: torch.nn.Module, epoch: int, metrics: dict) -> str | None:
        value = metrics.get(self.monitor, 0)
        improved = (self.mode == "min" and value < self.best_value) or                    (self.mode == "max" and value > self.best_value)
        if not self.save_best_only or improved:
            if improved:
                self.best_value = value
            path = self.dirpath / f"epoch_{epoch:03d}_{self.monitor}={value:.4f}.pt"
            torch.save(model.state_dict(), path)
            return str(path)
        return None


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, monitor: str = "val_loss"):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.counter = 0
        self.best_value = float("inf")
        self.stopped = False

    def step(self, metrics: dict) -> bool:
        value = metrics.get(self.monitor, 0)
        if value < self.best_value - self.min_delta:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stopped = True
        return self.stopped


class TensorBoardLogger:
    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"events_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.jsonl"

    def log(self, epoch: int, metrics: dict):
        entry = {"epoch": epoch, **metrics, "timestamp": datetime.utcnow().isoformat()}
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "
")
