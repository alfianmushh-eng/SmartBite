from __future__ import annotations
import numpy as np
from collections import defaultdict


class ConfusionMatrix:
    def __init__(self, num_classes: int):
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        self.num_classes = num_classes

    def update(self, preds: np.ndarray, targets: np.ndarray):
        for p, t in zip(preds, targets):
            self.matrix[t, p] += 1

    def accuracy(self) -> float:
        return float(np.trace(self.matrix)) / max(float(self.matrix.sum()), 1e-8)

    def precision(self, class_idx: int) -> float:
        tp = self.matrix[class_idx, class_idx]
        fp = self.matrix[:, class_idx].sum() - tp
        return tp / max(tp + fp, 1e-8)

    def recall(self, class_idx: int) -> float:
        tp = self.matrix[class_idx, class_idx]
        fn = self.matrix[class_idx, :].sum() - tp
        return tp / max(tp + fn, 1e-8)

    def f1(self, class_idx: int) -> float:
        p = self.precision(class_idx)
        r = self.recall(class_idx)
        return 2 * p * r / max(p + r, 1e-8)

    def mean_f1(self) -> float:
        return float(np.mean([self.f1(i) for i in range(self.num_classes)]))

    def reset(self):
        self.matrix.fill(0)


class MetricsTracker:
    def __init__(self):
        self.history: dict[str, list[float]] = defaultdict(list)
        self.best_value: dict[str, float] = {}
        self.best_epoch: dict[str, int] = {}

    def log(self, name: str, value: float, epoch: int):
        self.history[name].append(value)
        if name not in self.best_value or value > self.best_value[name]:
            self.best_value[name] = value
            self.best_epoch[name] = epoch

    def latest(self, name: str) -> float:
        vals = self.history.get(name, [])
        return vals[-1] if vals else 0.0
