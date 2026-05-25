from __future__ import annotations
import numpy as np
import cv2
from pathlib import Path
from typing import Callable, Optional
import json
import random


class FoodDataset:
    def __init__(self, root_dir: str, transform: Optional[Callable] = None,
                 split: str = "train", label_map: Optional[dict] = None):
        self.root = Path(root_dir)
        self.transform = transform
        self.split = split
        self.samples: list[tuple[str, int]] = []
        self.label_map = label_map or {}
        self._load()

    def _load(self):
        split_dir = self.root / self.split
        if not split_dir.exists():
            raise FileNotFoundError(f"{split_dir} not found")
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            label = self.label_map.get(class_dir.name, -1)
            for img_path in class_dir.glob("*.*"):
                if img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                    self.samples.append((str(img_path), label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:
        path, label = self.samples[idx]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        return img, label


class BatchIterator:
    def __init__(self, dataset: FoodDataset, batch_size: int = 32, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))
        self.pos = 0

    def __iter__(self):
        self.pos = 0
        if self.shuffle:
            random.shuffle(self.indices)
        return self

    def __next__(self) -> tuple[np.ndarray, np.ndarray]:
        if self.pos >= len(self.indices):
            raise StopIteration
        batch_idx = self.indices[self.pos:self.pos + self.batch_size]
        self.pos += self.batch_size
        batch_x, batch_y = [], []
        for idx in batch_idx:
            x, y = self.dataset[idx]
            batch_x.append(x)
            batch_y.append(y)
        return np.stack(batch_x), np.array(batch_y)
