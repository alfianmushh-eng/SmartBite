from __future__ import annotations
import cv2
import numpy as np
from pathlib import Path
from typing import Optional
import json
import pickle


class ImageLoader:
    supported = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

    def load(self, path: str) -> np.ndarray:
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Could not load image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def load_batch(self, paths: list[str]) -> list[np.ndarray]:
        return [self.load(p) for p in paths]

    def save(self, image: np.ndarray, path: str) -> None:
        ext = Path(path).suffix.lower()
        if ext not in self.supported:
            path += ".jpg"
        cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


class FileManager:
    def __init__(self, base_dir: str = "./"):
        self.base = Path(base_dir)

    def save_json(self, data: dict, name: str):
        path = self.base / f"{name}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        return str(path)

    def load_json(self, name: str) -> dict:
        with open(self.base / f"{name}.json") as f:
            return json.load(f)

    def save_pickle(self, obj, name: str):
        with open(self.base / f"{name}.pkl", "wb") as f:
            pickle.dump(obj, f)

    def list_checkpoints(self) -> list[str]:
        return sorted([str(p) for p in self.base.glob("*.pt")])
