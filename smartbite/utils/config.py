from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import json
import os


@dataclass
class ModelConfig:
    backbone: str = "efficientnet_b3"
    num_classes: int = 210
    input_size: tuple[int, int] = (224, 224)
    pretrained: bool = True
    dropout: float = 0.3
    freeze_backbone: bool = False


@dataclass
class TrainingConfig:
    batch_size: int = 64
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 500
    mixed_precision: bool = True
    gradient_clip: float = 1.0
    early_stopping_patience: int = 10
    val_split: float = 0.15
    test_split: float = 0.10


@dataclass
class DataConfig:
    image_size: tuple[int, int] = (224, 224)
    augment_train: bool = True
    normalize: bool = True
    mean: tuple[float, ...] = (0.485, 0.456, 0.406)
    std: tuple[float, ...] = (0.229, 0.224, 0.225)
    cache_images: bool = False
    max_samples: Optional[int] = None


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    device: str = "cuda"
    seed: int = 42
    project_name: str = "smartbite"
    log_dir: str = "./logs"
    checkpoint_dir: str = "./checkpoints"

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def to_dict(self) -> dict:
        return {
            "model": {k: v for k, v in self.model.__dict__.items()},
            "training": {k: v for k, v in self.training.__dict__.items()},
            "data": {k: v for k, v in self.data.__dict__.items()},
            "device": self.device,
            "seed": self.seed,
        }


def load_config(path: str) -> Config:
    with open(path) as f:
        data = json.load(f)
    cfg = Config()
    for k, v in data.get("model", {}).items():
        setattr(cfg.model, k, v)
    for k, v in data.get("training", {}).items():
        setattr(cfg.training, k, v)
    for k, v in data.get("data", {}).items():
        setattr(cfg.data, k, v)
    cfg.device = data.get("device", cfg.device)
    cfg.seed = data.get("seed", cfg.seed)
    return cfg
