"""Training script for SmartBite food quality models."""

from __future__ import annotations
import torch
from smartbite.utils.config import Config, load_config
from smartbite.models.classifier import FoodClassifier
from smartbite.training.trainer import Trainer
from smartbite.training.losses import CombinedLoss
from smartbite.training.callbacks import ModelCheckpoint, EarlyStopping, TensorBoardLogger


def main():
    config_path = "configs/default.json"
    cfg = load_config(config_path) if __import__("os").path.exists(config_path) else Config()

    model = FoodClassifier(num_classes=cfg.model.num_classes, pretrained=cfg.model.pretrained)
    criterion = CombinedLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate,
                                  weight_decay=cfg.training.weight_decay)

    callbacks = [
        ModelCheckpoint(cfg.checkpoint_dir),
        EarlyStopping(patience=cfg.training.early_stopping_patience),
        TensorBoardLogger(cfg.log_dir),
    ]

    trainer = Trainer(model, criterion, optimizer, cfg.device, callbacks)
    print("SmartBite training initialised. Use fit() with DataLoaders to start training.")


if __name__ == "__main__":
    main()