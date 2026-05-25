from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        return self.relu(self.bn2(self.conv2(x)))


class UNetSegmenter(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_filters: int = 64):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_filters)
        self.enc2 = ConvBlock(base_filters, base_filters * 2)
        self.enc3 = ConvBlock(base_filters * 2, base_filters * 4)
        self.pool = nn.MaxPool2d(2)
        self.bridge = ConvBlock(base_filters * 4, base_filters * 8)
        self.up3 = nn.ConvTranspose2d(base_filters * 8, base_filters * 4, 2, stride=2)
        self.dec3 = ConvBlock(base_filters * 8, base_filters * 4)
        self.up2 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, 2, stride=2)
        self.dec2 = ConvBlock(base_filters * 4, base_filters * 2)
        self.up1 = nn.ConvTranspose2d(base_filters * 2, base_filters, 2, stride=2)
        self.dec1 = ConvBlock(base_filters * 2, base_filters)
        self.out = nn.Conv2d(base_filters, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bridge(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)


class FoodSegmenter:
    @staticmethod
    def create(out_channels: int = 1) -> UNetSegmenter:
        return UNetSegmenter(in_channels=3, out_channels=out_channels)
