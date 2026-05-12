from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EnhancedAudioCNN(nn.Module):
    def __init__(self, num_classes: int = 8) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 32, 0.08),
            ConvBlock(32, 64, 0.12),
            ConvBlock(64, 128, 0.16),
            ConvBlock(128, 192, 0.20),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(384, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(0.40),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        pooled = torch.cat([self.avg_pool(x), self.max_pool(x)], dim=1)
        return self.classifier(pooled)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        pooled = torch.cat([self.avg_pool(x), self.max_pool(x)], dim=1)
        flattened = torch.flatten(pooled, 1)
        return self.classifier[:5](flattened)
