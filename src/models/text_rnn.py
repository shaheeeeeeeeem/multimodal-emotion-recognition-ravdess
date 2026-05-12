from __future__ import annotations

import torch
from torch import nn


class TextRNN(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int = 8, embedding_dim: int = 64, hidden_dim: int = 64) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        _, hidden = self.rnn(embedded)
        hidden_forward = hidden[-2]
        hidden_backward = hidden[-1]
        features = torch.cat([hidden_forward, hidden_backward], dim=1)
        return self.classifier(features)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        _, hidden = self.rnn(embedded)
        hidden_forward = hidden[-2]
        hidden_backward = hidden[-1]
        features = torch.cat([hidden_forward, hidden_backward], dim=1)
        return self.classifier[:3](features)
