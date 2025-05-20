"""Elastic Decision Transformer model."""

from __future__ import annotations

from typing import List

import torch
from torch import nn


class ElasticDecisionTransformer(nn.Module):
    """Simplified Decision Transformer with variable context."""

    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        hidden_size: int = 128,
        n_layer: int = 2,
        context_lengths: List[int] | None = None,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.context_lengths = context_lengths or [1, 5, 10, 20, 60]

        self.embed_state = nn.Linear(state_dim, hidden_size)
        self.embed_action = nn.Linear(act_dim, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(hidden_size, 4)
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layer)
        self.head = nn.Linear(hidden_size, act_dim)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.embed_state(states) + self.embed_action(actions)
        x = self.encoder(x)
        return self.head(x)
