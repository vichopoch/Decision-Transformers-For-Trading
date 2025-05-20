"""Elastic Trader package."""

from __future__ import annotations

import torch

# Ensure float32 precision for MPS
torch.set_default_dtype(torch.float32)

__all__: list[str] = [
    "data",
    "envs",
    "features",
    "models",
    "scripts",
    "utils",
    "evolve",
]
