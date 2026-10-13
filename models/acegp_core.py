"""Core neural network components for ACEGP models."""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal

class ResBlock(nn.Module):
    def __init__(self, w: int, p: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(w, w), 
            nn.GELU(), 
            nn.Dropout(p), 
            nn.Linear(w, w)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim: int, task_dim: int = 4, width: int = 256, blocks: int = 3, p: float = 0.1):
        super().__init__()
        self.inp = nn.Linear(latent_dim, width)
        self.blocks = nn.ModuleList([ResBlock(width, p) for _ in range(blocks)])
        self.out = nn.Linear(width, task_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.inp(z))
        for b in self.blocks: 
            h = b(h)
        return self.out(h)

class CoupledPredictor(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        rank: int,
        channel: Literal["coord", "random"],
        hidden: int = 256,
        generator: torch.Generator | None = None,
    ):
        super().__init__()
        assert channel in ("coord", "random")
        self.latent_dim = latent_dim
        self.rank = rank
        self.channel = channel

        if rank == 0:
            P = torch.zeros(latent_dim, 0)
        elif channel == "coord":
            P = torch.zeros(latent_dim, rank)
            idx = torch.arange(rank)
            P[idx, idx] = 1.0
        else:
            A = torch.randn(latent_dim, rank, generator=generator)
            Q, _ = torch.linalg.qr(A)
            P = Q

        self.register_buffer("P", P)
        self.head = nn.Sequential(
            nn.Linear(rank + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        if self.rank == 0:
            delta = self.head(a)
        else:
            z_proj = z @ self.P
            inp = torch.cat([z_proj, a], dim=-1)
            delta = self.head(inp)
        return z + delta
