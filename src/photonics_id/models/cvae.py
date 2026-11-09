from __future__ import annotations

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Spec2GeoCVAE(nn.Module):
    """
    Conditional Variational Autoencoder (CVAE) for photonic inverse design.
    Maps a target transmission spectrum (condition y) into candidate device geometries (x).

    Encoder: [y, x] -> (mu, logvar) in latent space z
    Decoder: [y, z] -> x_hat bounded in [0, 1] (unit-cube parameter bounds)
    """

    def __init__(
        self,
        y_dim: int = 20,
        x_dim: int = 3,
        z_dim: int = 16,
        width: int = 512,
        depth: int = 4,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.y_dim = y_dim
        self.x_dim = x_dim
        self.z_dim = z_dim

        # Encoder: [y, x] -> latent parameters
        enc_in = y_dim + x_dim
        enc_layers = [nn.Linear(enc_in, width), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(depth - 1):
            enc_layers.extend([nn.Linear(width, width), nn.ReLU(), nn.Dropout(dropout)])
        self.enc = nn.Sequential(*enc_layers)
        self.mu = nn.Linear(width, z_dim)
        self.logvar = nn.Linear(width, z_dim)

        # Decoder: [y, z] -> x_hat
        dec_in = y_dim + z_dim
        dec_layers = [nn.Linear(dec_in, width), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(depth - 1):
            dec_layers.extend([nn.Linear(width, width), nn.ReLU(), nn.Dropout(dropout)])
        dec_layers.extend([nn.Linear(width, x_dim), nn.Sigmoid()])
        self.dec = nn.Sequential(*dec_layers)

    def encode(self, y: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(torch.cat([y, x], dim=-1))
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.dec(torch.cat([y, z], dim=-1))

    def forward(
        self, y: torch.Tensor, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(y, x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(y, z)
        return x_hat, mu, logvar


def cvae_loss(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1e-3,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes CVAE reconstruction loss + beta-weighted KL divergence.
    """
    recon = F.l1_loss(x_hat, x)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon + beta * kl
    return total, recon, kl
