from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn


def freeze_parameters(model: nn.Module) -> None:
    """Freeze all model parameters in-place."""
    for p in model.parameters():
        p.requires_grad = False


class InputWarpAffineModel(nn.Module):
    """
    Physics-aware domain adaptation module:
        y_adapted(x) = a(x) * f_base(phi(x)) + b(x), where phi(x) = x + delta(x).

    The base model f_base is frozen, and only lightweight input-warp
    and calibration heads are trained on target domain samples.
    """

    def __init__(
        self,
        base_model: nn.Module,
        in_dim: int,
        out_dim: int,
        warp_hidden: int = 64,
        head_hidden: int = 64,
        warp_scale: float = 0.05,
    ):
        super().__init__()
        self.base_model = base_model
        freeze_parameters(self.base_model)

        self.warp_scale = float(warp_scale)
        self.warp_net = nn.Sequential(
            nn.Linear(in_dim, warp_hidden),
            nn.GELU(),
            nn.Linear(warp_hidden, warp_hidden),
            nn.GELU(),
            nn.Linear(warp_hidden, in_dim),
        )

        self.calib_net = nn.Sequential(
            nn.Linear(in_dim, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, 2 * out_dim),
        )

        # Initialize calibration heads near identity: a(x) ~ 1, b(x) ~ 0
        last_layer = self.calib_net[-1]
        nn.init.zeros_(last_layer.weight)
        bias_init = torch.cat([torch.zeros(out_dim), torch.zeros(out_dim)], dim=0)
        last_layer.bias.data.copy_(bias_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = torch.tanh(self.warp_net(x)) * self.warp_scale
        x_warped = torch.clamp(x + delta, 0.0, 1.0)

        with torch.no_grad():
            y_base = self.base_model(x_warped)

        calib = self.calib_net(x)
        out_dim = y_base.shape[-1]
        scale_raw, bias = calib[..., :out_dim], calib[..., out_dim:]
        scale = 1.0 + torch.tanh(scale_raw) * 0.2  # gentle scaling around 1.0

        return y_base * scale + bias


class ResidualAdapter(nn.Module):
    """
    Residual correction adapter:
        y_adapted(x) = f_base(x) + delta_f(x)
    """

    def __init__(
        self,
        base_model: nn.Module,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 64,
        depth: int = 2,
    ):
        super().__init__()
        self.base_model = base_model
        freeze_parameters(self.base_model)

        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.residual_net = nn.Sequential(*layers)

        # Zero-initialize the final layer so initial prediction matches base model
        nn.init.zeros_(self.residual_net[-1].weight)
        nn.init.zeros_(self.residual_net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            y_base = self.base_model(x)
        delta_y = self.residual_net(x)
        return y_base + delta_y
