from __future__ import annotations

from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsConstrainedMLP(nn.Module):
    """
    Forward surrogate network mapping photonic geometry parameters [L_c, W_c, P_t, (L_t)]
    to optical transmission spectra at upper and lower output ports.

    Physics Inductive Bias:
      - Uses Softplus to enforce strictly non-negative transmitted power.
      - Enforces the conservation of energy bound (T_up + T_down <= 1.0) via dynamic
        saturation clamping at each wavelength channel.
    """

    def __init__(
        self,
        in_dim: int = 3,
        n_wavelengths: int = 10,
        width: int = 512,
        depth: int = 5,
        dropout: float = 0.05,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.n_wavelengths = n_wavelengths
        self.out_dim = 2 * n_wavelengths
        self.eps = eps

        layers: List[nn.Module] = [
            nn.Linear(in_dim, width),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]
        for _ in range(depth - 1):
            layers.extend([
                nn.Linear(width, width),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
        layers.append(nn.Linear(width, self.out_dim))
        self.net = nn.Sequential(*layers)

    def forward(
        self, x: torch.Tensor, return_aux: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        raw = self.net(x)
        tu_raw, td_raw = torch.chunk(raw, 2, dim=-1)

        tu_pos = F.softplus(tu_raw)
        td_pos = F.softplus(td_raw)
        s_raw = tu_pos + td_pos

        # Energy conservation scale: scale down only if Tu + Td > 1
        scale = torch.clamp(1.0 / (s_raw + self.eps), max=1.0)
        tu = tu_pos * scale
        td = td_pos * scale

        out = torch.cat([tu, td], dim=-1)
        if return_aux:
            return out, s_raw
        return out


def physics_penalty_loss(s_raw: torch.Tensor) -> torch.Tensor:
    """Penalize raw unconstrained power predictions exceeding unity: (max(0, S_raw - 1))^2."""
    return F.relu(s_raw - 1.0).pow(2).mean()


class SurrogateEnsemble:
    """
    Manages an ensemble of forward surrogate models for epistemic uncertainty quantification
    and robust surrogate guidance during inverse design.
    """

    def __init__(self, models: List[nn.Module], device: Optional[torch.device] = None):
        self.models = models
        self.device = device or (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        for m in self.models:
            m.to(self.device)
            m.eval()

    def predict(
        self, x: Union[np.ndarray, torch.Tensor], return_uncertainty: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Compute mean prediction and epistemic standard deviation across ensemble members.
        """
        if isinstance(x, np.ndarray):
            xt = torch.from_numpy(x.astype(np.float32)).to(self.device)
        else:
            xt = x.to(self.device)

        with torch.no_grad():
            preds = [m(xt).detach().cpu().numpy() for m in self.models]

        stack = np.stack(preds, axis=0)  # [n_models, batch, out_dim]
        mean = np.mean(stack, axis=0)
        if return_uncertainty:
            std = np.std(stack, axis=0)
            return mean, std
        return mean
