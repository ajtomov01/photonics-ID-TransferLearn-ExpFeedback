from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from photonics_id.models.cvae import Spec2GeoCVAE
from photonics_id.pipeline.dataset import DataBundle


def sample_inverse_candidates(
    cvae: Spec2GeoCVAE,
    y_target: np.ndarray,
    n_samples: int = 256,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """
    Sample candidate geometries from CVAE conditioned on a target optical spectrum.
    Returns geometries in the unit cube [0, 1].
    """
    dev = device or next(cvae.parameters()).device
    cvae.eval()

    y_t = torch.from_numpy(np.atleast_2d(y_target).astype(np.float32)).to(dev)
    if y_t.shape[0] == 1 and n_samples > 1:
        y_t = y_t.repeat(n_samples, 1)

    with torch.no_grad():
        z = torch.randn(y_t.shape[0], cvae.z_dim, device=dev)
        x_unit = cvae.decode(y_t, z)

    return x_unit.cpu().numpy()


def rank_and_filter_candidates(
    surrogate_model: nn.Module,
    x_unit: np.ndarray,
    bundle: DataBundle,
    alpha_balance: float = 2.0,
    top_k: int = 10,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Score candidate geometries using the forward surrogate model and return the top-K designs.
    Objective: maximize total transmission throughput while penalizing port imbalance |Tu - Td|.
    """
    dev = device or next(surrogate_model.parameters()).device
    surrogate_model.eval()

    xt = torch.from_numpy(x_unit.astype(np.float32)).to(dev)
    with torch.no_grad():
        pred = surrogate_model(xt).cpu().numpy()

    n_w = pred.shape[1] // 2
    tu, td = pred[:, :n_w], pred[:, n_w:]

    throughput = (tu + td).mean(axis=1)
    imbalance = np.abs(tu - td).mean(axis=1)
    score = throughput - alpha_balance * imbalance

    ranked_idx = np.argsort(-score)[:top_k]
    best_x_unit = x_unit[ranked_idx]
    best_x_phys = bundle.unscale_X(best_x_unit)
    best_scores = score[ranked_idx]

    return best_x_phys, best_scores


def gradient_refinement(
    surrogate_model: nn.Module,
    x_seed_unit: np.ndarray,
    target_split: float = 0.5,
    steps: int = 50,
    lr: float = 0.01,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """
    Local gradient refinement starting from CVAE-proposed seeds.
    Optimizes geometry coordinates directly on the surrogate model's differentiable surface.
    """
    dev = device or next(surrogate_model.parameters()).device
    surrogate_model.eval()

    x = torch.tensor(x_seed_unit, dtype=torch.float32, device=dev, requires_grad=True)
    optimizer = torch.optim.Adam([x], lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()
        x_clamped = torch.clamp(x, 0.0, 1.0)
        pred = surrogate_model(x_clamped)

        n_w = pred.shape[-1] // 2
        tu, td = pred[..., :n_w], pred[..., n_w:]
        throughput = (tu + td).mean()
        split_error = (tu / (tu + td + 1e-6) - target_split).pow(2).mean()

        # Maximize throughput, minimize split error
        loss = -throughput + 2.0 * split_error
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        refined = torch.clamp(x, 0.0, 1.0).cpu().numpy()

    return refined
