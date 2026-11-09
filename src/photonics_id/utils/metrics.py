from __future__ import annotations

import numpy as np
import torch
from typing import Tuple, Union


def to_numpy(x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def compute_mae(y_pred: Union[np.ndarray, torch.Tensor], y_true: Union[np.ndarray, torch.Tensor]) -> float:
    """Mean absolute error across all elements."""
    p, t = to_numpy(y_pred), to_numpy(y_true)
    return float(np.mean(np.abs(p - t)))


def compute_per_channel_mae(y_pred: Union[np.ndarray, torch.Tensor], y_true: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Mean absolute error per spectral wavelength channel."""
    p, t = to_numpy(y_pred), to_numpy(y_true)
    return np.mean(np.abs(p - t), axis=0)


def split_transmission_ports(y: Union[np.ndarray, torch.Tensor]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split spectrum output into upper transmission port (T_up) and lower port (T_down).
    Assumes channels are arranged as [Tu_1, ..., Tu_N, Td_1, ..., Td_N].
    """
    arr = to_numpy(y)
    n_w = arr.shape[-1] // 2
    return arr[..., :n_w], arr[..., n_w:]


def compute_throughput(y: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Compute total transmission throughput T_total = T_up + T_down."""
    tu, td = split_transmission_ports(y)
    return tu + td


def compute_splitting_imbalance(y: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Compute absolute power split imbalance |T_up - T_down|."""
    tu, td = split_transmission_ports(y)
    return np.abs(tu - td)
