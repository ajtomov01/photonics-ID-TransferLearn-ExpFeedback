from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class DataBundle:
    X_train: np.ndarray
    Y_train: np.ndarray
    X_val: np.ndarray
    Y_val: np.ndarray
    X_test: np.ndarray
    Y_test: np.ndarray
    x_min: np.ndarray
    x_max: np.ndarray
    x_rng: np.ndarray
    feature_cols: List[str]
    target_cols: List[str]

    def scale_X(self, x: np.ndarray) -> np.ndarray:
        """Scale physical geometry parameters into [0, 1] unit cube."""
        return np.clip((x - self.x_min) / (self.x_rng + 1e-8), 0.0, 1.0).astype(np.float32)

    def unscale_X(self, x_unit: np.ndarray) -> np.ndarray:
        """Map unit cube parameters back to physical dimensions."""
        return x_unit * self.x_rng + self.x_min


def create_dataloaders(
    bundle: DataBundle, batch_size: int = 128
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create PyTorch DataLoaders for train, val, and test splits."""
    def _loader(x: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
        ds = TensorDataset(
            torch.from_numpy(bundle.scale_X(x)).float(),
            torch.from_numpy(y).float(),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    return (
        _loader(bundle.X_train, bundle.Y_train, shuffle=True),
        _loader(bundle.X_val, bundle.Y_val, shuffle=False),
        _loader(bundle.X_test, bundle.Y_test, shuffle=False),
    )


def generate_synthetic_mmi_dataset(
    n_samples: int = 1000,
    n_wavelengths: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic 2x2 MMI coupler data obeying physical bounds (T_up + T_down <= 1.0).
    Useful for demonstration, continuous integration, and architecture exploration
    without requiring proprietary or multi-gigabyte FDTD datasets.
    """
    rng = np.random.default_rng(seed)

    # Geometry: Lc (100 to 300 um), Wc (4 to 12 um), Pt (1 to 4 um)
    lc = rng.uniform(100.0, 300.0, size=n_samples)
    wc = rng.uniform(4.0, 12.0, size=n_samples)
    pt = rng.uniform(1.0, 4.0, size=n_samples)

    # Simplified optical physics proxy
    beat_length = 3.0 * (wc ** 2) / 1.55  # L_pi approximation
    phase = 2.0 * np.pi * lc / (beat_length + 1e-4)

    cols = {"L_c": lc, "W_c": wc, "P_t": pt}

    # Generate multi-wavelength transmission spectra (Tu and Td)
    wl_grid = np.linspace(1500.0, 1600.0, n_wavelengths)
    for i, wl in enumerate(wl_grid):
        dispersion = 1.0 + 0.05 * (wl - 1550.0) / 50.0
        coupling = 0.5 * (1.0 + np.sin(phase * dispersion))
        loss = np.exp(-0.001 * lc)  # propagation loss factor

        tu = np.clip(coupling * loss, 0.0, 0.5)
        td = np.clip((1.0 - coupling) * loss, 0.0, 0.5)

        cols[f"T_up_{int(wl)}"] = tu
        cols[f"T_down_{int(wl)}"] = td

    return pd.DataFrame(cols)
