#!/usr/bin/env python3
"""
Example 03: Sim-to-Real Transfer Adaptation
Demonstrates few-shot domain adaptation (e.g., wafer thickness variation)
using InputWarpAffineModel with a frozen base simulation model.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import torch.nn as nn
from photonics_id.models.surrogate import PhysicsConstrainedMLP
from photonics_id.models.adapters import InputWarpAffineModel
from photonics_id.utils.seed import set_global_seed


def main():
    set_global_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    in_dim = 3
    n_wl = 10
    out_dim = 2 * n_wl

    # 1. Base simulation model (representing trained FDTD model)
    base_model = PhysicsConstrainedMLP(in_dim=in_dim, n_wavelengths=n_wl, width=128, depth=3).to(device)
    base_model.eval()

    # 2. Wrap with InputWarpAffineModel for few-shot adaptation
    adapted_model = InputWarpAffineModel(
        base_model=base_model,
        in_dim=in_dim,
        out_dim=out_dim,
        warp_scale=0.05,
    ).to(device)

    # 3. Simulate a handful of experimental fabricated chip calibration measurements (few-shot)
    n_exp = 16
    x_exp = torch.rand(n_exp, in_dim, device=device)
    # Target domain has systematic spectral shift and attenuation from fabrication
    with torch.no_grad():
        y_sim = base_model(x_exp)
        y_exp = y_sim * 0.92 + 0.02 * torch.sin(x_exp[:, :1])  # fabrication shift

    optimizer = torch.optim.Adam(adapted_model.parameters(), lr=5e-3)

    print(f"Transfer Learning with {n_exp} experimental calibration samples...")
    for step in range(1, 51):
        optimizer.zero_grad()
        y_pred = adapted_model(x_exp)
        loss = nn.functional.mse_loss(y_pred, y_exp)
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(f"  Step {step:02d} | Transfer MSE: {loss.item():.6f}")

    print("\nTransfer adaptation completed successfully.")
    print("Base model weights remained frozen; calibration & warp heads adapted to fabrication.")


if __name__ == "__main__":
    main()
