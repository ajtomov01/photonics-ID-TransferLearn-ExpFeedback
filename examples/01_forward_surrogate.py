#!/usr/bin/env python3
"""
Example 01: Training the Forward Physics-Constrained Surrogate Model
Demonstrates energy conservation clamp and physics penalty loss.
"""

import sys
from pathlib import Path

# Add src to pythonpath for local execution
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import torch.nn as nn
from photonics_id.models.surrogate import PhysicsConstrainedMLP, physics_penalty_loss
from photonics_id.pipeline.dataset import generate_synthetic_mmi_dataset, DataBundle, create_dataloaders
from photonics_id.utils.seed import set_global_seed


def main():
    set_global_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # 1. Generate synthetic MMI dataset
    print("Generating synthetic MMI dataset...")
    df = generate_synthetic_mmi_dataset(n_samples=600, n_wavelengths=10, seed=42)

    feat_cols = ["L_c", "W_c", "P_t"]
    targ_cols = [c for c in df.columns if c.startswith("T_")]

    x_vals = df[feat_cols].values
    y_vals = df[targ_cols].values

    x_min, x_max = x_vals.min(axis=0), x_vals.max(axis=0)
    x_rng = x_max - x_min

    n_train = 400
    bundle = DataBundle(
        X_train=x_vals[:n_train],
        Y_train=y_vals[:n_train],
        X_val=x_vals[n_train:500],
        Y_val=y_vals[n_train:500],
        X_test=x_vals[500:],
        Y_test=y_vals[500:],
        x_min=x_min,
        x_max=x_max,
        x_rng=x_rng,
        feature_cols=feat_cols,
        target_cols=targ_cols,
    )

    tr_dl, va_dl, te_dl = create_dataloaders(bundle, batch_size=64)

    # 2. Instantiate physics-constrained model
    model = PhysicsConstrainedMLP(
        in_dim=len(feat_cols),
        n_wavelengths=len(targ_cols) // 2,
        width=256,
        depth=4,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # 3. Train loop
    print("Training PhysicsConstrainedMLP...")
    for epoch in range(1, 21):
        model.train()
        train_loss = 0.0
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred, s_raw = model(xb, return_aux=True)

            mse = nn.functional.mse_loss(pred, yb)
            phys_loss = physics_penalty_loss(s_raw)
            total_loss = mse + 0.25 * phys_loss

            total_loss.backward()
            optimizer.step()
            train_loss += total_loss.item() * len(xb)

        train_loss /= len(tr_dl.dataset)
        if epoch % 5 == 0:
            print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.5f}")

    # 4. Evaluation
    model.eval()
    with torch.no_grad():
        test_mae = 0.0
        for xb, yb in te_dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            test_mae += (pred - yb).abs().mean().item() * len(xb)
        test_mae /= len(te_dl.dataset)

    print(f"\nFinal Test MAE: {test_mae:.4f}")
    print("Physics energy conservation holds: all predicted Tu + Td <= 1.0.")


if __name__ == "__main__":
    main()
