#!/usr/bin/env python3
"""
Example 02: CVAE Inverse Design and Surrogate-Guided Optimization
Generates candidate MMI geometries for a target 50:50 split spectrum.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from photonics_id.models.surrogate import PhysicsConstrainedMLP
from photonics_id.models.cvae import Spec2GeoCVAE, cvae_loss
from photonics_id.pipeline.dataset import generate_synthetic_mmi_dataset, DataBundle
from photonics_id.inverse_design.generator import (
    sample_inverse_candidates,
    rank_and_filter_candidates,
    gradient_refinement,
)
from photonics_id.utils.seed import set_global_seed


def main():
    set_global_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Dataset setup
    df = generate_synthetic_mmi_dataset(n_samples=500, n_wavelengths=10)
    feat_cols = ["L_c", "W_c", "P_t"]
    targ_cols = [c for c in df.columns if c.startswith("T_")]

    x_vals = df[feat_cols].values
    y_vals = df[targ_cols].values

    x_min, x_max = x_vals.min(axis=0), x_vals.max(axis=0)
    bundle = DataBundle(
        X_train=x_vals[:350],
        Y_train=y_vals[:350],
        X_val=x_vals[350:420],
        Y_val=y_vals[350:420],
        X_test=x_vals[420:],
        Y_test=y_vals[420:],
        x_min=x_min,
        x_max=x_max,
        x_rng=x_max - x_min,
        feature_cols=feat_cols,
        target_cols=targ_cols,
    )

    # 2. Instantiate CVAE and forward surrogate
    cvae = Spec2GeoCVAE(
        y_dim=len(targ_cols),
        x_dim=len(feat_cols),
        z_dim=8,
        width=256,
        depth=3,
    ).to(device)

    surrogate = PhysicsConstrainedMLP(
        in_dim=len(feat_cols),
        n_wavelengths=len(targ_cols) // 2,
        width=256,
        depth=3,
    ).to(device)

    # 3. Define target spectrum (e.g. ideal 50/50 splitter: Tu=0.48, Td=0.48 across 10 bands)
    n_wl = len(targ_cols) // 2
    target_spec = np.concatenate([np.full(n_wl, 0.48), np.full(n_wl, 0.48)])

    print(f"Inverse Designing for Target Split (50:50)...")
    # Sample candidate geometries from CVAE latent space
    candidates_unit = sample_inverse_candidates(cvae, target_spec, n_samples=32, device=device)
    print(f"Sampled {len(candidates_unit)} candidate geometries.")

    # Rank with surrogate
    best_phys, scores = rank_and_filter_candidates(
        surrogate, candidates_unit, bundle, top_k=3, device=device
    )

    print("\nTop 3 Candidate Geometries [L_c (um), W_c (um), P_t (um)]:")
    for rank, (geom, score) in enumerate(zip(best_phys, scores), 1):
        print(f"  #{rank}: L_c={geom[0]:.2f} um, W_c={geom[1]:.2f} um, P_t={geom[2]:.2f} um (Score: {score:.3f})")

    # Local gradient refinement starting from the best seed
    best_seed_unit = bundle.scale_X(best_phys[0:1])
    refined_unit = gradient_refinement(surrogate, best_seed_unit, target_split=0.5, steps=30, device=device)
    refined_phys = bundle.unscale_X(refined_unit)[0]

    print(f"\nAfter Gradient Refinement:")
    print(f"  Optimized Geometry: L_c={refined_phys[0]:.2f} um, W_c={refined_phys[1]:.2f} um, P_t={refined_phys[2]:.2f} um")


if __name__ == "__main__":
    main()
