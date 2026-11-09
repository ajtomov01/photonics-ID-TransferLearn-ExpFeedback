# Photonics Inverse Design with Sim-to-Real Transfer Learning & Experimental Feedback

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A deep learning framework for the **physics-constrained forward surrogate modeling**, **generative inverse design (CVAE)**, **sim-to-real domain adaptation**, and **experimental cutback validation** of silicon nitride ($\text{Si}_3\text{N}_4$) multimode interference (MMI) photonic devices.

---

## Overview

Designing integrated photonic components typically relies on computationally expensive finite-difference time-domain (FDTD) electromagnetic simulations and suffers from fabrication reality gaps (e.g., etching variations, wafer thickness non-uniformity).

This repository provides an end-to-end machine learning pipeline addressing both challenges:

1. **Physics-Constrained Neural Forward Surrogate**:
   Predicts multi-wavelength transmission spectra $[T_{\text{up}}(\lambda), T_{\text{down}}(\lambda)]$ from device geometry $[L_c, W_c, P_t]$ while strictly enforcing energy conservation ($\sum T \le 1.0$) via dynamic saturation clamps and non-negativity via softplus activations.
2. **Deep Generative Inverse Design (Spec2Geo CVAE)**:
   A Conditional Variational Autoencoder that samples high-performing geometric candidates directly conditioned on arbitrary target spectral responses (e.g., 50:50 or 75:25 power splitters).
3. **Surrogate-Guided Local Gradient Refinement**:
   Differentiable local optimization on the surrogate energy landscape to fine-tune CVAE proposals to sub-nanometer precision.
4. **Sim-to-Real Transfer Adaptation (`InputWarpAffineModel`)**:
   Few-shot domain adaptation that bridges the simulation-to-fabrication gap using parameterized input coordinate warping $\phi(x) = x + \delta(x)$ and output calibration heads, keeping base physics models frozen.
5. **Experimental Feedback & Cutback De-Embedding**:
   Optical loss regression to extract waveguide propagation loss (dB/cm) and fiber-to-chip grating coupler insertion loss from experimental chip measurements.

---

## Pipeline Architecture

```text
                  [ Target Optical Spectrum Y ]
                                |
                                v
                   +-------------------------+
                   |  Spec2Geo CVAE Decoder  |  <-- Latent Sample z ~ N(0, I)
                   +-------------------------+
                                |
                                v
               [ Candidate Geometries X_seed ]
                                |
                                v
       +-------------------------------------------------+
       |  Physics-Constrained Forward Surrogate Ensemble |
       |     - Non-negative Softplus Output              |
       |     - Conservation of Energy Clamp (Tu+Td <= 1) |
       |     - Epistemic Uncertainty Quantification     |
       +-------------------------------------------------+
                                |
                    Gradient Refinement (dx/dLoss)
                                |
                                v
                 [ Optimized Device Geometries ]
                                |
                                v
                     (Chip Fabrication Run)
                                |
                                v
                [ Experimental Measurements / Loss ]
                                |
                                v
       +-------------------------------------------------+
       |           InputWarpAffine Transfer Model        |
       |  y_real = a(x) * f_sim(x + delta(x)) + b(x)    |
       +-------------------------------------------------+
```

---

## Model Weights & Confidentiality Notice

> **Model Framework vs. Production Weights**:
> This repository provides the **complete mathematical model framework, network architectures, physics loss formulations, and inverse design algorithms**.
>
> In accordance with institutional intellectual property and ongoing publication requirements, **full pre-trained production model weights are kept proprietary** and excluded from this repository.
>
> All examples and test suites include **lightweight synthetic data generators and modular initialization routines**, allowing users and reviewers to train, test, and inspect the entire pipeline immediately out of the box.

---

## Repository Structure

```text
photonics-ID-TransferLearn-ExpFeedback/
├── README.md                      # Documentation & project overview
├── LICENSE                        # MIT License
├── pyproject.toml                 # Package setup and build configuration
├── requirements.txt               # Pinned dependencies
├── .gitignore                     # Git ignore rules (weights, large FDTD caches)
│
├── configs/                       # Pipeline hyperparameter configurations
│   └── default_pipeline.yaml
│
├── src/
│   └── photonics_id/
│       ├── models/
│       │   ├── surrogate.py       # PhysicsConstrainedMLP, ensemble, energy penalty
│       │   ├── cvae.py            # Spec2GeoCVAE architecture & loss
│       │   └── adapters.py        # InputWarpAffineModel, ResidualAdapter
│       ├── pipeline/
│       │   ├── dataset.py         # DataBundle, scaling, loaders, synthetic generator
│       │   └── cutback.py         # Optical cutback loss analysis
│       ├── inverse_design/
│       │   ├── generator.py       # CVAE candidate generator & gradient refinement
│       │   └── active_learning.py # Active learning selection routines
│       └── utils/
│           ├── seed.py            # Reproducibility seed utilities
│           └── metrics.py         # Spectral MAE, throughput, and split balance metrics
│
├── examples/
│   ├── 01_forward_surrogate.py    # Train and evaluate forward physics surrogate
│   ├── 02_cvae_inverse_design.py   # Inverse design for target 50:50 splitter
│   └── 03_transfer_adaptation.py  # Sim-to-real transfer learning demo
│
└── tests/
    └── test_models.py             # Unit tests for models, loss, and adapters
```

---

## Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/ajtomov01/photonics-ID-TransferLearn-ExpFeedback.git
cd photonics-ID-TransferLearn-ExpFeedback
pip install -e .
```

To install development dependencies (for testing):

```bash
pip install -e ".[dev]"
```

---

## Quickstart

### 1. Train Forward Physics Surrogate

```python
from photonics_id.models.surrogate import PhysicsConstrainedMLP, physics_penalty_loss
import torch

# 3 geometry parameters -> 10 wavelength channels for Tu and Td (total 20 outputs)
model = PhysicsConstrainedMLP(in_dim=3, n_wavelengths=10, width=512, depth=5)
x = torch.rand(16, 3) # normalized [L_c, W_c, P_t] in unit cube

pred, s_raw = model(x, return_aux=True)
# Conservation of energy strictly enforced: pred[:, :10] + pred[:, 10:] <= 1.0
```

Run the standalone script:
```bash
python examples/01_forward_surrogate.py
```

### 2. Inverse Design a 50:50 Splitter with CVAE

```python
from photonics_id.models.cvae import Spec2GeoCVAE
from photonics_id.inverse_design.generator import sample_inverse_candidates
import numpy as np

cvae = Spec2GeoCVAE(y_dim=20, x_dim=3, z_dim=16)

# Condition on 50:50 transmission across all wavelengths
target_spec = np.full(20, 0.48)
candidates = sample_inverse_candidates(cvae, target_spec, n_samples=64)
```

Run the standalone script:
```bash
python examples/02_cvae_inverse_design.py
```

### 3. Sim-to-Real Domain Adaptation

```python
from photonics_id.models.adapters import InputWarpAffineModel

# Wrap pre-trained simulation base model with adaptive warping
transfer_model = InputWarpAffineModel(
    base_model=model,
    in_dim=3,
    out_dim=20,
    warp_scale=0.05
)
# Only warp and calibration heads are trainable; base model remains frozen.
```

Run the standalone script:
```bash
python examples/03_transfer_adaptation.py
```

---

## Testing

Run unit tests to verify the integrity of the network architectures, loss calculations, and physics bounds:

```bash
pytest tests/
```

---

## Citation

If you use this codebase or model architecture in your research, please cite:

```bibtex
@article{tomov2026photonics,
  title={Physics-Constrained Inverse Design and Sim-to-Real Domain Adaptation for Silicon Nitride MMI Couplers},
  author={Tomov, Aiden and Collaborators},
  journal={arXiv preprint},
  year={2026}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).
