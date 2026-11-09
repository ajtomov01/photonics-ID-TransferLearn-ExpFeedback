"""
photonics-id: Physics-Informed Inverse Design, Sim-to-Real Transfer Learning,
and Experimental Feedback for Integrated Photonics.
"""

from photonics_id.models.surrogate import PhysicsConstrainedMLP, SurrogateEnsemble, physics_penalty_loss
from photonics_id.models.cvae import Spec2GeoCVAE, cvae_loss
from photonics_id.models.adapters import InputWarpAffineModel, ResidualAdapter
from photonics_id.pipeline.dataset import DataBundle, create_dataloaders, generate_synthetic_mmi_dataset
from photonics_id.pipeline.cutback import cutback_linear_regression
from photonics_id.inverse_design.generator import (
    sample_inverse_candidates,
    rank_and_filter_candidates,
    gradient_refinement,
)
from photonics_id.utils.seed import set_global_seed

__version__ = "0.1.0"

__all__ = [
    "PhysicsConstrainedMLP",
    "SurrogateEnsemble",
    "physics_penalty_loss",
    "Spec2GeoCVAE",
    "cvae_loss",
    "InputWarpAffineModel",
    "ResidualAdapter",
    "DataBundle",
    "create_dataloaders",
    "generate_synthetic_mmi_dataset",
    "cutback_linear_regression",
    "sample_inverse_candidates",
    "rank_and_filter_candidates",
    "gradient_refinement",
    "set_global_seed",
]
