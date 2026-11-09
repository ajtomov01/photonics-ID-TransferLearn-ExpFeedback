import pytest
import torch
from photonics_id.models.surrogate import PhysicsConstrainedMLP, physics_penalty_loss
from photonics_id.models.cvae import Spec2GeoCVAE, cvae_loss
from photonics_id.models.adapters import InputWarpAffineModel, ResidualAdapter
from photonics_id.pipeline.dataset import generate_synthetic_mmi_dataset
from photonics_id.pipeline.cutback import cutback_linear_regression


def test_physics_constrained_mlp():
    in_dim = 3
    n_wl = 10
    model = PhysicsConstrainedMLP(in_dim=in_dim, n_wavelengths=n_wl, width=64, depth=2)
    x = torch.rand(8, in_dim)
    out, s_raw = model(x, return_aux=True)

    # Check output shape
    assert out.shape == (8, 2 * n_wl)
    # Check physics bounds: Tu >= 0, Td >= 0
    assert torch.all(out >= 0.0)
    # Check conservation of energy: Tu + Td <= 1.0 (with small numerical tolerance)
    tu = out[:, :n_wl]
    td = out[:, n_wl:]
    assert torch.all(tu + td <= 1.0001)

    # Test penalty
    loss = physics_penalty_loss(s_raw)
    assert loss.ndim == 0


def test_spec2geo_cvae():
    y_dim = 20
    x_dim = 3
    z_dim = 8
    model = Spec2GeoCVAE(y_dim=y_dim, x_dim=x_dim, z_dim=z_dim, width=64, depth=2)
    y = torch.rand(4, y_dim)
    x = torch.rand(4, x_dim)

    x_hat, mu, logvar = model(y, x)
    assert x_hat.shape == (4, x_dim)
    assert mu.shape == (4, z_dim)
    assert logvar.shape == (4, z_dim)

    # Check unit cube bound [0, 1]
    assert torch.all((x_hat >= 0.0) & (x_hat <= 1.0))

    total, recon, kl = cvae_loss(x_hat, x, mu, logvar, beta=1e-3)
    assert total.ndim == 0
    assert recon.ndim == 0
    assert kl.ndim == 0


def test_adapters():
    base_model = PhysicsConstrainedMLP(in_dim=3, n_wavelengths=5, width=32, depth=2)
    x = torch.rand(4, 3)

    # InputWarpAffineModel
    warp_model = InputWarpAffineModel(base_model, in_dim=3, out_dim=10)
    out_warp = warp_model(x)
    assert out_warp.shape == (4, 10)

    # ResidualAdapter
    res_model = ResidualAdapter(base_model, in_dim=3, out_dim=10, hidden_dim=32)
    out_res = res_model(x)
    assert out_res.shape == (4, 10)


def test_synthetic_data_and_cutback():
    df = generate_synthetic_mmi_dataset(n_samples=50, n_wavelengths=5)
    assert len(df) == 50
    assert "L_c" in df.columns
    assert "T_up_1550" in df.columns

    # Test cutback regression
    lengths = [0.5, 1.0, 2.0, 4.0]
    powers = [-10.0, -11.0, -13.0, -17.0]  # -2 dB/cm slope, -9 dBm intercept
    loss, coupling, r2 = cutback_linear_regression(lengths, powers)
    assert loss > 0.0
    assert r2 > 0.95
