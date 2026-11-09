from __future__ import annotations

from typing import Tuple
import numpy as np


def cutback_linear_regression(
    lengths_cm: np.ndarray,
    transmitted_power_dbm: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Perform optical cutback regression on fabricated spiral/waveguide structures.

    Parameters:
      lengths_cm: Array of waveguide lengths in centimeters.
      transmitted_power_dbm: Array of measured output powers in dBm.

    Returns:
      (loss_db_per_cm, coupling_loss_db, r_squared):
        - loss_db_per_cm: Waveguide propagation loss (positive dB/cm).
        - coupling_loss_db: Estimated fiber-to-chip insertion loss per facet / grating pair (dB).
        - r_squared: Linear fit goodness of fit (R^2).
    """
    x = np.asarray(lengths_cm, dtype=np.float64)
    y = np.asarray(transmitted_power_dbm, dtype=np.float64)

    # Linear fit: y = slope * x + intercept
    slope, intercept = np.polyfit(x, y, 1)

    y_pred = slope * x + intercept
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    ss_res = np.sum((y - y_pred) ** 2)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-12))

    # In photonics cutback, power drops with length (negative slope).
    loss_db_per_cm = -slope
    coupling_loss_db = -intercept  # total input/output baseline insertion loss

    return float(loss_db_per_cm), float(coupling_loss_db), float(r2)
