from __future__ import annotations

from interpax import interp1d
from jax.numpy import interp


def interpolate_data(x, x_data, y_data, cubic=False):
    """Cubic or linear interpolation function with JAX."""
    if cubic:
        return interp1d(x, x_data, y_data, method="cubic2")
    return interp(x, x_data, y_data)

