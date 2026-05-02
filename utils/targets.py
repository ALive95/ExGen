import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# JAX target functions  (used for training / evaluation)
# ---------------------------------------------------------------------------

def target_general_d(x):
    """
    Smooth blockwise sin/cos target for arbitrary even d; passes through last
    coordinate when d is odd.

    Args:
        x (jnp.ndarray): Input, shape (d,).

    Returns:
        jnp.ndarray: Output, shape (d,).
    """
    d = x.shape[0]
    out = []
    for j in range(0, d - 1, 2):
        out.append(jnp.sin(x[j]) * jnp.cos(x[j + 1]))
        out.append(jnp.cos(x[j]) * jnp.sin(x[j + 1]))
    if d % 2 == 1:
        out.append(x[d - 1])
    return jnp.stack(out)


def target_holder_d(x):
    """Hölder-1/2 target for arbitrary d."""
    return jnp.sign(x) * jnp.abs(x) ** 0.5


# ---------------------------------------------------------------------------
# NumPy versions  (used for estimator computations — no JAX tracing)
# ---------------------------------------------------------------------------

def target_general_d_np(x):
    """NumPy version of target_general_d."""
    d = len(x)
    out = []
    for j in range(0, d - 1, 2):
        out.append(np.sin(x[j]) * np.cos(x[j + 1]))
        out.append(np.cos(x[j]) * np.sin(x[j + 1]))
    if d % 2 == 1:
        out.append(x[d - 1])
    return np.array(out)


def target_holder_d_np(x):
    """NumPy version of target_holder_d."""
    return np.sign(x) * np.abs(x) ** 0.5