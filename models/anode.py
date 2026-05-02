import jax
import jax.numpy as jnp
import jax.nn as jnn
import equinox as eqx
import diffrax

from models.sanode import xavier_uniform


class ANODEField(eqx.Module):
    """
    Autonomous (time-independent) single-hidden-layer vector field.

    f(y) = sum_j w_j * relu(a_j . y + c_j),
    where w_j in R^d, a_j in R^d, c_j in R.

    Args:
        data_size (int): Dimension of y (= d).
        width_size (int): Number of hidden units p.
        key: JAX PRNG key.
        scale (float): Xavier init scale.
        activation (callable): Activation function (default: relu).
    """

    A: jnp.ndarray   # (p, d)  — a_j rows
    c: jnp.ndarray   # (p,)    — biases c_j
    W: jnp.ndarray   # (d, p)  — output columns w_j
    activation: callable

    def __init__(self, data_size, width_size, *, key,
                 scale=0.1, activation=jnn.relu):
        kA, kW = jax.random.split(key)
        self.A = xavier_uniform(kA, (width_size, data_size), scale=scale)
        self.c = jnp.zeros((width_size,))
        self.W = xavier_uniform(kW, (data_size, width_size), scale=scale)
        self.activation = activation

    def __call__(self, t, y, args):
        # t accepted to satisfy diffrax ODETerm interface but ignored
        return self.W @ self.activation(self.A @ y + self.c)


class ANODE(eqx.Module):
    """
    Autonomous Neural ODE with a single hidden layer of width p, solved with
    Dopri5 (adaptive step).

    Total trainable parameters (data dim d): p*d + p + d*p = p*(2d+1) + 0
    (no output bias).

    Args:
        data_size (int): Input/output dimension d.
        width_size (int): Hidden width p.
        key: JAX PRNG key.
        activation (callable): Activation function (default: relu).
    """

    func: ANODEField

    def __init__(self, data_size, width_size=64, *, key,
                 activation=jnn.relu):
        self.func = ANODEField(data_size, width_size, key=key,
                               activation=activation)

    def __call__(self, ts, y0):
        """
        Solve ODE from y0 over time points ts.

        Args:
            ts (jnp.ndarray): Time points to save, shape (T,).
            y0 (jnp.ndarray): Initial condition, shape (data_size,).

        Returns:
            jnp.ndarray: Solution at each time in ts, shape (T, data_size).
        """
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func),
            solver=diffrax.Dopri5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=y0,
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-4),
            saveat=diffrax.SaveAt(ts=ts),
        )
        return solution.ys
