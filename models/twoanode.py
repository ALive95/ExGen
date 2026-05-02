import jax
import jax.numpy as jnp
import jax.nn as jnn
import equinox as eqx
import diffrax

from models.sanode import xavier_uniform


class TwoANODEField(eqx.Module):
    """
    Autonomous two-hidden-layer vector field (2aNODE) of Theorem 2.6.

    f(y) = sum_{i=1}^q w_i * relu(sum_{j=1}^p a_ij * relu(b_ij . y + c_ij)),

    with q outer units, each containing an independent inner block of width p.
    There is no parameter sharing across outer units (block-diagonal first layer).

    Shapes:
        b: (q, p, d)   inner weights
        c: (q, p)      inner biases
        a: (q, p)      inner-to-outer weights
        w: (q, d)      outer-to-output weights

    Total params: q*p*d + q*p + q*p + q*d = q*p*(d+2) + q*d.

    Args:
        data_size (int): Dimension d of the state.
        p (int): Inner width (per outer unit).
        q (int): Number of outer units.
        key: JAX PRNG key.
        scale (float): Xavier init scale.
        activation (callable): Activation function (default: relu).
    """

    b: jnp.ndarray   # (q, p, d)
    c: jnp.ndarray   # (q, p)
    a: jnp.ndarray   # (q, p)
    w: jnp.ndarray   # (q, d)
    activation: callable

    def __init__(self, data_size, p, q, *, key,
                 scale=0.1, activation=jnn.relu):
        kb, ka, kw = jax.random.split(key, 3)
        self.b = xavier_uniform(kb, (q, p, data_size), scale=scale)
        self.c = jnp.zeros((q, p))
        self.a = xavier_uniform(ka, (q, p), scale=scale)
        self.w = xavier_uniform(kw, (q, data_size), scale=scale)
        self.activation = activation

    def __call__(self, t, y, args):
        # Inner pre-activations: (q, p) = (q, p, d) . (d,)
        h = jnp.einsum("qpd,d->qp", self.b, y) + self.c
        # Inner activations
        h = self.activation(h)
        # Inner-to-outer scalar combination: (q,) = sum_p (q, p) * (q, p)
        s = jnp.sum(self.a * h, axis=1)
        # Outer activation
        s = self.activation(s)
        # Output: (d,) = sum_i w_i * s_i, with w of shape (q, d)
        return self.w.T @ s


class TwoANODE(eqx.Module):
    """
    Autonomous two-layer Neural ODE (2aNODE), solved with Dopri5.

    Args:
        data_size (int): Input/output dimension d.
        p (int): Inner width per outer unit.
        q (int): Number of outer units.
        key: JAX PRNG key.
        activation (callable): Activation function (default: relu).
    """

    func: TwoANODEField

    def __init__(self, data_size, p=10, q=10, *, key,
                 activation=jnn.relu):
        self.func = TwoANODEField(data_size, p, q, key=key,
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
