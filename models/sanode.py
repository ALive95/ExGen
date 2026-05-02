import jax
import jax.numpy as jnp
import jax.nn as jnn
import equinox as eqx
import diffrax


def custom_activation(x):
    """sin²(x) + x activation — kept available but not used by default."""
    return jnp.sin(x) ** 2 + x


def xavier_uniform(key, shape, scale=1.0):
    """
    Xavier uniform initialization with an optional scaling factor.

    Args:
        key: JAX PRNG key.
        shape (tuple): Weight matrix shape (out, in).
        scale (float): Multiplier on the standard Xavier limit.

    Returns:
        jnp.ndarray: Initialized weight matrix.
    """
    fan_in = shape[0] + shape[1]
    limit = scale * jnp.sqrt(6 / fan_in)
    return jax.random.uniform(key, shape, minval=-limit, maxval=limit)


class Linear(eqx.Module):
    """Linear layer with Xavier-uniform weight init and zero bias."""

    weight: jnp.ndarray
    bias: jnp.ndarray

    def __init__(self, in_features: int, out_features: int, *, key, scale=0.1):
        self.weight = xavier_uniform(key, (out_features, in_features), scale=scale)
        self.bias = jnp.zeros((out_features,))

    def __call__(self, x):
        return self.weight @ x + self.bias


class MLP(eqx.Module):
    """
    Multi-layer perceptron with configurable depth and activation.

    Args:
        in_size (int): Input dimension.
        out_size (int): Output dimension.
        width_size (int): Hidden layer width.
        depth (int): Number of hidden layers (depth=1 → one hidden layer).
        key: JAX PRNG key.
        scale (float): Xavier init scale.
        activation (callable): Activation function (default: relu).
    """

    layers: list
    activation: callable

    def __init__(self, in_size, out_size, width_size, depth, *, key,
                 scale=0.1, activation=jnn.relu):
        keys = jax.random.split(key, depth + 1)
        self.activation = activation
        self.layers = [Linear(in_size, width_size, key=keys[0], scale=scale)]
        for i in range(depth - 1):
            self.layers.append(Linear(width_size, width_size, key=keys[i + 1], scale=scale))
        self.layers.append(Linear(width_size, out_size, key=keys[-1], scale=scale))

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)


class SANODEField(eqx.Module):
    """
    Time-dependent vector field f(t, y) for SANODE.
    Appends t as an extra input feature before the MLP.

    Args:
        data_size (int): Dimension of y.
        pad_size (int): Extra padding dimensions appended to y.
        width_size (int): MLP hidden width.
        depth (int): MLP depth.
        key: JAX PRNG key.
        scale (float): Xavier init scale.
        activation (callable): Activation function (default: relu).
    """

    mlp: MLP

    def __init__(self, data_size, pad_size, width_size, depth=1, *, key,
                 scale=0.1, activation=jnn.relu):
        self.mlp = MLP(
            in_size=data_size + pad_size + 1,  # +1 for time input
            out_size=data_size + pad_size,
            width_size=width_size,
            depth=depth,
            key=key,
            scale=scale,
            activation=activation,
        )

    def __call__(self, t, y, args):
        return self.mlp(jnp.concatenate([y, jnp.array([t])], axis=0))


class SANODE(eqx.Module):
    """
    Supervised Autonomous Neural ODE solved with Dopri5 (adaptive step).

    Args:
        data_size (int): Input/output dimension.
        pad_size (int): Extra padding dimensions (default: 0).
        width_size (int): Hidden width of the vector field MLP.
        depth (int): Depth of the vector field MLP (default: 1).
        key: JAX PRNG key.
        activation (callable): Activation function (default: relu).
    """

    func: SANODEField

    def __init__(self, data_size, pad_size=0, width_size=64, depth=1, *, key,
                 activation=jnn.relu):
        self.func = SANODEField(data_size, pad_size, width_size, depth,
                                key=key, activation=activation)

    def __call__(self, ts, y0):
        """
        Solve ODE from y0 over time points ts.

        Args:
            ts (jnp.ndarray): Time points to save, shape (T,). Must include t0 and t1.
            y0 (jnp.ndarray): Initial condition, shape (data_size + pad_size,).

        Returns:
            jnp.ndarray: Solution at each time in ts, shape (T, data_size + pad_size).
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
