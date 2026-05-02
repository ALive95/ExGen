import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_optimizer_and_state(model, lr):
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    return optimizer, opt_state


def _make_node_step_fn(model, optimizer, ts):
    """Step function for models taking (ts, x0) — SANODE, ANODE, TwoANODE."""
    @eqx.filter_jit
    def step_fn(model, opt_state, x_batch, y_batch):
        def loss_fn(m):
            y_pred = jax.vmap(lambda x0: m(ts, x0)[-1])(x_batch)
            return jnp.mean((y_pred - y_batch) ** 2)
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        return loss, eqx.apply_updates(model, updates), new_opt_state
    return step_fn


def _training_loop(model, optimizer, opt_state, step_fn,
                   x_train, y_train, num_steps, loader_key, batch_size,
                   patience, min_delta, loss_threshold, print_every):
    """
    Generic training loop with minibatching and early stopping.

    Returns:
        tuple: (model, loss_history) where loss_history is list of (step, loss).
    """
    N = x_train.shape[0]
    rng = loader_key
    best_loss = float("inf")
    no_improve = 0
    loss_val = float("inf")
    loss_history = []

    for step in range(num_steps):
        rng, subkey = jr.split(rng)
        idx = jr.choice(subkey, N, (batch_size,), replace=False)
        loss_val, model, opt_state = step_fn(model, opt_state, x_train[idx], y_train[idx])
        loss_val = float(loss_val)

        if print_every > 0 and (step % print_every == 0 or step == num_steps - 1):
            print(f"    step {step:>6}/{num_steps}  loss = {loss_val:.3e}")
            loss_history.append((step, loss_val))

        if loss_val < loss_threshold:
            print(f"    converged at step {step}: loss = {loss_val:.3e}")
            break
        if loss_val < best_loss - min_delta:
            best_loss = loss_val
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            print(f"    early stop at step {step}: no improvement for {patience} steps")
            break

    return model, loss_history


# ---------------------------------------------------------------------------
# SANODE training
# ---------------------------------------------------------------------------

def train_sanode(N, P, target_fn, num_steps=10000, seed=42, lr=1e-4,
                 data_size=2, pad_size=0, depth=1, T=1.0, batch_size=32,
                 patience=5000, min_delta=1e-8, loss_threshold=1e-6,
                 print_every=500):
    """
    Train a SANODE model on N samples drawn from target_fn.

    Args:
        N (int): Number of training points.
        P (int): Hidden width.
        target_fn (callable): JAX-compatible target x -> y.
        num_steps (int): Maximum gradient steps.
        seed (int): PRNG seed.
        lr (float): Adam learning rate.
        data_size (int): Input/output dimension.
        pad_size (int): Padding dimensions appended to state.
        depth (int): MLP depth.
        T (float): Integration horizon.
        batch_size (int): Minibatch size.
        patience (int): Early-stop patience.
        min_delta (float): Minimum improvement to reset patience.
        loss_threshold (float): Convergence threshold.
        print_every (int): Log interval (0 to silence).

    Returns:
        tuple: (model, loss_history).
    """
    from models.sanode import SANODE

    key = jr.PRNGKey(seed)
    data_key, model_key, loader_key = jr.split(key, 3)

    x_train = jr.uniform(data_key, (N, data_size), minval=-2.0, maxval=2.0)
    y_train = jax.vmap(target_fn)(x_train)

    model = SANODE(data_size, pad_size, P, depth, key=model_key)
    ts = jnp.array([0.0, T])

    optimizer, opt_state = _make_optimizer_and_state(model, lr)
    step_fn = _make_node_step_fn(model, optimizer, ts)

    print(f"  training SANODE: N={N}, P={P}, batch_size={batch_size}")
    return _training_loop(model, optimizer, opt_state, step_fn,
                          x_train, y_train, num_steps, loader_key, batch_size,
                          patience, min_delta, loss_threshold, print_every)


# ---------------------------------------------------------------------------
# ANODE training (single hidden layer)
# ---------------------------------------------------------------------------

def train_anode(N, P, target_fn, num_steps=10000, seed=42, lr=1e-3,
                data_size=2, T=1.0, batch_size=32,
                patience=1000, min_delta=1e-8, loss_threshold=1e-6,
                print_every=500):
    """
    Train an ANODE model on N samples drawn from target_fn.

    Args:
        N (int): Number of training points.
        P (int): Hidden width.
        target_fn (callable): JAX-compatible target x -> y.
        num_steps (int): Maximum gradient steps.
        seed (int): PRNG seed.
        lr (float): Adam learning rate.
        data_size (int): Input/output dimension.
        T (float): Integration horizon.
        batch_size (int): Minibatch size.
        patience (int): Early-stop patience.
        min_delta (float): Minimum improvement to reset patience.
        loss_threshold (float): Convergence threshold.
        print_every (int): Log interval (0 to silence).

    Returns:
        tuple: (model, loss_history).
    """
    from models.anode import ANODE

    key = jr.PRNGKey(seed)
    data_key, model_key, loader_key = jr.split(key, 3)

    x_train = jr.uniform(data_key, (N, data_size), minval=-2.0, maxval=2.0)
    y_train = jax.vmap(target_fn)(x_train)

    model = ANODE(data_size, P, key=model_key)
    ts = jnp.array([0.0, T])

    optimizer, opt_state = _make_optimizer_and_state(model, lr)
    step_fn = _make_node_step_fn(model, optimizer, ts)

    print(f"  training ANODE: N={N}, P={P}, batch_size={batch_size}")
    return _training_loop(model, optimizer, opt_state, step_fn,
                          x_train, y_train, num_steps, loader_key, batch_size,
                          patience, min_delta, loss_threshold, print_every)


# ---------------------------------------------------------------------------
# TwoANODE training (two hidden layers)
# ---------------------------------------------------------------------------

def train_twoanode(N, p, q, target_fn, num_steps=10000, seed=42, lr=1e-3,
                   data_size=2, T=1.0, batch_size=32,
                   patience=1000, min_delta=1e-8, loss_threshold=1e-6,
                   print_every=500):
    """
    Train a TwoANODE model on N samples drawn from target_fn.

    Args:
        N (int): Number of training points.
        p (int): Inner width per outer unit.
        q (int): Number of outer units.
        target_fn (callable): JAX-compatible target x -> y.
        num_steps (int): Maximum gradient steps.
        seed (int): PRNG seed.
        lr (float): Adam learning rate.
        data_size (int): Input/output dimension.
        T (float): Integration horizon.
        batch_size (int): Minibatch size.
        patience (int): Early-stop patience.
        min_delta (float): Minimum improvement to reset patience.
        loss_threshold (float): Convergence threshold.
        print_every (int): Log interval (0 to silence).

    Returns:
        tuple: (model, loss_history).
    """
    from models.twoanode import TwoANODE

    key = jr.PRNGKey(seed)
    data_key, model_key, loader_key = jr.split(key, 3)

    x_train = jr.uniform(data_key, (N, data_size), minval=-2.0, maxval=2.0)
    y_train = jax.vmap(target_fn)(x_train)

    model = TwoANODE(data_size, p, q, key=model_key)
    ts = jnp.array([0.0, T])

    optimizer, opt_state = _make_optimizer_and_state(model, lr)
    step_fn = _make_node_step_fn(model, optimizer, ts)

    print(f"  training TwoANODE: N={N}, p={p}, q={q}, batch_size={batch_size}")
    return _training_loop(model, optimizer, opt_state, step_fn,
                          x_train, y_train, num_steps, loader_key, batch_size,
                          patience, min_delta, loss_threshold, print_every)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_sanode(model, target_fn, data_size=2, T=1.0, N_test=1000, seed=999):
    """Compute test MSE for a trained SANODE."""
    key = jr.PRNGKey(seed)
    x_test = jr.uniform(key, (N_test, data_size), minval=-2.0, maxval=2.0)
    y_test = jax.vmap(target_fn)(x_test)
    ts = jnp.array([0.0, T])
    y_pred = jax.vmap(lambda x0: model(ts, x0)[-1])(x_test)
    return float(jnp.mean((y_pred - y_test) ** 2))


def evaluate_anode(model, target_fn, data_size=2, T=1.0, N_test=1000, seed=999):
    """Compute test MSE for a trained ANODE."""
    key = jr.PRNGKey(seed)
    x_test = jr.uniform(key, (N_test, data_size), minval=-2.0, maxval=2.0)
    y_test = jax.vmap(target_fn)(x_test)
    ts = jnp.array([0.0, T])
    y_pred = jax.vmap(lambda x0: model(ts, x0)[-1])(x_test)
    return float(jnp.mean((y_pred - y_test) ** 2))


def evaluate_twoanode(model, target_fn, data_size=2, T=1.0, N_test=1000, seed=999):
    """Compute test MSE for a trained TwoANODE."""
    key = jr.PRNGKey(seed)
    x_test = jr.uniform(key, (N_test, data_size), minval=-2.0, maxval=2.0)
    y_test = jax.vmap(target_fn)(x_test)
    ts = jnp.array([0.0, T])
    y_pred = jax.vmap(lambda x0: model(ts, x0)[-1])(x_test)
    return float(jnp.mean((y_pred - y_test) ** 2))
