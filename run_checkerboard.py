"""
run_checkerboard.py

Tests cell-wise controllability on a K x K checkerboard sorting task across
three architectures: saNODE (time-dependent), aNODE (autonomous, single layer),
and 2aNODE (autonomous, two layers). All three are parameter-matched.

Parameter counts (single hidden layer where applicable, d=2):
  saNODE  : (2d+2)*p + d                       = 6p + 2
  aNODE   : 2pd + p                            = 5p
  2aNODE  : qp(d+2) + qd, with p=q             = 4p^2 + 2p
"""

import math
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
import numpy as np

from models.sanode import SANODE
from models.anode import ANODE
from models.twoanode import TwoANODE
from utils.plotting import plot_checkerboard_sweep, plot_checkerboard_viz


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

D         = 2
R         = 1.0
S         = 0.8
K_VALUES  = [2, 3, 4]
P_SA      = 512
N_TRAIN   = 400
N_TEST    = 1000
N_SEEDS   = 3
NUM_STEPS = 30_000
PATIENCE  = 8000
MIN_DELTA = 1e-8
LR        = 5e-4
LOSS_THR  = 1e-6


# ---------------------------------------------------------------------------
# Parameter counts and matching
# ---------------------------------------------------------------------------

def param_count_sanode(p, d=D):
    """Single-layer SANODE parameter count: (2d+2)*p + d."""
    return (2 * d + 2) * p + d


def param_count_anode(p, d=D):
    """Single-layer ANODE parameter count (no output bias): 2*p*d + p."""
    return 2 * p * d + p


def param_count_twoanode(p, q, d=D):
    """TwoANODE parameter count: q*p*(d+2) + q*d."""
    return q * p * (d + 2) + q * d


def matched_p_anode(p_sa, d=D):
    """ANODE width matching SANODE total parameter count."""
    target = param_count_sanode(p_sa, d)
    # Solve 2*p*d + p = target  ->  p = target / (2d + 1)
    return math.ceil(target / (2 * d + 1))


def matched_p_twoanode(p_sa, d=D):
    """TwoANODE width (p = q) matching SANODE total parameter count.

    Solve (d+2)*p^2 + d*p = target  for p.
    """
    target = param_count_sanode(p_sa, d)
    a, b, c = (d + 2), d, -target
    disc = b * b - 4 * a * c
    return max(1, math.ceil((-b + math.sqrt(disc)) / (2 * a)))


# ---------------------------------------------------------------------------
# Checkerboard target
# ---------------------------------------------------------------------------

def cell_label(x, K, r=R):
    """Checkerboard cell label (0 or 1) for a single point x in R^2."""
    i = jnp.floor((x[0] + r) / (2 * r / K)).astype(jnp.int32)
    j = jnp.floor((x[1] + r) / (2 * r / K)).astype(jnp.int32)
    return (i + j) % 2


def checkerboard_target(x, K, s=S, r=R):
    """Map x to (+s,+s) or (-s,-s) based on checkerboard cell parity."""
    sign = jnp.where(cell_label(x, K, r) == 0, 1.0, -1.0)
    return jnp.array([sign * s, sign * s])


def cell_label_np(x, K, r=R):
    """NumPy version of cell_label for visualization."""
    i = min(int((x[0] + r) / (2 * r / K)), K - 1)
    j = min(int((x[1] + r) / (2 * r / K)), K - 1)
    return (i + j) % 2


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _train_model(model, K, seed, n_steps=NUM_STEPS):
    """Train a NODE model on the checkerboard target.

    Returns:
        tuple: (test_mse, model, loss_history).
    """
    key = jr.PRNGKey(seed)
    data_key, loader_key = jr.split(key)

    x_train = jr.uniform(data_key, (N_TRAIN, D), minval=-R, maxval=R)
    target_fn = lambda x: checkerboard_target(x, K)
    y_train = jax.vmap(target_fn)(x_train)

    optimizer = optax.adam(LR)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    ts = jnp.array([0.0, 1.0])

    @eqx.filter_jit
    def step_fn(model, opt_state, x, y):
        def loss_fn(m):
            y_pred = jax.vmap(lambda x0: m(ts, x0)[-1])(x)
            return jnp.mean((y_pred - y) ** 2)
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = optimizer.update(grads, opt_state)
        return loss, eqx.apply_updates(model, updates), opt_state

    rng = loader_key
    best_loss = float("inf")
    no_improve = 0
    loss_val = float("inf")
    loss_history = []

    for step in range(n_steps):
        perm = jr.permutation(rng, jnp.arange(N_TRAIN))
        rng, = jr.split(rng, 1)
        loss_val, model, opt_state = step_fn(model, opt_state,
                                             x_train[perm], y_train[perm])
        loss_val = float(loss_val)

        if step % 500 == 0 or step == n_steps - 1:
            loss_history.append((step, loss_val))
            print(f"      step={step:5d}, loss={loss_val:.6e}")

        if loss_val < LOSS_THR:
            break
        if loss_val < best_loss - MIN_DELTA:
            best_loss = loss_val; no_improve = 0
        else:
            no_improve += 1
        if no_improve >= PATIENCE:
            break

    test_key = jr.PRNGKey(9999)
    x_test = jr.uniform(test_key, (N_TEST, D), minval=-R, maxval=R)
    y_test = jax.vmap(target_fn)(x_test)
    y_pred = jax.vmap(lambda x0: model(ts, x0)[-1])(x_test)
    mse = float(jnp.mean((y_pred - y_test) ** 2))

    print(f"    K={K}, seed={seed}: train_loss={loss_val:.3e}, test_mse={mse:.3e}")
    return mse, model, loss_history


def _make_model(arch, p_sa, key):
    """Instantiate a model parameter-matched to SANODE(p_sa)."""
    if arch == "sanode":
        return SANODE(D, pad_size=0, width_size=p_sa, depth=1, key=key)
    elif arch == "anode":
        return ANODE(D, width_size=matched_p_anode(p_sa), key=key)
    elif arch == "twoanode":
        p = matched_p_twoanode(p_sa)
        return TwoANODE(D, p=p, q=p, key=key)
    else:
        raise ValueError(f"Unknown arch: {arch}")


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

def load_or_train(arch, K, seed, cache_dir, p_sa=P_SA):
    """Load cached result and model, or train from scratch."""
    cache_path = cache_dir / f"{arch}_K{K}_s{seed}.pkl"
    model_path = cache_dir / f"{arch}_K{K}_s{seed}.eqx"
    key = jr.PRNGKey(seed * 100 + K)
    model = _make_model(arch, p_sa, key)

    if cache_path.exists() and model_path.exists():
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        mse = cached["mse"] if isinstance(cached, dict) else cached
        model = eqx.tree_deserialise_leaves(model_path, model)
        print(f"    [cached] {arch} K={K}, seed={seed}: mse={mse:.3e}")
        return mse, model

    mse, model, loss_history = _train_model(model, K, seed)
    with open(cache_path, "wb") as f:
        pickle.dump({"mse": mse, "loss_history": loss_history}, f)
    eqx.tree_serialise_leaves(model_path, model)
    return mse, model


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def run_sweep(cache_dir, fig_dir):
    """Train all three architectures across K and seeds; plot the comparison."""
    p_a = matched_p_anode(P_SA)
    p_2a = matched_p_twoanode(P_SA)
    print(f"\nsaNODE : p={P_SA}, params={param_count_sanode(P_SA)}")
    print(f"aNODE  : p={p_a},  params={param_count_anode(p_a)}")
    print(f"2aNODE : p=q={p_2a}, params={param_count_twoanode(p_2a, p_2a)}")

    archs = ("sanode", "anode", "twoanode")
    results = {arch: {} for arch in archs}
    for K in K_VALUES:
        print(f"\n--- K = {K} ---")
        for arch in archs:
            mses = []
            for seed in range(N_SEEDS):
                mse, _ = load_or_train(arch, K, seed, cache_dir)
                mses.append(mse)
            results[arch][K] = np.array(mses)

    arch_widths = {
        "sanode":   f"p={P_SA}",
        "anode":    f"p={p_a}",
        "twoanode": f"p=q={p_2a}",
    }
    arch_param_counts = {
        "sanode":   param_count_sanode(P_SA),
        "anode":    param_count_anode(p_a),
        "twoanode": param_count_twoanode(p_2a, p_2a),
    }
    plot_checkerboard_sweep(K_VALUES, results, arch_widths, arch_param_counts,
                            out_path=fig_dir / "checkerboard_sweep.pdf")
    return results


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def make_visualization(cache_dir, fig_dir, k_values=K_VALUES, p_sa=P_SA, seed=0):
    """Generate five-panel scatter figure for each K in k_values."""
    rng = np.random.default_rng(42)
    x_np = rng.uniform(-R, R, (600, D))
    ts = jnp.array([0.0, 1.0])

    for K in k_values:
        _, sanode_model   = load_or_train("sanode",   K, seed, cache_dir, p_sa)
        _, anode_model    = load_or_train("anode",    K, seed, cache_dir, p_sa)
        _, twoanode_model = load_or_train("twoanode", K, seed, cache_dir, p_sa)

        labels   = np.array([cell_label_np(xi, K) for xi in x_np])
        x_jax    = jnp.array(x_np)
        y_sa  = np.array(jax.vmap(lambda x0: sanode_model(ts, x0)[-1])(x_jax))
        y_a   = np.array(jax.vmap(lambda x0: anode_model(ts, x0)[-1])(x_jax))
        y_2a  = np.array(jax.vmap(lambda x0: twoanode_model(ts, x0)[-1])(x_jax))
        y_target = np.array(jax.vmap(lambda x: checkerboard_target(x, K))(x_jax))

        plot_checkerboard_viz(
            x_np, labels, y_target, y_sa, y_a, y_2a,
            K=K, D=D, S=S,
            out_path=fig_dir / f"checkerboard_viz_K{K}.pdf",
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cache_dir = Path("saved_models/checkerboard")
    fig_dir   = Path("figures/checkerboard")
    cache_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    run_sweep(cache_dir, fig_dir)
    make_visualization(cache_dir, fig_dir)


if __name__ == "__main__":
    main()
