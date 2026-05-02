"""
run_width_experiment.py

Width sweep experiment for saNODE in d=3.

Fix N; train saNODE with increasing width P; compare test MSE against
histogram and Voronoi baselines. Shows that saNODE matches estimator
errors at a width P* much smaller than the theoretical threshold.

Results cached in saved_models/width_experiment/.
"""

import csv
import pickle
from pathlib import Path

import numpy as np
from utils.estimators import histogram_estimator_error, voronoi_estimator_error
from utils.training import train_sanode
from utils.plotting import plot_width_vs_error


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

D          = 8
ALPHA      = 1.0
N_VALUES   = [5000]
P_VALUES   = [5, 11, 23, 51, 109, 237, 512, 1024]
N_SEEDS    = 1
SEEDS      = list([777, 6602, 1346])
N_TEST     = 10000
NUM_STEPS  = 40_000
LR         = 5e-4
BATCH_SIZE = 1024

from utils.targets import target_holder_d, target_holder_d_np, target_general_d, target_general_d_np

TARGETS = [
    ("smooth", target_general_d, target_general_d_np, ALPHA),
    ("holder", target_holder_d, target_holder_d_np, 0.5),
]


# ---------------------------------------------------------------------------
# Parameter count
# ---------------------------------------------------------------------------

def sanode_param_count(P, d, pad_size=0, depth=1):
    """
    Total trainable parameters in a saNODE with a single hidden layer.

    The vector field MLP has input size (d + pad_size + 1) for the time
    concatenation, hidden width P, and output size (d + pad_size).

    Args:
        P (int): Hidden width.
        d (int): Data dimension.
        pad_size (int): Padding dimensions (default 0).
        depth (int): Number of hidden layers (default 1).

    Returns:
        int: Total parameter count.
    """
    in_size  = d + pad_size + 1   # +1 for time input
    out_size = d + pad_size
    # First hidden layer
    params = P * in_size + P      # weight + bias
    # Additional hidden layers
    params += (depth - 1) * (P * P + P)
    # Output layer (no bias counted separately — Linear has bias)
    params += out_size * P + out_size
    return params


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

def load_or_train(tag, N, P, target_fn, seed, cache_dir, lr, batch_size):
    """
    Load cached MSE or train and cache.

    Args:
        tag (str): Target label ('smooth' or 'holder').
        N (int): Training set size.
        P (int): Width.
        target_fn (callable): JAX target function.
        seed (int): PRNG seed.
        cache_dir (Path): Cache directory.
        lr (float): Adam learning rate.
        batch_size (int): Minibatch size.

    Returns:
        float: Test MSE.
    """
    cache_path = cache_dir / f"{tag}_N{N}_p{P}_s{seed}.pkl"
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        mse = cached["mse"] if isinstance(cached, dict) else cached
        print(f"    [cached] {tag} N={N}, p={P}, seed={seed}: mse={mse:.3e}")
        return mse

    model, loss_hist = train_sanode(
        N, P, target_fn,
        num_steps=NUM_STEPS, seed=seed, data_size=D,
        lr=lr, batch_size=batch_size,
    )

    import jax
    import jax.numpy as jnp
    import jax.random as jr
    key = jr.PRNGKey(999)
    x_test = jr.uniform(key, (N_TEST, D), minval=-2.0, maxval=2.0)
    y_test = jax.vmap(target_fn)(x_test)
    ts = jnp.array([0.0, 1.0])
    y_pred = jax.vmap(lambda x0: model(ts, x0)[-1])(x_test)
    mse = float(jnp.mean((y_pred - y_test) ** 2))

    with open(cache_path, "wb") as f:
        pickle.dump({"mse": mse, "loss_history": loss_hist}, f)
    return mse


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

def print_and_save_table(N, P_values, results_by_target, baselines, fig_dir):
    """
    Print and save a CSV table of mean MSE, std, and parameter count per model,
    alongside histogram and Voronoi baseline errors and parameter counts.

    Voronoi stores N input points + N labels → 2*N*D parameters.
    Histogram stores one value per cell; number of cells = ceil(2R/h)^D
    with h = N^{-1/(2*alpha+D)} and R=2, but has no trainable params in the
    classical sense — reported as N (number of training points used).

    Args:
        N (int): Training set size.
        P_values (list[int]): Width values swept.
        results_by_target (dict): {tag: np.ndarray of shape (n_p, n_seeds)}.
        baselines (dict): {tag: {"hist": float, "vor": float}}.
        fig_dir (Path): Output directory for the CSV.
    """
    # Voronoi stores N (x, y) pairs in R^D → 2*N*D scalars
    vor_params  = 2 * N * D
    # Histogram has no parameters beyond the training set itself
    hist_params = N

    header = ["target", "P", "sanode_params", "mean_mse", "std_mse",
              "hist_params", "hist_mse", "vor_params", "vor_mse"]
    rows = []

    for tag, data in results_by_target.items():
        hist_mse = baselines[tag]["hist"]
        vor_mse  = baselines[tag]["vor"]
        for i, P in enumerate(P_values):
            mean = float(data[i].mean())
            std  = float(data[i].std())
            n_params = sanode_param_count(P, D)
            rows.append((tag, P, n_params, mean, std,
                         hist_params, hist_mse, vor_params, vor_mse))

    # Print to stdout
    col_w = [8, 6, 14, 12, 12, 12, 12, 12, 12]
    fmt_h = "  ".join(f"{h:<{w}}" for h, w in zip(header, col_w))
    sep = "=" * sum(col_w + [2 * (len(col_w) - 1)])
    print(f"\n{sep}")
    print(f"Results table  (N={N}, d={D})")
    print(sep)
    print(fmt_h)
    print("-" * len(fmt_h))
    prev_tag = None
    for tag, P, n_params, mean, std, hp, hm, vp, vm in rows:
        if prev_tag and tag != prev_tag:
            print("-" * len(fmt_h))
        print(f"  {tag:<6}  {P:<6}  {n_params:<14}  {mean:<12.4e}  {std:<12.4e}"
              f"  {hp:<12}  {hm:<12.4e}  {vp:<12}  {vm:<12.4e}")
        prev_tag = tag
    print(sep)
    print(sep)

    # Save CSV
    csv_path = fig_dir / f"width_results_N{N}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"Table saved to {csv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cache_dir = Path("saved_models/width_experiment")
    fig_dir   = Path("figures/width_experiment")
    cache_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"Width grid: {P_VALUES}")

    targets_meta = [(tag, alpha) for tag, _, _, alpha in TARGETS]

    for N in N_VALUES:
        print(f"\n{'='*60}\nN = {N}\n{'='*60}")

        baselines = {}
        for tag, _, target_np, alpha in TARGETS:
            h_err = histogram_estimator_error(N, target_np, alpha, D)
            v_err = voronoi_estimator_error(N, target_np, D)
            baselines[tag] = {"hist": h_err, "vor": v_err}
            print(f"  [{tag}] histogram={h_err:.3e}  voronoi={v_err:.3e}")

        results_by_target = {}
        for tag, target_fn, _, _ in TARGETS:
            print(f"\n  -- {tag} target --")
            data = np.zeros((len(P_VALUES), len(SEEDS)))
            for i, P in enumerate(P_VALUES):
                for j, s in enumerate(SEEDS):
                    data[i, j] = load_or_train(tag, N, P, target_fn,
                                               seed=s, cache_dir=cache_dir,
                                               lr=LR, batch_size=BATCH_SIZE)
            results_by_target[tag] = data

        print_and_save_table(N, P_VALUES, results_by_target, baselines, fig_dir)
        plot_width_vs_error(N, P_VALUES, results_by_target, baselines,
                            targets_meta, D, fig_dir)


if __name__ == "__main__":
    main()