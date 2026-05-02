import numpy as np


def histogram_estimator_error(N, target_fn_np, alpha, d, R=2.0, N_test=2000, seed=0):
    """
    MSE of the histogram estimator with optimal bandwidth h_N = N^{-1/(2*alpha+d)}.

    Builds a d-dimensional grid of cells of side h, averages training labels
    per cell, and evaluates on a fixed test set.

    Args:
        N (int): Number of training points.
        target_fn_np (callable): Target function x -> y (NumPy, no JAX tracing).
        alpha (float): Holder regularity exponent.
        d (int): Input dimension.
        R (float): Domain half-width; points drawn from [-R, R]^d.
        N_test (int): Number of test points.
        seed (int): RNG seed for reproducibility.

    Returns:
        float: MSE on the test set.
    """
    rng = np.random.default_rng(seed)
    x_test = rng.uniform(-R, R, (N_test, d))
    y_test = np.stack([target_fn_np(x) for x in x_test])

    rng_tr = np.random.default_rng(seed + N)
    x_train = rng_tr.uniform(-R, R, (N, d))
    y_train = np.stack([target_fn_np(x) for x in x_train])

    h = min(2 * R, N ** (-1.0 / (2 * alpha + d)))
    n_cells = [max(1, int(np.ceil(2 * R / h)))] * d

    cell_sum = np.zeros(n_cells + [y_train.shape[1]])
    cell_cnt = np.zeros(n_cells)

    for xi, yi in zip(x_train, y_train):
        idx = tuple(min(int((xi[k] + R) / h), n_cells[k] - 1) for k in range(d))
        cell_sum[idx] += yi
        cell_cnt[idx] += 1

    y_pred = np.zeros_like(y_test)
    for j, xi in enumerate(x_test):
        idx = tuple(min(int((xi[k] + R) / h), n_cells[k] - 1) for k in range(d))
        if cell_cnt[idx] > 0:
            y_pred[j] = cell_sum[idx] / cell_cnt[idx]

    mse = float(np.mean((y_pred - y_test) ** 2))
    print(f"  histogram: N={N:>5}, d={d}, h={h:.3f}, mse={mse:.4e}")
    return mse


def voronoi_estimator_error(N, target_fn_np, d, R=2.0, N_test=2000, seed=0):
    """
    MSE of the nearest-neighbour (Voronoi) estimator.

    Assigns each test point the label of its closest training point.

    Args:
        N (int): Number of training points.
        target_fn_np (callable): Target function x -> y (NumPy, no JAX tracing).
        d (int): Input dimension.
        R (float): Domain half-width; points drawn from [-R, R]^d.
        N_test (int): Number of test points.
        seed (int): RNG seed for reproducibility.

    Returns:
        float: MSE on the test set.
    """
    rng = np.random.default_rng(seed)
    x_test = rng.uniform(-R, R, (N_test, d))
    y_test = np.stack([target_fn_np(x) for x in x_test])

    rng_tr = np.random.default_rng(seed + N)
    x_train = rng_tr.uniform(-R, R, (N, d))
    y_train = np.stack([target_fn_np(x) for x in x_train])

    # Pairwise squared distances: (N_test, N)
    diff = x_test[:, None, :] - x_train[None, :, :]
    dists = np.sum(diff ** 2, axis=-1)
    y_pred = y_train[np.argmin(dists, axis=1)]

    mse = float(np.mean((y_pred - y_test) ** 2))
    print(f"  voronoi:   N={N:>5}, d={d}, mse={mse:.4e}")
    return mse


def empirical_constant(estimator_errors, N_values, rate_fn):
    """
    Estimate the constant C such that estimator_error(N) <= C * rate_fn(N).
    Taken as the maximum ratio over all N.

    Args:
        estimator_errors (dict): {N: mse} from an estimator.
        N_values (list[int]): List of N values.
        rate_fn (callable): Theoretical rate function N -> float.

    Returns:
        float: Empirical constant C.
    """
    return max(estimator_errors[N] / rate_fn(N) for N in N_values)
