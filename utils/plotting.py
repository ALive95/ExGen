import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ---------------------------------------------------------------------------
# Width experiment plot
# ---------------------------------------------------------------------------

def plot_width_vs_error(N, P_values, results_by_target, baselines,
                        targets_meta, D, fig_dir):
    """
    Plot test MSE vs width P for smooth and Holder targets.

    Args:
        N (int): Training set size (for title).
        P_values (list[int]): Width values swept.
        results_by_target (dict): {tag: np.ndarray of shape (n_p, n_seeds)}.
        baselines (dict): {tag: {"hist": float, "vor": float}}.
        targets_meta (list): List of (tag, alpha) tuples.
        D (int): Input dimension.
        fig_dir (Path): Output directory.
    """
    fig, axes = plt.subplots(1, len(targets_meta),
                             figsize=(6 * len(targets_meta), 4), sharey=False)
    if len(targets_meta) == 1:
        axes = [axes]

    p_arr = np.array(P_values)
    colors = {"smooth": "tab:blue", "holder": "tab:green"}

    for ax, (tag, _) in zip(axes, targets_meta):
        data = results_by_target[tag]
        mean = data.mean(axis=1); std = data.std(axis=1)

        ax.fill_between(p_arr, mean - std, mean + std, alpha=0.2, color=colors[tag])
        ax.loglog(p_arr, mean, "o-", color=colors[tag], linewidth=2,
                  markersize=5, label="saNODE (mean ± std)")
        ax.axhline(baselines[tag]["hist"], color="tab:orange", linestyle="--",
                   linewidth=1.5, label="Histogram estimator")
        ax.axhline(baselines[tag]["vor"], color="tab:red", linestyle="--",
                   linewidth=1.5, label="Voronoi estimator")
        ax.set_xlabel("Width $p$"); ax.set_ylabel("Test MSE")
        title = f"{'Smooth' if tag == 'smooth' else 'Hölder-1/2'} target, $N={N}$, $d={D}$"
        ax.set_title(title)
        ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    out = fig_dir / f"width_vs_error_N{N}.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight", format="pdf")
    plt.close(fig)
    print(f"Figure saved: {out}")


# ---------------------------------------------------------------------------
# Checkerboard plots — three architectures
# ---------------------------------------------------------------------------

ARCH_COLORS = {"sanode": "tab:blue", "anode": "tab:red", "twoanode": "tab:green"}
ARCH_LABELS = {"sanode": "saNODE", "anode": "aNODE", "twoanode": "2aNODE"}


def plot_checkerboard_sweep(K_values, results, arch_widths, arch_param_counts,
                            out_path):
    """
    Semilog plot of checkerboard test MSE vs K for saNODE, aNODE, 2aNODE.

    Args:
        K_values (list[int]): Checkerboard resolutions.
        results (dict): {arch: {K: np.ndarray}} for arch in
            {"sanode", "anode", "twoanode"}.
        arch_widths (dict): {arch: width_str}, e.g. {"twoanode": "p=q=12"}.
        arch_param_counts (dict): {arch: int} total parameter counts.
        out_path (Path): Save path.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    for arch in ("sanode", "anode", "twoanode"):
        if arch not in results:
            continue
        K_arr = np.array(K_values)
        means = np.array([results[arch][K].mean() for K in K_values])
        stds  = np.array([results[arch][K].std()  for K in K_values])
        ax.fill_between(K_arr, means - stds, means + stds,
                        alpha=0.15, color=ARCH_COLORS[arch])
        label = (f"{ARCH_LABELS[arch]} (${arch_widths[arch]}$, "
                 f"{arch_param_counts[arch]} params)")
        ax.semilogy(K_arr, means, "o-", color=ARCH_COLORS[arch],
                    linewidth=2, markersize=5, label=label)
    ax.set_xlabel("Checkerboard resolution $K$"); ax.set_ylabel("Test MSE")
    ax.set_title("Cell-wise controllability")
    ax.set_xticks(K_values); ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Sweep figure saved: {out_path}")


def plot_checkerboard_viz(x_np, labels, y_target, y_sa, y_a, y_2a,
                          K, D, S, out_path):
    """
    Five-panel scatter figure: input, target, saNODE, aNODE, 2aNODE.

    Args:
        x_np (np.ndarray): Input points, shape (N, 2).
        labels (np.ndarray): Checkerboard cell labels, shape (N,).
        y_target (np.ndarray): Target outputs, shape (N, 2).
        y_sa (np.ndarray): saNODE outputs, shape (N, 2).
        y_a (np.ndarray): aNODE outputs, shape (N, 2).
        y_2a (np.ndarray): 2aNODE outputs, shape (N, 2).
        K (int): Checkerboard resolution.
        D (int): Dimension (expected 2).
        S (float): Target cluster half-distance.
        out_path (Path): Save path.
    """
    LABEL_COLORS = {0: "#2166ac", 1: "#d6604d"}

    def scatter_panel(ax, points, labels, title):
        for lab, color in LABEL_COLORS.items():
            mask = labels == lab
            ax.scatter(points[mask, 0], points[mask, 1],
                       c=color, s=8, alpha=0.6, linewidths=0)
        ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2)
        ax.set_aspect("equal"); ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    def draw_grid(ax, K, R=1.0):
        h = 2 * R / K
        for k in range(K + 1):
            v = -R + k * h
            ax.axhline(v, color="grey", linewidth=0.4, alpha=0.5)
            ax.axvline(v, color="grey", linewidth=0.4, alpha=0.5)

    fig, axes = plt.subplots(1, 5, figsize=(17.5, 3.5))
    scatter_panel(axes[0], x_np, labels, "Input (initial)")
    draw_grid(axes[0], K)
    scatter_panel(axes[1], y_target, labels, "Target")
    axes[1].scatter([S, -S], [S, -S], marker="*", s=120,
                    c=["#2166ac", "#d6604d"], zorder=5)
    scatter_panel(axes[2], y_sa, labels, f"saNODE output  ($K={K}$)")
    scatter_panel(axes[3], y_a,  labels, f"aNODE output   ($K={K}$)")
    scatter_panel(axes[4], y_2a, labels, f"2aNODE output  ($K={K}$)")

    handles = [mpatches.Patch(color=LABEL_COLORS[0], label="Label 0 (even cells)"),
               mpatches.Patch(color=LABEL_COLORS[1], label="Label 1 (odd cells)")]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               fontsize=9, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(f"Checkerboard sorting ($K={K}$, $d={D}$)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Visualization figure saved: {out_path}")
