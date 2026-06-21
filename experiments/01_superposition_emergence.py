"""Experiment 1 — the emergence of superposition.

We fix a tiny model with n = 5 features and a bottleneck of m = 2 dimensions,
and sweep the sparsity from 0 (dense world) to nearly 1 (very sparse world).
For each sparsity we plot the two learned feature directions in the 2D
bottleneck.

The story the figure tells:

  * Dense world  -> the network can only afford to store 2 of the 5 features,
    and it stores them along orthogonal axes (no interference, no superposition).
  * Sparse world -> the network stores ALL 5 features by spreading them out as a
    regular pentagon. The directions overlap (interference), but because
    features rarely co-occur, that interference rarely bites.

This is the canonical "Toy Models of Superposition" result, reproduced from
scratch.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from _common import figure_path, log, train_best_of
from superposition import TrainConfig
from superposition.metrics import feature_dimensionality, num_represented
from superposition.viz import plot_feature_vectors_2d

SPARSITIES = [0.0, 0.5, 0.7, 0.9, 0.97]


def main() -> None:
    # First pass: train every panel and collect results, so we can choose ONE
    # axis scale shared by all panels. With a shared scale the dashed unit circle
    # renders identically everywhere and arrow lengths are directly comparable
    # across sparsities (per-panel auto-scaling would make the same length look
    # different from panel to panel).
    panels = []
    for sparsity in SPARSITIES:
        cfg = TrainConfig(
            n_features=5,
            n_hidden=2,
            sparsity=sparsity,
            importance=0.9,  # features decay in importance; breaks the dense-case symmetry
            steps=6000,
            lr=2e-3,
        )
        model, loss = train_best_of(cfg, n_seeds=5)
        W = model.W.detach()
        dims = feature_dimensionality(W)
        k = num_represented(W)
        log(
            f"sparsity={sparsity:.2f}  represented={k}  loss={loss:.4f}  "
            f"D_i={[round(d, 2) for d in dims.tolist()]}"
        )
        panels.append((sparsity, W, dims, k))

    # One shared half-range for all 2D panels (keep the unit circle well inside).
    shared_lim = max(1.3, 1.15 * max(float(W.abs().max()) for _, W, _, _ in panels))

    fig, axes = plt.subplots(2, len(SPARSITIES), figsize=(3 * len(SPARSITIES), 6))
    for col, (sparsity, W, dims, k) in enumerate(panels):
        ax_top = axes[0, col]
        plot_feature_vectors_2d(ax_top, W, lim=shared_lim)
        ax_top.set_title(
            f"sparsity = {sparsity:.2f}\n{k} of 5 features stored", fontsize=10
        )

        ax_bot = axes[1, col]
        sorted_dims = dims.sort(descending=True).values.numpy()
        ax_bot.bar(range(len(sorted_dims)), sorted_dims, color="#3b6", width=0.9)
        ax_bot.set_ylim(0, 1.05)
        ax_bot.set_xticks(range(5))
        if col == 0:
            ax_bot.set_ylabel(r"dimensionality $D_i$")
        ax_bot.set_xlabel("feature")

    fig.suptitle(
        "Emergence of superposition: dense world stores few features on orthogonal axes;\n"
        "sparse world packs all features into a pentagon (overlapping directions)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = figure_path("01_superposition_emergence.png")
    fig.savefig(out, dpi=150)
    log(f"saved {out}")


if __name__ == "__main__":
    main()
