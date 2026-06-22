"""Experiment 2 — quantized geometry and the packing-problem connection.

Two ideas, one figure.

(A) Geometric phases. As we sweep sparsity for the small model (n = 5, m = 2),
    the per-feature dimensionality D_i does not vary smoothly: it locks onto a
    discrete ladder of preferred values,

        1            a feature alone on its own axis
        2/3 ≈ 0.67   three features at 120°  (a triangle)
        1/2 = 0.50   two antipodal features  (a "digon" / a pair)
        2/5 = 0.40   five features at 72°     (a pentagon)

    Each value corresponds to a specific regular arrangement of feature vectors.
    The flat plateaus separated by jumps are the signature of distinct
    geometric *phases*, just like the discrete plateaus in a physical system.

(B) A packing problem. At high sparsity the five features form a regular
    pentagon. We show that this arrangement minimises a "frustration energy"
    E = Σ_{i<j} (Ŵ_i·Ŵ_j)² — the same flavour of objective as the Thomson
    problem (point charges repelling on a sphere). The network, just by
    minimising reconstruction error, discovers the optimal packing.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from _common import figure_path, log, train_best_of
from superposition import TrainConfig, importance_weights
from superposition.metrics import (
    feature_dimensionality,
    feature_norms,
    frustration_energy,
)

IMPORTANCE = 0.9
from superposition.viz import plot_feature_vectors_2d

SPARSITIES = np.round(np.linspace(0.0, 0.97, 12), 3)
STICKY = {"1": 1.0, "2/3 (triangle)": 2 / 3, "1/2 (pair)": 1 / 2, "2/5 (pentagon)": 2 / 5}


def regular_polygon_energy(k: int) -> float:
    """Frustration energy of k unit vectors equally spaced around the circle."""
    angles = 2 * np.pi * np.arange(k) / k
    v = np.stack([np.cos(angles), np.sin(angles)])  # (2, k)
    g = v.T @ v
    iu = np.triu_indices(k, k=1)
    return float((g[iu] ** 2).sum())


def main() -> None:
    # ---- (A) sweep: collect every feature's dimensionality at each sparsity ----
    xs, ys = [], []
    pentagon_W = None
    for sparsity in SPARSITIES:
        cfg = TrainConfig(
            n_features=5, n_hidden=2, sparsity=float(sparsity),
            importance=IMPORTANCE, steps=6000, lr=2e-3,
        )
        model, loss = train_best_of(cfg, n_seeds=6)
        W = model.W.detach()
        dims = feature_dimensionality(W)
        for d in dims.tolist():
            if d > 0.02:  # ignore unrepresented features
                xs.append(float(sparsity))
                ys.append(d)
        pentagon_W = W  # keep the last (highest-sparsity) one
        log(f"sparsity={sparsity:.2f}  D_i={[round(d,2) for d in dims.tolist()]}  loss={loss:.4f}")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.5))

    axA.scatter(xs, ys, s=40, color="#36c", alpha=0.7, edgecolor="white", zorder=3)
    for label, val in STICKY.items():
        axA.axhline(val, ls="--", color="0.6", lw=1)
        axA.text(0.985, val, label, va="center", ha="left", fontsize=8, color="0.4")
    axA.set_xlim(-0.03, 1.12)
    axA.set_ylim(0, 1.08)
    axA.set_xlabel("sparsity")
    axA.set_ylabel(r"feature dimensionality $D_i$")
    axA.set_title("(A) Geometry is quantized:\n$D_i$ locks onto values set by regular polygons")

    # ---- (B) the pentagon as a solved packing problem ----
    plot_feature_vectors_2d(axB, pentagon_W)
    k = int((feature_norms(pentagon_W) > 0.5).sum())
    E_learned = frustration_energy(pentagon_W)
    E_ideal = regular_polygon_energy(k)
    # generalized (importance-weighted) energy: E_I = sum (I_i + I_j)(Ŵ_i·Ŵ_j)^2.
    # The clean unweighted match below holds because at high sparsity the geometry
    # is essentially importance-independent (the uniform-importance Thomson regime).
    I_vec = importance_weights(5, IMPORTANCE, "cpu")
    E_weighted = frustration_energy(pentagon_W, importance=I_vec)
    axB.set_title(
        f"(B) Optimal packing at high sparsity\n"
        f"{k} features form a regular {k}-gon\n"
        f"frustration energy: learned={E_learned:.3f}, ideal {k}-gon={E_ideal:.3f}"
    )
    log(
        f"pentagon: k={k}  E_learned={E_learned:.4f}  E_ideal={E_ideal:.4f}  "
        f"E_weighted(r={IMPORTANCE})={E_weighted:.4f}"
    )

    fig.tight_layout()
    out = figure_path("02_feature_geometry.png")
    fig.savefig(out, dpi=150)
    log(f"saved {out}")


if __name__ == "__main__":
    main()
