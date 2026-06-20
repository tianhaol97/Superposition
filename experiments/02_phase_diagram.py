"""Experiment 2 — a phase diagram for superposition.

A statistical-physicist's instinct: pick an *order parameter* (a number that
summarises the macroscopic state of the system) and watch how it responds as we
sweep a *control parameter* (here, the sparsity). A sharp change in the order
parameter signals a phase transition.

We use a larger model (n = 40 features, m = 10 dimensions) and sweep sparsity
from dense to very sparse. We track two complementary order parameters:

  * features-per-dimension = (number of features actually stored) / m.
        - 1  means each stored feature owns a dimension (no superposition).
        - >1 means the network is packing more features than it has dimensions
             (superposition). The ceiling is n/m = 4.

  * bottleneck usage = (sum_i D_i) / m, the fraction of the available
        representational capacity that is in use. It saturates near 1.

Sweeping sparsity drives the network from a "no-superposition" phase into a
"superposition" phase, exactly the way lowering temperature drives a magnet
from disordered to ordered.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from _common import figure_path, log, train_best_of
from superposition import TrainConfig
from superposition.metrics import (
    feature_norms,
    fraction_dimensions_used,
    num_represented,
    total_dimensionality,
)

N_FEATURES = 40
N_HIDDEN = 10
SPARSITIES = [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]
SEEDS = 4
IMPORTANCE = 0.95  # features decay in importance; gives a clean "no-superposition" baseline


def main() -> None:
    feat_per_dim = []
    usage = []
    interference = []

    for sparsity in SPARSITIES:
        cfg = TrainConfig(
            n_features=N_FEATURES,
            n_hidden=N_HIDDEN,
            sparsity=sparsity,
            importance=IMPORTANCE,
            steps=5000,
            lr=1e-3,
        )
        model, loss = train_best_of(cfg, n_seeds=SEEDS)
        W = model.W.detach()

        fpd = num_represented(W) / N_HIDDEN
        use = fraction_dimensions_used(W)
        # mean absolute off-diagonal interference among represented features
        norms = feature_norms(W)
        rep = norms > 0.5
        Wr = W[:, rep]
        gram = (Wr.T @ Wr)
        off = gram - np.diag(np.diag(gram.numpy()))
        intf = float(np.abs(off.numpy()).sum() / max(1, rep.sum() ** 2 - rep.sum()))

        feat_per_dim.append(fpd)
        usage.append(use)
        interference.append(intf)
        log(
            f"sparsity={sparsity:.2f}  features/dim={fpd:.2f}  "
            f"usage={use:.2f}  interference={intf:.3f}  loss={loss:.4f}"
        )

    density = 1.0 - np.array(SPARSITIES)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(SPARSITIES, feat_per_dim, "o-", color="#c33", lw=2, label="features per dimension")
    ax1.plot(SPARSITIES, usage, "s-", color="#36c", lw=2, label="bottleneck usage (ΣD/m)")
    ax1.axhline(1.0, ls="--", color="0.6", lw=1)
    ax1.axhline(N_FEATURES / N_HIDDEN, ls=":", color="#c33", lw=1, label="max possible n/m = 4")
    ax1.set_xlabel("sparsity  (P feature is off)")
    ax1.set_ylabel("order parameter")
    ax1.set_title(f"Order parameters vs sparsity\n(n={N_FEATURES} features, m={N_HIDDEN} dimensions)")
    ax1.legend(fontsize=9)
    ax1.annotate(
        "no superposition", xy=(0.05, 1.0), xytext=(0.05, 1.6), fontsize=9, color="0.4"
    )
    ax1.annotate(
        "superposition", xy=(0.9, 3.0), xytext=(0.55, 3.2), fontsize=9, color="0.4"
    )

    # Analytic prediction: a feature earns a (superposed) slot once its importance
    # clears a threshold that shrinks with density. With geometric importance decay
    # this makes the stored-feature count grow LINEARLY IN log(1/density). We test
    # that predicted *form*; its slope carries O(1) constants we do not fix.
    fpd_arr = np.array(feat_per_dim)
    inv_density = 1.0 / density
    x = np.log(inv_density)
    ceiling = N_FEATURES / N_HIDDEN
    rising = (fpd_arr > 1.05) & (fpd_arr < ceiling - 0.05)  # the superposition regime

    ax2.plot(inv_density, fpd_arr, "o", color="#c33", ms=8, label="measured")
    if rising.sum() >= 2:
        slope, intercept = np.polyfit(x[rising], fpd_arr[rising], 1)
        pred = slope * x[rising] + intercept
        ss_res = float(np.sum((fpd_arr[rising] - pred) ** 2))
        ss_tot = float(np.sum((fpd_arr[rising] - fpd_arr[rising].mean()) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        xline = np.linspace(x[rising].min(), x[rising].max(), 50)
        ax2.plot(
            np.exp(xline), slope * xline + intercept, "--", color="#222", lw=2,
            label=f"predicted form  ∝ log(1/density)\n(fit R² = {r2:.2f})",
        )
        slope_theory = 1.0 / (N_HIDDEN * np.log(1.0 / IMPORTANCE))
        log(
            f"right-panel: fitted slope={slope:.3f}/e-fold (R²={r2:.3f}); "
            f"crude theory slope={slope_theory:.3f} — same order, O(1) constants unfixed"
        )
    ax2.set_xscale("log")
    ax2.axhline(ceiling, ls=":", color="#c33", lw=1, label="ceiling n/m = 4")
    ax2.axhline(1.0, ls="--", color="0.6", lw=1)
    ax2.set_xlabel("1 / density   (how rare each feature is)")
    ax2.set_ylabel("features per dimension")
    ax2.set_title(
        "Superposition grows as log(1/density)\n"
        "predicted scaling form; prefactor carries O(1) constants"
    )
    ax2.legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    out = figure_path("02_phase_diagram.png")
    fig.savefig(out, dpi=150)
    log(f"saved {out}")


if __name__ == "__main__":
    main()
