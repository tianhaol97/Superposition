"""Experiment 4 — the n=2, m=1 phase diagram (reproducing the paper's phase change).

We train the minimal model (2 features, 1 hidden dimension) across a grid of

    * I  = relative importance of the "extra" 2nd feature (the 1st has importance 1)
    * p  = density = 1 - sparsity

classify each learned solution into one of three phases, and overlay the
analytic boundary derived in Appendix B of the README. This is the direct
counterpart of the paper's phase-change diagram, and serves as a check that the
closed-form analysis matches what the model actually does.

Phases (by the learned weights W = [w1, w2]):
    * superposition   — both features stored (antipodal): |w1|, |w2| both large
    * not represented — only feature 1 kept, the extra feature dropped
    * dedicated       — only the extra feature kept (its own dimension)
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from _common import figure_path, log, train_best_of
from superposition import TrainConfig

N_SEEDS = 4
STEPS = 4000
IMPORTANCES = np.geomspace(0.3, 3.3, 11)   # relative importance of the extra feature
DENSITIES = np.linspace(0.1, 0.9, 11)       # p = 1 - sparsity
THRESHOLD = 0.4

SUP, NOT_REP, DED = 0, 1, 2


def classify(W) -> int:
    w = W.detach().flatten().numpy()
    a, b = abs(float(w[0])), abs(float(w[1]))
    if a > THRESHOLD and b > THRESHOLD:
        return SUP
    return NOT_REP if a >= b else DED  # which single feature survived


def p_star(I: float) -> float:
    """Analytic superposition boundary (Appendix B): superpose iff p < p_star."""
    return (4 * I) / (2 + 5 * I) if I <= 1 else 4.0 / (5 + 2 * I)


def main() -> None:
    phase = np.zeros((len(DENSITIES), len(IMPORTANCES)), dtype=int)
    for j, I in enumerate(IMPORTANCES):
        for i, p in enumerate(DENSITIES):
            cfg = TrainConfig(
                n_features=2, n_hidden=1, sparsity=float(1 - p),
                importance=float(I), steps=STEPS, lr=3e-3,
            )
            model, _ = train_best_of(cfg, n_seeds=N_SEEDS)
            phase[i, j] = classify(model.W)
        log(f"I={I:5.2f}  " + "".join("SND"[c] for c in phase[:, j]))

    fig, ax = plt.subplots(figsize=(7.5, 6))
    cmap = ListedColormap(["#bcd9ff", "#ffd6a8", "#cdeec0"])  # SUP, NOT_REP, DED
    ax.pcolormesh(IMPORTANCES, DENSITIES, phase, cmap=cmap, vmin=0, vmax=2,
                  shading="nearest", alpha=0.9)

    # analytic boundary p*(I) and the I=1 line separating the two dense phases
    I_fine = np.geomspace(IMPORTANCES[0], IMPORTANCES[-1], 200)
    ax.plot(I_fine, [p_star(I) for I in I_fine], "k-", lw=2.5,
            label=r"analytic boundary  $p^\star(I)$")
    ax.plot([1, 1], [p_star(1.0), DENSITIES[-1]], "k--", lw=1.5)

    ax.set_xscale("log")
    ax.set_xlim(IMPORTANCES[0], IMPORTANCES[-1])
    ax.set_ylim(DENSITIES[0], DENSITIES[-1])
    ax.set_xlabel("relative importance of the extra feature,  $I$")
    ax.set_ylabel("density  $p = 1 - $ sparsity")
    ax.set_title(
        "n = 2, m = 1 phase diagram\n"
        "cells = trained model; curve = analytic boundary (Appendix B)"
    )
    legend = [
        Patch(facecolor="#bcd9ff", label="superposition (antipodal)"),
        Patch(facecolor="#ffd6a8", label="not represented (extra dropped)"),
        Patch(facecolor="#cdeec0", label="dedicated (extra gets the dim)"),
    ]
    ax.legend(handles=legend + [plt.Line2D([], [], color="k", lw=2.5,
              label=r"analytic $p^\star(I)$")], loc="upper right", fontsize=9)

    fig.tight_layout()
    out = figure_path("04_phase_diagram_n2m1.png")
    fig.savefig(out, dpi=150)
    log(f"saved {out}")


if __name__ == "__main__":
    main()
