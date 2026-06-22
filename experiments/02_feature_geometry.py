"""Experiment 2 — quantized geometry and the packing-problem connection.

Two panels.

(A) Geometric phases. Sweeping sparsity for the small model (n = 5, m = 2), the
    per-feature dimensionality D_i locks onto a discrete ladder of values
    (1, 2/3, 1/2, 2/5, ...), each a regular polygon — flat plateaus separated by
    jumps, the signature of distinct geometric phases.

(B) Learning dynamics: the network *discovers* the packing. During a single
    high-sparsity run we track the frustration / generalized-Thomson energy
    E = Σ_{i<j} (Ŵ_i·Ŵ_j)² of the learned feature *directions*. As the
    reconstruction loss falls, E is driven down to the ideal regular-pentagon
    value E_5 = 3.75 — the network solves the packing problem as a byproduct of
    minimising reconstruction error. This is the dynamical view of the paper's
    "energy-level" picture.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import matplotlib.pyplot as plt

from _common import figure_path, log, train_best_of
from superposition import TrainConfig, train
from superposition.metrics import feature_dimensionality, frustration_energy

IMPORTANCE = 0.9
SPARSITIES = np.round(np.linspace(0.0, 0.97, 12), 3)
STICKY = {"1": 1.0, "2/3 (triangle)": 2 / 3, "1/2 (pair)": 1 / 2, "2/5 (pentagon)": 2 / 5}


def regular_polygon_energy(k: int) -> float:
    """Frustration energy of k unit vectors equally spaced around the circle = k(k-2)/4."""
    angles = 2 * np.pi * np.arange(k) / k
    v = np.stack([np.cos(angles), np.sin(angles)])
    g = v.T @ v
    iu = np.triu_indices(k, k=1)
    return float((g[iu] ** 2).sum())


def best_energy_trace(base_cfg, n_seeds=6, record_every=40):
    """Train ``n_seeds`` runs, recording the frustration energy of the learned
    directions every ``record_every`` steps, and return the lowest-final-loss
    run as ``(final_loss, steps, energies, loss_history)``.

    Energy is taken over *all* feature directions (threshold = -1) so the
    trajectory reflects how the directions organise; the metric is scale-free,
    so it is well defined even while the weights are still small.
    """
    best = None
    for s in range(n_seeds):
        cfg = replace(base_cfg, seed=s, history=[])
        steps: list[int] = []
        energies: list[float] = []

        def rec(step, model, _s=steps, _e=energies, _cfg=cfg):
            if step % record_every == 0 or step == _cfg.steps - 1:
                _s.append(step)
                _e.append(frustration_energy(model.W.detach(), threshold=-1.0))

        train(cfg, on_step=rec)
        if best is None or cfg.history[-1] < best[0]:
            best = (cfg.history[-1], steps, energies, list(cfg.history))
    return best


def main() -> None:
    # ---- (A) sweep: per-feature dimensionality vs sparsity ----
    xs, ys = [], []
    for sparsity in SPARSITIES:
        cfg = TrainConfig(n_features=5, n_hidden=2, sparsity=float(sparsity),
                          importance=IMPORTANCE, steps=6000, lr=2e-3)
        model, loss = train_best_of(cfg, n_seeds=6)
        dims = feature_dimensionality(model.W.detach())
        for d in dims.tolist():
            if d > 0.02:  # ignore unrepresented features
                xs.append(float(sparsity))
                ys.append(d)
        log(f"sparsity={sparsity:.2f}  D_i={[round(d, 2) for d in dims.tolist()]}  loss={loss:.4f}")

    # ---- (B) learning dynamics at high sparsity ----
    dyn_cfg = TrainConfig(n_features=5, n_hidden=2, sparsity=0.97,
                          importance=IMPORTANCE, steps=6000, lr=2e-3)
    final_loss, steps, energies, loss_hist = best_energy_trace(dyn_cfg, n_seeds=6)
    E_ideal = regular_polygon_energy(5)
    log(f"dynamics: final E={energies[-1]:.4f}  ideal pentagon E={E_ideal:.4f}  loss={final_loss:.4f}")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.5))

    # panel A
    axA.scatter(xs, ys, s=40, color="#36c", alpha=0.7, edgecolor="white", zorder=3)
    for label, val in STICKY.items():
        axA.axhline(val, ls="--", color="0.6", lw=1)
        axA.text(0.985, val, label, va="center", ha="left", fontsize=8, color="0.4")
    axA.set_xlim(-0.03, 1.12)
    axA.set_ylim(0, 1.08)
    axA.set_xlabel("sparsity")
    axA.set_ylabel(r"feature dimensionality $D_i$")
    axA.set_title("(A) Geometry is quantized:\n$D_i$ locks onto values set by regular polygons")

    # panel B: energy of learned features (left) and reconstruction loss (right)
    h_E, = axB.plot(steps, energies, color="#c33", lw=2,
                    label=r"energy of learned features $E(W)$")
    h_ideal = axB.axhline(E_ideal, ls="--", color="#333", lw=1.5,
                          label=rf"ideal pentagon  $E_5 = {E_ideal:.2f}$")
    axB.set_xlabel("training step")
    axB.set_ylabel(r"frustration / Thomson energy  $E(W)$", color="#c33")
    axB.tick_params(axis="y", colors="#c33")
    axB.set_ylim(E_ideal - 0.4, max(energies) + 0.3)

    axL = axB.twinx()
    # the per-batch loss is noisy; show a moving average for legibility
    w = 50
    loss_smooth = np.convolve(np.array(loss_hist), np.ones(w) / w, mode="valid")
    h_L, = axL.plot(np.arange(len(loss_smooth)) + w // 2, loss_smooth, color="0.5", lw=1.5,
                    label="reconstruction loss (smoothed)")
    axL.set_yscale("log")
    axL.set_ylabel("reconstruction loss  (log scale)", color="0.4")
    axL.tick_params(axis="y", colors="0.4")

    axB.set_title("(B) Training discovers the packing:\n"
                  "energy of learned features falls to the ideal regular-pentagon value")
    axB.legend([h_E, h_ideal, h_L], [h.get_label() for h in (h_E, h_ideal, h_L)],
               fontsize=8, loc="upper right")

    fig.tight_layout()
    out = figure_path("02_feature_geometry.png")
    fig.savefig(out, dpi=150)
    log(f"saved {out}")


if __name__ == "__main__":
    main()
