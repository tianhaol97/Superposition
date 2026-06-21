"""Experiment 2 — a phase diagram for superposition.

A statistical-physicist's instinct: pick an *order parameter* (a number that
summarises the macroscopic state of the system) and watch how it responds as we
sweep a *control parameter* (here, the sparsity). A sharp change in the order
parameter signals a phase transition.

We use a larger model (n = 40 features, m = 10 dimensions) and sweep sparsity
from dense to very sparse. We track two complementary order parameters:

  * features-per-dimension  φ = (number of features actually stored) / m.
        A feature counts as "stored" if its representation vector exceeds a
        norm threshold τ = 0.5. This is well defined because the learned norms
        are bimodal — stored features have ||W_i|| ≈ 1, dropped ones ≈ 0 — so any
        τ inside that gap gives the same count.
        - φ = 1  means each stored feature owns a dimension (no superposition).
        - φ > 1  means superposition. The ceiling is n/m = 4.

  * bottleneck usage = (sum_i D_i) / m, the fraction of representational
        capacity in use. It saturates near 1.

Unlike Experiments 1 and 3 (which show a single clean ground state via
best-of-seeds), a phase diagram should *average over disorder*: for each
sparsity we train many independent seeds and report the mean order parameter
with its standard deviation. The sparsities are spaced logarithmically in
1/density so the sparse (superposition) regime — where the scaling law lives —
is well sampled.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from _common import figure_path, log
from superposition import TrainConfig, train
from superposition.metrics import fraction_dimensions_used, num_represented

N_FEATURES = 40
N_HIDDEN = 10
IMPORTANCE = 0.95          # geometric importance decay; clean no-superposition baseline
THRESHOLD = 0.5           # a feature is "stored" if ||W_i|| > THRESHOLD
SEEDS = 10                # independent runs averaged per sparsity (disorder average)
STEPS = 4000
INV_DENSITY = np.logspace(0.0, 2.0, 15)   # 1 .. 100, log-spaced
SPARSITIES = 1.0 - 1.0 / INV_DENSITY        # 0 .. 0.99


def main() -> None:
    phi_mean, phi_std, usage_mean = [], [], []

    for sparsity in SPARSITIES:
        phis, usages = [], []
        for seed in range(SEEDS):
            cfg = TrainConfig(
                n_features=N_FEATURES,
                n_hidden=N_HIDDEN,
                sparsity=float(sparsity),
                importance=IMPORTANCE,
                steps=STEPS,
                lr=1e-3,
                seed=seed,
            )
            W = train(cfg).W.detach()
            phis.append(num_represented(W, THRESHOLD) / N_HIDDEN)
            usages.append(fraction_dimensions_used(W))
        phis, usages = np.array(phis), np.array(usages)
        phi_mean.append(phis.mean())
        phi_std.append(phis.std())
        usage_mean.append(usages.mean())
        log(
            f"sparsity={sparsity:.3f}  phi={phis.mean():.2f}±{phis.std():.2f}  "
            f"usage={usages.mean():.2f}"
        )

    phi_mean = np.array(phi_mean)
    phi_std = np.array(phi_std)
    usage_mean = np.array(usage_mean)
    density = 1.0 - SPARSITIES
    inv_density = 1.0 / density
    ceiling = N_FEATURES / N_HIDDEN

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ---- left: order parameters vs sparsity (with error bars) ----
    ax1.errorbar(
        SPARSITIES, phi_mean, yerr=phi_std, fmt="o-", color="#c33", lw=2, capsize=3,
        label=r"features per dimension $\phi$ (mean $\pm$ std)",
    )
    ax1.plot(SPARSITIES, usage_mean, "s-", color="#36c", lw=2,
             label=r"bottleneck usage  $\frac{1}{m}\sum_i D_i$")
    ax1.axhline(1.0, ls="--", color="0.6", lw=1)
    ax1.axhline(ceiling, ls=":", color="#c33", lw=1, label="max possible n/m = 4")
    ax1.set_xlabel("sparsity  (P feature is off)")
    ax1.set_ylabel("order parameter")
    ax1.set_title(
        f"Order parameters vs sparsity\n"
        f"(n={N_FEATURES}, m={N_HIDDEN}, averaged over {SEEDS} seeds)"
    )
    ax1.legend(fontsize=9)
    ax1.annotate("no superposition", xy=(0.05, 1.0), xytext=(0.05, 1.5), fontsize=9, color="0.4")
    ax1.annotate("superposition", xy=(0.9, 3.0), xytext=(0.5, 3.2), fontsize=9, color="0.4")

    # ---- right: scaling law on a log axis ----
    # Analytic prediction: a feature earns a (superposed) slot once its importance
    # clears a threshold that shrinks with density; with geometric importance decay
    # the stored-feature count grows LINEARLY IN log(1/density). We test that
    # predicted *form*; its slope carries O(1) constants we do not fix.
    x = np.log(inv_density)
    rising = (phi_mean > 1.05) & (phi_mean < ceiling - 0.05)   # superposition regime

    ax2.errorbar(inv_density, phi_mean, yerr=phi_std, fmt="o", color="#c33", ms=6,
                 capsize=3, label="measured (mean $\\pm$ std)")
    if rising.sum() >= 2:
        slope, intercept = np.polyfit(x[rising], phi_mean[rising], 1)
        pred = slope * x[rising] + intercept
        ss_res = float(np.sum((phi_mean[rising] - pred) ** 2))
        ss_tot = float(np.sum((phi_mean[rising] - phi_mean[rising].mean()) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        xline = np.linspace(x[rising].min(), x[rising].max(), 50)
        ax2.plot(np.exp(xline), slope * xline + intercept, "--", color="#222", lw=2,
                 label=f"predicted form  $\\propto$ log(1/density)\n(fit R² = {r2:.2f})")
        slope_theory = 1.0 / (N_HIDDEN * np.log(1.0 / IMPORTANCE))
        log(
            f"right-panel fit: slope={slope:.3f}/e-fold (R²={r2:.3f}); "
            f"crude theory slope={slope_theory:.3f} — same order, O(1) constants unfixed"
        )
    ax2.set_xscale("log")
    ax2.axhline(ceiling, ls=":", color="#c33", lw=1, label="ceiling n/m = 4")
    ax2.axhline(1.0, ls="--", color="0.6", lw=1)
    ax2.set_xlabel("1 / density   (how rare each feature is)")
    ax2.set_ylabel(r"features per dimension $\phi$")
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
