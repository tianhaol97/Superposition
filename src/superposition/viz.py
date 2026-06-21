"""Plotting helpers for visualising what the network learned.

All functions take an axis (or create a figure) and a model/weight matrix, so
the experiment scripts stay short and the plotting logic lives in one place.
"""

from __future__ import annotations

import numpy as np
import torch
from matplotlib.axes import Axes

from .metrics import feature_dimensionality, feature_norms


def _to_numpy(W) -> np.ndarray:
    if isinstance(W, torch.Tensor):
        return W.detach().cpu().numpy()
    return np.asarray(W)


def plot_feature_vectors_2d(ax: Axes, W, title: str = "", lim: float | None = None) -> None:
    """Draw each feature's 2D representation vector as an arrow from the origin.

    Only valid when the bottleneck dimension ``m == 2``. The colour encodes the
    feature index, so you can see which features ended up sharing directions.

    Args:
        lim: axis half-range. If ``None``, auto-scales to this panel's vectors.
            Pass a fixed value so several panels share one scale, which keeps the
            dashed unit-circle reference the same apparent size across them.
    """
    Wn = _to_numpy(W)
    if Wn.shape[0] != 2:
        raise ValueError("plot_feature_vectors_2d requires a 2D bottleneck (m=2)")

    n = Wn.shape[1]
    colors = _index_colors(n)
    if lim is None:
        lim = 1.2 * max(1e-6, np.abs(Wn).max())
    for i in range(n):
        ax.plot([0, Wn[0, i]], [0, Wn[1, i]], "-", color=colors[i], lw=2)
        ax.plot(Wn[0, i], Wn[1, i], "o", color=colors[i], ms=5)

    circle = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(circle), np.sin(circle), "--", color="0.7", lw=0.8)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=10)


def plot_gram_matrix(ax: Axes, W, title: str = "") -> None:
    """Heatmap of ``WᵀW``: the diagonal is squared norms, off-diagonal is interference."""
    Wn = _to_numpy(W)
    gram = Wn.T @ Wn
    vmax = max(1e-6, np.abs(gram).max())
    im = ax.imshow(gram, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=10)
    return im


def plot_dimensionality_bars(ax: Axes, W, title: str = "") -> None:
    """Bar chart of per-feature dimensionality ``D_i``, sorted descending."""
    dims = feature_dimensionality(torch.as_tensor(_to_numpy(W))).numpy()
    dims = np.sort(dims)[::-1]
    ax.bar(np.arange(len(dims)), dims, color="#3b6", width=0.9)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("feature (sorted)")
    ax.set_ylabel(r"dimensionality $D_i$")
    if title:
        ax.set_title(title, fontsize=10)


def _index_colors(n: int):
    import matplotlib.cm as cm

    return [cm.viridis(i / max(1, n - 1)) for i in range(n)]
