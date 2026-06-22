"""Training the toy model.

We never use a fixed dataset: on every optimisation step we draw a fresh batch
of sparse feature vectors and ask the model to reconstruct them. The model is
therefore learning the *statistics* of the sparse world, not memorising
examples.

The loss is an importance-weighted mean squared reconstruction error,

    L = mean over batch and features of  I_i * (x_i - x'_i)^2

where ``I_i`` is the (optional) importance of feature ``i``. Unequal importances
let some features matter more than others, which is what produces the richer
geometric arrangements; with uniform importance all features are interchangeable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .data import generate_batch
from .model import ToyModel


@dataclass
class TrainConfig:
    n_features: int = 5
    n_hidden: int = 2
    sparsity: float = 0.0
    importance: float = 1.0          # geometric decay base; 1.0 = uniform
    steps: int = 10_000
    batch_size: int = 1024
    lr: float = 1e-3
    seed: int = 0
    device: str = "cpu"
    log_every: int = 0               # 0 disables progress logging
    history: list = field(default_factory=list, repr=False)


def importance_weights(n_features: int, base: float, device) -> torch.Tensor:
    """Return ``(n,)`` importance vector ``[base^0, base^1, ...]``.

    ``base == 1.0`` gives uniform importance.
    """
    powers = torch.arange(n_features, device=device, dtype=torch.float32)
    return base ** powers


def train(config: TrainConfig, on_step=None) -> ToyModel:
    """Train a :class:`ToyModel` according to ``config`` and return it.

    The per-step loss is appended to ``config.history`` so callers can inspect
    convergence. If ``on_step`` is given, it is called as ``on_step(step, model)``
    after every optimiser step — useful for recording a metric trajectory (e.g.
    the frustration energy of the learned features) during training.
    """
    device = torch.device(config.device)
    gen = torch.Generator(device=device).manual_seed(config.seed)
    torch.manual_seed(config.seed)

    model = ToyModel(config.n_features, config.n_hidden).to(device)
    importance = importance_weights(config.n_features, config.importance, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    for step in range(config.steps):
        x = generate_batch(
            config.batch_size,
            config.n_features,
            config.sparsity,
            generator=gen,
            device=device,
        )
        x_hat = model(x)
        loss = (importance * (x - x_hat) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        config.history.append(float(loss))
        if on_step is not None:
            on_step(step, model)
        if config.log_every and (step % config.log_every == 0 or step == config.steps - 1):
            print(f"step {step:6d}  loss {float(loss):.6f}")

    return model
