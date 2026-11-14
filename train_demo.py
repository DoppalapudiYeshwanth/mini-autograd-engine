"""Training demo for a pure-Python autograd engine.

Trains a small MLP with an SVM max-margin hinge loss on a synthetic
two-moons dataset, then exports computation-graph visualizations.
"""

import math
import random
from pathlib import Path
from typing import List, Tuple
from autograd.engine import Value
from autograd.nn import MLP
from autograd.visualize import render_graph


def generate_moons_dataset(
    n_samples: int = 40,
    noise: float = 0.12,
    seed: int = 42,
) -> Tuple[List[List[float]], List[float]]:
    """Generates two interleaved crescent moons in 2D using pure Python.
    Args:
        n_samples: Total number of samples (split between the two moons).
        noise: Standard deviation of Gaussian noise added to each coordinate.
        seed: Seed for the local random number generator.

    Returns:
        A tuple ``(X, y)`` where ``X`` is a list of 2D points and ``y`` is a
        list of binary targets in ``{-1.0, 1.0}``.
    """
    rng = random.Random(seed)
    n_outer = n_samples // 2
    n_inner = n_samples - n_outer

    X: List[List[float]] = []
    y: List[float] = []

    # Upper moon (label -1.0).
    for i in range(n_outer):
        t = math.pi * i / max(n_outer - 1, 1)
        x1 = math.cos(t) + rng.gauss(0.0, noise)
        x2 = math.sin(t) + rng.gauss(0.0, noise)
        X.append([x1, x2])
        y.append(-1.0)

    # Lower, shifted moon (label +1.0).
    for i in range(n_inner):
        t = math.pi * i / max(n_inner - 1, 1)
        x1 = 1.0 - math.cos(t) + rng.gauss(0.0, noise)
        x2 = 0.5 - math.sin(t) + rng.gauss(0.0, noise)
        X.append([x1, x2])
        y.append(1.0)

    # Shuffle samples while keeping features and labels aligned.
    order = list(range(n_samples))
    rng.shuffle(order)
    X = [X[i] for i in order]
    y = [y[i] for i in order]
    return X, y


def compute_batch_loss(
    model: MLP,
    X: List[List[float]],
    y: List[float],
    l2_reg: float = 1e-4,
) -> Tuple[Value, float]:
    """Computes SVM hinge loss with L2 regularization over a batch.
    Args:
        model: The MLP producing a single scalar score per sample.
        X: Input feature vectors.
        y: Targets in ``{-1.0, 1.0}``.
        l2_reg: L2 regularization coefficient.

    Returns:
        A tuple ``(total_loss, accuracy)`` where ``total_loss`` is a ``Value``
        and ``accuracy`` is the fraction of correctly classified samples.
    """
    losses: List[Value] = []
    correct = 0

    for xi, yi in zip(X, y):
        score = model([Value(float(v)) for v in xi])
        losses.append((1.0 + (-yi * score)).relu())
        if (score.data > 0) == (yi > 0):
            correct += 1

    data_loss = sum(losses, Value(0.0)) / float(len(losses))
    reg_loss = l2_reg * sum((p * p for p in model.parameters()), Value(0.0))
    total_loss = data_loss + reg_loss
    accuracy = correct / len(y)
    return total_loss, accuracy


def train() -> MLP:
    """Trains the MLP on the moons dataset.
    Returns:
        The trained ``MLP`` model.
    """
    random.seed(42)
    X, y = generate_moons_dataset(n_samples=40)
    model = MLP(in_features=2, layer_sizes=[8, 8, 1], hidden_activation="tanh", output_activation=None)
    velocities = [0.0 for _ in model.parameters()]

    epochs = 60
    lr_start, lr_end = 0.15, 0.02

    for epoch in range(1, epochs + 1):
        lr = lr_start + (lr_end - lr_start) * (epoch - 1) / (epochs - 1)

        model.zero_grad()
        loss, acc = compute_batch_loss(model, X, y)
        loss.backward()

        for i, p in enumerate(model.parameters()):
            velocities[i] = 0.9 * velocities[i] - lr * p.grad
            p.data += velocities[i]

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d}/{epochs:03d} | "
                f"Loss: {loss.data:.4f} | "
                f"Acc: {acc * 100:.1f}% | "
                f"LR: {lr:.4f}"
            )

    return model


def export_visualizations(model: MLP) -> None:
    """Exports computation-graph visualizations to the ``artifacts`` folder.
    Renders a compact single-neuron expression DAG and a single-sample
    forward/backward pass through the trained model.

    Args:
        model: The trained MLP.
    """
    Path("artifacts").mkdir(parents=True, exist_ok=True)

    # Compact single-neuron graph: x1*w1 + x2*w2 + b -> tanh.
    x1 = Value(0.5, label="x1")
    x2 = Value(-0.8, label="x2")
    w1 = Value(0.7, label="w1")
    w2 = Value(-0.3, label="w2")
    b = Value(0.1, label="b")

    x1w1 = x1 * w1
    x1w1.label = "x1*w1"
    x2w2 = x2 * w2
    x2w2.label = "x2*w2"
    summed = x1w1 + x2w2
    summed.label = "x1*w1 + x2*w2"
    pre_act = summed + b
    pre_act.label = "n"
    neuron_out = pre_act.tanh()
    neuron_out.label = "tanh(n)"
    neuron_out.backward()
    render_graph(
        neuron_out,
        output_path="artifacts/neuron_graph",
        graph_format="svg",
        view=False,
    )

    # Single-sample forward + backward pass through the trained model.
    model.zero_grad()
    x = [Value(0.5, label="x0"), Value(-0.8, label="x1")]
    out = model(x)
    out.label = "out"
    out.backward()
    render_graph(
        out,
        output_path="artifacts/trained_mlp_pass",
        graph_format="svg",
        view=False,
    )


def main() -> None:
    """Runs training followed by graph export."""
    model = train()
    export_visualizations(model)
    print("Saved graphs to artifacts/neuron_graph.svg and artifacts/trained_mlp_pass.svg")


if __name__ == "__main__":
    main()