"""Neural network building blocks on top of the scalar autograd engine."""

from __future__ import annotations
import random
from abc import ABC, abstractmethod
from typing import Sequence

from autograd.engine import Value

_ACTIVATIONS = ("tanh", "relu", None)


class Module(ABC):
    """Base class for all neural network components."""

    def zero_grad(self) -> None:
        """Reset the gradient of every parameter to zero."""
        for param in self.parameters():
            param.grad = 0.0

    @abstractmethod
    def parameters(self) -> list[Value]:
        """Return all trainable parameters of the module."""
        ...


class Neuron(Module):
    """A single neuron computing an affine map followed by an activation.
    Attributes:
        weights: Trainable weights, one per input feature.
        bias: Trainable bias term.
        activation: Name of the activation ("tanh", "relu") or None for linear.
    """

    __slots__ = ("weights", "bias", "activation")

    def __init__(self, in_features: int, activation: str | None = "tanh") -> None:
        """Initialize the neuron.
        Args:
            in_features: Number of input features.
            activation: One of "tanh", "relu", or None for linear output.
        Raises:
            ValueError: If `activation` is not supported.
        """
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"Unsupported activation {activation!r}; "
                f"expected one of {_ACTIVATIONS}."
            )
        self.weights: list[Value] = [
            Value(random.uniform(-1.0, 1.0), label=f"w{i}")
            for i in range(in_features)
        ]
        self.bias: Value = Value(0.0, label="b")
        self.activation = activation

    def __call__(self, inputs: Sequence[Value | float]) -> Value:
        """Compute the neuron output.
        Args:
            inputs: Input values, one per weight.
        Returns:
            The activated output as a `Value`.
        Raises:
            ValueError: If the number of inputs does not match the weights.
        """
        if len(inputs) != len(self.weights):
            raise ValueError(
                f"Expected {len(self.weights)} inputs, got {len(inputs)}."
            )
        out = sum((w * x for w, x in zip(self.weights, inputs)), self.bias)
        if self.activation == "tanh":
            return out.tanh()
        if self.activation == "relu":
            return out.relu()
        return out

    def parameters(self) -> list[Value]:
        """Return the weights followed by the bias."""
        return [*self.weights, self.bias]

    def __repr__(self) -> str:
        kind = self.activation if self.activation is not None else "linear"
        return f"Neuron({kind}, in_features={len(self.weights)})"


class Layer(Module):
    """A fully connected layer of neurons.
    Attributes:
        neurons: The neurons composing the layer.
    """

    __slots__ = ("neurons",)

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: str | None = "tanh",
    ) -> None:
        """Initialize the layer.
        Args:
            in_features: Number of input features per neuron.
            out_features: Number of neurons in the layer.
            activation: One of "tanh", "relu", or None for linear output.
        """
        self.neurons: list[Neuron] = [
            Neuron(in_features, activation) for _ in range(out_features)
        ]

    def __call__(self, inputs: Sequence[Value | float]) -> list[Value] | Value:
        """Apply every neuron to the inputs.
        Args:
            inputs: Input values shared by all neurons.
        Returns:
            A single `Value` if the layer has one neuron, else a list of them.
        """
        outs = [neuron(inputs) for neuron in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self) -> list[Value]:
        """Return the flattened parameters of all neurons."""
        return [p for neuron in self.neurons for p in neuron.parameters()]

    def __repr__(self) -> str:
        inner = ", ".join(repr(n) for n in self.neurons)
        return f"Layer([{inner}])"


class MLP(Module):
    """A multi-layer perceptron built from sequential `Layer` instances.
    Attributes:
        layers: The layers applied in order during the forward pass.
    """

    __slots__ = ("layers",)

    def __init__(
        self,
        in_features: int,
        layer_sizes: Sequence[int],
        hidden_activation: str = "tanh",
        output_activation: str | None = None,
    ) -> None:
        """Initialize the network.

        Args:
            in_features: Number of input features.
            layer_sizes: Output size of each successive layer.
            hidden_activation: Activation used by all hidden layers.
            output_activation: Activation used by the final layer.
        """
        sizes = [in_features, *layer_sizes]
        last = len(layer_sizes) - 1
        self.layers: list[Layer] = [
            Layer(
                sizes[i],
                sizes[i + 1],
                output_activation if i == last else hidden_activation,
            )
            for i in range(len(layer_sizes))
        ]

    def __call__(self, inputs: Sequence[Value | float]) -> list[Value] | Value:
        """Run a forward pass through all layers.
        Args:
            inputs: Input values for the first layer.
        Returns:
            A single `Value` if the final layer has one neuron, else a list.
        """
        out: list[Value] | Value = list(inputs)
        for layer in self.layers:
            out = layer(out if isinstance(out, list) else [out])
        return out

    def parameters(self) -> list[Value]:
        """Return the flattened parameters of all layers."""
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self) -> str:
        inner = ", ".join(repr(layer) for layer in self.layers)
        return f"MLP([{inner}])"