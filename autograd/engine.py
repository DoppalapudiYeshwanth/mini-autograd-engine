"""Scalar reverse-mode automatic differentiation engine.

Provides a minimal ``Value`` class that records arithmetic operations in a
dynamic computation graph and computes gradients via backpropagation.
"""

from __future__ import annotations

import math
from typing import Callable


def _ensure_value(other: Value | int | float) -> Value:
    """Coerces a number to a ``Value``, passing existing ``Value`` through.
    Args:
        other: A ``Value``, ``int``, or ``float``.
    Returns:
        ``other`` itself if already a ``Value``, otherwise a new ``Value``.
    Raises:
        TypeError: If ``other`` is not a ``Value``, ``int``, or ``float``.
    """

    if isinstance(other, Value):
        return other
    if isinstance(other, (int, float)):
        return Value(other)
    raise TypeError(
        f"unsupported operand type: {type(other).__name__!r}; "
        "expected Value, int, or float"
    )


class Value:
    """A scalar node in a reverse-mode autodiff computation graph.
    Each instance stores a float, its accumulated gradient, and a closure
    that propagates gradients to the operands that produced it.
    Attributes:
        data: The scalar value held by this node.
        grad: The gradient of the backpropagated output w.r.t. this node.
        label: Optional human-readable name used in ``repr``.
    """

    __slots__ = ("data", "grad", "_backward", "_prev", "_op", "label")

    def __init__(
        self,
        data: float,
        _children: tuple[Value, ...] = (),
        _op: str = "",
        label: str = "",
    ) -> None:
        self.data: float = float(data)
        self.grad: float = 0.0
        self._backward: Callable[[], None] = lambda: None
        self._prev: set[Value] = set(_children)
        self._op: str = _op
        self.label: str = label

    def __repr__(self) -> str:
        name = f"label={self.label!r}, " if self.label else ""
        return f"Value({name}data={self.data:.4f}, grad={self.grad:.4f})"

    def __add__(self, other: Value | int | float) -> Value:
        other = _ensure_value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other: Value | int | float) -> Value:
        return self + other

    def __mul__(self, other: Value | int | float) -> Value:
        other = _ensure_value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other: Value | int | float) -> Value:
        return self * other

    def __neg__(self) -> Value:
        return self * -1.0

    def __sub__(self, other: Value | int | float) -> Value:
        return self + (-_ensure_value(other))

    def __rsub__(self, other: Value | int | float) -> Value:
        return _ensure_value(other) + (-self)

    def __truediv__(self, other: Value | int | float) -> Value:
        return self * (_ensure_value(other) ** -1)

    def __rtruediv__(self, other: Value | int | float) -> Value:
        return _ensure_value(other) * (self**-1)

    def __pow__(self, exponent: int | float) -> Value:
        if not isinstance(exponent, (int, float)):
            raise TypeError(
                f"exponent must be int or float, got {type(exponent).__name__!r}"
            )
        out = Value(math.pow(self.data, exponent), (self,), f"**{exponent}")

        def _backward() -> None:
            self.grad += exponent * math.pow(self.data, exponent - 1) * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> Value:
        """Applies the hyperbolic tangent activation."""
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward() -> None:
            self.grad += (1.0 - t * t) * out.grad

        out._backward = _backward
        return out

    def relu(self) -> Value:
        """Applies the rectified linear unit activation."""
        out = Value(self.data if self.data > 0.0 else 0.0, (self,), "relu")

        def _backward() -> None:
            self.grad += (1.0 if out.data > 0.0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def exp(self) -> Value:
        """Applies the natural exponential function."""
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward() -> None:
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def log(self) -> Value:
        """Applies the natural logarithm.

        Raises:
            ValueError: If the value is not strictly positive.
        """
        if self.data <= 0.0:
            raise ValueError(f"log requires a positive value, got {self.data}")
        out = Value(math.log(self.data), (self,), "log")

        def _backward() -> None:
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    def backward(self) -> None:
        """Backpropagates gradients from this node through the graph.
        Builds a topological ordering with an iterative post-order depth-first
        search, seeds this node's gradient with 1.0, and invokes each node's
        backward closure in reverse order. Gradients accumulate across calls,
        so reset ``grad`` on all nodes before re-running on the same graph.
        """
        topo: list[Value] = []
        visited: set[Value] = {self}
        stack = [(self, iter(self._prev))]

        while stack:
            node, children = stack[-1]
            for child in children:
                if child not in visited:
                    visited.add(child)
                    stack.append((child, iter(child._prev)))
                    break
            else:
                stack.pop()
                topo.append(node)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()