"""Unit tests for ``autograd.engine.Value`` and ``autograd.nn.MLP``."""

import math
from typing import Callable
import pytest
from autograd.engine import Value
from autograd.nn import MLP

TOL = 1e-5


def numerical_grad(
    fn: Callable[[list[float]], float],
    inputs: list[float],
    eps: float = 1e-6,
) -> list[float]:
    """Estimate the gradient of ``fn`` at ``inputs`` via central differences."""
    grads = []
    for i in range(len(inputs)):
        plus = list(inputs)
        minus = list(inputs)
        plus[i] += eps
        minus[i] -= eps
        grads.append((fn(plus) - fn(minus)) / (2 * eps))
    return grads


def _check_against_numerical(
    build: Callable[..., "Value | float"],
    inputs: list[float],
) -> None:
    """Compare autograd gradients of ``build`` with numerical gradients."""
    values = [Value(v) for v in inputs]
    out = build(*values)
    out.backward()

    def scalar_fn(xs: list[float]) -> float:
        result = build(*[Value(v) for v in xs])
        return result.data

    expected = numerical_grad(scalar_fn, inputs)
    for value, exp in zip(values, expected):
        assert value.grad == pytest.approx(exp, abs=TOL)


def test_diamond_graph_accumulation():
    """Gradients accumulate correctly when a node feeds several branches."""
    x0, y0 = 1.7, -0.6

    def build(x, y):
        return (x * y + x ** 2) / (y + 3.0)

    x = Value(x0)
    y = Value(y0)
    z = build(x, y)
    z.backward()

    assert z.data == pytest.approx((x0 * y0 + x0 ** 2) / (y0 + 3.0))

    expected = numerical_grad(
        lambda xs: (xs[0] * xs[1] + xs[0] ** 2) / (xs[1] + 3.0),
        [x0, y0],
    )
    assert x.grad == pytest.approx(expected[0], abs=TOL)
    assert y.grad == pytest.approx(expected[1], abs=TOL)


def test_transcendentals_and_activations():
    """Chained activations and reverse operators match numerical gradients."""

    def chain(x):
        return ((x * 0.5).tanh() + 1.5).log()

    def exp_relu(x):
        return (x.exp() * 0.3).relu() + (2.0 / (x + 4.0))

    def reverse_ops(x):
        return (3.0 - x) * (2.0 / x) + (5.0 + x)

    def combined(x, y):
        return (3.0 - x).relu().exp() * (y * 0.2).tanh() + (5.0 + y).log()

    _check_against_numerical(chain, [0.8])
    _check_against_numerical(exp_relu, [0.4])
    _check_against_numerical(reverse_ops, [1.3])
    _check_against_numerical(lambda x: 3.0 - x, [0.9])
    _check_against_numerical(lambda x: 2.0 / x, [1.7])
    _check_against_numerical(lambda x: 5.0 + x, [-2.0])
    _check_against_numerical(combined, [1.1, 0.7])


def test_relu_kink_and_negative_branch():
    """ReLU yields gradient 0.0 for negative inputs and 1.0 for positive."""
    neg = Value(-2.5)
    neg_out = neg.relu()
    neg_out.backward()
    assert neg_out.data == 0.0
    assert neg.grad == 0.0

    pos = Value(3.5)
    pos_out = pos.relu()
    pos_out.backward()
    assert pos_out.data == 3.5
    assert pos.grad == 1.0


def test_mlp_forward_backward_and_zero_grad():
    """MLP backward populates every parameter gradient; zero_grad resets."""
    model = MLP(2, [4, 1])
    params = model.parameters()
    assert len(params) == (2 * 4 + 4) + (4 * 1 + 1)

    out = model([Value(0.7), Value(-1.3)])
    if isinstance(out, (list, tuple)):
        assert len(out) == 1
        out = out[0]
    assert isinstance(out, Value)
    assert math.isfinite(out.data)

    out.backward()
    assert all(p.grad != 0.0 for p in params)

    model.zero_grad()
    assert all(p.grad == 0.0 for p in model.parameters())


def test_invalid_type_errors():
    """Unsupported operand types and domain violations raise errors."""
    with pytest.raises(TypeError):
        Value(2.0) + "bad"
    with pytest.raises(TypeError):
        Value(2.0) ** "bad"
    with pytest.raises(ValueError):
        Value(-1.0).log()