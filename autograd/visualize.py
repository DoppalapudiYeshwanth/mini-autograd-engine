"""Visualization utilities for rendering autograd computation graphs."""

from __future__ import annotations
from pathlib import Path
from graphviz import Digraph
from autograd.engine import Value


def _escape_record(text: str) -> str:
    """Escapes characters that have special meaning in Graphviz record labels.
    Args:
        text: Raw label text.
    Returns:
        The text with record-label metacharacters backslash-escaped.
    """
    for char in ("\\", "{", "}", "|", "<", ">"):
        text = text.replace(char, "\\" + char)
    return text


def trace_graph(root: Value) -> tuple[set[Value], set[tuple[Value, Value]]]:
    """Collects all nodes and edges reachable backwards from ``root``.
    Args:
        root: The output node from which to traverse the graph.
    Returns:
        A tuple ``(nodes, edges)`` where ``nodes`` is the set of unique
        ``Value`` nodes and ``edges`` is the set of directed
        ``(parent, child)`` pairs, with ``parent`` an operand and ``child``
        the result computed from it.
    """
    nodes: set[Value] = set()
    edges: set[tuple[Value, Value]] = set()
    visited: set[Value] = set()
    stack: list[Value] = [root]

    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        nodes.add(node)
        for parent in node._prev:
            edges.add((parent, node))
            if parent not in visited:
                stack.append(parent)

    return nodes, edges


def build_dot(
    root: Value, graph_format: str = "svg", rankdir: str = "LR"
) -> Digraph:
    """Builds a Graphviz ``Digraph`` for the computation graph of ``root``.
    Args:
        root: The output node of the computation graph.
        graph_format: Output format passed to Graphviz (e.g. ``"svg"``).
        rankdir: Graph layout direction (e.g. ``"LR"`` or ``"TB"``).
    Returns:
        A configured ``Digraph`` containing value nodes, operation nodes,
        and the edges connecting them.
    """
    dot = Digraph(
        format=graph_format,
        graph_attr={"rankdir": rankdir, "bgcolor": "white"},
    )
    nodes, edges = trace_graph(root)

    for node in nodes:
        node_id = str(id(node))
        if node.label:
            label = (
                f"{{ {_escape_record(node.label)} | data {node.data:.4f} "
                f"| grad {node.grad:.4f} }}"
            )
        else:
            label = f"{{ data {node.data:.4f} | grad {node.grad:.4f} }}"

        dot.node(
            node_id,
            label=label,
            shape="record",
            style="filled",
            fillcolor="#F8FAFC",
            color="#475569",
            fontname="Helvetica",
        )

        if node._op:
            op_id = f"{id(node)}_op"
            dot.node(
                op_id,
                label=node._op,
                shape="circle",
                style="filled",
                fillcolor="#E2E8F0",
                fontname="Helvetica",
            )
            dot.edge(op_id, node_id)

    for parent, child in edges:
        dot.edge(str(id(parent)), f"{id(child)}_op")

    return dot


def render_graph(
    root: Value,
    output_path: str | Path = "computation_graph",
    graph_format: str = "svg",
    view: bool = False,
) -> Path:
    """Renders the computation graph of ``root`` to an image file.
    Args:
        root: The output node of the computation graph.
        output_path: Output path without extension; Graphviz appends the
            format suffix.
        graph_format: Output image format (e.g. ``"svg"`` or ``"png"``).
        view: Whether to open the rendered file with the default viewer.
    Returns:
        The resolved path to the generated image file.
    """
    dot = build_dot(root, graph_format=graph_format)
    rendered = dot.render(filename=str(output_path), cleanup=True, view=view)
    return Path(rendered).resolve()