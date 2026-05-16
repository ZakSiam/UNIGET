"""Canonical ordering helpers for deterministic graph tokenization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from torch_geometric.data import Data


AttrTuple = Tuple[int, ...]
EdgeRecord = Tuple[int, int, AttrTuple]


@dataclass
class CanonicalizationResult:
    """Canonical graph order plus deterministic edge linearization metadata."""

    order: List[int]
    path: List[Tuple[int, int]]
    real_edges: List[Tuple[int, int]]
    jump_edges: List[Tuple[int, int]]


def _as_attr_tuple(values) -> AttrTuple:
    if values is None:
        return ()
    if hasattr(values, "detach"):
        values = values.detach().cpu()
    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, list):
        return tuple(int(v) for v in values)
    return (int(values),)


def _node_attrs(graph: Data) -> List[AttrTuple]:
    x = getattr(graph, "x", None)
    if x is None:
        return [() for _ in range(int(graph.num_nodes))]
    return [_as_attr_tuple(x[idx]) for idx in range(int(graph.num_nodes))]


def _unique_undirected_edges(graph: Data) -> List[EdgeRecord]:
    edge_attrs = getattr(graph, "edge_attr", None)
    records: Dict[Tuple[int, int], AttrTuple] = {}
    src_nodes = graph.edge_index[0].tolist()
    dst_nodes = graph.edge_index[1].tolist()
    for idx, (src, dst) in enumerate(zip(src_nodes, dst_nodes)):
        if src == dst:
            key = (int(src), int(dst))
        else:
            key = tuple(sorted((int(src), int(dst))))
        attr = _as_attr_tuple(edge_attrs[idx]) if edge_attrs is not None else ()
        previous = records.get(key)
        if previous is None or attr < previous:
            records[key] = attr
    return [(src, dst, attr) for (src, dst), attr in sorted(records.items())]


def _build_adjacency(num_nodes: int, edges: Iterable[EdgeRecord]):
    adjacency: List[List[Tuple[int, AttrTuple]]] = [[] for _ in range(num_nodes)]
    for src, dst, attr in edges:
        adjacency[src].append((dst, attr))
        if src != dst:
            adjacency[dst].append((src, attr))
    for neighbors in adjacency:
        neighbors.sort()
    return adjacency


def _compress_signatures(signatures: Sequence[Tuple]) -> Tuple[int, ...]:
    palette = {sig: idx for idx, sig in enumerate(sorted(set(signatures)))}
    return tuple(palette[sig] for sig in signatures)


def _refine_colors(
    colors: Tuple[int, ...],
    adjacency: Sequence[Sequence[Tuple[int, AttrTuple]]],
) -> Tuple[int, ...]:
    while True:
        signatures = []
        for node, color in enumerate(colors):
            neighborhood = tuple(
                sorted((colors[neighbor], edge_attr) for neighbor, edge_attr in adjacency[node])
            )
            signatures.append((color, neighborhood))
        refined = _compress_signatures(signatures)
        if refined == colors:
            return refined
        colors = refined


def _certificate(
    order: Sequence[int],
    node_attrs: Sequence[AttrTuple],
    edges: Sequence[EdgeRecord],
) -> Tuple:
    rank = {node: idx for idx, node in enumerate(order)}
    node_part = tuple(node_attrs[node] for node in order)
    edge_part = tuple(
        sorted(
            (
                min(rank[src], rank[dst]),
                max(rank[src], rank[dst]),
                attr,
            )
            for src, dst, attr in edges
        )
    )
    return node_part, edge_part


def canonical_node_order(graph: Data, *, search_budget: int = 100_000) -> List[int]:
    """Return an isomorphism-consistent node order for a small attributed graph."""

    num_nodes = int(graph.num_nodes)
    node_attrs = _node_attrs(graph)
    edges = _unique_undirected_edges(graph)
    adjacency = _build_adjacency(num_nodes, edges)
    initial_colors = _compress_signatures(node_attrs)
    states_visited = 0

    def search(colors: Tuple[int, ...]):
        nonlocal states_visited
        states_visited += 1
        if states_visited > search_budget:
            raise RuntimeError(
                "canonical search budget exceeded; increase search_budget for this graph"
            )

        colors = _refine_colors(colors, adjacency)
        classes: Dict[int, List[int]] = {}
        for node, color in enumerate(colors):
            classes.setdefault(color, []).append(node)

        tied_classes = [
            (color, members) for color, members in classes.items() if len(members) > 1
        ]
        if not tied_classes:
            order = [node for node, _ in sorted(enumerate(colors), key=lambda item: item[1])]
            return _certificate(order, node_attrs, edges), order

        _, target = min(tied_classes, key=lambda item: (len(item[1]), item[0]))
        individualized_color = max(colors) + 1
        best = None
        for node in target:
            individualized = list(colors)
            individualized[node] = individualized_color
            candidate = search(tuple(individualized))
            if best is None or candidate[0] < best[0]:
                best = candidate
        return best

    _, order = search(initial_colors)
    return order


def canonicalize_graph(graph: Data) -> CanonicalizationResult:
    """Create a deterministic edge-once path suitable for stacked tokenization."""

    order = canonical_node_order(graph)
    rank = {node: idx for idx, node in enumerate(order)}
    edge_records = _unique_undirected_edges(graph)
    sorted_edges = sorted(
        (
            min((src, dst), key=rank.get),
            max((src, dst), key=rank.get),
            attr,
        )
        for src, dst, attr in edge_records
    )
    sorted_edges.sort(key=lambda item: (rank[item[0]], rank[item[1]], item[2]))

    path: List[Tuple[int, int]] = []
    real_edges: List[Tuple[int, int]] = []
    jump_edges: List[Tuple[int, int]] = []
    current = None
    seen_nodes = set()

    for src, dst, _ in sorted_edges:
        if current is not None and current != src:
            path.append((current, src))
            jump_edges.append((current, src))
        path.append((src, dst))
        real_edges.append((src, dst))
        current = dst
        seen_nodes.update((src, dst))

    for node in order:
        if node in seen_nodes:
            continue
        if current is not None and current != node:
            path.append((current, node))
            jump_edges.append((current, node))
        current = node
        seen_nodes.add(node)

    return CanonicalizationResult(
        order=order,
        path=path,
        real_edges=real_edges,
        jump_edges=jump_edges,
    )
