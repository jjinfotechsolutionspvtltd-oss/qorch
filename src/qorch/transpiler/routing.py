"""Qubit routing: insert SWAP gates to satisfy coupling constraints.

Supports both topology-only routing (BFS) and noise-aware routing that
prefers paths through higher-fidelity qubits.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from collections import deque

from qorch.ir import Circuit, Gate


@dataclass(frozen=True)
class CouplingMap:
    """Directed coupling graph: edges are (control, target) pairs."""

    edges: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class QubitQuality:
    """Per-qubit quality metrics for noise-aware routing."""

    gate_fidelity: float  # single-qubit gate fidelity (0-1)
    readout_fidelity: float = 1.0
    t1: float = 0.0  # microseconds
    t2: float = 0.0


def _build_adjacency(edges: tuple[tuple[int, int], ...]) -> dict[int, set[int]]:
    adj: dict[int, set[int]] = {}
    for c, t in edges:
        adj.setdefault(c, set()).add(t)
        adj.setdefault(t, set()).add(c)
    return adj


def _shortest_path(adj: dict[int, set[int]], start: int, end: int) -> list[int]:
    """BFS shortest path between two qubits."""
    if start == end:
        return [start]
    q: deque[tuple[int, list[int]]] = deque()
    q.append((start, [start]))
    visited = {start}
    while q:
        node, path = q.popleft()
        for neighbor in adj.get(node, set()):
            if neighbor not in visited:
                if neighbor == end:
                    return path + [neighbor]
                visited.add(neighbor)
                q.append((neighbor, path + [neighbor]))
    raise ValueError(f"no path between qubits {start} and {end}")


def _apply_swap(state: dict[int, int], q0: int, q1: int) -> None:
    """Swap logical qubits in the mapping."""
    state[q0], state[q1] = state[q1], state[q0]


def _best_swap_path(
    adj: dict[int, set[int]],
    edges_set: set[tuple[int, int]],
    start: int,
    end: int,
    quality: dict[int, QubitQuality] | None = None,
) -> list[int]:
    """Find the best path between start and end for SWAP insertion.

    With quality info, prefers paths through high-fidelity qubits.
    Without quality, uses BFS shortest path.
    """
    if quality is None:
        return _shortest_path(adj, start, end)

    # Weighted shortest path: edge cost = 1 - avg qubit fidelity
    weights: dict[tuple[int, int], float] = {}
    for a in adj:
        for b in adj.get(a, set()):
            f_a = quality.get(a, QubitQuality(1.0)).gate_fidelity
            f_b = quality.get(b, QubitQuality(1.0)).gate_fidelity
            weights[(a, b)] = 1.0 - 0.5 * (f_a + f_b)

    # Dijkstra
    import heapq
    INF = float("inf")
    dist: dict[int, float] = {start: 0.0}
    prev: dict[int, int | None] = {start: None}
    pq: list[tuple[float, int]] = [(0.0, start)]
    while pq:
        d, node = heapq.heappop(pq)
        if node == end:
            path: list[int] = []
            cur: int | None = node
            while cur is not None:
                path.append(cur)
                nxt = prev.get(cur)
                if nxt is None:
                    break
                cur = nxt
            return list(reversed(path))
        if d > dist.get(node, INF):
            continue  # pragma: no cover
        for neighbor in adj.get(node, set()):
            w = weights.get((node, neighbor), 1.0)
            nd = d + w
            if nd < dist.get(neighbor, INF):
                dist[neighbor] = nd
                prev[neighbor] = node
                heapq.heappush(pq, (nd, neighbor))
    raise ValueError(f"no path between qubits {start} and {end}")


def route(
    circuit: Circuit,
    coupling_map: CouplingMap,
    qubit_quality: dict[int, QubitQuality] | None = None,
) -> Circuit:
    """Route a circuit to satisfy coupling constraints by inserting SWAPs.

    Uses a greedy SWAP insertion strategy: for each 2-qubit gate, if the
    physical qubits are not directly connected, find the shortest path and
    insert SWAP gates to bring them adjacent.

    When ``qubit_quality`` is provided, the path is weighted by gate fidelity
    so that SWAPs prefer higher-quality qubits.

    Returns a new Circuit with SWAP gates inserted.
    """
    if not coupling_map.edges:
        return circuit

    adj = _build_adjacency(coupling_map.edges)
    edges_set = set(coupling_map.edges)

    physical: dict[int, int] = {i: i for i in range(circuit.num_qubits)}

    new_gates: list[Gate] = []

    def _logical_to_physical(lq: int) -> int:
        for phys, log in physical.items():
            if log == lq:
                return phys
        # All qubits are initialized in physical mapping; fallback shouldn't be needed
        return lq  # pragma: no cover

    for g in circuit.gates:
        if len(g.qubits) < 2:
            new_gates.append(g)
            continue

        q0_log = g.qubits[0]
        q1_log = g.qubits[1]
        q0_phys = _logical_to_physical(q0_log)
        q1_phys = _logical_to_physical(q1_log)

        if (q0_phys, q1_phys) in edges_set:
            new_gates.append(g)
            continue

        path = _best_swap_path(adj, edges_set, q0_phys, q1_phys, qubit_quality)
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            if (a, b) not in edges_set and (b, a) not in edges_set:
                continue  # pragma: no cover
            new_gates.append(Gate("swap", (a, b)))
            _apply_swap(physical, a, b)

        q0_phys = _logical_to_physical(q0_log)
        q1_phys = _logical_to_physical(q1_log)
        new_gates.append(Gate(g.name, (q0_phys, q1_phys), g.params))

    return replace(circuit, gates=tuple(new_gates))
