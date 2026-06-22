"""Qubit routing: insert SWAP gates to satisfy coupling constraints."""

from __future__ import annotations

from dataclasses import dataclass, replace
from collections import deque

from qorch.ir import Circuit, Gate


@dataclass(frozen=True)
class CouplingMap:
    """Directed coupling graph: edges are (control, target) pairs."""

    edges: tuple[tuple[int, int], ...]


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


def _swap_cost(adj: dict[int, set[int]], q0: int, q1: int) -> int:
    """Minimum SWAP gates needed to bring q0 and q1 adjacent."""
    return len(_shortest_path(adj, q0, q1)) - 1


def _apply_swap(state: dict[int, int], q0: int, q1: int) -> None:
    """Swap logical qubits in the mapping."""
    state[q0], state[q1] = state[q1], state[q0]


def route(circuit: Circuit, coupling_map: CouplingMap) -> Circuit:
    """Route a circuit to satisfy coupling constraints by inserting SWAPs.

    Uses a greedy SWAP insertion strategy: for each 2-qubit gate, if the
    physical qubits are not directly connected, find the shortest path and
    insert SWAP gates to bring them adjacent.

    Returns a new Circuit with SWAP gates inserted.
    """
    if not coupling_map.edges:
        return circuit  # all-to-all, no routing needed

    adj = _build_adjacency(coupling_map.edges)
    edges_set = set(coupling_map.edges)

    # Physical → logical mapping (identity start)
    physical: dict[int, int] = {i: i for i in range(circuit.num_qubits)}

    new_gates: list[Gate] = []

    def _logical_to_physical(lq: int) -> int:
        for phys, log in physical.items():
            if log == lq:
                return phys
        return lq  # fallback

    for g in circuit.gates:
        if len(g.qubits) < 2:
            new_gates.append(g)
            continue

        q0_log = g.qubits[0]
        q1_log = g.qubits[1]
        q0_phys = _logical_to_physical(q0_log)
        q1_phys = _logical_to_physical(q1_log)

        # Check if directly connected (either direction)
        if (q0_phys, q1_phys) in edges_set:
            new_gates.append(g)
            continue

        # Find swap path
        path = _shortest_path(adj, q0_phys, q1_phys)
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            if (a, b) not in edges_set and (b, a) not in edges_set:
                continue
            new_gates.append(Gate("swap", (a, b)))
            _apply_swap(physical, a, b)

        # Now qubits should be adjacent; re-resolve physical qubits
        q0_phys = _logical_to_physical(q0_log)
        q1_phys = _logical_to_physical(q1_log)
        new_gates.append(
            Gate(g.name, (q0_phys, q1_phys), g.params)
        )

    return replace(circuit, gates=tuple(new_gates))
