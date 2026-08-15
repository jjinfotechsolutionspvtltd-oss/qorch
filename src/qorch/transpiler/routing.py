"""Qubit routing: insert SWAP gates to satisfy coupling constraints.

Supports both topology-only routing (BFS) and noise-aware routing that
prefers paths through higher-fidelity qubits.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from collections import deque

from qorch.ir import Circuit, Gate, Measure, Operation, with_qubits


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

    Routing is **semantically transparent**: single-qubit gates and the final
    ``measured`` read-out are remapped through the running logical→physical
    permutation, so the output distribution over *logical* qubits is unchanged
    (only the physical wiring differs). The final layout is recorded so callers
    can recover which physical wire each logical qubit ended on.

    Dynamic circuits route too: mid-circuit ``Measure``/``Reset`` follow their
    qubit onto its current wire, and classical conditions ride along untouched.
    Inserted SWAPs are deliberately *unconditional* even when they precede a
    conditioned gate — the layout permutation must stay deterministic, or the
    router would no longer know where a logical qubit lives after a branch.

    Returns a new Circuit with SWAP gates inserted and read-out remapped.
    """
    if not coupling_map.edges:
        return circuit

    adj = _build_adjacency(coupling_map.edges)
    edges_set = set(coupling_map.edges)

    physical: dict[int, int] = {i: i for i in range(circuit.num_qubits)}

    new_gates: list[Operation] = []

    def _logical_to_physical(lq: int) -> int:
        for phys, log in physical.items():
            if log == lq:
                return phys
        return lq  # pragma: no cover

    for g in circuit.gates:
        if len(g.qubits) < 2:
            # remap the 1-qubit op (gate, measure, or reset) onto its current wire
            phys = _logical_to_physical(g.qubits[0])
            new_gates.append(with_qubits(g, (phys,)))
            continue

        q0_log = g.qubits[0]
        q1_log = g.qubits[1]
        q0_phys = _logical_to_physical(q0_log)
        q1_phys = _logical_to_physical(q1_log)

        if (q0_phys, q1_phys) in edges_set:
            new_gates.append(with_qubits(g, (q0_phys, q1_phys)))
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
        new_gates.append(with_qubits(g, (q0_phys, q1_phys)))

    # Remap read-out: to read logical qubit q, measure the physical wire it
    # now lives on. Output-bitstring order stays in logical order.
    new_measured = tuple(_logical_to_physical(q) for q in circuit.readout_qubits)
    return replace(circuit, gates=tuple(new_gates), measured=new_measured)


# ── SabreSWAP lookahead router ───────────────────────────────────────────


def _build_dag_deps(
    gates: tuple[Operation, ...],
) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
    """Dependency DAG over ops: quantum wires *and* classical bits.

    The router is free to reorder anything the DAG leaves unordered, so every
    real ordering constraint has to appear here. Beyond same-qubit ordering,
    classical bits impose the three data hazards: a conditioned op reads bits a
    ``Measure`` writes (RAW), and a ``Measure`` overwriting a bit must stay after
    everything that read (WAR) or wrote (WAW) it. Without these, feed-forward
    corrections could float ahead of the measurement that decides them.
    """
    last_gate_on_qubit: dict[int, int] = {}
    last_writer: dict[int, int] = {}
    readers_since_write: dict[int, set[int]] = {}
    preds: dict[int, set[int]] = {}
    succs: dict[int, set[int]] = {}

    def _link(pred: int, node: int) -> None:
        preds[node].add(pred)
        succs[pred].add(node)

    for i, g in enumerate(gates):
        preds[i] = set()
        succs[i] = set()
        for q in g.qubits:
            if q in last_gate_on_qubit:
                _link(last_gate_on_qubit[q], i)
            last_gate_on_qubit[q] = i

        reads = {cbit for cbit, _value in (g.condition or ())}
        for c in reads:
            if c in last_writer:
                _link(last_writer[c], i)          # RAW
        if isinstance(g, Measure):
            if g.cbit in last_writer:
                _link(last_writer[g.cbit], i)     # WAW
            for r in readers_since_write.get(g.cbit, set()):
                _link(r, i)                       # WAR
            last_writer[g.cbit] = i
            readers_since_write[g.cbit] = set()
        for c in reads:
            readers_since_write.setdefault(c, set()).add(i)
    return preds, succs


def _front_layer(
    gates: tuple[Operation, ...],
    preds: dict[int, set[int]],
    executed: set[int],
) -> set[int]:
    front: set[int] = set()
    for i in range(len(gates)):
        if i in executed:
            continue
        if preds[i].issubset(executed):
            front.add(i)
    return front


def _try_execute_front(
    gates: tuple[Operation, ...],
    front: set[int],
    executed: set[int],
    edges_set: set[tuple[int, int]],
    phys_map: dict[int, int],
    output: list[Operation],
) -> int:
    n = 0
    for i in sorted(front):
        g = gates[i]
        if len(g.qubits) < 2:
            phys = _logical_to_physical_inv(phys_map, g.qubits[0])
            output.append(with_qubits(g, (phys,)))
            executed.add(i)
            front.discard(i)
            n += 1
        else:
            q0 = _logical_to_physical_inv(phys_map, g.qubits[0])
            q1 = _logical_to_physical_inv(phys_map, g.qubits[1])
            if (q0, q1) in edges_set:
                output.append(with_qubits(g, (q0, q1)))
                executed.add(i)
                front.discard(i)
                n += 1
    return n


def _logical_to_physical_inv(m: dict[int, int], logical: int) -> int:
    for phys, log in m.items():
        if log == logical:
            return phys
    return logical  # pragma: no cover


def _extended_set(
    gates: tuple[Operation, ...],
    front: set[int],
    lookahead: int,
) -> set[int]:
    seen: set[int] = set(front)
    ext: set[int] = set()
    for i in range(len(gates)):
        if i in seen:
            continue
        seen.add(i)
        ext.add(i)
        if len(ext) >= lookahead:
            break
    return ext


def _swap_score(
    edge: tuple[int, int],
    gates: tuple[Operation, ...],
    front: set[int],
    extended: set[int],
    edges_set: set[tuple[int, int]],
    phys_map: dict[int, int],
    decay: float,
    dist_cache: dict[tuple[int, int], float],
) -> float:
    a, b = edge
    if a not in phys_map or b not in phys_map:
        return -1.0
    _apply_swap(phys_map, a, b)
    f_score = 0
    f_dist = 0.0
    for i in front:
        g = gates[i]
        if len(g.qubits) >= 2:
            q0 = _logical_to_physical_inv(phys_map, g.qubits[0])
            q1 = _logical_to_physical_inv(phys_map, g.qubits[1])
            if (q0, q1) in edges_set:
                f_score += 1
            else:
                f_dist -= dist_cache.get((q0, q1), 5)
    e_score = 0
    for i in extended:
        g = gates[i]
        if len(g.qubits) >= 2:
            q0 = _logical_to_physical_inv(phys_map, g.qubits[0])
            q1 = _logical_to_physical_inv(phys_map, g.qubits[1])
            if (q0, q1) in edges_set:
                e_score += 1
    _apply_swap(phys_map, a, b)
    if f_score + e_score > 0:
        return decay * f_score + (1.0 - decay) * e_score
    return f_dist


def _build_dist_cache(
    adj: dict[int, set[int]],
    nq: int,
    quality: dict[int, QubitQuality] | None = None,
) -> dict[tuple[int, int], float]:
    """Precompute shortest distances between all pairs.

    With ``quality``, edge cost = 1 - avg(gate_fidelity of endpoints),
    so high-fidelity paths have lower cost. Without quality, each hop costs 1.
    """
    cache: dict[tuple[int, int], float] = {}
    for src in range(nq):
        if src not in adj:
            cache[(src, src)] = 0.0
            continue

        import heapq
        INF = float("inf")
        dist: dict[int, float] = {src: 0.0}
        pq: list[tuple[float, int]] = [(0.0, src)]
        while pq:
            d, node = heapq.heappop(pq)
            if d > dist.get(node, INF):
                continue
            cache[(src, node)] = d
            for nb in adj.get(node, set()):
                if quality:
                    f_a = quality.get(node, QubitQuality(1.0)).gate_fidelity
                    f_b = quality.get(nb, QubitQuality(1.0)).gate_fidelity
                    w = 1.0 - 0.5 * (f_a + f_b)
                else:
                    w = 1.0
                nd = d + w
                if nd < dist.get(nb, INF):
                    dist[nb] = nd
                    heapq.heappush(pq, (nd, nb))
    return cache


def route_lookahead(
    circuit: Circuit,
    coupling_map: CouplingMap,
    lookahead: int = 20,
    decay: float = 0.5,
    qubit_quality: dict[int, QubitQuality] | None = None,
) -> Circuit:
    """SabreSWAP-style routing with a lookahead window.

    Dynamic circuits are supported: the dependency DAG carries classical-bit
    hazards as well as qubit ordering, so measurements and the feed-forward
    gates that consume them keep their relative order under reordering.
    """
    if not coupling_map.edges or not circuit.gates:
        return circuit

    edges_set = set(coupling_map.edges)
    adj = _build_adjacency(coupling_map.edges)

    for g in circuit.gates:
        if len(g.qubits) >= 2:
            for q in g.qubits:
                if q not in adj:
                    raise ValueError(
                        f"qubit {q} is not in the coupling map "
                        f"(qubits with edges: {sorted(adj)})"
                    )

    gates = circuit.gates
    phys_map: dict[int, int] = {i: i for i in range(circuit.num_qubits)}
    output: list[Operation] = []

    preds, _succs = _build_dag_deps(gates)
    executed: set[int] = set()
    dist_cache = _build_dist_cache(adj, circuit.num_qubits, quality=qubit_quality)

    while len(executed) < len(gates):
        front = _front_layer(gates, preds, executed)

        n_ex = _try_execute_front(gates, front, executed, edges_set, phys_map, output)
        if n_ex > 0:
            continue

        extended = _extended_set(gates, front, lookahead)

        best_edge: tuple[int, int] | None = None
        best_score = -1e9
        for a, b in edges_set:
            score = _swap_score((a, b), gates, front, extended, edges_set, phys_map, decay, dist_cache)
            if score > best_score:
                best_score = score
                best_edge = (a, b)

        if best_edge is None:
            raise ValueError("no candidate SWAP edge found — is the coupling map connected?")

        output.append(Gate("swap", (best_edge[0], best_edge[1])))
        _apply_swap(phys_map, best_edge[0], best_edge[1])

    # Remap read-out through the final logical→physical permutation so the
    # routed circuit is semantically transparent (logical-order bitstrings).
    new_measured = tuple(
        _logical_to_physical_inv(phys_map, q) for q in circuit.readout_qubits
    )
    return replace(circuit, gates=tuple(output), measured=new_measured)
