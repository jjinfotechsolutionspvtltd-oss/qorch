"""Layout: choose *where* logical qubits start before routing moves them.

Routing has always begun from the identity placement — logical qubit 0 on
physical wire 0, and so on — which is an arbitrary choice that routing then pays
to correct. If two logical qubits interact constantly but start at opposite ends
of a line, every one of their gates costs SWAPs that a better starting placement
would have avoided outright.

The passes here pick that placement. They are ordinary permutations applied
before routing, so they compose with it rather than competing: layout decides
where qubits *begin*, routing handles where they must *move*.

Scope: a layout here permutes the circuit's own qubits. Selecting a good
*subset* of a device larger than the circuit is a further step — it needs the
circuit widened to the device's width first — and is deliberately not attempted,
because a layout that silently changed a circuit's qubit count would be a
surprising thing for a compiler pass to do.
"""

from __future__ import annotations

from dataclasses import replace

from qorch.ir import Circuit, Operation, with_qubits
from qorch.transpiler.routing import CouplingMap, QubitQuality

# layout[logical] = physical wire the logical qubit starts on.
Layout = tuple[int, ...]


def apply_layout(circuit: Circuit, layout: Layout) -> Circuit:
    """Relabel ``circuit`` so logical qubit ``q`` acts on wire ``layout[q]``.

    Read-out follows: a caller asking for logical qubit ``q`` still gets logical
    qubit ``q``, measured on whichever wire now carries it. The circuit is
    semantically identical — only the wiring changed.
    """
    if len(layout) != circuit.num_qubits:
        raise ValueError(
            f"layout has {len(layout)} entries for a {circuit.num_qubits}-qubit "
            f"circuit"
        )
    if sorted(layout) != list(range(circuit.num_qubits)):
        raise ValueError(f"layout is not a permutation of the qubits: {layout}")

    ops: list[Operation] = [
        with_qubits(op, tuple(layout[q] for q in op.qubits)) for op in circuit.gates
    ]
    measured = tuple(layout[q] for q in circuit.readout_qubits)
    return replace(circuit, gates=tuple(ops), measured=measured)


def interaction_graph(circuit: Circuit) -> dict[tuple[int, int], int]:
    """How often each pair of logical qubits interacts, keyed by sorted pair.

    This is what a layout is trying to satisfy: pairs that interact often want
    to sit next to each other.
    """
    weights: dict[tuple[int, int], int] = {}
    for op in circuit.gates:
        if len(op.qubits) == 2:
            pair = (min(op.qubits), max(op.qubits))
            weights[pair] = weights.get(pair, 0) + 1
    return weights


def trivial_layout(circuit: Circuit, *_args, **_kwargs) -> Layout:
    """The identity placement — what routing used to assume unconditionally."""
    return tuple(range(circuit.num_qubits))


def _physical_adjacency(
    coupling_map: CouplingMap, num_qubits: int
) -> dict[int, set[int]]:
    adjacency: dict[int, set[int]] = {q: set() for q in range(num_qubits)}
    for a, b in coupling_map.edges:
        if a < num_qubits and b < num_qubits:
            adjacency[a].add(b)
            adjacency[b].add(a)
    return adjacency


def dense_layout(
    circuit: Circuit,
    coupling_map: CouplingMap,
    qubit_quality: dict[int, QubitQuality] | None = None,
) -> Layout:
    """Place heavily-interacting logical qubits on well-connected physical ones.

    Greedy, and deliberately so — optimal placement is graph isomorphism, and a
    good greedy seed is worth far more to routing than an exact answer is worth
    the search. The busiest logical qubit is seeded onto the best physical one,
    then each remaining logical qubit is placed, in order of how strongly it
    interacts with what is already placed, onto the free physical qubit adjacent
    to its partners.

    With ``qubit_quality``, "best" accounts for gate fidelity as well as degree,
    so a well-connected but noisy qubit loses to a slightly less connected clean
    one — the noise-adaptive variant of the same pass.
    """
    n = circuit.num_qubits
    weights = interaction_graph(circuit)
    if not weights:
        return trivial_layout(circuit)

    adjacency = _physical_adjacency(coupling_map, n)

    def physical_score(p: int) -> float:
        score = float(len(adjacency[p]))
        if qubit_quality is not None:
            # Degree is the dominant term; fidelity breaks ties and demotes a
            # well-connected qubit that is measurably worse than its neighbours.
            score += 4.0 * qubit_quality.get(p, QubitQuality(1.0)).gate_fidelity
        return score

    logical_weight: dict[int, int] = {q: 0 for q in range(n)}
    for (a, b), w in weights.items():
        logical_weight[a] += w
        logical_weight[b] += w

    # Deterministic ordering: heaviest first, ties by index.
    logical_order = sorted(range(n), key=lambda q: (-logical_weight[q], q))
    physical_order = sorted(range(n), key=lambda p: (-physical_score(p), p))

    placement: dict[int, int] = {}
    used: set[int] = set()

    def affinity(logical: int, candidate: int) -> tuple[int, float, int]:
        """Rank a candidate wire for a logical qubit: satisfied pairs, then quality."""
        satisfied = 0
        for other, other_physical in placement.items():
            pair = (min(logical, other), max(logical, other))
            weight = weights.get(pair, 0)
            if weight and other_physical in adjacency[candidate]:
                satisfied += weight
        return (satisfied, physical_score(candidate), -candidate)

    for logical in logical_order:
        if not placement:
            chosen = physical_order[0]
        else:
            free = [p for p in range(n) if p not in used]
            chosen = max(free, key=lambda p: affinity(logical, p))
        placement[logical] = chosen
        used.add(chosen)

    return tuple(placement[q] for q in range(n))


def noise_adaptive_layout(
    circuit: Circuit,
    coupling_map: CouplingMap,
    qubit_quality: dict[int, QubitQuality] | None = None,
) -> Layout:
    """:func:`dense_layout`, with qubit fidelity weighed alongside connectivity."""
    return dense_layout(circuit, coupling_map, qubit_quality or {})


LAYOUT_METHODS = {
    "trivial": trivial_layout,
    "dense": dense_layout,
    "noise-adaptive": noise_adaptive_layout,
}


def select_layout(
    method: str,
    circuit: Circuit,
    coupling_map: CouplingMap,
    qubit_quality: dict[int, QubitQuality] | None = None,
) -> Layout:
    """Run a named layout method, with an error listing the real options."""
    try:
        chooser = LAYOUT_METHODS[method]
    except KeyError:
        raise ValueError(
            f"unknown layout method {method!r}; options: {sorted(LAYOUT_METHODS)}"
        ) from None
    return chooser(circuit, coupling_map, qubit_quality)
