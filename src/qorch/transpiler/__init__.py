"""Transpiler: gate decomposition + qubit routing for Indian-native gate sets."""

from qorch.transpiler.gateset import (
    IndianGateSet,
    IIT_JODHPUR_ION_TRAP,
    TIFR_SUPERCONDUCTING,
    DRDO_MIRAI,
    CLIFFORD_T,
)
from qorch.transpiler.decompose import DECOMPOSITION_RULES, decompose, decompose_to_clifford_t
from qorch.transpiler.optimizer import optimize
from qorch.transpiler.routing import (
    CouplingMap,
    QubitQuality,
    RoutingResult,
    fix_gate_directions,
    route,
    route_lookahead,
    route_lookahead_with_layout,
    route_with_layout,
)

from dataclasses import dataclass

from qorch.ir import Circuit


@dataclass(frozen=True)
class TranspileResult:
    """A transpiled circuit plus where each logical qubit physically ended up.

    ``final_layout[q]`` is the physical wire holding logical qubit ``q`` after
    the pipeline runs. Needed to correlate results with per-qubit calibration
    data, attribute errors to specific hardware qubits, or debug a layout —
    none of which the circuit alone can tell you, because routing's permutation
    is not otherwise recoverable from the output.
    """

    circuit: Circuit
    final_layout: tuple[int, ...]


def _lower_to_target(
    circuit: Circuit,
    target: IndianGateSet,
    coupling_map: CouplingMap | None,
) -> Circuit:
    """Lower every op to ``target.basis_gates`` without breaking connectivity.

    Run *after* routing, not just before it: routing emits ``swap`` gates the
    first decomposition never saw, and a SWAP is native to almost no real
    target. Skipping this is how a ``swap`` used to survive into the output of a
    cx/sx/rz/x target.

    Order matters. Decompose first so SWAPs become CX triples, *then* fix
    directions, because it is that expansion which introduces the reversed CX a
    directed edge forbids. The Hadamards used to flip it are themselves not
    native, so decompose once more. That last pass can only rewrite single-qubit
    gates, so it cannot reintroduce a direction violation.
    """
    lowered = decompose(circuit, target)
    if coupling_map is not None and coupling_map.edges:
        lowered = decompose(fix_gate_directions(lowered, coupling_map), target)
    return lowered


def transpile(
    circuit: Circuit,
    target: IndianGateSet,
    coupling_map: CouplingMap | None = None,
    qubit_quality: dict[int, QubitQuality] | None = None,
    dd_sequence: str | None = None,
    do_optimize: bool = True,
    use_lookahead: bool = False,
    lookahead: int = 20,
    decay: float = 0.5,
) -> Circuit:
    """Full transpile pipeline: decompose → route → lower → optimize → DD.

    Every stage is dynamic-circuit aware, so mid-circuit measurement, reset, and
    feed-forward survive compilation to a native gate set and topology.

    Decomposition runs twice on purpose. Routing inserts ``swap`` gates *after*
    the first pass, and a SWAP is native to almost no real target, so the result
    has to be lowered again before it can be called executable.

    Args:
        circuit: Input circuit (any supported gates).
        target: Native gate set for the target QPU.
        coupling_map: Qubit connectivity (None = all-to-all, no routing).
        qubit_quality: Per-qubit fidelities for noise-aware routing.
        dd_sequence: DD sequence to insert (``'xy4'``, ``'xy8'``, etc., or ``None``).
        do_optimize: Run optimizer passes after all other steps.
        use_lookahead: Use SabreSWAP lookahead router instead of greedy.
        lookahead: Lookahead window for SabreSWAP.
        decay: Front-layer vs extended-layer weight for SabreSWAP.

    Returns:
        Transpiled circuit ready for execution on the target: every gate is in
        ``target.basis_gates``, and every two-qubit gate sits on an edge of
        ``coupling_map`` in the direction the hardware implements it.

        Use :func:`transpile_with_layout` when you also need to know which
        physical wire each logical qubit ended on.
    """
    return transpile_with_layout(
        circuit, target, coupling_map, qubit_quality, dd_sequence,
        do_optimize, use_lookahead, lookahead, decay,
    ).circuit


def transpile_with_layout(
    circuit: Circuit,
    target: IndianGateSet,
    coupling_map: CouplingMap | None = None,
    qubit_quality: dict[int, QubitQuality] | None = None,
    dd_sequence: str | None = None,
    do_optimize: bool = True,
    use_lookahead: bool = False,
    lookahead: int = 20,
    decay: float = 0.5,
) -> TranspileResult:
    """:func:`transpile`, additionally reporting the final logical→physical layout.

    Routing permutes qubits, and only routing knows the permutation. Every later
    stage — the second lowering, direction fixing, the optimizer, DD — rewrites
    gates in place on the wires they are already on, so the layout routing
    produced is still valid at the end of the pipeline.

    Without a coupling map nothing is routed and the layout is the identity.
    """
    c = decompose(circuit, target)
    final_layout = tuple(range(circuit.num_qubits))
    if coupling_map and coupling_map.edges:
        if use_lookahead:
            routed = route_lookahead_with_layout(
                c, coupling_map, lookahead=lookahead, decay=decay,
                qubit_quality=qubit_quality,
            )
        else:
            routed = route_with_layout(c, coupling_map, qubit_quality)
        c, final_layout = routed.circuit, routed.final_layout
    # Routing injects SWAPs, which are not native to most targets, so lower
    # again. The optimizer only merges or drops gates, never introduces new
    # kinds, so it cannot undo this.
    c = _lower_to_target(c, target, coupling_map)
    if do_optimize:
        c = optimize(c)
    if dd_sequence:
        # DD runs last, after the optimizer. Its refocusing pulses are logically
        # the identity (xy4 = XYXY = I, hahn = XX = I), so an optimizer that saw
        # them would cancel away exactly the pulses the caller asked for — which
        # is what used to happen to the hahn and cpmg sequences. Inserting here
        # also measures idle windows against the final circuit. The pulses are
        # single-qubit, so lowering them cannot affect connectivity.
        from qorch.mitigation.dd import insert_dd
        c = decompose(insert_dd(c, sequence=dd_sequence), target)
    return TranspileResult(circuit=c, final_layout=final_layout)


__all__ = [
    "IndianGateSet",
    "IIT_JODHPUR_ION_TRAP",
    "TIFR_SUPERCONDUCTING",
    "DRDO_MIRAI",
    "CLIFFORD_T",
    "decompose",
    "fix_gate_directions",
    "route",
    "route_lookahead",
    "route_with_layout",
    "route_lookahead_with_layout",
    "RoutingResult",
    "decompose_to_clifford_t",
    "transpile",
    "transpile_with_layout",
    "TranspileResult",
    "CouplingMap",
    "QubitQuality",
    "DECOMPOSITION_RULES",
    "optimize",
]
