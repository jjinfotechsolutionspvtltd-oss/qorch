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
from qorch.transpiler.routing import CouplingMap, QubitQuality, route, route_lookahead

from qorch.ir import Circuit


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
    """Full transpile pipeline: decompose → route → DD → optimize.

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
        Transpiled circuit ready for execution on the target.
    """
    c = decompose(circuit, target)
    if coupling_map and coupling_map.edges:
        if use_lookahead:
            c = route_lookahead(c, coupling_map, lookahead=lookahead, decay=decay, qubit_quality=qubit_quality)
        else:
            c = route(c, coupling_map, qubit_quality)
    if dd_sequence:
        from qorch.mitigation.dd import insert_dd
        c = insert_dd(c, sequence=dd_sequence)
    if do_optimize:
        c = optimize(c)
    return c


__all__ = [
    "IndianGateSet",
    "IIT_JODHPUR_ION_TRAP",
    "TIFR_SUPERCONDUCTING",
    "DRDO_MIRAI",
    "CLIFFORD_T",
    "decompose",
    "route",
    "route_lookahead",
    "decompose_to_clifford_t",
    "transpile",
    "CouplingMap",
    "QubitQuality",
    "DECOMPOSITION_RULES",
    "optimize",
]
