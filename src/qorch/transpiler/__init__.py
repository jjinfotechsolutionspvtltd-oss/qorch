"""Transpiler: gate decomposition + qubit routing for Indian-native gate sets."""

from qorch.transpiler.gateset import (
    IndianGateSet,
    IIT_JODHPUR_ION_TRAP,
    TIFR_SUPERCONDUCTING,
    DRDO_MIRAI,
)
from qorch.transpiler.decompose import DECOMPOSITION_RULES, decompose
from qorch.transpiler.optimizer import optimize
from qorch.transpiler.routing import CouplingMap, QubitQuality, route

from qorch.ir import Circuit


def transpile(
    circuit: Circuit,
    target: IndianGateSet,
    coupling_map: CouplingMap | None = None,
    qubit_quality: dict[int, QubitQuality] | None = None,
    dd_sequence: str | None = None,
    do_optimize: bool = True,
) -> Circuit:
    """Full transpile pipeline: decompose → route → DD → optimize.

    Args:
        circuit: Input circuit (any supported gates).
        target: Native gate set for the target QPU.
        coupling_map: Qubit connectivity (None = all-to-all, no routing).
        qubit_quality: Per-qubit fidelities for noise-aware routing.
        dd_sequence: DD sequence to insert (``'xy4'``, ``'xy8'``, etc., or ``None``).
        do_optimize: Run optimizer passes after all other steps.

    Returns:
        Transpiled circuit ready for execution on the target.
    """
    c = decompose(circuit, target)
    if coupling_map and coupling_map.edges:
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
    "decompose",
    "route",
    "transpile",
    "CouplingMap",
    "QubitQuality",
    "DECOMPOSITION_RULES",
    "optimize",
]
