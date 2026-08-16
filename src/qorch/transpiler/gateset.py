"""Native gate sets: what each target can actually execute.

A gate set is the pair of facts compilation needs about a device — which gates
are physically implemented, and which qubits can interact. Everything the
transpiler does is in service of reaching one of these.

The sets here describe real Indian hardware plus Clifford+T, the fault-tolerant
set that is not a device at all but a resource-accounting target. A neutral-atom
set is built per-arrangement instead, since its connectivity comes from geometry
rather than fabrication — see :mod:`qorch.neutral_atom`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IndianGateSet:
    """Native gate set + coupling topology for an Indian quantum processor.

    ``basis_gates`` are the physically implemented gates.
    ``coupling_map`` is a list of directed (control, target) qubit pairs.
    ``None`` means all-to-all (e.g. ion trap).
    """

    name: str
    description: str
    basis_gates: tuple[str, ...]
    coupling_map: tuple[tuple[int, int], ...] | None = None
    num_qubits: int = 0


# IIT Jodhpur Ion Trap: all-to-all connectivity, MS gate as entangler
# Reference: IIT Jodhpur's trapped-ion quantum processor publications
IIT_JODHPUR_ION_TRAP = IndianGateSet(
    name="iit-jodhpur-ion-trap",
    description="IIT Jodhpur trapped-ion processor: all-to-all, MS gate",
    basis_gates=("rx", "ry", "ms"),
    coupling_map=None,
    num_qubits=6,
)

# TIFR Superconducting (Mumbai): nearest-neighbor, CX-based
# Reference: TIFR's superconducting qubit roadmap
TIFR_SUPERCONDUCTING = IndianGateSet(
    name="tifr-superconducting",
    description="TIFR Mumbai superconducting processor: linear nearest-neighbor",
    basis_gates=("cx", "sx", "rz", "x"),
    coupling_map=((0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2), (3, 4), (4, 3)),
    num_qubits=5,
)

# DRDO MIRAI Lab: superconducting, grid-like coupling
DRDO_MIRAI = IndianGateSet(
    name="drdo-mirai",
    description="DRDO MIRAI superconducting processor: grid topology",
    basis_gates=("cx", "rx", "rz", "x"),
    coupling_map=(
        (0, 1), (1, 0), (1, 2), (2, 1),
        (3, 4), (4, 3), (4, 5), (5, 4),
        (0, 3), (3, 0), (1, 4), (4, 1), (2, 5), (5, 2),
    ),
    num_qubits=6,
)

# Clifford+T: fault-tolerant gate set
# T = Rz(pi/4) — expensive (magic state distillation)
# H, CX are cheap Clifford gates
CLIFFORD_T = IndianGateSet(
    name="clifford-t",
    description="Clifford+T fault-tolerant gate set",
    basis_gates=("h", "cx", "t"),
)
