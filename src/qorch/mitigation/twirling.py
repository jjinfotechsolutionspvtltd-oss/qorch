"""Pauli Twirling — converts coherent noise to stochastic Pauli noise.

For each Clifford gate, insert a random Pauli P before and G·P·G† after.
This preserves the logical operation (P² = I for Paulis) while randomizing
any coherent error component into Pauli noise, which is tractable for PEC/ZNE.

Reference: Wallman & Emerson, "Noise tailoring for scalable quantum
computation via randomized compiling" (2016), Phys. Rev. A 94, 052325.
"""

from __future__ import annotations

import random
from dataclasses import replace

from qorch.ir import Circuit, Gate, static_gates

# Pauli conjugation: for a Clifford gate G and Pauli P, compute G·P·G†
# For our gate set, we precompute the Pauli-to-Pauli mapping.
# Key: (gate_name, pauli) → conjugated_pauli (on each qubit the gate touches)
# Paulis: 0=I, 1=X, 2=Y, 3=Z

_PAULI_NAMES = ("i", "x", "y", "z")
_PAULI_GATES = ("x", "y", "z")

# Conjugation tables for single-qubit Clifford gates:
# G·P·G† for P ∈ {I, X, Y, Z}
# h: H·X·H = Z, H·Y·H = -Y, H·Z·H = X (global phase ignored)
# x: X·X·X = X, X·Y·X = -Y, X·Z·X = -Z (we preserve up to global phase)
# y: Y·X·Y = -X, Y·Y·Y = Y, Y·Z·Y = -Z
# z: Z·X·Z = -X, Z·Y·Z = -Y, Z·Z·Z = Z

_CONJ_1Q: dict[str, tuple[int, ...]] = {
    "h": (0, 3, 2, 1),   # I→I, X→Z, Y→Y, Z→X (H·Y·H = -Y but phase ignored)
    "x": (0, 1, 2, 3),   # I→I, X→X, Y→Y, Z→Z → X commutes with all Paulis
    "y": (0, 1, 2, 3),   # Y commutes with all Paulis (up to phase)
    "z": (0, 1, 2, 3),   # Z commutes with all Paulis
    "sx": (0, 3, 2, 1),  # SX·X·SX† = Z, similar to H
}

# 2-qubit conjugation: CX·(P⊗Q)·CX†
# CX: X⊗I → X⊗X, I⊗X → I⊗X, X⊗X → X⊗I, etc.
_CONJ_CX: dict[tuple[int, int], tuple[int, int]] = {
    (0, 0): (0, 0),  # I⊗I → I⊗I
    (1, 0): (1, 1),  # X⊗I → X⊗X
    (2, 0): (2, 2),  # Y⊗I → Y⊗X  (Y⊗X = -Z⊗Y, but phase ignored)
    (3, 0): (3, 0),  # Z⊗I → Z⊗I
    (0, 1): (0, 1),  # I⊗X → I⊗X
    (1, 1): (1, 0),  # X⊗X → X⊗I
    (2, 1): (2, 3),  # Y⊗X → Z⊗Y
    (3, 1): (3, 1),  # Z⊗X → Z⊗X
    (0, 2): (0, 2),  # I⊗Y → I⊗Y
    (1, 2): (1, 3),  # X⊗Y → Z⊗Z
    (2, 2): (2, 1),  # Y⊗Y → Y⊗X
    (3, 2): (3, 2),  # Z⊗Y → Z⊗Y
    (0, 3): (0, 3),  # I⊗Z → I⊗Z
    (1, 3): (1, 2),  # X⊗Z → Y⊗Z
    (2, 3): (2, 0),  # Y⊗Z → Y⊗I
    (3, 3): (3, 3),  # Z⊗Z → Z⊗Z
}

# SWAP conjugation: SWAP·(P⊗Q)·SWAP† = Q⊗P (just swaps)
_CONJ_SWAP: dict[tuple[int, int], tuple[int, int]] = {
    (a, b): (b, a) for a in range(4) for b in range(4)
}


def _random_pauli(rng: random.Random) -> int:
    """Return 0=I, 1=X, 2=Y, or 3=Z."""
    return rng.randint(0, 3)


def _conjugate_1q(gate_name: str, pauli: int) -> int:
    """Compute G·P·G† for a single-qubit Clifford gate."""
    return _CONJ_1Q.get(gate_name, (0, 1, 2, 3))[pauli]


def _conjugate_2q(gate_name: str, p0: int, p1: int) -> tuple[int, int]:
    """Compute G·(P⊗Q)·G† for a 2-qubit Clifford gate."""
    if gate_name == "cx":
        return _CONJ_CX.get((p0, p1), (p0, p1))
    if gate_name == "swap":
        return _CONJ_SWAP.get((p0, p1), (p0, p1))
    return (p0, p1)


def twirl_circuit(circuit: Circuit, seed: int | None = None) -> Circuit:
    """Insert random Pauli twirling gates preserving the logical operation.

    Each Clifford gate gets a random Pauli before and its conjugate after.
    Non-Clifford gates (rx, ry, rz, ms) are left untouched.

    The resulting circuit has the same logical effect but its noise is
    twirled into Pauli noise.
    """
    rng = random.Random(seed)

    # Gates that we know how to twirl (Clifford)
    _TWIRLABLE = frozenset({"h", "x", "y", "z", "sx", "cx", "swap"})

    new_gates: list[Gate] = []
    for g in static_gates(circuit.gates):
        if g.name not in _TWIRLABLE:
            new_gates.append(g)
            continue

        if len(g.qubits) == 1:
            q = g.qubits[0]
            p = _random_pauli(rng)
            if p != 0:
                new_gates.append(Gate(_PAULI_GATES[p - 1], (q,)))
            new_gates.append(g)
            cp = _conjugate_1q(g.name, p)
            if cp != 0:
                new_gates.append(Gate(_PAULI_GATES[cp - 1], (q,)))

        elif len(g.qubits) == 2:
            q0, q1 = g.qubits
            p0 = _random_pauli(rng)
            p1 = _random_pauli(rng)
            if p0 != 0:
                new_gates.append(Gate(_PAULI_GATES[p0 - 1], (q0,)))
            if p1 != 0:
                new_gates.append(Gate(_PAULI_GATES[p1 - 1], (q1,)))
            new_gates.append(g)
            cp0, cp1 = _conjugate_2q(g.name, p0, p1)
            if cp0 != 0:
                new_gates.append(Gate(_PAULI_GATES[cp0 - 1], (q0,)))
            if cp1 != 0:
                new_gates.append(Gate(_PAULI_GATES[cp1 - 1], (q1,)))

    return replace(circuit, gates=tuple(new_gates))
