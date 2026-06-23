"""Quantum state tomography — reconstruct density matrices from measurement data.

Supports 1-qubit and 2-qubit tomographic reconstruction via linear inversion.
Useful for QPU characterization and verifying state preparation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from qorch import Circuit
from qorch.backends.base import Backend


_PAULI: dict[str, tuple[complex, ...]] = {
    "I": (1, 0, 0, 1),
    "X": (0, 1, 1, 0),
    "Y": (0, -1j, 1j, 0),
    "Z": (1, 0, 0, -1),
}


def _pauli_label(i: int) -> tuple[str, str]:
    return ("I", "I")


def _rotate_to_basis(circuit: Circuit, qubit: int, basis: str) -> Circuit:
    """Prepend rotation so Z-measurement gives the desired Pauli expectation."""
    if basis == "Z":
        return circuit
    if basis == "X":
        return circuit.h(qubit)
    if basis == "Y":
        return circuit.rz(qubit, -math.pi / 2).h(qubit)
    raise ValueError(f"unknown basis: {basis!r}")


def _expectation_1q(counts: dict[str, int], shots: int, qubit: int = 0) -> float:
    """⟨P⟩ from counts = P(0) - P(1)."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    signed = sum((1 if key[qubit] == "0" else -1) * c for key, c in counts.items())
    return signed / total


def _expectation_2q(counts: dict[str, int], shots: int) -> float:
    """⟨P₁⊗P₂⟩ for two-qubit Pauli product."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    corr = 0.0
    for bits, c in counts.items():
        if len(bits) >= 2:
            parity = 1 if bits[0] == bits[1] else -1
        else:
            parity = 1 if bits == "0" else -1
        corr += parity * c
    return corr / total


def _measure_expectation_1q(
    backend: Backend, circuit: Circuit, basis: str, qubit: int, shots: int
) -> float:
    """Run circuit rotated to *basis* and return the Pauli expectation."""
    rot = _rotate_to_basis(circuit, qubit, basis).measure(qubit, qubit)
    counts = backend.run(rot, shots=shots).counts
    return _expectation_1q(counts, shots, qubit)


@dataclass
class StateTomographyResult:
    """Result of quantum state tomography."""

    num_qubits: int
    rho: list[list[complex]]
    pauli_expectations: dict[str, float]


def state_tomography_1q(
    backend: Backend,
    circuit: Circuit,
    shots: int = 8192,
    qubit: int = 0,
) -> StateTomographyResult:
    """Reconstruct single-qubit density matrix via Pauli measurements.

    ρ = ½(I + ⟨X⟩σ_x + ⟨Y⟩σ_y + ⟨Z⟩σ_z)
    """
    ex_z = _measure_expectation_1q(backend, circuit, "Z", qubit, shots)
    ex_x = _measure_expectation_1q(backend, circuit, "X", qubit, shots)
    ex_y = _measure_expectation_1q(backend, circuit, "Y", qubit, shots)

    sx, sy, sz = _PAULI["X"], _PAULI["Y"], _PAULI["Z"]
    rho = [
        [(1.0 + ex_z * sz[0] + ex_x * sx[0] + ex_y * sy[0]) / 2.0,
         (ex_z * sz[1] + ex_x * sx[1] + ex_y * sy[1]) / 2.0],
        [(ex_z * sz[2] + ex_x * sx[2] + ex_y * sy[2]) / 2.0,
         (1.0 + ex_z * sz[3] + ex_x * sx[3] + ex_y * sy[3]) / 2.0],
    ]

    return StateTomographyResult(
        num_qubits=1,
        rho=rho,
        pauli_expectations={"<X>": ex_x, "<Y>": ex_y, "<Z>": ex_z},
    )


def state_tomography_2q(
    backend: Backend,
    circuit: Circuit,
    shots: int = 8192,
) -> StateTomographyResult:
    """Reconstruct two-qubit density matrix via 9 Pauli-basis measurements.

    ρ = ¼ Σ_{i,j∈{I,X,Y,Z}} ⟨P_i⊗P_j⟩ · P_i⊗P_j
    """
    pauli_labels = ["I", "X", "Y", "Z"]
    bases = [(b0, b1) for b0 in pauli_labels for b1 in pauli_labels]
    expectations: dict[str, float] = {}

    for b0, b1 in bases:
        c = circuit
        if b0 != "I":
            c = _rotate_to_basis(c, 0, b0)
        if b1 != "I":
            c = _rotate_to_basis(c, 1, b1)
        c = c.measure(0, 1)
        counts = backend.run(c, shots=shots).counts
        total = sum(counts.values())
        if (b0, b1) == ("I", "I"):
            ex = 1.0
        elif b0 == "I":
            ex = _expectation_1q(counts, total, 1)
        elif b1 == "I":
            ex = _expectation_1q(counts, total, 0)
        else:
            ex = _expectation_2q(counts, total)
        expectations[f"<{b0}{b1}>"] = ex

    _I, _X, _Y, _Z = (
        _PAULI["I"], _PAULI["X"], _PAULI["Y"], _PAULI["Z"],
    )
    _pauli_2q = [_I, _X, _Y, _Z]

    rho = [[0j] * 4 for _ in range(4)]
    for i0, p0 in enumerate(_pauli_2q):
        for i1, p1 in enumerate(_pauli_2q):
            label = f"<{pauli_labels[i0]}{pauli_labels[i1]}>"
            ex = expectations[label]
            p = [
                [p0[0] * p1[0], p0[0] * p1[1], p0[1] * p1[0], p0[1] * p1[1]],
                [p0[0] * p1[2], p0[0] * p1[3], p0[1] * p1[2], p0[1] * p1[3]],
                [p0[2] * p1[0], p0[2] * p1[1], p0[3] * p1[0], p0[3] * p1[1]],
                [p0[2] * p1[2], p0[2] * p1[3], p0[3] * p1[2], p0[3] * p1[3]],
            ]
            for r in range(4):
                for c_idx in range(4):
                    rho[r][c_idx] += ex * p[r][c_idx] / 4.0

    return StateTomographyResult(
        num_qubits=2,
        rho=rho,
        pauli_expectations=expectations,
    )


def format_density_matrix(rho: list[list[complex]], decimals: int = 4) -> str:
    """Format a density matrix as a readable string."""
    lines: list[str] = []
    for i, row in enumerate(rho):
        cells = []
        for v in row:
            r, im = round(v.real, decimals), round(v.imag, decimals)
            if abs(im) < 1e-12:
                cells.append(f"{r:{decimals+6}.{decimals}f}")
            elif abs(r) < 1e-12:
                cells.append(f"{im:{decimals+6}.{decimals}f}j")
            else:
                sign = "+" if im >= 0 else "-"
                cells.append(f"{r:{decimals+6}.{decimals}f}{sign}{abs(im):.{decimals}f}j")
        lines.append("  " + "  ".join(cells))
    return "\n".join(lines)


def trace_rho(rho: list[list[complex]]) -> float:
    """Compute Tr(ρ). Should be 1 for valid density matrices."""
    return sum(abs(rho[i][i]) for i in range(len(rho)))


def purity(rho: list[list[complex]]) -> float:
    """Compute Tr(ρ²). 1 for pure states, ≤1 for mixed."""
    n = len(rho)
    total = 0.0
    for i in range(n):
        for j in range(n):
            total += abs(rho[i][j]) ** 2
    return total
