"""Entanglement benchmarks: Bell-state fidelity, CHSH S-value, witness.

These are the standard metrics DST/DRDO evaluation reports require
to certify a quantum processor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from qorch import Circuit
from qorch.backends.base import Backend


@dataclass
class BellFidelityResult:
    """Bell-state fidelity measurement."""

    fidelity: float | None
    counts_zz: dict[str, int]
    counts_xx: dict[str, int]
    counts_yy: dict[str, int]


@dataclass
class CHSHResult:
    """CHSH S-value measurement. S > 2 violates Bell inequality."""

    s_value: float | None
    e_ab: float
    e_aB: float
    e_Ab: float
    e_AB: float
    violation: bool | None


@dataclass
class EntanglementWitnessResult:
    """Entanglement witness measurement. W < 0 → entangled."""

    witness_value: float | None
    entangled: bool | None


def _expectation_from_counts(
    counts: dict[str, int], shots: int,
) -> float:
    """Compute ⟨Z₁Z₂⟩ from measurement counts."""
    corr = 0.0
    for bits, c in counts.items():
        if len(bits) >= 2:
            parity = 1 if bits[0] == bits[1] else -1
        else:
            parity = 1 if bits == "0" else -1
        corr += parity * c
    return corr / shots if shots > 0 else 0.0


def _run_and_expect(
    backend: Backend,
    circuit: Circuit,
    shots: int,
) -> float:
    """Run circuit, return ⟨Z₁Z₂⟩ correlation."""
    result = backend.run(circuit, shots=shots)
    return _expectation_from_counts(result.counts, shots)


def _y_basis(circuit: Circuit, q: int) -> Circuit:
    """Rotate qubit q so that measurement in Z-basis gives Y-basis info.
    
    Y-basis eigenstates: |±_y⟩ = (|0⟩ ± i|1⟩) / √2.
    Rotation: rz(-π/2) then H maps |+_y⟩ → |0⟩, |-_y⟩ → |1⟩.
    """
    return circuit.rz(q, -math.pi / 2).h(q)


def bell_state_fidelity(
    backend: Backend,
    shots: int = 8192,
) -> BellFidelityResult:
    """Measure Bell-state fidelity F = (1 + ⟨ZZ⟩ + ⟨XX⟩ - ⟨YY⟩) / 4.

    Creates |Φ⁺⟩ = (|00⟩ + |11⟩) / √2.
    Fidelity close to 1 indicates high-quality entanglement.
    """
    bell = Circuit(2).h(0).cx(0, 1)

    c_zz = bell.measure(0, 1)
    counts_zz = backend.run(c_zz, shots=shots).counts
    e_zz = _expectation_from_counts(counts_zz, shots)

    c_xx = bell.h(0).h(1).measure(0, 1)
    counts_xx = backend.run(c_xx, shots=shots).counts
    e_xx = _expectation_from_counts(counts_xx, shots)

    # Y-basis: rz(-π/2) H on each qubit
    c_yy = _y_basis(_y_basis(bell, 0), 1).measure(0, 1)
    counts_yy = backend.run(c_yy, shots=shots).counts
    e_yy = _expectation_from_counts(counts_yy, shots)

    fidelity = (1.0 + e_zz + e_xx - e_yy) / 4.0

    return BellFidelityResult(
        fidelity=fidelity,
        counts_zz=counts_zz,
        counts_xx=counts_xx,
        counts_yy=counts_yy,
    )


def chsh_s_value(
    backend: Backend,
    shots: int = 8192,
) -> CHSHResult:
    """Measure CHSH S-value for a Bell state.

    Measurement settings (optimal for |Φ⁺⟩):
      A₀ = Z,  A₁ = X
      B₀ = (Z+X)/√2,  B₁ = (Z-X)/√2

    Apply R_y(-θ_B) on q1 to rotate B_b to Z, then measure ⟨Z₁Z₂⟩:
      A₀B₀: nothing on q0, ry(-π/4) on q1  →  θ_B = π/4
      A₀B₁: nothing on q0, ry(π/4) on q1   →  θ_B = -π/4
      A₁B₀: H on q0, ry(-π/4) on q1
      A₁B₁: H on q0, ry(π/4) on q1
    """
    bell = Circuit(2).h(0).cx(0, 1)
    pi4 = math.pi / 4

    e_ab = _run_and_expect(backend, bell.ry(1, -pi4).measure(0, 1), shots)
    e_aB = _run_and_expect(backend, bell.ry(1, pi4).measure(0, 1), shots)
    e_Ab = _run_and_expect(backend, bell.h(0).ry(1, -pi4).measure(0, 1), shots)
    e_AB = _run_and_expect(backend, bell.h(0).ry(1, pi4).measure(0, 1), shots)

    s = e_ab + e_aB + e_Ab - e_AB
    violation = s > 2.0 if s is not None else None

    return CHSHResult(
        s_value=s,
        e_ab=e_ab,
        e_aB=e_aB,
        e_Ab=e_Ab,
        e_AB=e_AB,
        violation=violation,
    )


def entanglement_witness(
    backend: Backend,
    shots: int = 8192,
) -> EntanglementWitnessResult:
    """Measure entanglement witness W for |Φ⁺⟩ = (|00⟩ + |11⟩)/√2.

    W = ½ - F where F = (1 + ⟨ZZ⟩ + ⟨XX⟩ - ⟨YY⟩)/4.
    W < 0 signals entanglement (fidelity > ½).
    """
    bell = Circuit(2).h(0).cx(0, 1)

    e_zz = _run_and_expect(backend, bell.measure(0, 1), shots)
    c_xx = bell.h(0).h(1).measure(0, 1)
    e_xx = _run_and_expect(backend, c_xx, shots)
    c_yy = _y_basis(_y_basis(bell, 0), 1).measure(0, 1)
    e_yy = _run_and_expect(backend, c_yy, shots)

    fidelity = (1.0 + e_zz + e_xx - e_yy) / 4.0
    w = 0.5 - fidelity
    entangled = w < 0.0 if w is not None else None

    return EntanglementWitnessResult(
        witness_value=w,
        entangled=entangled,
    )
