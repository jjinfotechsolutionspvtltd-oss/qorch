"""The hardware-abstraction layer (HAL).

Every backend — local simulator or real QPU — implements ``Backend`` (ADR-2). The
scheduler and mitigation layers program against this interface only, so adding a real
QPU or future indigenous hardware is one new adapter with zero core changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from qorch.ir import Circuit


@dataclass(frozen=True)
class BackendProperties:
    """What a backend can do — used by the scheduler to route jobs."""

    num_qubits: int
    basis_gates: tuple[str, ...]
    is_simulator: bool
    # noise hint: per-qubit readout fidelity, if known (empty for ideal sims)
    readout_fidelity: tuple[float, ...] = ()


@dataclass(frozen=True)
class QubitCalibration:
    """Per-qubit calibration data for one physical qubit (Backend API v2).

    Times in microseconds; errors are average per-operation error rates (0–1).
    """

    t1_us: float = 0.0
    t2_us: float = 0.0
    readout_fidelity: float = 1.0
    single_qubit_error: float = 0.0
    frequency_ghz: float = 0.0


@dataclass(frozen=True)
class DeviceCalibration:
    """A device's calibration snapshot — the source a calibration-aware
    transpiler / noise-adaptive scheduler reads from (Backend API v2).

    This is the structured replacement for the single ``readout_fidelity`` hint
    on :class:`BackendProperties`: it carries T1/T2, per-edge two-qubit error,
    connectivity, and gate timing so compilation can optimize against real noise.
    """

    qubits: tuple[QubitCalibration, ...]
    two_qubit_error: dict[tuple[int, int], float] = field(default_factory=dict)
    coupling_map: tuple[tuple[int, int], ...] = ()
    basis_gates: tuple[str, ...] = ()
    gate_durations_us: dict[str, float] = field(default_factory=dict)

    @property
    def num_qubits(self) -> int:
        return len(self.qubits)


@dataclass(frozen=True)
class JobResult:
    """Measurement histogram plus provenance, and what can be derived from it.

    ``counts`` maps result bitstrings (leftmost char = qubit 0) to occurrence counts.

    The optional fields exist because a histogram alone loses information a
    caller often needs and cannot reconstruct:

    - ``memory`` — per-shot outcomes in order. Counts discard shot ordering, so
      anything looking for drift or time correlation cannot work from them.
    - ``quasi_probabilities`` — what readout mitigation produces. These are not
      counts: they can be negative, which is the whole point, so they cannot be
      squeezed back into an integer histogram without discarding the correction.
    - ``final_layout`` — which physical wire held each logical qubit. Without it
      results cannot be tied to per-qubit calibration.
    - ``expectation_values`` — precomputed observables, when a backend produced
      them directly rather than by sampling.

    Every one defaults to ``None``, so a backend that only returns counts is
    still a complete implementation.
    """

    counts: dict[str, int]
    shots: int
    backend_name: str
    metadata: dict[str, object] = field(default_factory=dict)
    memory: tuple[str, ...] | None = None
    quasi_probabilities: dict[str, float] | None = None
    expectation_values: dict[str, float] | None = None
    final_layout: tuple[int, ...] | None = None

    @property
    def probabilities(self) -> dict[str, float]:
        """Counts normalized to a distribution, or the quasi-probabilities.

        Prefers ``quasi_probabilities`` when present: if mitigation has run, its
        corrected distribution is the better answer and returning the raw counts
        would silently discard the correction.
        """
        if self.quasi_probabilities is not None:
            return dict(self.quasi_probabilities)
        if self.shots <= 0:
            return {}
        return {key: value / self.shots for key, value in self.counts.items()}

    def expectation_z(self, qubit: int) -> float:
        """⟨Z⟩ on one qubit: P(0) - P(1)."""
        return self.parity_expectation((qubit,))

    def parity_expectation(self, qubits: tuple[int, ...]) -> float:
        """⟨Z⊗Z⊗…⟩ over ``qubits`` — the parity of the selected bits.

        Computed from ``probabilities``, so it follows mitigation when mitigation
        has been applied.
        """
        if not qubits:
            raise ValueError("parity_expectation needs at least one qubit")
        total = 0.0
        for bits, weight in self.probabilities.items():
            for q in qubits:
                if q >= len(bits):
                    raise ValueError(
                        f"qubit {q} is outside the {len(bits)}-bit result strings"
                    )
            ones = sum(1 for q in qubits if bits[q] == "1")
            total += weight * (1 if ones % 2 == 0 else -1)
        return total


class Backend(ABC):
    """Abstract execution target. Synchronous ``run`` is the common denominator;
    cloud adapters add async submit/poll on top but still satisfy this contract."""

    name: str

    @abstractmethod
    def properties(self) -> BackendProperties:
        """Advertise capabilities (qubit count, basis gates, noise hints)."""

    @abstractmethod
    def run(self, circuit: Circuit, shots: int = 1024) -> JobResult:
        """Execute ``circuit`` for ``shots`` and return measurement counts."""

    def validate(self, circuit: Circuit) -> None:
        """Boundary check before execution — never forward an oversized circuit."""
        props = self.properties()
        if circuit.num_qubits > props.num_qubits:
            raise ValueError(
                f"circuit needs {circuit.num_qubits} qubits, "
                f"{self.name} has {props.num_qubits}"
            )

    # --- optional capability hooks (Backend API v2) -----------------------
    # Default implementations keep simple/simulator backends a 3-method adapter;
    # real hardware overrides these to advertise calibration and connectivity.
    def calibration(self) -> "DeviceCalibration | None":
        """Return the device's calibration snapshot, or ``None`` if unknown."""
        return None

    def coupling_map(self) -> tuple[tuple[int, int], ...] | None:
        """Return the qubit connectivity, or ``None`` for all-to-all/unknown."""
        return None
