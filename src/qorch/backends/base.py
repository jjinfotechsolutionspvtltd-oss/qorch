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
class JobResult:
    """Measurement histogram plus provenance.

    ``counts`` maps result bitstrings (leftmost char = qubit 0) to occurrence counts.
    """

    counts: dict[str, int]
    shots: int
    backend_name: str
    metadata: dict[str, object] = field(default_factory=dict)


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
