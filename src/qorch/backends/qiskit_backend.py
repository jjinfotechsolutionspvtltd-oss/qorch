"""Qiskit-backed adapter — the M2 proof that the HAL is genuinely hardware-agnostic.

The *same* ``Circuit`` that runs on ``LocalSimulator`` runs here on Qiskit Aer or on real
IBM Quantum hardware, with no change to the scheduler or mitigation layers. Qiskit is an
optional dependency (``pip install qorch[qiskit]``): it is imported lazily so ``import qorch``
never requires it, and this module's top level stays Qiskit-free.

Endianness: Qiskit reports a measured classical register with bit ``c0`` rightmost, and we
measure qubit ``i`` into ``c[i]`` — so Qiskit's rightmost char is qubit 0. Our convention is
leftmost = qubit 0 (architecture §5), so we reverse each key at the boundary. That reversal is
the single most common cross-tool bug in quantum software, so it lives in a pure, tested
function below.
"""

from __future__ import annotations

import os
from typing import Any

from qorch.backends.base import Backend, BackendProperties, JobResult
from qorch.ir import Circuit

# Map our IR gate names to Qiskit QuantumCircuit method names. Covers the full
# qorch gate set (defect A3 fix) — previously only h/x/z/cx were handled and any
# rotation/sx/swap/ms/t/y/id raised KeyError.
_GATE_METHODS = {
    "h": "h", "x": "x", "y": "y", "z": "z", "sx": "sx", "t": "t", "id": "id",
    "cx": "cx", "swap": "swap",
    "rx": "rx", "ry": "ry", "rz": "rz",
    "ms": "rxx",
}


def qiskit_instruction(gate) -> tuple[str, tuple[float, ...], tuple[int, ...]]:
    """Translate a qorch ``Gate`` to ``(qiskit_method, params, qubits)``.

    Pure and Qiskit-free so the whole gate-coverage surface is unit-testable
    without the SDK. The Mølmer–Sørensen gate ``ms(θ)=exp(-iθ X⊗X)`` maps to
    Qiskit's ``rxx(φ)=exp(-iφ/2 X⊗X)`` with ``φ = 2θ``.
    """
    if gate.name not in _GATE_METHODS:
        raise ValueError(f"no Qiskit mapping for gate {gate.name!r}")
    method = _GATE_METHODS[gate.name]
    if gate.name == "ms":
        theta = gate.params[0] if gate.params else 0.0
        return method, (2.0 * theta,), gate.qubits
    return method, gate.params, gate.qubits

# Environment variable holding the IBM Quantum API token (a secret; never committed).
IBM_TOKEN_ENV = "QISKIT_IBM_TOKEN"


def reorder_counts_qiskit_to_qorch(counts: dict[str, int]) -> dict[str, int]:
    """Translate Qiskit count keys (rightmost = qubit 0) to qorch keys (leftmost = qubit 0).

    Pure and Qiskit-free so it is unit-testable without the SDK installed. Qiskit may include
    spaces between registers; we strip them before reversing.
    """
    out: dict[str, int] = {}
    for key, value in counts.items():
        qorch_key = key.replace(" ", "")[::-1]
        out[qorch_key] = out.get(qorch_key, 0) + value
    return out


class QiskitBackend(Backend):
    """Wraps any Qiskit execution target (Aer simulator or IBM hardware) as a qorch backend."""

    def __init__(self, qiskit_backend: Any, name: str | None = None) -> None:
        # qiskit_backend is a BackendV2-like object; typed as Any to avoid a hard import.
        self._backend: Any = qiskit_backend
        self.name = str(name or getattr(qiskit_backend, "name", "qiskit-backend"))

    # --- constructors -----------------------------------------------------
    @classmethod
    def aer(cls, *, seed: int | None = None) -> "QiskitBackend":
        """Local Qiskit Aer simulator — same interface as real hardware."""
        from qiskit_aer import AerSimulator  # lazy: optional dependency

        backend = AerSimulator(seed_simulator=seed)
        return cls(backend, name="qiskit-aer")

    @classmethod
    def ibm(cls, backend_name: str, token: str | None = None) -> "QiskitBackend":
        """Real IBM Quantum hardware via Qiskit Runtime.

        Token resolves from the ``token`` argument or the ``QISKIT_IBM_TOKEN`` env var —
        never hard-coded. Raises if neither is present.
        """
        # validate the boundary (token is a secret) before the heavy optional import
        token = token or os.environ.get(IBM_TOKEN_ENV)
        if not token:
            raise ValueError(
                f"no IBM token: pass token= or set {IBM_TOKEN_ENV} (it is a secret)"
            )
        from qiskit_ibm_runtime import QiskitRuntimeService  # lazy: optional dependency

        service = QiskitRuntimeService(channel="ibm_quantum", token=token)
        backend = service.backend(backend_name)
        return cls(backend, name=backend_name)

    # --- Backend interface ------------------------------------------------
    def properties(self) -> BackendProperties:
        num_qubits = int(getattr(self._backend, "num_qubits", 0))
        target = getattr(self._backend, "target", None)
        basis = tuple(target.operation_names) if target is not None else ()
        is_sim = "aer" in self.name.lower() or "simulator" in self.name.lower()
        return BackendProperties(
            num_qubits=num_qubits or 64,  # some Aer sims report 0; treat as ample
            basis_gates=basis,
            is_simulator=is_sim,
        )

    def run(self, circuit: Circuit, shots: int = 1024) -> JobResult:
        self.validate(circuit)
        qc = self._to_qiskit(circuit)
        raw_counts = self._execute(qc, shots)
        return JobResult(
            counts=reorder_counts_qiskit_to_qorch(raw_counts),
            shots=shots,
            backend_name=self.name,
            metadata={"transpiled_to": self.name},
        )

    # --- translation ------------------------------------------------------
    def _to_qiskit(self, circuit: Circuit):
        from qiskit import QuantumCircuit  # lazy

        readout = circuit.readout_qubits
        qc = QuantumCircuit(circuit.num_qubits, len(readout))
        for gate in circuit.gates:
            method, params, qubits = qiskit_instruction(gate)
            getattr(qc, method)(*params, *qubits)
        for classical_bit, qubit in enumerate(readout):
            qc.measure(qubit, classical_bit)
        return qc

    def _execute(self, qc, shots: int) -> dict[str, int]:
        from qiskit import transpile  # lazy

        transpiled = transpile(qc, self._backend)
        job = self._backend.run(transpiled, shots=shots)
        return job.result().get_counts()
