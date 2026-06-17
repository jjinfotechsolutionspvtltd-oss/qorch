"""qorch — the QS-004 Quantum Orchestration Layer.

A hardware-agnostic control plane for quantum execution: IR ingestion, a backend HAL,
error mitigation, and scheduling. See ``../architecture.md``.
"""

from qorch.backends.base import Backend, BackendProperties, JobResult
from qorch.backends.qiskit_backend import QiskitBackend, reorder_counts_qiskit_to_qorch
from qorch.backends.simulator import GateNoise, LocalSimulator, ReadoutNoise
from qorch.ir import Circuit, Gate, from_qasm3
from qorch.mitigation.readout import ReadoutMitigator
from qorch.mitigation.zne import (
    ZNEResult,
    expectation_z,
    fold_circuit,
    zne_expectation,
)
from qorch.scheduler import Scheduler, first_available

__version__ = "0.1.0"

__all__ = [
    "Backend",
    "BackendProperties",
    "JobResult",
    "LocalSimulator",
    "ReadoutNoise",
    "GateNoise",
    "QiskitBackend",
    "reorder_counts_qiskit_to_qorch",
    "Circuit",
    "Gate",
    "from_qasm3",
    "ReadoutMitigator",
    "ZNEResult",
    "expectation_z",
    "fold_circuit",
    "zne_expectation",
    "Scheduler",
    "first_available",
]
