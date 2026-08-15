"""qorch — the QS-004 Quantum Orchestration Layer.

A hardware-agnostic control plane for quantum execution: IR ingestion, a backend HAL,
error mitigation, and scheduling. See ``../architecture.md``.
"""

from qorch.backends.base import (
    Backend,
    BackendProperties,
    DeviceCalibration,
    JobResult,
    QubitCalibration,
)
from qorch.backends.qiskit_backend import QiskitBackend, reorder_counts_qiskit_to_qorch
from qorch.backends.indian_backend import IndianQPU, INDIAN_QPU_CONFIGS
from qorch.backends.simulator import GateNoise, LocalSimulator, ReadoutNoise
from qorch.backends.density_simulator import DensitySimulator, NoiseChannel
from qorch.backends.stabilizer import StabilizerSimulator
from qorch.ir import (
    Circuit,
    Gate,
    Measure,
    Parameter,
    Reset,
    from_qasm3,
    to_qasm3,
)
from qorch.mitigation.dd import DD_SEQUENCES, apply_dd_mitigation, insert_dd
from qorch.pulse import (
    Waveform,
    calibrate_gaussian,
    pulse_unitary,
    rotation_angle,
    sx_pulse,
    x_pulse,
    y_pulse,
)
from qorch.transpiler import transpile
from qorch.mitigation.pec import PECResult, pec_expectation
from qorch.mitigation.readout import ReadoutMitigator
from qorch.mitigation.twirling import twirl_circuit
from qorch.mitigation.zne import (
    ZNEResult,
    expectation_z,
    fold_circuit,
    zne_expectation,
)
from qorch.scheduler import Scheduler, first_available
from qorch.surface_code import ToricCode, toric_logical_error_rate

__version__ = "0.1.0"

__all__ = [
    "Backend",
    "BackendProperties",
    "DeviceCalibration",
    "QubitCalibration",
    "JobResult",
    "LocalSimulator",
    "DensitySimulator",
    "StabilizerSimulator",
    "NoiseChannel",
    "ReadoutNoise",
    "GateNoise",
    "IndianQPU",
    "INDIAN_QPU_CONFIGS",
    "QiskitBackend",
    "reorder_counts_qiskit_to_qorch",
    "Circuit",
    "Gate",
    "Measure",
    "Reset",
    "Parameter",
    "from_qasm3",
    "to_qasm3",
    "ReadoutMitigator",
    "ZNEResult",
    "expectation_z",
    "fold_circuit",
    "zne_expectation",
    "PECResult",
    "pec_expectation",
    "DD_SEQUENCES",
    "apply_dd_mitigation",
    "insert_dd",
    "twirl_circuit",
    "transpile",
    "Waveform",
    "pulse_unitary",
    "rotation_angle",
    "calibrate_gaussian",
    "x_pulse",
    "y_pulse",
    "sx_pulse",
    "Scheduler",
    "first_available",
    "ToricCode",
    "toric_logical_error_rate",
]
