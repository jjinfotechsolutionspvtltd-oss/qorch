"""Tests for circuit optimization (gate cancellation + rotation merging)."""

import math

from qorch import Circuit, LocalSimulator
from qorch.transpiler.optimizer import optimize


def test_cancel_adjacent_x():
    """x x → identity."""
    circuit = Circuit(num_qubits=1).x(0).x(0)
    opt = optimize(circuit)
    assert len(opt.gates) == 0


def test_cancel_adjacent_h():
    circuit = Circuit(num_qubits=1).h(0).h(0)
    opt = optimize(circuit)
    assert len(opt.gates) == 0


def test_cancel_adjacent_y():
    circuit = Circuit(num_qubits=1).y(0).y(0)
    opt = optimize(circuit)
    assert len(opt.gates) == 0


def test_cancel_adjacent_z():
    circuit = Circuit(num_qubits=1).z(0).z(0)
    opt = optimize(circuit)
    assert len(opt.gates) == 0


def test_cancel_adjacent_cx():
    circuit = Circuit(num_qubits=2).cx(0, 1).cx(0, 1)
    opt = optimize(circuit)
    assert len(opt.gates) == 0


def test_cancel_adjacent_swap():
    circuit = Circuit(num_qubits=2).swap(0, 1).swap(0, 1)
    opt = optimize(circuit)
    assert len(opt.gates) == 0


def test_merge_rz():
    """rz(a) rz(b) → rz(a+b)."""
    circuit = Circuit(num_qubits=1).rz(0, 0.3).rz(0, 0.4)
    opt = optimize(circuit)
    assert len(opt.gates) == 1
    assert abs(opt.gates[0].params[0] - 0.7) < 1e-12


def test_merge_rx():
    circuit = Circuit(num_qubits=1).rx(0, 1.0).rx(0, 2.0)
    opt = optimize(circuit)
    assert len(opt.gates) == 1
    assert abs(opt.gates[0].params[0] - 3.0) < 1e-12


def test_rotate_identity_removed():
    """rz(2π) → removed."""
    circuit = Circuit(num_qubits=1).rz(0, 2 * math.pi)
    opt = optimize(circuit)
    assert len(opt.gates) == 0


def test_no_cancel_different_qubits():
    """Same gates on different qubits should not cancel."""
    circuit = Circuit(num_qubits=2).x(0).x(1)
    opt = optimize(circuit)
    assert len(opt.gates) == 2


def test_optimize_preserves_bell():
    """Optimized Bell state still produces correct correlations."""
    bell = Circuit(num_qubits=2).h(0).cx(0, 1)
    opt = optimize(bell)
    result = LocalSimulator(seed=42).run(opt, shots=2000)
    assert set(result.counts) <= {"00", "11"}


def test_optimize_full_stack():
    """Transpile → optimize → simulate on Indian backend."""
    from qorch import IndianQPU
    bell = Circuit(num_qubits=2).h(0).cx(0, 1)
    qpu = IndianQPU.from_preset("tifr-superconducting", seed=42)
    from qorch.transpiler import decompose
    from qorch.transpiler.gateset import TIFR_SUPERCONDUCTING
    decomposed = decompose(bell, TIFR_SUPERCONDUCTING)
    opt = optimize(decomposed)
    assert len(opt.gates) <= len(decomposed.gates)
    result = qpu.run(opt, shots=2000)
    assert result.counts.get("00", 0) + result.counts.get("11", 0) > 0
