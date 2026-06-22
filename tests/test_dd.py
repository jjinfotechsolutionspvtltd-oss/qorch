"""Tests for Dynamical Decoupling mitigation."""

from qorch import Circuit
from qorch.mitigation.dd import insert_dd


def test_xy4_inserts_four_gates_on_idle_qubit():
    """H on q0 then CX on q0,q1 — q1 sits idle during H; DD should fill."""
    circuit = Circuit(num_qubits=2).h(0).cx(0, 1)
    dd = insert_dd(circuit, sequence="xy4", qubits=(1,))
    # Original: 2 gates. XY-4 inserts 4 gates in q1's idle gap.
    assert len(dd.gates) == 6


def test_xy8_inserts_eight_gates():
    circuit = Circuit(num_qubits=2).h(0).cx(0, 1)
    dd = insert_dd(circuit, sequence="xy8", qubits=(1,))
    assert len(dd.gates) == 10


def test_cpmg_inserts_two_gates():
    circuit = Circuit(num_qubits=2).h(0).cx(0, 1)
    dd = insert_dd(circuit, sequence="cpmg", qubits=(1,))
    assert len(dd.gates) == 4


def test_hahn_echo_inserts_two_x_gates():
    circuit = Circuit(num_qubits=2).h(0).cx(0, 1)
    dd = insert_dd(circuit, sequence="hahn", qubits=(1,))
    assert len(dd.gates) == 4


def test_no_idle_means_no_insertion():
    """Continuous gates on a qubit → no DD insertion."""
    circuit = Circuit(num_qubits=1).h(0).x(0).z(0)
    dd = insert_dd(circuit, sequence="xy4", qubits=(0,))
    assert len(dd.gates) == 3  # unchanged


def test_dd_preserves_circuit_logic():
    """DD on an idle qubit should not change the Bell-state outcome on noiseless sim."""
    from qorch import LocalSimulator
    circuit = Circuit(num_qubits=2).h(0).cx(0, 1)
    dd = insert_dd(circuit, sequence="xy4", qubits=(1,))
    result = LocalSimulator(seed=42).run(dd, shots=2000)
    assert set(result.counts) <= {"00", "11"}


def test_unknown_sequence_raises():
    circuit = Circuit(num_qubits=1).h(0)
    try:
        insert_dd(circuit, sequence="nonexistent")
        assert False, "expected ValueError"
    except ValueError:
        pass
