"""Tests for QASM emitter (to_qasm3) and round-trip consistency."""

from __future__ import annotations

from qorch import Circuit, from_qasm3, to_qasm3


class TestToQasm3:
    def test_empty_circuit(self):
        c = Circuit(2)
        qasm = to_qasm3(c)
        assert "OPENQASM 3.0" in qasm
        assert "qubit[2] q" in qasm

    def test_single_gate(self):
        c = Circuit(1).h(0)
        qasm = to_qasm3(c)
        assert "h q[0];" in qasm

    def test_cx_gate(self):
        c = Circuit(2).cx(0, 1)
        qasm = to_qasm3(c)
        assert "cx q[0], q[1];" in qasm

    def test_param_gate(self):
        c = Circuit(1).rx(0, 3.14159)
        qasm = to_qasm3(c)
        assert "rx" in qasm
        assert "3.14159" in qasm

    def test_measurement(self):
        c = Circuit(2).h(0).cx(0, 1).measure(0, 1)
        qasm = to_qasm3(c)
        assert "measure q[0];" in qasm
        assert "measure q[1];" in qasm

    def test_round_trip(self):
        c = Circuit(3).h(0).cx(0, 1).rx(2, 0.5).measure(0, 2)
        qasm = to_qasm3(c)
        c2 = from_qasm3(qasm)
        assert c2.num_qubits == 3
        assert c2.readout_qubits == (0, 2)

    def test_swap_gate(self):
        c = Circuit(2).swap(0, 1)
        qasm = to_qasm3(c)
        assert "swap q[0], q[1];" in qasm

    def test_ms_gate(self):
        c = Circuit(2).ms(0, 1, 0.25)
        qasm = to_qasm3(c)
        assert "ms(" in qasm
        assert "0.25" in qasm
