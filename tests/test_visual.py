"""Tests for ASCII circuit drawer."""

from __future__ import annotations

from qorch import Circuit
from qorch.visual import draw_circuit


class TestDrawCircuit:
    def test_empty_circuit(self):
        c = Circuit(1)
        out = draw_circuit(c)
        assert "q[0]" in out

    def test_single_gate(self):
        c = Circuit(1).h(0)
        out = draw_circuit(c)
        assert "q[0]" in out
        assert "H" in out

    def test_two_qubits(self):
        c = Circuit(2).h(0).cx(0, 1)
        out = draw_circuit(c)
        assert "q[0]" in out
        assert "q[1]" in out

    def test_cx_displayed(self):
        c = Circuit(2).cx(0, 1)
        out = draw_circuit(c)
        assert "●" in out

    def test_three_qubit_circuit(self):
        c = Circuit(3).h(0).cx(0, 1).cx(1, 2)
        out = draw_circuit(c)
        assert "q[0]" in out
        assert "q[1]" in out
        assert "q[2]" in out

    def test_param_gate(self):
        c = Circuit(1).rx(0, 3.14159)
        out = draw_circuit(c)
        assert "RX" in out

    def test_no_crash_big_circuit(self):
        c = Circuit(5)
        for i in range(4):
            c = c.h(i).cx(i, i + 1)
        out = draw_circuit(c)
        assert "q[0]" in out
        assert "q[4]" in out
