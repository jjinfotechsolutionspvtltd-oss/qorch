"""Tests for QMI binary encoding."""

from __future__ import annotations

from qorch import Circuit
from qorch.qmi import QMIEncoder, from_qmi, to_qmi


class TestQMIRoundTrip:
    def test_empty_circuit(self):
        c = Circuit(2)
        data = to_qmi(c)
        c2 = from_qmi(data)
        assert c2.num_qubits == 2
        assert len(c2.gates) == 0

    def test_bell_state(self):
        c = Circuit(2).h(0).cx(0, 1)
        data = to_qmi(c)
        c2 = from_qmi(data)
        assert c2.num_qubits == 2
        assert len(c2.gates) == 2
        assert c2.gates[0].name == "h"
        assert c2.gates[0].qubits == (0,)
        assert c2.gates[1].name == "cx"
        assert c2.gates[1].qubits == (0, 1)

    def test_param_gates(self):
        c = Circuit(2).rx(0, 1.5).ry(1, -0.5).rz(0, 3.14159)
        data = to_qmi(c)
        c2 = from_qmi(data)
        for g1, g2 in zip(c.gates, c2.gates):
            assert g1.name == g2.name
            assert g1.qubits == g2.qubits
            assert len(g1.params) == len(g2.params)
            for p1, p2 in zip(g1.params, g2.params):
                assert abs(p1 - p2) < 1e-6

    def test_ms_gate(self):
        c = Circuit(2).ms(0, 1, 0.25)
        data = to_qmi(c)
        c2 = from_qmi(data)
        assert c2.gates[0].name == "ms"
        assert abs(c2.gates[0].params[0] - 0.25) < 1e-6

    def test_all_gates(self):
        c = Circuit(4)
        c = c.h(0).x(1).y(2).z(3)
        c = c.sx(0).id(1).swap(2, 3)
        c = c.cx(0, 1).ms(2, 3, 0.25)
        c = c.rx(0, 0.1).ry(1, 0.2).rz(2, 0.3)
        data = to_qmi(c)
        c2 = from_qmi(data)
        assert len(c2.gates) == len(c.gates)
        for g1, g2 in zip(c.gates, c2.gates):
            assert g1.name == g2.name
            assert g1.qubits == g2.qubits

    def test_with_measurements(self):
        c = Circuit(3).h(0).cx(0, 1).measure(0, 1)
        data = to_qmi(c)
        c2 = from_qmi(data)
        assert c2.measured == (0, 1)

    def test_encoder_class(self):
        c = Circuit(1).h(0)
        data = QMIEncoder.encode(c)
        c2 = QMIEncoder.decode(data)
        assert c2.num_qubits == 1
        assert c2.gates[0].name == "h"

    def test_hexdump_returns_string(self):
        c = Circuit(2).h(0).cx(0, 1)
        data = to_qmi(c)
        dump = QMIEncoder.hexdump(data)
        assert isinstance(dump, str)
        assert "QMI" in dump

    def test_bad_magic_raises(self):
        try:
            from_qmi(b"XXXX" + b"\x00" * 20)
            assert False
        except ValueError:
            pass
