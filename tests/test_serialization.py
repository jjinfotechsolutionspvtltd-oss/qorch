"""Tests for Circuit JSON serialization."""

from __future__ import annotations

from qorch import Circuit


class TestCircuitSerialization:
    def test_to_json_basic(self):
        c = Circuit(2).h(0).cx(0, 1)
        j = c.to_json()
        assert '"num_qubits": 2' in j
        assert '"name": "h"' in j
        assert '"name": "cx"' in j

    def test_round_trip(self):
        c = Circuit(3).h(0).cx(0, 1).rx(2, 0.5).measure(0, 2)
        j = c.to_json()
        c2 = Circuit.from_json(j)
        assert c2.num_qubits == 3
        assert c2.readout_qubits == (0, 2)
        assert len(c2.gates) == 3

    def test_parametric_gates(self):
        c = Circuit(1).rx(0, 3.14159)
        j = c.to_json()
        assert "3.14159" in j
        c2 = Circuit.from_json(j)
        assert c2.gates[0].params[0] == 3.14159

    def test_empty_circuit(self):
        c = Circuit(1)
        j = c.to_json()
        c2 = Circuit.from_json(j)
        assert c2.num_qubits == 1
        assert len(c2.gates) == 0

    def test_measured_qubits(self):
        c = Circuit(4).h(0).cx(0, 1).measure(0, 3)
        j = c.to_json()
        c2 = Circuit.from_json(j)
        assert c2.measured == (0, 3)

    def test_to_json_parseable(self):
        import json
        c = Circuit(5).h(0).cx(1, 2).swap(0, 3).ms(0, 1, 0.25)
        j = c.to_json()
        data = json.loads(j)
        assert data["num_qubits"] == 5
        assert len(data["gates"]) == 4

    def test_round_trip_ms(self):
        c = Circuit(2).ms(0, 1, 0.25)
        j = c.to_json()
        c2 = Circuit.from_json(j)
        assert c2.gates[0].name == "ms"
        assert abs(c2.gates[0].params[0] - 0.25) < 1e-10
