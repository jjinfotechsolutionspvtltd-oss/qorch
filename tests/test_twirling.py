"""Tests for Pauli Twirling mitigation."""

from __future__ import annotations

from qorch import Circuit
from qorch.mitigation.twirling import twirl_circuit


class TestTwirling:
    def test_twirl_preserves_qubit_count(self):
        c = Circuit(3).h(0).cx(0, 1).x(2)
        t = twirl_circuit(c, seed=42)
        assert t.num_qubits == 3

    def test_twirl_may_add_gates(self):
        c = Circuit(4).h(0).h(1).h(2).h(3)
        t = twirl_circuit(c, seed=42)
        assert len(t.gates) >= len(c.gates)

    def test_twirl_non_clifford_untouched(self):
        c = Circuit(1).rx(0, 0.5)
        t = twirl_circuit(c, seed=42)
        assert t.gates == c.gates

    def test_twirl_empty_circuit(self):
        c = Circuit(1)
        t = twirl_circuit(c, seed=42)
        assert len(t.gates) == 0

    def test_twirl_deterministic_with_seed(self):
        c = Circuit(2).h(0).cx(0, 1)
        t1 = twirl_circuit(c, seed=42)
        t2 = twirl_circuit(c, seed=42)
        assert t1.gates == t2.gates

    def test_twirl_eventually_inserts_paulis(self):
        c = Circuit(4).h(0).h(1).h(2).h(3)
        for s in range(100):
            t = twirl_circuit(c, seed=s)
            if len(t.gates) > len(c.gates):
                return
        raise AssertionError("No non-identity Pauli found in 100 seeds")

    def test_twirl_cx_preserves_qubits(self):
        c = Circuit(3).cx(0, 2)
        t = twirl_circuit(c, seed=42)
        cx = next(g for g in t.gates if g.name == "cx")
        assert cx.qubits == (0, 2)

    def test_twirl_swap_preserved(self):
        c = Circuit(3).swap(0, 2)
        t = twirl_circuit(c, seed=42)
        sw = next(g for g in t.gates if g.name == "swap")
        assert sw is not None

    def test_twirl_paulis_pair_with_conjugate_on_h(self):
        for s in range(100):
            c = Circuit(1).h(0)
            t = twirl_circuit(c, seed=s)
            paulis = [g for g in t.gates if g.name in ("x", "y", "z")]
            if len(paulis) == 2:
                assert t.gates[0].name in ("x", "y", "z")
                assert t.gates[2].name in ("x", "y", "z")
                return
        raise AssertionError("No seed produced paired Paulis around H")

    def test_twirl_cx_has_even_pauli_count(self):
        for s in range(100):
            c = Circuit(2).cx(0, 1)
            t = twirl_circuit(c, seed=s)
            paulis = [g for g in t.gates if g.name in ("x", "y", "z")]
            if len(paulis) > 0:
                assert len(paulis) % 4 == 0
                return
        raise AssertionError("No seed produced Paulis around CX")
