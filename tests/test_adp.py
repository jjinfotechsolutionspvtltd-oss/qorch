"""Tests for ADP — Application Development Platform."""

from __future__ import annotations

import math

from qorch import Circuit, LocalSimulator


class TestQFT:
    def test_qft_circuit_builds(self):
        from qorch.adp import qft_circuit

        c = qft_circuit(3)
        assert c.num_qubits == 3
        assert len(c.gates) > 0

    def test_qft_circuit_2q(self):
        from qorch.adp import qft_circuit

        c = qft_circuit(2)
        names = [g.name for g in c.gates]
        assert "h" in names
        assert "cx" in names

    def test_run_qft_returns_result(self):
        from qorch.adp import run_qft

        sim = LocalSimulator(seed=0)
        result = run_qft(sim, num_qubits=2, shots=128)
        assert result.num_qubits == 2
        assert sum(result.counts.values()) == 128

    def test_run_qft_with_input_state(self):
        from qorch.adp import run_qft

        sim = LocalSimulator(seed=0)
        inp = Circuit(2).x(0)  # |10>
        result = run_qft(sim, num_qubits=2, input_state=inp, shots=128)
        assert result.num_qubits == 2


class TestGrover:
    def test_grover_diffusion_builds(self):
        from qorch.adp import grover_diffusion

        c = grover_diffusion(3)
        assert c.num_qubits == 3

    def test_oracle_by_bitstring_builds(self):
        from qorch.adp import oracle_by_bitstring

        c = oracle_by_bitstring(3, "101")
        assert c.num_qubits == 3

    def test_run_grover_2qubit(self):
        from qorch.adp import run_grover

        sim = LocalSimulator(seed=0)
        result = run_grover(sim, num_qubits=2, marked="11", shots=256)
        assert result.iterations >= 1
        assert result.marked_states == ("11",)

    def test_run_grover_multi_marked(self):
        from qorch.adp import run_grover

        sim = LocalSimulator(seed=0)
        result = run_grover(sim, num_qubits=3, marked=("000", "111"), shots=256)
        assert len(result.marked_states) == 2
        assert len(result.top_outcomes) <= 5


class TestQAOA:
    def test_qaoa_cost_circuit_builds(self):
        from qorch.adp import qaoa_cost_circuit

        c = qaoa_cost_circuit(3, 0.5, [(0, 1), (1, 2)])
        assert c.num_qubits == 3
        assert "cx" in [g.name for g in c.gates]

    def test_qaoa_mixer_circuit_builds(self):
        from qorch.adp import qaoa_mixer_circuit

        c = qaoa_mixer_circuit(3, 0.5)
        assert c.num_qubits == 3
        assert "rx" in [g.name for g in c.gates]

    def test_run_qaoa_returns_result(self):
        from qorch.adp import run_qaoa

        sim = LocalSimulator(seed=0)
        edges = [(0, 1), (1, 2), (0, 2)]
        result = run_qaoa(sim, num_qubits=3, edges=edges, gamma=0.5, beta=0.5, shots=256)
        assert result.num_qubits == 3
        assert result.best_solution != ""

    def test_maxcut_value(self):
        from qorch.adp import _maxcut_value

        val = _maxcut_value("00", [(0, 1)])
        assert val == 0.0
        val = _maxcut_value("01", [(0, 1)])
        assert val == 1.0
        val = _maxcut_value("10", [(0, 1)])
        assert val == 1.0
        val = _maxcut_value("11", [(0, 1)])
        assert val == 0.0


class TestVQE:
    def test_vqe_runs_and_returns_result(self):
        from qorch.adp import run_vqe

        sim = LocalSimulator(seed=0)
        result = run_vqe(sim, shots=512, max_iterations=3, initial_guess=(0.5,))
        assert result.num_qubits == 2
        assert isinstance(result.optimal_value, float)


class TestQPE:
    def test_qpe_circuit_builds(self):
        from qorch.adp import qpe_circuit

        u = Circuit(1).rz(0, math.pi / 2)
        c = qpe_circuit(3, u)
        assert c.num_qubits == 4

    def test_run_qpe_returns_result(self):
        from qorch.adp import run_qpe

        sim = LocalSimulator(seed=0)
        u = Circuit(1).rz(0, math.pi / 4)
        result = run_qpe(sim, phase_qubits=3, unitary=u, shots=256)
        assert result.num_qubits == 3
        assert isinstance(result.counts, dict)
