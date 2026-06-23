"""Tests for quantum state tomography."""

from __future__ import annotations

from qorch import Circuit, LocalSimulator
from qorch.tomography import (
    format_density_matrix,
    purity,
    state_tomography_1q,
    state_tomography_2q,
    trace_rho,
)


class TestStateTomography1Q:
    def test_h_zero_gives_plus_state(self):
        sim = LocalSimulator(seed=0)
        circ = Circuit(1).h(0)
        result = state_tomography_1q(sim, circ, shots=4096)
        assert abs(result.pauli_expectations["<X>"] - 1.0) < 0.1
        assert abs(result.pauli_expectations["<Z>"] - 0.0) < 0.1
        assert abs(purity(result.rho) - 1.0) < 0.05
        assert abs(trace_rho(result.rho) - 1.0) < 0.05

    def test_zero_state(self):
        sim = LocalSimulator(seed=0)
        circ = Circuit(1)
        result = state_tomography_1q(sim, circ, shots=4096)
        assert abs(result.pauli_expectations["<Z>"] - 1.0) < 0.1

    def test_one_state(self):
        sim = LocalSimulator(seed=0)
        circ = Circuit(1).x(0)
        result = state_tomography_1q(sim, circ, shots=4096)
        assert abs(result.pauli_expectations["<Z>"] + 1.0) < 0.1


class TestStateTomography2Q:
    def test_bell_state_fidelity(self):
        sim = LocalSimulator(seed=0)
        circ = Circuit(2).h(0).cx(0, 1)
        result = state_tomography_2q(sim, circ, shots=4096)
        assert abs(result.rho[0][0].real - 0.5) < 0.05
        assert abs(result.rho[0][3].real - 0.5) < 0.05
        assert abs(result.rho[3][3].real - 0.5) < 0.05
        assert abs(result.rho[3][0].real - 0.5) < 0.05
        assert abs(purity(result.rho) - 1.0) < 0.05
        assert abs(trace_rho(result.rho) - 1.0) < 0.05

    def test_separable_state(self):
        sim = LocalSimulator(seed=0)
        circ = Circuit(2).h(0)
        result = state_tomography_2q(sim, circ, shots=4096)
        diag_sum = sum(abs(result.rho[i][i].real) for i in range(4))
        assert abs(diag_sum - 1.0) < 0.05


class TestFormatDensityMatrix:
    def test_format_returns_string(self):
        rho = [[1 + 0j, 0j], [0j, 0j]]
        result = format_density_matrix(rho)
        assert isinstance(result, str)
        assert "1." in result

    def test_format_imaginary_only(self):
        rho = [[1j, 0j], [0j, 0j]]
        result = format_density_matrix(rho)
        assert "j" in result

    def test_format_mixed_real_imag(self):
        rho = [[0.5 + 0.3j, 0j], [0j, 0.5 - 0.3j]]
        result = format_density_matrix(rho)
        assert "+" in result or "-" in result

    def test_trace_and_purity_edge_cases(self):
        assert trace_rho([[1j, 0j], [0j, 0j]]) == 1.0
        rho = [[0.5 + 0j, 0.5 + 0j], [0.5 + 0j, 0.5 + 0j]]
        assert abs(purity(rho) - 1.0) < 1e-12


class TestTomographyEdgeCases:
    def test_rotate_to_basis_unknown(self):
        from qorch.tomography import _rotate_to_basis
        try:
            _rotate_to_basis(Circuit(1), 0, "W")
            assert False
        except ValueError:
            pass

    def test_expectation_1q_empty(self):
        from qorch.tomography import _expectation_1q
        assert _expectation_1q({}, 0, 0) == 0.0

    def test_expectation_2q_empty(self):
        from qorch.tomography import _expectation_2q
        assert _expectation_2q({}, 0) == 0.0

    def test_expectation_2q_single_qubit_count(self):
        from qorch.tomography import _expectation_2q
        # Single-qubit counts should still be handled
        result = _expectation_2q({"0": 10, "1": 5}, 15)
        assert abs(result - (10 - 5) / 15) < 1e-12
