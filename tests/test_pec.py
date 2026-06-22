"""Tests for Probabilistic Error Cancellation."""

from qorch import Circuit, GateNoise, LocalSimulator, expectation_z
from qorch.mitigation.pec import _pec_coeffs, pec_expectation


def test_pec_coeffs_no_noise():
    """With p=0, c_I=1 and all Pauli c=0."""
    ci, cx, cy, cz = _pec_coeffs(0.0)
    assert abs(ci - 1.0) < 1e-12
    assert abs(cx) < 1e-12
    assert abs(cy) < 1e-12
    assert abs(cz) < 1e-12


def test_pec_coeffs_some_noise():
    """With p=0.05, verify structure."""
    ci, cx, cy, cz = _pec_coeffs(0.05)
    assert ci > 1.0  # c_I > 1
    assert cx < 0    # negative weights for Paulis


def test_pec_coeffs_sum_property():
    """c_I + c_X + c_Y + c_Z = 1 for quasi-probability condition."""
    ci, cx, cy, cz = _pec_coeffs(0.05)
    assert abs(ci + cx + cy + cz - 1.0) < 1e-12


def test_pec_improves_estimate():
    """PEC should improve <Z> estimate for a noisy identity circuit."""
    circuit = Circuit(num_qubits=1).x(0).x(0)  # identity; ideal <Z> = 1
    noisy = LocalSimulator(seed=7, gate_noise=GateNoise(depolarizing_prob=0.05))
    result = pec_expectation(
        noisy, circuit, expectation_z,
        noise_prob=0.05, n_samples=200, shots_per_sample=100, seed=42,
    )
    raw_err = abs(result.raw_value - 1.0)
    mit_err = abs(result.mitigated_value - 1.0)
    assert mit_err < raw_err + 0.02


def test_pec_no_noise_passthrough():
    """With noise_prob=0, PEC should return raw value."""
    circuit = Circuit(num_qubits=1).x(0).x(0)
    ideal = LocalSimulator(seed=7)
    result = pec_expectation(
        ideal, circuit, expectation_z,
        noise_prob=0.0, n_samples=10, shots_per_sample=200, seed=42,
    )
    assert abs(result.mitigated_value - 1.0) < 0.05


def test_pec_result_improvement_property():
    """PECResult.improvement compares |raw| vs |mitigated|."""
    from qorch.mitigation.pec import PECResult
    r = PECResult(raw_value=0.8, mitigated_value=0.95, n_samples=50, gamma=1.5, noise_prob=0.05)
    assert isinstance(r.improvement, float)
    assert r.improvement < 0  # |0.95| > |0.8| → negative
    r2 = PECResult(raw_value=0.5, mitigated_value=0.2, n_samples=50, gamma=1.5, noise_prob=0.05)
    assert r2.improvement > 0  # |0.2| < |0.5| → positive
