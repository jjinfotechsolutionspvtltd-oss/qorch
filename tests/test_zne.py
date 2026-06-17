"""QS-004 M3: zero-noise extrapolation, the depth-scaling mitigation technique."""

from __future__ import annotations

import pytest

from qorch import (
    Circuit,
    GateNoise,
    LocalSimulator,
    expectation_z,
    fold_circuit,
    zne_expectation,
)


# --- folding ----------------------------------------------------------------
def test_fold_multiplies_gate_count_by_scale():
    circuit = Circuit(num_qubits=1).x(0).x(0)  # 2 gates
    assert len(fold_circuit(circuit, 1).gates) == 2
    assert len(fold_circuit(circuit, 3).gates) == 6
    assert len(fold_circuit(circuit, 5).gates) == 10


def test_fold_rejects_even_or_zero_scale():
    circuit = Circuit(num_qubits=1).x(0)
    for bad in (0, 2, 4):
        with pytest.raises(ValueError):
            fold_circuit(circuit, bad)


def test_fold_preserves_logic_noiselessly():
    """Folding is identity-padding: on a noiseless sim the result is unchanged."""
    bell = Circuit(num_qubits=2).h(0).cx(0, 1)
    folded = fold_circuit(bell, 5)
    assert set(LocalSimulator(seed=1).run(folded, 2000).counts) <= {"00", "11"}


# --- observable -------------------------------------------------------------
def test_expectation_z_extremes_and_superposition():
    assert expectation_z({"0": 1000}) == pytest.approx(1.0)
    assert expectation_z({"1": 1000}) == pytest.approx(-1.0)
    assert expectation_z({"0": 500, "1": 500}) == pytest.approx(0.0)


# --- the payoff -------------------------------------------------------------
def test_zne_recovers_value_under_depolarizing_noise():
    """A circuit that ideally leaves |0> (so <Z> = 1) decays under gate noise; ZNE recovers it."""
    circuit = Circuit(num_qubits=1).x(0).x(0).x(0).x(0)  # 4 X's = identity; ideal <Z> = 1
    noisy = LocalSimulator(seed=7, gate_noise=GateNoise(depolarizing_prob=0.05))

    result = zne_expectation(noisy, circuit, expectation_z, scales=(1, 3, 5), shots=8192)

    ideal = 1.0
    raw_err = abs(result.raw - ideal)
    mit_err = abs(result.mitigated - ideal)

    assert raw_err > 0.05, "sanity: noise must actually degrade the raw value"
    assert mit_err < raw_err, "ZNE must move the estimate toward the zero-noise value"
    assert result.mitigated <= 1.02, "extrapolation must not blow past the physical bound"
