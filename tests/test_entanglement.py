"""Tests for entanglement benchmarks."""

from __future__ import annotations

from qorch import LocalSimulator
from qorch.entanglement import (
    BellFidelityResult,
    CHSHResult,
    EntanglementWitnessResult,
    bell_state_fidelity,
    chsh_s_value,
    entanglement_witness,
)


class TestBellFidelity:
    def test_noiseless_simulator(self):
        sim = LocalSimulator(seed=0)
        result = bell_state_fidelity(sim, shots=4096)
        assert isinstance(result, BellFidelityResult)
        assert result.fidelity is not None
        assert result.fidelity > 0.9


class TestCHSH:
    def test_noiseless_simulator(self):
        sim = LocalSimulator(seed=0)
        result = chsh_s_value(sim, shots=4096)
        assert isinstance(result, CHSHResult)
        # Ideal Bell state gives S = 2√2 ≈ 2.828
        if result.s_value is not None:
            assert result.s_value > 2.0


class TestEntanglementWitness:
    def test_noiseless_simulator(self):
        sim = LocalSimulator(seed=0)
        result = entanglement_witness(sim, shots=4096)
        assert isinstance(result, EntanglementWitnessResult)
        if result.witness_value is not None:
            assert result.witness_value < 0.0
