"""End-to-end dynamic-circuit algorithms: teleportation + repetition code."""

from __future__ import annotations

import math

from qorch import Circuit, LocalSimulator
from qorch.dynamic import (
    run_repetition_code,
    run_teleportation,
    teleportation_circuit,
)


class TestTeleportation:
    def test_teleport_zero(self):
        sim = LocalSimulator(seed=1)
        marg = run_teleportation(sim, state_prep=None, shots=4000)
        assert marg["0"] > 0.97  # |0> teleported faithfully

    def test_teleport_one(self):
        sim = LocalSimulator(seed=1)
        marg = run_teleportation(sim, state_prep=Circuit(1).x(0), shots=4000)
        assert marg["1"] > 0.97  # |1> teleported faithfully

    def test_teleport_rotated_state_matches_direct(self):
        """A partially-excited state teleports with the right population."""
        sim = LocalSimulator(seed=2)
        theta = math.pi / 3  # P(1) = sin^2(theta/2)
        marg = run_teleportation(sim, state_prep=Circuit(1).ry(0, theta), shots=8000)
        expected_p1 = math.sin(theta / 2) ** 2
        assert abs(marg["1"] - expected_p1) < 0.05

    def test_teleportation_is_dynamic(self):
        assert teleportation_circuit().is_dynamic


class TestRepetitionCode:
    def test_no_error_preserves_logical_one(self):
        sim = LocalSimulator(seed=1)
        res = run_repetition_code(sim, state_prep=Circuit(1).x(0), error_qubit=None, shots=2000)
        assert res.logical_distribution["1"] > 0.97

    def test_single_bit_flip_is_corrected(self):
        """A bit-flip on any one data qubit is detected and corrected."""
        sim = LocalSimulator(seed=3)
        for err in (0, 1, 2):
            res = run_repetition_code(
                sim, state_prep=Circuit(1).x(0), error_qubit=err, shots=2000
            )
            assert res.logical_distribution["1"] > 0.97, f"error on q{err} not corrected"

    def test_logical_zero_with_error_corrected(self):
        sim = LocalSimulator(seed=4)
        res = run_repetition_code(sim, state_prep=None, error_qubit=2, shots=2000)
        assert res.logical_distribution["0"] > 0.97

    def test_syndrome_identifies_error_location(self):
        """Each single-qubit error produces its distinct, deterministic syndrome."""
        sim = LocalSimulator(seed=5)
        syndromes = {}
        for err in (0, 1, 2):
            res = run_repetition_code(
                sim, state_prep=Circuit(1).x(0), error_qubit=err, shots=1000
            )
            # deterministic syndrome ⇒ exactly one syndrome string with weight 1.0
            top = max(res.syndrome_distribution, key=res.syndrome_distribution.get)
            assert res.syndrome_distribution[top] > 0.97
            syndromes[err] = top
        # the three errors map to three distinct non-zero syndromes
        assert len(set(syndromes.values())) == 3
        assert "00" not in syndromes.values()
