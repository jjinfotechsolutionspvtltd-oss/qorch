"""Quantum error correction: repetition code, Steane code, threshold behavior."""

from __future__ import annotations

import pytest

from qorch.qec import (
    decode_repetition,
    repetition_logical_error_rate,
    run_repetition,
    run_steane,
    threshold_sweep,
)


class TestRepetitionCode:
    @pytest.mark.parametrize("distance", [3, 5, 7])
    @pytest.mark.parametrize("logical", [0, 1])
    def test_corrects_every_single_bit_flip(self, distance, logical):
        for q in range(distance):
            res = run_repetition(distance, logical=logical, errors=(q,))
            assert res.corrected, f"d={distance} L={logical} error q{q} not corrected"

    def test_distance_5_corrects_two_errors(self):
        res = run_repetition(5, logical=1, errors=(1, 3))
        assert res.corrected

    def test_no_error_preserves_logical(self):
        assert run_repetition(5, logical=1, errors=()).logical_out == 1
        assert run_repetition(5, logical=0, errors=()).logical_out == 0

    def test_decoder_minimum_weight(self):
        # syndrome "10" on d=3 → single error on qubit 0 (weight 1, not weight 2)
        assert decode_repetition("10") == [1, 0, 0]
        assert decode_repetition("00") == [0, 0, 0]


class TestSteaneCode:
    @pytest.mark.parametrize("logical", [0, 1])
    @pytest.mark.parametrize("pauli", ["x", "y"])
    def test_corrects_every_single_x_or_y_error(self, logical, pauli):
        for q in range(7):
            res = run_steane(logical=logical, error=(pauli, q))
            assert res.corrected, f"L={logical} {pauli} on q{q} not corrected"
            assert res.error_qubit == q, f"syndrome mislocated error: {res.error_qubit} != {q}"

    def test_no_error_zero_syndrome(self):
        res = run_steane(logical=1, error=None)
        assert res.syndrome == 0
        assert res.corrected

    def test_z_error_harmless_for_logical_z(self):
        # Z errors are invisible to a logical-Z read-out (no logical flip)
        assert run_steane(logical=0, error=("z", 3)).corrected

    def test_distinct_syndromes_per_qubit(self):
        syndromes = {run_steane(0, ("x", q)).syndrome for q in range(7)}
        assert syndromes == set(range(1, 8))  # 7 distinct non-zero Hamming indices


class TestThreshold:
    def test_distance_suppresses_logical_error_below_threshold(self):
        p = 0.08  # well below the 0.5 code-capacity pseudo-threshold
        r3 = repetition_logical_error_rate(3, p, trials=4000, seed=1)
        r5 = repetition_logical_error_rate(5, p, trials=4000, seed=1)
        r7 = repetition_logical_error_rate(7, p, trials=4000, seed=1)
        assert r3 > r5 > r7, f"logical error not suppressed with distance: {r3,r5,r7}"

    def test_threshold_sweep_shape(self):
        sweep = threshold_sweep(distances=(3, 5), physical_errors=(0.05, 0.1),
                                trials=1000, seed=2)
        assert set(sweep.logical_error) == {3, 5}
        assert set(sweep.logical_error[3]) == {0.05, 0.1}
        for d in (3, 5):
            for p in (0.05, 0.1):
                assert 0.0 <= sweep.logical_error[d][p] <= 1.0
