"""Toric surface code: geometry, MWPM decoder, distance, threshold, syndrome."""

from __future__ import annotations

import itertools

import pytest

from qorch.surface_code import (
    ToricCode,
    measured_toric_syndrome,
    toric_logical_error_rate,
    toric_threshold_sweep,
)


class TestHomology:
    def test_trivial_chains_not_logical(self):
        c = ToricCode(3)
        assert not c.logical_failure(set())

    def test_winding_loops_are_logical(self):
        c, L = ToricCode(3), 3
        # vertical-winding loop: a column of horizontal (dual) edges
        assert c.logical_failure({c.h(r, 0) for r in range(L)})
        # horizontal-winding loop: a row of vertical (dual) edges
        assert c.logical_failure({c.v(0, col) for col in range(L)})


class TestDecoderDistance:
    def test_net_chain_is_always_closed(self):
        """error ⊕ correction must have empty syndrome (a valid cycle)."""
        import random
        for L in (3, 4, 5):
            code = ToricCode(L)
            rng = random.Random(L)
            for _ in range(200):
                err = {e for e in range(code.num_edges) if rng.random() < 0.12}
                net = err ^ code.decode(code.syndrome(err))
                assert code.syndrome(net) == []

    @pytest.mark.parametrize("distance,max_weight", [(3, 1), (5, 2)])
    def test_corrects_all_errors_up_to_distance(self, distance, max_weight):
        """A distance-d code must correct every error of weight ≤ ⌊(d-1)/2⌋."""
        code = ToricCode(distance)
        E = code.num_edges
        for w in range(max_weight + 1):
            for combo in itertools.combinations(range(E), w):
                err = set(combo)
                net = err ^ code.decode(code.syndrome(err))
                assert not code.logical_failure(net), f"weight-{w} error {combo} failed"


class TestThreshold:
    def test_distance_suppresses_below_threshold(self):
        """Below the ~10% threshold, larger distance lowers the logical error."""
        p = 0.05
        r3 = toric_logical_error_rate(3, p, trials=6000, seed=1)
        r5 = toric_logical_error_rate(5, p, trials=6000, seed=1)
        assert r5 < r3, f"distance did not help below threshold: L3={r3}, L5={r5}"

    def test_distance_hurts_above_threshold(self):
        """Above threshold, larger distance makes things worse — the crossover."""
        p = 0.15
        r3 = toric_logical_error_rate(3, p, trials=6000, seed=1)
        r5 = toric_logical_error_rate(5, p, trials=6000, seed=1)
        assert r5 > r3, f"no crossover above threshold: L3={r3}, L5={r5}"

    def test_sweep_shape(self):
        sweep = toric_threshold_sweep(distances=(3, 5), physical_errors=(0.05, 0.12),
                                      trials=1500, seed=2)
        assert set(sweep.logical_error) == {3, 5}
        assert set(sweep.logical_error[3]) == {0.05, 0.12}


class TestQuantumSyndrome:
    def test_quantum_syndrome_matches_classical(self):
        """Syndrome extraction on the stabilizer simulator matches the model."""
        import random
        for L in (3, 4):
            code = ToricCode(L)
            rng = random.Random(L)
            for _ in range(15):
                err = {e for e in range(code.num_edges) if rng.random() < 0.15}
                assert measured_toric_syndrome(L, err) == set(code.syndrome(err))
