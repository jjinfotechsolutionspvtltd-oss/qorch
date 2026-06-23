"""Tests for the benchmarking suite (RB, QV, XEB)."""

from __future__ import annotations

from qorch import LocalSimulator


class TestRandomizedBenchmarking:
    def test_rb_returns_result(self):
        sim = LocalSimulator(seed=0)
        from qorch.benchmarking import randomized_benchmarking
        result = randomized_benchmarking(sim, num_qubits=1, depths=(1, 2),
                                          circuits_per_depth=2, shots=128, seed=42)
        assert len(result.depths) == 2
        assert len(result.survival_probabilities) == 2
        for p in result.survival_probabilities:
            assert 0.0 <= p <= 1.0


class TestQuantumVolume:
    def test_qv_returns_result(self):
        sim = LocalSimulator(seed=0)
        from qorch.benchmarking import quantum_volume
        result = quantum_volume(sim, width=2, shots=256, trials=3, seed=42)
        assert result.width == 2
        assert result.depth == 2
        if result.heavy_output_probability is not None:
            assert 0.0 <= result.heavy_output_probability <= 1.0


class TestCrossEntropyBenchmarking:
    def test_xeb_returns_result(self):
        sim = LocalSimulator(seed=0)
        from qorch.benchmarking import cross_entropy_benchmarking
        result = cross_entropy_benchmarking(sim, num_qubits=2, depth=1,
                                             num_circuits=2, shots=128, seed=42)
        assert result.depth == 1
        assert result.num_circuits == 2
