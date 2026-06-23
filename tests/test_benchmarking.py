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

    def test_qv_sweep_returns_result(self):
        sim = LocalSimulator(seed=0)
        from qorch.benchmarking import qv_sweep
        result = qv_sweep(sim, start_width=2, end_width=3, trials=10, shots=256, seed=42, stop_on_fail=False)
        assert len(result.results) >= 1
        assert isinstance(result.quantum_volume, int)
        assert result.quantum_volume >= 1

    def test_qv_sweep_stops_on_fail(self):
        class FailingBackend:
            name = "failing"
            def properties(self):
                from qorch.backends.base import BackendProperties
                return BackendProperties(num_qubits=10, basis_gates=("h", "cx"), is_simulator=False)
            def run(self, circuit, shots=1024):
                from qorch.backends.base import JobResult
                return JobResult(counts={"0" * circuit.num_qubits: shots}, shots=shots, backend_name="failing")
        from qorch.benchmarking import qv_sweep
        # HOP of all-zeros is 0 for width > 1, so should fail
        result = qv_sweep(FailingBackend(), start_width=2, end_width=5, trials=2, shots=128, stop_on_fail=True)
        assert result.max_passing_width <= 1

    def test_qv_sweep_cli_runs(self):
        from qorch.cli import cmd_qv_sweep
        import argparse
        args = argparse.Namespace(backend="local-simulator", start=2, end=2, trials=2, shots=64, seed=42)
        cmd_qv_sweep(args)


class TestCrossEntropyBenchmarking:
    def test_xeb_returns_result(self):
        sim = LocalSimulator(seed=0)
        from qorch.benchmarking import cross_entropy_benchmarking
        result = cross_entropy_benchmarking(sim, num_qubits=2, depth=1,
                                             num_circuits=2, shots=128, seed=42)
        assert result.depth == 1
        assert result.num_circuits == 2
