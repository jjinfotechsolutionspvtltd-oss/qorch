"""Tests for batch scheduling."""

from __future__ import annotations

from qorch import Circuit, LocalSimulator
from qorch.scheduler import BatchJob, Scheduler, _best_fit


class TestBestFit:
    def test_best_fit_picks_smallest_adequate(self):
        big = LocalSimulator(seed=0)
        small = LocalSimulator(seed=0)
        # Override properties via monkey-patch
        small.properties = lambda: type("P", (), {"num_qubits": 2, "basis_gates": (), "is_simulator": True, "readout_fidelity": ()})()
        big.properties = lambda: type("P", (), {"num_qubits": 10, "basis_gates": (), "is_simulator": True, "readout_fidelity": ()})()
        c = Circuit(2)
        result = _best_fit(c, [big, small])
        assert result.properties().num_qubits == 2

    def test_best_fit_no_backend_raises(self):
        tiny = LocalSimulator(seed=0)
        tiny.properties = lambda: type("P", (), {"num_qubits": 1, "basis_gates": (), "is_simulator": True, "readout_fidelity": ()})()
        c = Circuit(5)
        try:
            _best_fit(c, [tiny])
            assert False
        except ValueError:
            pass


class TestRunBatch:
    def test_run_batch_two_circuits(self):
        sched = Scheduler(backends=[LocalSimulator(seed=0)])
        jobs = [
            BatchJob(circuit=Circuit(1).h(0).measure(0, 0), shots=64, label="bell"),
            BatchJob(circuit=Circuit(2).h(0).cx(0, 1).measure(0, 1), shots=64, label="ghz"),
        ]
        results = sched.run_batch(jobs)
        assert len(results) == 2
        for r in results:
            assert r.result is not None
            assert r.error is None

    def test_run_batch_with_error(self):
        sched = Scheduler(backends=[])
        c = Circuit(10)
        results = sched.run_batch([BatchJob(circuit=c, shots=64, label="fails")])
        assert len(results) == 1
        assert results[0].result is None
        assert results[0].error is not None
