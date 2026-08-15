"""Fan-out scheduling and a job store that survives the process.

The FIFO scheduler blocks on each circuit in turn. Against queued hardware the
wait dominates, so the useful move is to submit everything first and collect
afterwards — and to write the handles down, because a handle held only in memory
is lost with the process, along with a job that may be hours into a queue.
"""

from __future__ import annotations

import pytest

from qorch import Circuit
from qorch.async_scheduler import (
    AsyncScheduler,
    JobStore,
    cost_based_policy,
    first_fit_policy,
)
from qorch.backends.async_backend import JobStatus, LocalAsyncSimulator
from qorch.backends.base import DeviceCalibration, QubitCalibration


def _bell() -> Circuit:
    return Circuit(2).h(0).cx(0, 1).measure(0, 1)


def _jobs(n: int = 3):
    return [(_bell(), 100, f"job-{i}") for i in range(n)]


# ── fan-out ──────────────────────────────────────────────────────────────


def test_a_batch_runs_and_returns_a_result_per_circuit() -> None:
    scheduler = AsyncScheduler(backends=[LocalAsyncSimulator(seed=1)])
    results = scheduler.run_batch(_jobs(3))

    assert len(results) == 3
    assert [r.label for r in results] == ["job-0", "job-1", "job-2"]
    assert all(sum(r.result.counts.values()) == 100 for r in results)


def test_everything_is_submitted_before_anything_is_collected() -> None:
    """That ordering is the entire point of fanning out."""
    scheduler = AsyncScheduler(backends=[LocalAsyncSimulator(seed=2)])
    handles = scheduler.submit_batch(_jobs(4))

    assert len(handles) == 4
    assert len({h.job_id for h in handles}) == 4
    assert len(scheduler._pending) == 4          # none collected yet

    assert len(scheduler.collect()) == 4
    assert scheduler._pending == []


def test_results_preserve_submission_order() -> None:
    scheduler = AsyncScheduler(backends=[LocalAsyncSimulator(seed=3)])
    results = scheduler.run_batch([(Circuit(1).x(0).measure(0), 10, f"j{i}")
                                   for i in range(5)])
    assert [r.label for r in results] == [f"j{i}" for i in range(5)]


def test_a_failing_job_is_reported_without_sinking_the_batch() -> None:
    """One bad circuit must not cost the results of the others."""
    from qorch.backends.simulator import LocalSimulator

    class Flaky(LocalSimulator):
        def run(self, circuit, shots=1024):
            if circuit.num_qubits == 1:
                raise ValueError("device offline")
            return super().run(circuit, shots)

    scheduler = AsyncScheduler(backends=[LocalAsyncSimulator(backend=Flaky())])
    results = scheduler.run_batch([
        (_bell(), 50, "good"),
        (Circuit(1).x(0).measure(0), 50, "bad"),
    ])

    by_label = {r.label: r for r in results}
    assert by_label["good"].result is not None
    assert by_label["bad"].result is None
    assert "device offline" in by_label["bad"].error


# ── selection policy ─────────────────────────────────────────────────────


def test_first_fit_rejects_a_circuit_that_does_not_fit() -> None:
    small = LocalAsyncSimulator()
    with pytest.raises(ValueError, match="no backend can run"):
        first_fit_policy(Circuit(40), [small])


def test_cost_based_policy_prefers_the_better_device() -> None:
    """Fitting is not the same as good: rank by predicted fidelity."""
    def device(error: float, readout: float) -> DeviceCalibration:
        return DeviceCalibration(
            qubits=tuple(
                QubitCalibration(t1_us=90, t2_us=80, readout_fidelity=readout,
                                 single_qubit_error=error)
                for _ in range(4)
            ),
            two_qubit_error={(0, 1): error * 10},
        )

    class Calibrated(LocalAsyncSimulator):
        def __init__(self, calibration, name):
            super().__init__()
            self._calibration = calibration
            self.name = name

        def calibration(self):
            return self._calibration

    noisy = Calibrated(device(0.05, 0.85), "noisy")
    clean = Calibrated(device(0.0005, 0.999), "clean")

    chosen = cost_based_policy(_bell(), [noisy, clean])
    assert chosen.name == "clean"


def test_cost_based_policy_still_respects_capacity() -> None:
    with pytest.raises(ValueError, match="no backend can run"):
        cost_based_policy(Circuit(40), [LocalAsyncSimulator()])


def test_an_uncalibrated_backend_ranks_below_a_calibrated_one() -> None:
    """An unknown device is not evidence of a good one."""
    class Calibrated(LocalAsyncSimulator):
        def __init__(self):
            super().__init__()
            self.name = "known"

        def calibration(self):
            return DeviceCalibration(qubits=tuple(
                QubitCalibration(readout_fidelity=0.999, single_qubit_error=0.0001)
                for _ in range(4)
            ))

    unknown = LocalAsyncSimulator()
    unknown.name = "unknown"
    assert cost_based_policy(_bell(), [unknown, Calibrated()]).name == "known"


# ── persistence ──────────────────────────────────────────────────────────


def test_submitted_jobs_are_written_down() -> None:
    store = JobStore()
    scheduler = AsyncScheduler(backends=[LocalAsyncSimulator(seed=4)], store=store)
    handle = scheduler.submit(_bell(), shots=100, label="persisted")

    stored = store.get(handle.job_id)
    assert stored is not None
    assert stored.label == "persisted"
    assert stored.status is JobStatus.QUEUED


def test_completion_updates_the_stored_status() -> None:
    store = JobStore()
    scheduler = AsyncScheduler(backends=[LocalAsyncSimulator(seed=5)], store=store)
    handle = scheduler.submit(_bell(), shots=50)
    scheduler.collect()

    assert store.get(handle.job_id).status is JobStatus.DONE
    assert store.unfinished() == []


def test_unfinished_lists_what_a_restart_should_reclaim() -> None:
    store = JobStore()
    scheduler = AsyncScheduler(backends=[LocalAsyncSimulator(seed=6)], store=store)
    scheduler.submit(_bell(), shots=10, label="a")
    scheduler.submit(_bell(), shots=10, label="b")

    unfinished = store.unfinished()
    assert {job.label for job in unfinished} == {"a", "b"}


def test_a_store_survives_being_reopened(tmp_path) -> None:
    """The point of persisting: a new process can pick the job back up."""
    path = tmp_path / "jobs.db"
    store = JobStore(path)
    scheduler = AsyncScheduler(backends=[LocalAsyncSimulator(seed=7)], store=store)
    handle = scheduler.submit(_bell(), shots=25, label="overnight")
    store.close()

    reopened = JobStore(path)
    recovered = reopened.get(handle.job_id)
    assert recovered is not None
    assert recovered.label == "overnight"
    assert recovered.handle == handle
    reopened.close()


def test_recording_the_same_job_twice_does_not_duplicate_it() -> None:
    store = JobStore()
    scheduler = AsyncScheduler(backends=[LocalAsyncSimulator(seed=8)], store=store)
    handle = scheduler.submit(_bell(), shots=10)
    store.record(handle, "again")

    assert len(store.all_jobs()) == 1


def test_an_unknown_job_id_returns_nothing() -> None:
    assert JobStore().get("not-a-job") is None


def test_a_scheduler_without_a_store_still_works() -> None:
    """Persistence is optional; it must not be load-bearing for execution."""
    scheduler = AsyncScheduler(backends=[LocalAsyncSimulator(seed=9)])
    assert len(scheduler.run_batch(_jobs(2))) == 2
