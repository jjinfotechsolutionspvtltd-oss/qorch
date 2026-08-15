"""Async execution: submit, poll, fetch, cancel.

``run`` blocks, which is right for a simulator and wrong for hardware — real
QPUs queue for minutes or hours. These tests pin the lifecycle every caller has
to handle, including the awkward states: cancelling something already finished,
fetching a job that failed, and using a handle that was serialized and revived
in what could have been another process.
"""

from __future__ import annotations

import json

import pytest

from qorch import Circuit, LocalSimulator
from qorch.backends.async_backend import (
    AsyncBackend,
    AuthenticatedBackend,
    JobHandle,
    JobNotFoundError,
    JobStatus,
    LocalAsyncSimulator,
)


def _bell() -> Circuit:
    return Circuit(2).h(0).cx(0, 1).measure(0, 1)


# ── lifecycle ────────────────────────────────────────────────────────────


def test_submit_returns_a_handle_without_running() -> None:
    """The point of submitting: it comes back before the work is done."""
    backend = LocalAsyncSimulator(seed=1)
    handle = backend.submit(_bell(), shots=100)

    assert isinstance(handle, JobHandle)
    assert handle.backend_name == backend.name
    assert handle.shots == 100
    assert backend._jobs[handle.job_id].result is None


def test_a_job_completes_and_yields_a_result() -> None:
    backend = LocalAsyncSimulator(seed=1)
    handle = backend.submit(_bell(), shots=500)

    result = backend.wait(handle)
    assert backend.status(handle) is JobStatus.DONE
    assert sum(result.counts.values()) == 500
    assert set(result.counts) == {"00", "11"}


def test_status_reaches_a_terminal_state() -> None:
    backend = LocalAsyncSimulator(seed=2)
    handle = backend.submit(_bell(), shots=50)
    backend.wait(handle)
    assert backend.status(handle).is_terminal


def test_result_can_be_fetched_without_polling_first() -> None:
    """A caller that just wants the answer should not have to drive the machine."""
    backend = LocalAsyncSimulator(seed=3)
    handle = backend.submit(_bell(), shots=100)
    assert sum(backend.result(handle).counts.values()) == 100


def test_run_is_submit_then_wait() -> None:
    """The async backend is still an ordinary Backend."""
    backend = LocalAsyncSimulator(seed=4)
    assert isinstance(backend, AsyncBackend)
    result = backend.run(_bell(), shots=200)
    assert sum(result.counts.values()) == 200


def test_async_results_match_the_synchronous_backend() -> None:
    """Going through the queue must not change the physics."""
    circuit = _bell()
    direct = LocalSimulator(seed=9).run(circuit, shots=1000).counts
    queued = LocalAsyncSimulator(seed=9).run(circuit, shots=1000).counts
    assert direct == queued


# ── cancellation ─────────────────────────────────────────────────────────


def test_a_queued_job_can_be_cancelled() -> None:
    backend = LocalAsyncSimulator(seed=5)
    handle = backend.submit(_bell(), shots=100)

    assert backend.cancel(handle) is True
    assert backend.status(handle) is JobStatus.CANCELLED


def test_cancelling_a_finished_job_reports_that_it_did_nothing() -> None:
    """Returning True here would tell the caller work was stopped that was not."""
    backend = LocalAsyncSimulator(seed=6)
    handle = backend.submit(_bell(), shots=100)
    backend.wait(handle)

    assert backend.cancel(handle) is False
    assert backend.status(handle) is JobStatus.DONE


def test_a_cancelled_job_has_no_result() -> None:
    backend = LocalAsyncSimulator(seed=7)
    handle = backend.submit(_bell(), shots=100)
    backend.cancel(handle)

    with pytest.raises(RuntimeError, match="cancelled"):
        backend.result(handle)


# ── failure and unknown jobs ─────────────────────────────────────────────


def test_a_failing_job_surfaces_as_an_error_status() -> None:
    class Broken(LocalSimulator):
        def run(self, circuit, shots=1024):
            raise ValueError("device offline")

    backend = LocalAsyncSimulator(backend=Broken())
    handle = backend.submit(_bell(), shots=10)

    assert backend.status(handle) is JobStatus.ERROR
    with pytest.raises(RuntimeError, match="device offline"):
        backend.result(handle)


def test_an_unknown_handle_is_rejected_clearly() -> None:
    backend = LocalAsyncSimulator(seed=8)
    stray = JobHandle(job_id="does-not-exist", backend_name=backend.name, shots=10)

    with pytest.raises(JobNotFoundError, match="does-not-exist"):
        backend.status(stray)


# ── handles are plain, serializable data ─────────────────────────────────


def test_a_handle_survives_a_round_trip_through_json() -> None:
    """This is what lets an overnight job be reclaimed after a restart."""
    backend = LocalAsyncSimulator(seed=10)
    handle = backend.submit(_bell(), shots=250)

    revived = JobHandle.from_dict(json.loads(json.dumps(handle.to_dict())))
    assert revived == handle
    assert sum(backend.result(revived).counts.values()) == 250


def test_handles_are_unique_per_submission() -> None:
    backend = LocalAsyncSimulator(seed=11)
    first = backend.submit(_bell(), shots=10)
    second = backend.submit(_bell(), shots=10)
    assert first.job_id != second.job_id


# ── status semantics ─────────────────────────────────────────────────────


@pytest.mark.parametrize("status,terminal", [
    (JobStatus.QUEUED, False),
    (JobStatus.RUNNING, False),
    (JobStatus.DONE, True),
    (JobStatus.CANCELLED, True),
    (JobStatus.ERROR, True),
])
def test_terminal_states_are_the_ones_that_stop_a_poll_loop(status, terminal) -> None:
    assert status.is_terminal is terminal


def test_wait_times_out_rather_than_hanging() -> None:
    """A poll loop with no deadline is how a client hangs forever."""
    class NeverFinishes(LocalAsyncSimulator):
        def status(self, handle):
            return JobStatus.RUNNING

    backend = NeverFinishes(seed=12)
    handle = backend.submit(_bell(), shots=10)
    with pytest.raises(TimeoutError, match="still running"):
        backend.wait(handle, timeout=0.15, poll_interval=0.01)


# ── authentication ───────────────────────────────────────────────────────


class _NeedsToken(AuthenticatedBackend):
    credential_env_var = "QORCH_TEST_TOKEN"

    def __init__(self, present: bool) -> None:
        self._present = present

    def is_authenticated(self) -> bool:
        return self._present


def test_missing_credentials_raise_a_clear_error() -> None:
    with pytest.raises(PermissionError, match="QORCH_TEST_TOKEN"):
        _NeedsToken(present=False).require_authentication()


def test_authenticated_backend_passes_when_a_credential_exists() -> None:
    _NeedsToken(present=True).require_authentication()      # must not raise


def test_the_error_message_never_contains_a_secret() -> None:
    """It names the variable, never a value — errors end up in logs and tickets."""
    try:
        _NeedsToken(present=False).require_authentication()
    except PermissionError as exc:
        message = str(exc)
    assert "QORCH_TEST_TOKEN" in message
    assert "never logged or stored" in message
