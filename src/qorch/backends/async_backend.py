"""Asynchronous execution: submit, poll, fetch, cancel.

``Backend.run`` blocks until results come back, which is the right model for a
simulator and the wrong one for hardware. Real QPUs are queued: a job waits
behind other people's jobs, for minutes or hours, and the submitting process
should not be held open for it. A synchronous-only HAL forces every caller into
a blocking thread, and gives them nothing to persist if the process dies.

:class:`AsyncBackend` adds the four operations that lifecycle needs — submit,
poll, result, cancel — around a :class:`JobHandle` that is plain data. That last
point matters: a handle can be written to disk and used to reclaim a job from a
different process, which is what makes an overnight queue survivable.

Authentication is separate, in :class:`AuthenticatedBackend`, because the two
are independent: a local async simulator needs no credentials, and a synchronous
cloud backend might need them.
"""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

from qorch.backends.base import Backend, JobResult
from qorch.ir import Circuit


class JobStatus(str, Enum):
    """Where a submitted job is in its lifecycle."""

    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    CANCELLED = "cancelled"
    ERROR = "error"

    @property
    def is_terminal(self) -> bool:
        """True once the status can no longer change — stop polling."""
        return self in (JobStatus.DONE, JobStatus.CANCELLED, JobStatus.ERROR)


@dataclass(frozen=True)
class JobHandle:
    """A reference to a submitted job.

    Deliberately plain data with no live connection inside it, so it can be
    serialized, stored, and used later — possibly by a different process — to
    reclaim a job. A handle holding an open socket could not do that, and
    reclaiming work after a crash is most of the point of asynchronous
    submission.
    """

    job_id: str
    backend_name: str
    shots: int
    submitted_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, object]:
        return {
            "job_id": self.job_id,
            "backend_name": self.backend_name,
            "shots": self.shots,
            "submitted_at": self.submitted_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "JobHandle":
        shots = data["shots"]
        submitted = data["submitted_at"]
        if not isinstance(shots, (int, float, str)):
            raise ValueError(f"handle has a non-numeric shots field: {shots!r}")
        if not isinstance(submitted, (int, float, str)):
            raise ValueError(f"handle has a non-numeric timestamp: {submitted!r}")
        return cls(
            job_id=str(data["job_id"]),
            backend_name=str(data["backend_name"]),
            shots=int(shots),
            submitted_at=float(submitted),
        )


class JobNotFoundError(KeyError):
    """Raised when a handle names a job the backend does not know about."""


class AsyncBackend(Backend, ABC):
    """A backend whose execution can be started and collected separately.

    Implementations remain ordinary :class:`Backend` objects — ``run`` still
    works and is expected to be submit-then-wait — so nothing that already
    accepts a Backend needs to change to accept an async one.
    """

    @abstractmethod
    def submit(self, circuit: Circuit, shots: int = 1024) -> JobHandle:
        """Queue a circuit and return immediately."""

    @abstractmethod
    def status(self, handle: JobHandle) -> JobStatus:
        """Where the job is now. Cheap enough to call in a poll loop."""

    @abstractmethod
    def result(self, handle: JobHandle) -> JobResult:
        """Fetch a finished job's result, raising if it is not finished."""

    @abstractmethod
    def cancel(self, handle: JobHandle) -> bool:
        """Attempt to cancel. Returns whether the job was actually stopped."""

    def wait(
        self,
        handle: JobHandle,
        timeout: float | None = None,
        poll_interval: float = 0.05,
    ) -> JobResult:
        """Block until the job reaches a terminal state, then return its result.

        Provided so the common case does not have to hand-roll a poll loop, and
        so the timeout is enforced in one place rather than each caller's.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            current = self.status(handle)
            if current.is_terminal:
                return self.result(handle)
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(
                    f"job {handle.job_id} still {current.value} after {timeout}s"
                )
            time.sleep(poll_interval)

    def run(self, circuit: Circuit, shots: int = 1024) -> JobResult:
        """Synchronous execution, expressed as submit-then-wait."""
        return self.wait(self.submit(circuit, shots))


class AuthenticatedBackend(ABC):
    """Mixin for backends that require credentials.

    Credentials are never stored on the instance and never accepted as a
    constructor argument in this interface. They are read from the environment
    at the point of use, so a token cannot end up in a repr, a pickle, a log
    line, or a traceback — all of which happen readily to constructor arguments.
    """

    #: Environment variable holding the credential.
    credential_env_var: str = ""

    @abstractmethod
    def is_authenticated(self) -> bool:
        """Whether a usable credential is currently available."""

    def require_authentication(self) -> None:
        """Raise a clear, secret-free error if no credential is available."""
        if not self.is_authenticated():
            raise PermissionError(
                f"backend requires authentication: set the "
                f"{self.credential_env_var or '<credential>'} environment "
                f"variable (its value is never logged or stored)"
            )


@dataclass
class _Job:
    circuit: Circuit
    shots: int
    status: JobStatus = JobStatus.QUEUED
    result: JobResult | None = None
    error: str | None = None


class LocalAsyncSimulator(AsyncBackend):
    """An in-process async backend, so the protocol is exercised, not just declared.

    Execution is deferred rather than threaded: a job stays ``QUEUED`` until it
    is polled, then runs and becomes ``DONE``. That is enough to reproduce the
    lifecycle every caller has to handle — including cancelling something that
    has not started — without introducing concurrency into a library whose core
    is deliberately simple.
    """

    def __init__(self, backend: Backend | None = None, seed: int | None = None) -> None:
        from qorch.backends.simulator import LocalSimulator

        self._backend = backend or LocalSimulator(seed=seed)
        self._jobs: dict[str, _Job] = {}
        self.name = "local-async-simulator"

    def properties(self):
        return self._backend.properties()

    def validate(self, circuit: Circuit) -> None:
        self._backend.validate(circuit)

    def submit(self, circuit: Circuit, shots: int = 1024) -> JobHandle:
        self.validate(circuit)
        job_id = uuid.uuid4().hex
        self._jobs[job_id] = _Job(circuit=circuit, shots=shots)
        return JobHandle(job_id=job_id, backend_name=self.name, shots=shots)

    def _job(self, handle: JobHandle) -> _Job:
        try:
            return self._jobs[handle.job_id]
        except KeyError:
            raise JobNotFoundError(
                f"no job {handle.job_id!r} on {self.name}"
            ) from None

    def status(self, handle: JobHandle) -> JobStatus:
        job = self._job(handle)
        if job.status is JobStatus.QUEUED:
            # Deferred execution: polling is what advances the job.
            job.status = JobStatus.RUNNING
            try:
                job.result = self._backend.run(job.circuit, job.shots)
            except Exception as exc:                       # noqa: BLE001
                job.status = JobStatus.ERROR
                job.error = str(exc)
            else:
                job.status = JobStatus.DONE
        return job.status

    def result(self, handle: JobHandle) -> JobResult:
        job = self._job(handle)
        if job.status is JobStatus.QUEUED:
            self.status(handle)
        if job.status is JobStatus.ERROR:
            raise RuntimeError(f"job {handle.job_id} failed: {job.error}")
        if job.status is JobStatus.CANCELLED:
            raise RuntimeError(f"job {handle.job_id} was cancelled")
        if job.result is None:                             # pragma: no cover
            raise RuntimeError(f"job {handle.job_id} has no result")
        return job.result

    def cancel(self, handle: JobHandle) -> bool:
        job = self._job(handle)
        if job.status is JobStatus.QUEUED:
            job.status = JobStatus.CANCELLED
            return True
        return False
