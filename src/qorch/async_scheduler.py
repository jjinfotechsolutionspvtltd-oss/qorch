"""Fan-out scheduling over async backends, with a job store that survives a crash.

The FIFO scheduler runs one circuit at a time and blocks on each. That is fine
against simulators and wrong against queued hardware, where the useful thing is
to put every circuit into every relevant queue at once and collect them as they
land — the queue wait dominates, and waiting on them sequentially adds up.

Two things are needed beyond :class:`~qorch.backends.async_backend.AsyncBackend`:

  - **Fan-out.** Submit the whole batch, then poll. Submitting is cheap; the
    waiting is what costs, and it should be shared.
  - **Persistence.** A handle held only in memory is lost when the process dies,
    and with it a job that may already be hours into a queue. :class:`JobStore`
    writes handles to SQLite (stdlib — the core stays dependency-free) so a new
    process can pick them up.

Backend selection is pluggable, and :func:`cost_based_policy` is the interesting
one: it ranks backends by *predicted fidelity* from their own calibration rather
than by whether the circuit merely fits.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from qorch.backends.async_backend import AsyncBackend, JobHandle, JobStatus
from qorch.backends.base import JobResult
from qorch.ir import Circuit

# A policy picks one backend for a circuit from the registered set.
AsyncPolicy = Callable[[Circuit, list[AsyncBackend]], AsyncBackend]


def first_fit_policy(circuit: Circuit, backends: list[AsyncBackend]) -> AsyncBackend:
    """First backend with enough qubits — the cheap default."""
    for backend in backends:
        if backend.properties().num_qubits >= circuit.num_qubits:
            return backend
    raise ValueError(f"no backend can run {circuit.num_qubits} qubits")


def cost_based_policy(
    circuit: Circuit, backends: list[AsyncBackend]
) -> AsyncBackend:
    """Pick the backend predicted to give the best result, not merely a fitting one.

    Uses each backend's own calibration through the cost model. A backend that
    publishes no calibration cannot be scored, so it is ranked below any that
    can — an unknown device is not evidence of a good one.
    """
    from qorch.transpiler.cost import estimate_cost

    best: AsyncBackend | None = None
    best_score = -1.0
    for backend in backends:
        if backend.properties().num_qubits < circuit.num_qubits:
            continue
        calibration = backend.calibration() if hasattr(backend, "calibration") else None
        score = 0.0
        if calibration is not None:
            score = estimate_cost(circuit, calibration).success_probability
        if score > best_score:
            best_score, best = score, backend
    if best is None:
        raise ValueError(f"no backend can run {circuit.num_qubits} qubits")
    return best


@dataclass(frozen=True)
class StoredJob:
    """A persisted submission: the handle, plus what it was for."""

    handle: JobHandle
    label: str
    status: JobStatus


class JobStore:
    """SQLite-backed record of submitted jobs.

    Exists so a submitted job is not lost with the process that submitted it.
    On real hardware a handle can represent hours of queue position, and holding
    it only in memory means a restart throws that away.

    ``sqlite3`` is in the standard library, so this adds no dependency.
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        self._connection = sqlite3.connect(str(path))
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                job_id       TEXT PRIMARY KEY,
                backend_name TEXT NOT NULL,
                shots        INTEGER NOT NULL,
                submitted_at REAL NOT NULL,
                label        TEXT NOT NULL DEFAULT '',
                status       TEXT NOT NULL
            )
            """
        )
        self._connection.commit()

    def record(self, handle: JobHandle, label: str = "",
               status: JobStatus = JobStatus.QUEUED) -> None:
        self._connection.execute(
            "INSERT OR REPLACE INTO jobs VALUES (?, ?, ?, ?, ?, ?)",
            (handle.job_id, handle.backend_name, handle.shots,
             handle.submitted_at, label, status.value),
        )
        self._connection.commit()

    def update_status(self, handle: JobHandle, status: JobStatus) -> None:
        self._connection.execute(
            "UPDATE jobs SET status = ? WHERE job_id = ?",
            (status.value, handle.job_id),
        )
        self._connection.commit()

    def get(self, job_id: str) -> StoredJob | None:
        row = self._connection.execute(
            "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        return self._to_job(row) if row else None

    def unfinished(self) -> list[StoredJob]:
        """Jobs that were still live when they were last written down.

        What a restarting process asks for: everything that may still be running
        somewhere and is worth reclaiming.
        """
        rows = self._connection.execute(
            "SELECT * FROM jobs WHERE status IN (?, ?)",
            (JobStatus.QUEUED.value, JobStatus.RUNNING.value),
        ).fetchall()
        return [self._to_job(row) for row in rows]

    def all_jobs(self) -> list[StoredJob]:
        return [self._to_job(row)
                for row in self._connection.execute("SELECT * FROM jobs").fetchall()]

    @staticmethod
    def _to_job(row: tuple) -> StoredJob:
        job_id, backend_name, shots, submitted_at, label, status = row
        return StoredJob(
            handle=JobHandle(
                job_id=job_id, backend_name=backend_name,
                shots=shots, submitted_at=submitted_at,
            ),
            label=label,
            status=JobStatus(status),
        )

    def close(self) -> None:
        self._connection.close()


@dataclass
class AsyncBatchResult:
    """Outcome of one circuit in a fanned-out batch."""

    label: str
    backend_name: str
    result: JobResult | None = None
    error: str | None = None


@dataclass
class AsyncScheduler:
    """Submit a batch across async backends, then collect as results land."""

    backends: list[AsyncBackend]
    policy: AsyncPolicy = first_fit_policy
    store: JobStore | None = None
    _pending: list[tuple[str, AsyncBackend, JobHandle]] = field(
        default_factory=list, init=False
    )

    def submit(self, circuit: Circuit, shots: int = 1024, label: str = "") -> JobHandle:
        """Queue one circuit on the backend the policy chooses."""
        backend = self.policy(circuit, self.backends)
        handle = backend.submit(circuit, shots)
        self._pending.append((label, backend, handle))
        if self.store is not None:
            self.store.record(handle, label)
        return handle

    def submit_batch(
        self, jobs: list[tuple[Circuit, int, str]]
    ) -> list[JobHandle]:
        """Submit every circuit before waiting for any of them.

        The point of fan-out: with a queued device the wait dominates, so all
        the jobs should be queued at once rather than one after another.
        """
        return [self.submit(circuit, shots, label) for circuit, shots, label in jobs]

    def collect(self, timeout: float | None = None) -> list[AsyncBatchResult]:
        """Wait for everything submitted so far and return results in order."""
        results: list[AsyncBatchResult] = []
        for label, backend, handle in self._pending:
            try:
                result = backend.wait(handle, timeout=timeout)
            except Exception as exc:                        # noqa: BLE001
                results.append(AsyncBatchResult(
                    label=label, backend_name=backend.name, error=str(exc)
                ))
                if self.store is not None:
                    self.store.update_status(handle, JobStatus.ERROR)
            else:
                results.append(AsyncBatchResult(
                    label=label, backend_name=backend.name, result=result
                ))
                if self.store is not None:
                    self.store.update_status(handle, JobStatus.DONE)
        self._pending.clear()
        return results

    def run_batch(
        self, jobs: list[tuple[Circuit, int, str]], timeout: float | None = None
    ) -> list[AsyncBatchResult]:
        """Fan out, then collect — submit everything first, wait once."""
        self.submit_batch(jobs)
        return self.collect(timeout=timeout)
