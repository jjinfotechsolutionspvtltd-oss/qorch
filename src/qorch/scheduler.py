"""Job scheduler / backend router.

M1 is intentionally a dumb FIFO with a pluggable selection policy (charter: "the scheduler
is allowed to be dumb"). The policy hook is what M4 grows into cost/latency/availability
routing — the interface stays the same.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

from qorch.backends.base import Backend, JobResult
from qorch.ir import Circuit

# A policy picks one backend for a circuit from the registered set.
SelectionPolicy = Callable[[Circuit, list[Backend]], Backend]


def first_available(circuit: Circuit, backends: list[Backend]) -> Backend:
    """Default policy: first backend with enough qubits."""
    for backend in backends:
        if backend.properties().num_qubits >= circuit.num_qubits:
            return backend
    raise ValueError(f"no registered backend can run {circuit.num_qubits} qubits")


def _best_fit(circuit: Circuit, backends: list[Backend]) -> Backend:
    """Pick the backend that best fits the circuit (minimal qubit waste)."""
    nq = circuit.num_qubits
    best: Backend | None = None
    best_waste = float("inf")
    for backend in backends:
        avail = backend.properties().num_qubits
        if avail >= nq:
            waste = avail - nq
            if waste < best_waste:
                best_waste = waste
                best = backend
    if best is None:
        raise ValueError(f"no registered backend can run {nq} qubits")
    return best


@dataclass
class Job:
    circuit: Circuit
    shots: int


@dataclass
class BatchJob:
    circuit: Circuit
    shots: int
    label: str = ""


@dataclass
class BatchResult:
    label: str
    backend_name: str
    result: JobResult | None = None
    error: str | None = None


@dataclass
class Scheduler:
    backends: list[Backend]
    policy: SelectionPolicy = first_available
    _queue: deque[Job] = field(default_factory=deque, init=False)

    def submit(self, circuit: Circuit, shots: int = 1024) -> None:
        self._queue.append(Job(circuit, shots))

    def run_all(self) -> list[JobResult]:
        """Drain the queue in FIFO order, routing each job via the policy."""
        results: list[JobResult] = []
        while self._queue:
            job = self._queue.popleft()
            backend = self.policy(job.circuit, self.backends)
            results.append(backend.run(job.circuit, job.shots))
        return results

    def run_batch(self, jobs: list[BatchJob]) -> list[BatchResult]:
        """Run a batch of circuits, grouping by best-fit backend.

        Each circuit is routed to the backend that fits it best (minimum qubit waste).
        Returns a list of ``BatchResult`` with results or error messages.
        """
        results: list[BatchResult] = []
        for bj in jobs:
            try:
                backend = _best_fit(bj.circuit, self.backends)
                result = backend.run(bj.circuit, bj.shots)
                results.append(BatchResult(label=bj.label, backend_name=backend.name, result=result))
            except ValueError as e:
                results.append(BatchResult(label=bj.label, backend_name="", error=str(e)))
        return results
