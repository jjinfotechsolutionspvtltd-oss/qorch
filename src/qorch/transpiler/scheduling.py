"""Scheduling: when each operation runs, in nanoseconds rather than gate slots.

Everything upstream of this counts *gates*. Hardware does not care how many
gates there are; it cares how long the qubits are alive. Those differ sharply
because gate durations differ by more than an order of magnitude — a CX is
around 300 ns where an ``rz`` is a frame change costing nothing at all — so a
circuit with fewer gates can easily take longer than one with more.

That matters most for decoherence. A qubit sitting idle is decaying, and "idle"
is a duration, not a count of intervening gates. The existing DD insertion
measures idle windows in *gate slots*, which is why it can put a refocusing
sequence into a gap too short to hold it, and miss a long gap that happens to
contain a single slow gate on another qubit.

Durations come from :class:`~qorch.backends.base.DeviceCalibration` when a device
supplies them, and from the gate registry's advisory defaults otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass

from qorch.backends.base import DeviceCalibration
from qorch.gates import gate_duration_ns
from qorch.ir import Circuit, Gate, Operation

# Measurement and reset are not in the gate registry but do take real time.
_MEASURE_NS = 1000.0
_RESET_NS = 1000.0


def op_duration_ns(op: Operation, calibration: DeviceCalibration | None = None) -> float:
    """How long ``op`` occupies its qubits.

    Prefers the device's own numbers when it publishes them; a real machine's
    calibration beats a plausible default every time.
    """
    if calibration is not None:
        durations = getattr(calibration, "gate_durations_us", None)
        if durations:
            value = durations.get(op.name)
            if value is not None:
                return float(value) * 1000.0        # calibration is in microseconds
    if isinstance(op, Gate):
        return gate_duration_ns(op.name)
    return _MEASURE_NS if op.name == "measure" else _RESET_NS


@dataclass(frozen=True)
class ScheduledOp:
    """An operation with the window it occupies."""

    op: Operation
    start_ns: float
    duration_ns: float

    @property
    def end_ns(self) -> float:
        return self.start_ns + self.duration_ns


@dataclass(frozen=True)
class Schedule:
    """A circuit with a start time for every operation."""

    ops: tuple[ScheduledOp, ...]
    num_qubits: int

    @property
    def duration_ns(self) -> float:
        """Wall-clock length of the circuit — what decoherence is measured against."""
        return max((s.end_ns for s in self.ops), default=0.0)

    def busy_intervals(self, qubit: int) -> list[tuple[float, float]]:
        """When ``qubit`` is occupied, in order."""
        return [
            (s.start_ns, s.end_ns) for s in self.ops if qubit in s.op.qubits
        ]

    def idle_windows(self, qubit: int) -> list[tuple[float, float]]:
        """Gaps during which ``qubit`` holds a state it is not using.

        Includes the **trailing** gap from a qubit's last operation to the end of
        the circuit. That window is usually the largest one and it is real: with
        terminal read-out the qubit holds its state until the whole circuit
        finishes, decaying the entire time. An ASAP schedule packs operations
        early, so for most qubits the trailing window is the *only* idleness
        there is — excluding it would report a circuit as having no idle qubits
        while most of them wait.

        The **leading** gap, before a qubit's first operation, is excluded. Until
        then the qubit sits in |0>, which T1 relaxation leaves alone; there is no
        superposition yet to lose.
        """
        busy = self.busy_intervals(qubit)
        if not busy:
            return []
        windows: list[tuple[float, float]] = []
        for (_, end), (next_start, _) in zip(busy, busy[1:]):
            if next_start > end:
                windows.append((end, next_start))
        total = self.duration_ns
        last_end = busy[-1][1]
        if total > last_end:
            windows.append((last_end, total))
        return windows

    def total_idle_ns(self, qubit: int) -> float:
        return sum(end - start for start, end in self.idle_windows(qubit))


def schedule_asap(
    circuit: Circuit, calibration: DeviceCalibration | None = None
) -> Schedule:
    """As-soon-as-possible schedule: every op starts the moment its qubits free up.

    Operations keep their program order on each qubit — reordering is routing's
    and the optimizer's job, not the scheduler's. All this decides is *when*
    each op runs given that order.
    """
    ready: dict[int, float] = {}
    scheduled: list[ScheduledOp] = []
    for op in circuit.gates:
        duration = op_duration_ns(op, calibration)
        start = max((ready.get(q, 0.0) for q in op.qubits), default=0.0)
        for q in op.qubits:
            ready[q] = start + duration
        scheduled.append(ScheduledOp(op=op, start_ns=start, duration_ns=duration))
    return Schedule(ops=tuple(scheduled), num_qubits=circuit.num_qubits)


def schedule_alap(
    circuit: Circuit, calibration: DeviceCalibration | None = None
) -> Schedule:
    """As-late-as-possible schedule: every op runs as late as it can.

    The same total duration as ASAP, but idle time moves to the *start* of each
    qubit's life rather than the end. That is the better shape for a state
    prepared early and used late: the qubit spends less time waiting in a
    superposition it has to hold.
    """
    asap = schedule_asap(circuit, calibration)
    total = asap.duration_ns
    latest: dict[int, float] = {}
    reversed_ops: list[ScheduledOp] = []
    for scheduled in reversed(asap.ops):
        op, duration = scheduled.op, scheduled.duration_ns
        end = min((latest.get(q, total) for q in op.qubits), default=total)
        start = end - duration
        for q in op.qubits:
            latest[q] = start
        reversed_ops.append(ScheduledOp(op=op, start_ns=start, duration_ns=duration))
    return Schedule(ops=tuple(reversed(reversed_ops)), num_qubits=circuit.num_qubits)


def circuit_duration_ns(
    circuit: Circuit, calibration: DeviceCalibration | None = None
) -> float:
    """Wall-clock length of ``circuit`` — the number decoherence cares about."""
    return schedule_asap(circuit, calibration).duration_ns


def idle_report(
    circuit: Circuit, calibration: DeviceCalibration | None = None
) -> dict[int, float]:
    """Total idle time per qubit, longest first when formatted.

    The direct answer to "which qubit is decohering while it waits?", which gate
    counts cannot express.
    """
    schedule = schedule_asap(circuit, calibration)
    return {q: schedule.total_idle_ns(q) for q in range(circuit.num_qubits)}
