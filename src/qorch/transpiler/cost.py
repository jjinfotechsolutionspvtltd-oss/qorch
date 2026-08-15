"""A cost model: what a circuit is predicted to cost *on a particular device*.

Every compiler decision so far has optimized a proxy. Routing minimizes SWAPs,
layout minimizes distance — both reasonable, both blind to the fact that qubits
differ. A SWAP across two excellent qubits can be cheaper than a direct gate on
a bad pair, and no amount of SWAP counting will discover that.

This module estimates the quantity actually worth minimizing: the probability
the circuit produces the right answer, given a device's calibration. Three
sources of loss are accounted for, because on real hardware all three matter and
which dominates depends on the circuit:

  - **gate error** — per-qubit for single-qubit gates, per-edge for two-qubit
    ones, since a device's worst pair is often several times its best
  - **read-out error** — charged once per measured qubit
  - **decoherence** — from the *schedule*, not the gate count: a qubit idling
    865 ns is decaying whether or not any gate is applied to it

The result is an estimate, not a simulation. It assumes errors are independent
and multiply, which is the standard first-order model and is wrong in the third
decimal place. It is used for *ranking* candidate compilations, where being
consistently directionally right is what matters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from qorch.backends.base import DeviceCalibration, QubitCalibration
from qorch.ir import Circuit, Gate
from qorch.transpiler.scheduling import schedule_asap


@dataclass(frozen=True)
class CostEstimate:
    """Predicted fidelity of a circuit on a device, and where it was lost."""

    success_probability: float
    gate_error: float
    readout_error: float
    decoherence_error: float
    duration_ns: float

    @property
    def total_error(self) -> float:
        return 1.0 - self.success_probability

    def format(self) -> str:
        return "\n".join([
            "Cost estimate",
            "=============",
            f"  Success probability: {self.success_probability:.4f}",
            f"  Gate error:          {self.gate_error:.4f}",
            f"  Read-out error:      {self.readout_error:.4f}",
            f"  Decoherence:         {self.decoherence_error:.4f}",
            f"  Duration:            {self.duration_ns:.0f} ns",
        ])


def _qubit(calibration: DeviceCalibration, index: int) -> QubitCalibration:
    if index < len(calibration.qubits):
        return calibration.qubits[index]
    return QubitCalibration()


def _two_qubit_error(calibration: DeviceCalibration, pair: tuple[int, ...]) -> float:
    """Error for a two-qubit gate, trying both operand orders.

    A calibration table keyed on physical edges usually lists each pair once, so
    looking up only the given order silently reports a perfect gate for half of
    them.
    """
    a, b = pair[0], pair[1]
    errors = calibration.two_qubit_error
    if (a, b) in errors:
        return errors[(a, b)]
    if (b, a) in errors:
        return errors[(b, a)]
    # Fall back to the worse of the two qubits' single-qubit errors, scaled:
    # two-qubit gates are the dominant error source on every real device.
    worst = max(_qubit(calibration, a).single_qubit_error,
                _qubit(calibration, b).single_qubit_error)
    return min(1.0, worst * 10.0)


def _decoherence_error(circuit: Circuit, calibration: DeviceCalibration) -> float:
    """Loss from qubits sitting idle, computed from the schedule.

    Uses ``1 - exp(-t/T)`` against the shorter of T1 and T2, which is the
    conservative choice: whichever mechanism decays faster is the one that
    limits you.
    """
    schedule = schedule_asap(circuit, calibration)
    survival = 1.0
    for q in range(circuit.num_qubits):
        idle_ns = schedule.total_idle_ns(q)
        if idle_ns <= 0:
            continue
        cal = _qubit(calibration, q)
        times = [t for t in (cal.t1_us, cal.t2_us) if t > 0]
        if not times:
            continue
        coherence_ns = min(times) * 1000.0
        survival *= math.exp(-idle_ns / coherence_ns)
    return 1.0 - survival


def estimate_cost(circuit: Circuit, calibration: DeviceCalibration) -> CostEstimate:
    """Predict how well ``circuit`` will run on the device ``calibration`` describes."""
    gate_survival = 1.0
    for op in circuit.gates:
        if not isinstance(op, Gate):
            continue
        if len(op.qubits) == 1:
            error = _qubit(calibration, op.qubits[0]).single_qubit_error
        else:
            error = _two_qubit_error(calibration, op.qubits)
        gate_survival *= max(0.0, 1.0 - error)

    readout_survival = 1.0
    for q in circuit.readout_qubits:
        readout_survival *= _qubit(calibration, q).readout_fidelity

    decoherence = _decoherence_error(circuit, calibration)
    total = gate_survival * readout_survival * (1.0 - decoherence)

    return CostEstimate(
        success_probability=total,
        gate_error=1.0 - gate_survival,
        readout_error=1.0 - readout_survival,
        decoherence_error=decoherence,
        duration_ns=schedule_asap(circuit, calibration).duration_ns,
    )


def compare_costs(
    candidates: dict[str, Circuit], calibration: DeviceCalibration
) -> list[tuple[str, CostEstimate]]:
    """Rank compilations by predicted success, best first.

    The intended use of the model: not to believe the absolute number, but to
    choose between compilations of the same circuit.
    """
    scored = [(name, estimate_cost(c, calibration)) for name, c in candidates.items()]
    return sorted(scored, key=lambda item: -item[1].success_probability)
