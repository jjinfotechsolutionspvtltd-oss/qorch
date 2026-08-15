"""Timing: when operations run, in nanoseconds rather than gate slots.

Everything upstream counts gates. Hardware cares how long qubits are alive, and
the two differ sharply — a CX is ~300 ns where an ``rz`` is a frame change
costing nothing — so a circuit with fewer gates can take longer than one with
more.

The tests that matter here are the ones where slot counting and real time
disagree, because those are the cases the old model got wrong.
"""

from __future__ import annotations

import pytest

from qorch import Circuit
from qorch.gates import gate_duration_ns
from qorch.mitigation.dd import insert_dd, insert_dd_timed
from qorch.transpiler.scheduling import (
    circuit_duration_ns,
    idle_report,
    op_duration_ns,
    schedule_alap,
    schedule_asap,
)


# ── durations ────────────────────────────────────────────────────────────


def test_rz_is_free_and_cx_is_not() -> None:
    """The whole reason slot counting misleads: gate costs are not comparable."""
    assert gate_duration_ns("rz") == 0.0
    assert gate_duration_ns("cx") > 100.0


def test_measure_and_reset_cost_real_time() -> None:
    from qorch.ir import Measure, Reset

    assert op_duration_ns(Measure((0,), 0)) > 0
    assert op_duration_ns(Reset((0,))) > 0


def test_calibration_overrides_the_advisory_default() -> None:
    """A real device's numbers beat a plausible guess."""
    from qorch.backends.base import DeviceCalibration

    calibration = DeviceCalibration(qubits=(), gate_durations_us={"cx": 1.5})
    from qorch.ir import Gate

    assert op_duration_ns(Gate("cx", (0, 1)), calibration) == 1500.0


# ── scheduling ───────────────────────────────────────────────────────────


def test_an_empty_circuit_takes_no_time() -> None:
    assert circuit_duration_ns(Circuit(2)) == 0.0


def test_serial_gates_on_one_qubit_add_up() -> None:
    c = Circuit(1).x(0).x(0).x(0)
    assert circuit_duration_ns(c) == pytest.approx(3 * gate_duration_ns("x"))


def test_gates_on_different_qubits_run_in_parallel() -> None:
    """Three x gates on three qubits take as long as one, not three."""
    parallel = Circuit(3).x(0).x(1).x(2)
    assert circuit_duration_ns(parallel) == pytest.approx(gate_duration_ns("x"))


def test_a_two_qubit_gate_blocks_both_of_its_qubits() -> None:
    c = Circuit(2).cx(0, 1).x(0)
    assert circuit_duration_ns(c) == pytest.approx(
        gate_duration_ns("cx") + gate_duration_ns("x")
    )


def test_rz_gates_take_no_wall_clock_time() -> None:
    """A hundred frame changes cost nothing — slot counting says otherwise."""
    c = Circuit(1)
    for _ in range(100):
        c = c.rz(0, 0.1)
    assert circuit_duration_ns(c) == 0.0


def test_alap_matches_asap_in_total_duration() -> None:
    """Scheduling moves idleness around; it does not change the length."""
    c = Circuit(3).h(0).h(2).cx(0, 1).cx(0, 1).h(2)
    assert schedule_alap(c).duration_ns == pytest.approx(schedule_asap(c).duration_ns)


def test_alap_starts_operations_no_earlier_than_asap() -> None:
    c = Circuit(3).h(0).h(2).cx(0, 1).cx(0, 1).h(2)
    asap, alap = schedule_asap(c), schedule_alap(c)
    for early, late in zip(asap.ops, alap.ops):
        assert late.start_ns >= early.start_ns - 1e-9


def test_scheduling_preserves_program_order_per_qubit() -> None:
    """Reordering is the router's and optimizer's job, not the scheduler's."""
    c = Circuit(2).h(0).x(0).cx(0, 1)
    starts = [s.start_ns for s in schedule_asap(c).ops]
    assert starts == sorted(starts)


# ── idle time ────────────────────────────────────────────────────────────


def test_a_qubit_waiting_for_the_circuit_to_end_is_reported_as_idle() -> None:
    """The trailing window is usually the largest, and gate counts show none of it.

    q2 finishes early while q0/q1 grind through CX gates; with terminal read-out
    it holds its state — and decays — until the circuit ends.
    """
    c = Circuit(3).h(0).h(2).cx(0, 1).cx(0, 1).cx(0, 1).h(2)
    idle = idle_report(c)

    assert idle[2] > 800.0
    assert idle[0] == pytest.approx(0.0)


def test_a_qubit_that_is_never_used_reports_no_idle_time() -> None:
    """Untouched means still in |0>, which T1 leaves alone — not an idle window."""
    c = Circuit(3).h(0).cx(0, 1)
    assert idle_report(c)[2] == 0.0


def test_idle_windows_exclude_the_leading_gap() -> None:
    """Before its first gate a qubit sits in |0>; there is no superposition to lose."""
    c = Circuit(2).cx(0, 1).cx(0, 1).x(1)
    windows = schedule_asap(c).idle_windows(1)
    assert all(start > 0 for start, _ in windows)


# ── duration-aware DD ────────────────────────────────────────────────────


def test_timed_dd_refuses_a_zero_duration_gap() -> None:
    """Slot counting sees a gap of three rz as an opportunity; it takes no time.

    This is the case the old model got wrong: it packs refocusing pulses into a
    window that does not exist.
    """
    c = Circuit(2).h(0).rz(1, 0.1).rz(1, 0.2).h(0).x(1)

    assert len(insert_dd(c, "xy4").gates) > len(c.gates)
    assert insert_dd_timed(c, "xy4").gates == c.gates


def test_timed_dd_protects_a_genuinely_long_window() -> None:
    c = Circuit(3).h(0).h(2).cx(0, 1).cx(0, 1).cx(0, 1).h(2)
    protected = insert_dd_timed(c, "xy4")

    assert len(protected.gates) > len(c.gates)
    added = len(protected.gates) - len(c.gates)
    assert added % 4 == 0, "xy4 inserts whole sequences"


def test_timed_dd_can_be_restricted_to_chosen_qubits() -> None:
    c = Circuit(3).h(0).h(2).cx(0, 1).cx(0, 1).cx(0, 1).h(2)
    only_q1 = insert_dd_timed(c, "xy4", qubits=(1,))
    inserted = [g for g in only_q1.gates if g.qubits == (2,) and g.name in ("x", "y")]
    assert not inserted


def test_timed_dd_threshold_is_configurable() -> None:
    c = Circuit(3).h(0).h(2).cx(0, 1).cx(0, 1).cx(0, 1).h(2)
    assert insert_dd_timed(c, "xy4", min_idle_ns=1e9).gates == c.gates


def test_timed_dd_rejects_an_unknown_sequence() -> None:
    with pytest.raises(ValueError, match="unknown DD sequence"):
        insert_dd_timed(Circuit(2).h(0), "nonsense")


def test_timed_dd_inserts_far_less_than_slot_based_dd() -> None:
    """Fewer, better-placed pulses: every one lands in a window that exists."""
    c = Circuit(3).h(0).h(2).cx(0, 1).cx(0, 1).cx(0, 1).h(2)
    assert len(insert_dd_timed(c, "xy4").gates) < len(insert_dd(c, "xy4").gates)
