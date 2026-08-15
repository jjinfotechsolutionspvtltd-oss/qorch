"""Two-qubit gates are not single-qubit gates, and the presets now say so.

``IndianQPUConfig`` carried one ``gate_time_us`` and one ``gate_fidelity``
applied to every gate. On every real platform the entangler is both slower and
less reliable — on an ion trap the MS gate runs 10–20× longer than a single-qubit
rotation — so collapsing them understated the cost of exactly the operation that
dominates it. Both the scheduler and the cost model read those numbers.
"""

from __future__ import annotations

import pytest

from qorch import Circuit, IndianQPU
from qorch.transpiler.cost import estimate_cost
from qorch.transpiler.scheduling import circuit_duration_ns

_TWO_QUBIT = {"cx", "ms", "swap"}


@pytest.mark.parametrize("preset", [
    "iit-jodhpur-ion-trap", "tifr-superconducting", "drdo-mirai",
])
def test_two_qubit_gates_take_longer_than_single_qubit_ones(preset: str) -> None:
    calibration = IndianQPU.from_preset(preset).calibration()
    durations = calibration.gate_durations_us

    two_qubit = {g: t for g, t in durations.items() if g in _TWO_QUBIT}
    single = {g: t for g, t in durations.items() if g not in _TWO_QUBIT}
    assert two_qubit, f"{preset} publishes no two-qubit gate duration"

    if preset == "iit-jodhpur-ion-trap":
        # The MS gate is the slow one on an ion trap, by a wide margin.
        assert min(two_qubit.values()) > max(single.values()) * 5


def test_the_ion_trap_ms_gate_is_far_slower_than_a_rotation() -> None:
    """It used to be reported as exactly as fast, which is off by ~20x."""
    durations = IndianQPU.from_preset("iit-jodhpur-ion-trap").calibration().gate_durations_us
    assert durations["ms"] >= 10 * durations["rx"]


@pytest.mark.parametrize("preset", [
    "tifr-superconducting", "drdo-mirai",
])
def test_two_qubit_error_exceeds_single_qubit_error(preset: str) -> None:
    calibration = IndianQPU.from_preset(preset).calibration()
    single = calibration.qubits[0].single_qubit_error
    assert calibration.two_qubit_error, f"{preset} publishes no per-edge error"
    assert min(calibration.two_qubit_error.values()) > single


def test_the_scheduler_now_charges_for_a_slow_entangler() -> None:
    """A circuit of MS gates must be dominated by them, not by the rotations."""
    calibration = IndianQPU.from_preset("iit-jodhpur-ion-trap").calibration()
    rotations = Circuit(2).rx(0, 0.5).rx(0, 0.5).rx(0, 0.5)
    entangler = Circuit(2).ms(0, 1, 0.25)

    assert (circuit_duration_ns(entangler, calibration)
            > circuit_duration_ns(rotations, calibration))


def test_the_cost_model_now_penalizes_two_qubit_gates_more() -> None:
    calibration = IndianQPU.from_preset("tifr-superconducting").calibration()
    single = estimate_cost(Circuit(2).x(0).x(0).measure(0, 1), calibration)
    entangling = estimate_cost(Circuit(2).cx(0, 1).cx(0, 1).measure(0, 1), calibration)
    assert entangling.gate_error > single.gate_error


# ── the matrix table is no longer duplicated ─────────────────────────────


@pytest.mark.parametrize("name,params", [
    ("x", ()), ("sx", ()), ("h", ()), ("rz", (0.7,)), ("rx", (0.3,)), ("ry", (1.1,)),
])
def test_backend_matrices_come_from_the_registry(name: str, params) -> None:
    from qorch.backends.indian_backend import _indian_gate_matrix
    from qorch.gates import gate_matrix

    assert _indian_gate_matrix(name, params) == gate_matrix(name, params)


def test_an_unknown_gate_is_rejected_rather_than_silently_ignored() -> None:
    """It used to return the identity — an unrecognized gate became a no-op."""
    from qorch.backends.indian_backend import _indian_gate_matrix

    with pytest.raises(ValueError, match="unknown gate"):
        _indian_gate_matrix("not-a-gate", ())


# ── presets still run ────────────────────────────────────────────────────


@pytest.mark.parametrize("preset", [
    "iit-jodhpur-ion-trap", "tifr-superconducting", "drdo-mirai",
])
def test_every_preset_still_executes_a_bell_pair(preset: str) -> None:
    qpu = IndianQPU.from_preset(preset, seed=1)
    counts = qpu.run(Circuit(2).h(0).cx(0, 1).measure(0, 1), shots=500).counts
    assert sum(counts.values()) == 500
    assert counts.get("00", 0) + counts.get("11", 0) > 300


def test_exact_noise_mode_still_runs() -> None:
    """3.4: the SC presets route through the density simulator's T1/T2."""
    qpu = IndianQPU.from_preset("tifr-superconducting", seed=1, exact_noise=True)
    counts = qpu.run(Circuit(2).h(0).cx(0, 1).measure(0, 1), shots=200).counts
    assert sum(counts.values()) == 200
