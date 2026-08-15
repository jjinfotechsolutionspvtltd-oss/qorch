"""Predicting what a circuit costs on a *particular* device.

Every earlier compiler decision optimizes a proxy — SWAP count, interaction
distance — and every proxy is blind to the fact that qubits differ. A SWAP
across two excellent qubits can beat a direct gate on a bad pair, and no amount
of SWAP counting discovers that.

These tests check the model is directionally right on each error source
separately, and then that it is good enough to *choose* between compilations,
which is what it is for.
"""

from __future__ import annotations

import pytest

from qorch import Circuit
from qorch.backends.base import DeviceCalibration, QubitCalibration
from qorch.transpiler import (
    TIFR_SUPERCONDUCTING,
    CouplingMap,
    compare_costs,
    estimate_cost,
    transpile_with_layout,
)

_LINE_5 = CouplingMap(TIFR_SUPERCONDUCTING.coupling_map)


def _device(
    single_error: float = 0.001,
    readout: float = 0.99,
    t1: float = 80.0,
    t2: float = 60.0,
    two_qubit: dict | None = None,
    n: int = 5,
) -> DeviceCalibration:
    return DeviceCalibration(
        qubits=tuple(
            QubitCalibration(t1_us=t1, t2_us=t2, readout_fidelity=readout,
                             single_qubit_error=single_error)
            for _ in range(n)
        ),
        two_qubit_error=two_qubit or {},
        coupling_map=TIFR_SUPERCONDUCTING.coupling_map,
    )


# ── each error source moves the estimate the right way ───────────────────


def test_a_perfect_device_yields_certainty() -> None:
    perfect = _device(single_error=0.0, readout=1.0, t1=0.0, t2=0.0)
    estimate = estimate_cost(Circuit(2).h(0).cx(0, 1).measure(0, 1), perfect)
    assert estimate.success_probability == pytest.approx(1.0)
    assert estimate.total_error == pytest.approx(0.0)


def test_worse_gates_lower_the_estimate() -> None:
    circuit = Circuit(2).h(0).h(0).h(0).measure(0, 1)
    good = estimate_cost(circuit, _device(single_error=0.001))
    bad = estimate_cost(circuit, _device(single_error=0.05))
    assert bad.success_probability < good.success_probability
    assert bad.gate_error > good.gate_error


def test_worse_readout_lowers_the_estimate() -> None:
    circuit = Circuit(2).h(0).measure(0, 1)
    good = estimate_cost(circuit, _device(readout=0.999))
    bad = estimate_cost(circuit, _device(readout=0.85))
    assert bad.readout_error > good.readout_error
    assert bad.success_probability < good.success_probability


def test_more_gates_cost_more() -> None:
    device = _device(single_error=0.01)
    short = estimate_cost(Circuit(1).h(0).measure(0), device)
    long = estimate_cost(
        Circuit(1).h(0).h(0).h(0).h(0).h(0).h(0).measure(0), device
    )
    assert long.success_probability < short.success_probability


def test_two_qubit_error_is_looked_up_in_either_operand_order() -> None:
    """Edge tables usually list a pair once; missing the reverse reports perfection."""
    device = _device(two_qubit={(0, 1): 0.2})
    forward = estimate_cost(Circuit(2).cx(0, 1).measure(0, 1), device)
    reverse = estimate_cost(Circuit(2).cx(1, 0).measure(0, 1), device)
    assert forward.gate_error == pytest.approx(reverse.gate_error)
    assert forward.gate_error > 0.1


def test_a_bad_edge_costs_more_than_a_good_one() -> None:
    device = _device(two_qubit={(0, 1): 0.001, (2, 3): 0.15})
    good = estimate_cost(Circuit(4).cx(0, 1).measure(0, 1, 2, 3), device)
    bad = estimate_cost(Circuit(4).cx(2, 3).measure(0, 1, 2, 3), device)
    assert bad.success_probability < good.success_probability


# ── decoherence comes from the schedule, not the gate count ──────────────


def test_an_idling_qubit_costs_coherence() -> None:
    """q2 finishes early and waits; gate counting sees no cost at all."""
    circuit = Circuit(3).h(0).h(2).cx(0, 1).cx(0, 1).cx(0, 1)
    estimate = estimate_cost(circuit, _device(single_error=0.0, readout=1.0,
                                              t1=1.0, t2=1.0))
    assert estimate.decoherence_error > 0.0


def test_shorter_coherence_times_cost_more() -> None:
    circuit = Circuit(3).h(0).h(2).cx(0, 1).cx(0, 1).cx(0, 1)
    patient = estimate_cost(circuit, _device(t1=500.0, t2=500.0))
    fragile = estimate_cost(circuit, _device(t1=1.0, t2=1.0))
    assert fragile.decoherence_error > patient.decoherence_error


def test_a_device_with_no_coherence_data_is_not_penalized() -> None:
    """Missing data must not be read as instant decoherence."""
    circuit = Circuit(3).h(0).h(2).cx(0, 1).cx(0, 1)
    assert estimate_cost(circuit, _device(t1=0.0, t2=0.0)).decoherence_error == 0.0


def test_the_shorter_of_t1_and_t2_governs() -> None:
    """Whichever mechanism decays faster is the one that limits you."""
    circuit = Circuit(3).h(0).h(2).cx(0, 1).cx(0, 1).cx(0, 1)
    a = estimate_cost(circuit, _device(t1=100.0, t2=1.0))
    b = estimate_cost(circuit, _device(t1=1.0, t2=100.0))
    assert a.decoherence_error == pytest.approx(b.decoherence_error)


# ── using the model to choose ────────────────────────────────────────────


def test_compare_costs_ranks_best_first() -> None:
    device = _device(single_error=0.02)
    ranked = compare_costs({
        "short": Circuit(1).h(0).measure(0),
        "long": Circuit(1).h(0).h(0).h(0).h(0).h(0).measure(0),
    }, device)
    assert [name for name, _ in ranked] == ["short", "long"]


def test_cost_aware_layout_beats_swap_minimal_on_a_lopsided_device() -> None:
    """The point of 4.3: fewest SWAPs is not the same as best result.

    Qubit 2 is far worse than the rest. A layout that minimizes SWAPs routes
    straight through it; the cost model sees that and prefers a placement with
    more SWAPs on better hardware.
    """
    qubits = list(_device().qubits)
    qubits[2] = QubitCalibration(t1_us=5.0, t2_us=4.0, readout_fidelity=0.80,
                                 single_qubit_error=0.05)
    device = DeviceCalibration(
        qubits=tuple(qubits),
        two_qubit_error={(0, 1): 0.01, (1, 2): 0.09, (2, 3): 0.09, (3, 4): 0.01},
        coupling_map=TIFR_SUPERCONDUCTING.coupling_map,
    )
    circuit = Circuit(5).h(0).cx(0, 1).cx(1, 2).cx(3, 4)

    compiled = {
        method: transpile_with_layout(
            circuit, TIFR_SUPERCONDUCTING, coupling_map=_LINE_5,
            layout_method=method, calibration=device,
        ).circuit
        for method in ("dense", "cost-aware")
    }
    scores = dict(compare_costs(compiled, device))
    assert (scores["cost-aware"].success_probability
            >= scores["dense"].success_probability)


def test_cost_aware_layout_falls_back_without_calibration() -> None:
    """With no device data there is nothing to be calibration-aware about."""
    from qorch.transpiler import cost_aware_layout, dense_layout

    circuit = Circuit(5).cx(0, 4).cx(1, 3)
    assert cost_aware_layout(circuit, _LINE_5) == dense_layout(circuit, _LINE_5)


def test_cost_aware_layout_returns_a_permutation() -> None:
    from qorch.transpiler import cost_aware_layout

    layout = cost_aware_layout(Circuit(5).cx(0, 4).cx(1, 3), _LINE_5,
                               calibration=_device())
    assert sorted(layout) == list(range(5))


def test_estimate_is_reported_readably() -> None:
    text = estimate_cost(Circuit(2).h(0).cx(0, 1).measure(0, 1), _device()).format()
    for label in ("Success probability", "Gate error", "Read-out", "Decoherence"):
        assert label in text
