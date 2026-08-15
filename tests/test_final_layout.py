"""Routing reports where each logical qubit physically ended up.

Routing permutes qubits as it inserts SWAPs, and only routing knows the
permutation. The routed circuit alone cannot tell you: it remaps ``measured`` so
bitstrings stay in logical order, which is exactly what *hides* the permutation
from the caller. Recovering it matters for anything that has to name a physical
qubit — correlating results with per-qubit calibration, attributing an error to
a specific device qubit, debugging a layout.

The central test here reads the routed circuit's *raw physical wires* and decodes
them through the reported layout, so a wrong layout produces a wrong answer. A
test that merely compared the layout against ``measured`` would be checking that
one line of code agrees with another.
"""

from __future__ import annotations

import pytest

from qorch import Circuit, LocalSimulator
from qorch.transpiler import (
    TIFR_SUPERCONDUCTING,
    CouplingMap,
    TranspileResult,
    route_lookahead_with_layout,
    route_with_layout,
    transpile,
    transpile_with_layout,
)

_LINE_3 = CouplingMap(((0, 1), (1, 0), (1, 2), (2, 1)))
_LINE_5 = CouplingMap(TIFR_SUPERCONDUCTING.coupling_map)


def _physical_readout(circuit: Circuit, shots: int = 200) -> str:
    """Run measuring every physical wire in physical order; return the bitstring."""
    raw = Circuit(
        num_qubits=circuit.num_qubits,
        gates=circuit.gates,
        measured=tuple(range(circuit.num_qubits)),
        num_clbits=circuit.num_clbits,
    )
    counts = LocalSimulator(seed=5).run(raw, shots=shots).counts
    assert len(counts) == 1, f"expected a deterministic outcome, got {counts}"
    return next(iter(counts))


# ── the layout is physically correct ─────────────────────────────────────


def test_final_layout_locates_each_logical_qubit_on_the_wire_holding_it() -> None:
    """Decode raw physical wires through the layout and recover the logical state.

    A deterministic circuit sets logical qubits 0 and 2 to |1> and leaves 1 at
    |0>, then a non-adjacent CX forces SWAPs. If ``final_layout`` is wrong, the
    decoded bits come back permuted.
    """
    c = Circuit(3).x(0).x(2).cx(0, 2)          # cx(0,2) is not adjacent on a line
    result = route_with_layout(c, _LINE_3)

    physical_bits = _physical_readout(result.circuit)
    decoded = "".join(physical_bits[result.final_layout[q]] for q in range(3))

    # x(0) x(2) -> |1,0,1>; then cx(0,2) flips q2 -> |1,0,0>
    assert decoded == "100"


def test_layout_is_needed_because_raw_physical_order_differs() -> None:
    """Guard the guard: if routing never permuted, the test above proves nothing."""
    c = Circuit(3).x(0).x(2).cx(0, 2)
    result = route_with_layout(c, _LINE_3)

    assert result.final_layout != (0, 1, 2), "no permutation — test is vacuous"
    assert _physical_readout(result.circuit) != "100"


def test_lookahead_router_reports_a_correct_layout_too() -> None:
    c = Circuit(3).x(0).x(2).cx(0, 2)
    result = route_lookahead_with_layout(c, _LINE_3)

    physical_bits = _physical_readout(result.circuit)
    decoded = "".join(physical_bits[result.final_layout[q]] for q in range(3))
    assert decoded == "100"


# ── shape and consistency ────────────────────────────────────────────────


@pytest.mark.parametrize("router", [route_with_layout, route_lookahead_with_layout])
def test_layout_is_a_permutation(router) -> None:
    c = Circuit(5).h(0).cx(0, 4).cx(1, 3).cx(0, 2)
    result = router(c, _LINE_5)
    assert sorted(result.final_layout) == list(range(5))


@pytest.mark.parametrize("router", [route_with_layout, route_lookahead_with_layout])
def test_layout_agrees_with_the_remapped_readout(router) -> None:
    """``measured`` is the layout applied to the read-out qubits; they must match."""
    c = Circuit(5).h(0).cx(0, 4).measure(0, 4)
    result = router(c, _LINE_5)
    assert result.circuit.measured == tuple(result.final_layout[q] for q in (0, 4))


def test_no_coupling_map_gives_the_identity_layout() -> None:
    c = Circuit(4).h(0).cx(0, 3)
    assert route_with_layout(c, CouplingMap(())).final_layout == (0, 1, 2, 3)


# ── the layout survives the whole pipeline ───────────────────────────────


def test_transpile_reports_the_layout_through_every_later_pass() -> None:
    """Lowering, direction fixing, the optimizer and DD rewrite gates in place.

    None of them permutes wires, so routing's layout must still be valid at the
    end of the pipeline — including with DD inserted, which adds gates last.
    """
    c = Circuit(5).x(0).cx(0, 4)
    result = transpile_with_layout(
        c, TIFR_SUPERCONDUCTING, coupling_map=_LINE_5, dd_sequence="xy4"
    )

    assert isinstance(result, TranspileResult)
    assert sorted(result.final_layout) == list(range(5))

    physical_bits = _physical_readout(result.circuit)
    decoded = "".join(physical_bits[result.final_layout[q]] for q in range(5))
    assert decoded == "10001"          # x(0) then cx(0,4) -> q0 and q4 set


def test_transpile_without_routing_gives_the_identity_layout() -> None:
    result = transpile_with_layout(Circuit(3).h(0).cx(0, 1), TIFR_SUPERCONDUCTING)
    assert result.final_layout == (0, 1, 2)


# ── the lookahead router terminates ──────────────────────────────────────


def test_lookahead_router_terminates_on_distant_operands() -> None:
    """Regression: this circuit used to hang the router forever.

    Two bugs compounded. ``_extended_set`` included already-executed gates, so a
    SWAP that moved their operands back together scored as progress; and
    ``_swap_score`` counted only gates a SWAP made *immediately* adjacent, which
    gives no signal when the front gate's operands are four hops apart on a line.
    With no real signal the router took a phantom-scoring SWAP, undid it, and
    oscillated. Scoring by total distance reduction — the actual SabreSWAP cost
    function — gives every SWAP a gradient to descend.
    """
    c = Circuit(5).h(0).cx(0, 4).cx(1, 3).cx(0, 2)
    result = route_lookahead_with_layout(c, _LINE_5)

    assert sorted(result.final_layout) == list(range(5))
    two_qubit = [g for g in result.circuit.gates if len(g.qubits) == 2]
    assert all(q in set(_LINE_5.edges) for q in (g.qubits for g in two_qubit))


def test_lookahead_routing_preserves_the_distribution_on_distant_operands() -> None:
    """Terminating is not enough — it has to route the circuit correctly."""
    c = Circuit(5).h(0).cx(0, 4).cx(1, 3).cx(0, 2).measure(0, 1, 2, 3, 4)
    routed = route_lookahead_with_layout(c, _LINE_5).circuit

    before = LocalSimulator(seed=3).run(c, shots=4000).counts
    after = LocalSimulator(seed=3).run(routed, shots=4000).counts
    for key in set(before) | set(after):
        assert abs(before.get(key, 0) - after.get(key, 0)) / 4000 < 0.05


def test_transpile_still_returns_a_bare_circuit() -> None:
    """The original API is unchanged; the layout is additive."""
    c = Circuit(5).x(0).cx(0, 4)
    plain = transpile(c, TIFR_SUPERCONDUCTING, coupling_map=_LINE_5)
    withl = transpile_with_layout(c, TIFR_SUPERCONDUCTING, coupling_map=_LINE_5)
    assert isinstance(plain, Circuit)
    assert plain == withl.circuit
