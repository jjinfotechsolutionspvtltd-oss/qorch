"""The transpile pipeline is an inspectable list of passes with attributable cost.

A transpiled circuit is much larger than its input, and before this the growth
could only be inferred from the final gate count. These tests pin the two things
that make the pipeline auditable: the pass list is a value you can read, and the
metrics attribute each circuit's size to the pass that produced it.
"""

from __future__ import annotations

import pytest

from qorch import Circuit
from qorch.transpiler import (
    TIFR_SUPERCONDUCTING,
    CouplingMap,
    PassManager,
    TranspileMetrics,
    build_pass_manager,
    circuit_pass,
    transpile,
    transpile_with_layout,
)
from qorch.transpiler.passes import PassState

_LINE_5 = CouplingMap(TIFR_SUPERCONDUCTING.coupling_map)


def _circuit() -> Circuit:
    return Circuit(5).h(0).cx(0, 4).cx(1, 3)


# ── the pipeline is a value ──────────────────────────────────────────────


def test_pipeline_order_is_inspectable() -> None:
    names = [n for n, _ in build_pass_manager(TIFR_SUPERCONDUCTING, _LINE_5).passes]
    assert names == ["decompose", "route", "fuse", "lower", "optimize"]


def test_lookahead_selects_a_different_router_pass() -> None:
    names = [n for n, _ in build_pass_manager(
        TIFR_SUPERCONDUCTING, _LINE_5, use_lookahead=True
    ).passes]
    assert "route-lookahead" in names
    assert "route" not in names


def test_routing_is_skipped_without_a_coupling_map() -> None:
    names = [n for n, _ in build_pass_manager(TIFR_SUPERCONDUCTING).passes]
    assert not any(n.startswith("route") for n in names)


def test_optional_passes_appear_only_when_requested() -> None:
    without = [n for n, _ in build_pass_manager(TIFR_SUPERCONDUCTING, _LINE_5).passes]
    assert "dd" not in without

    with_dd = [n for n, _ in build_pass_manager(
        TIFR_SUPERCONDUCTING, _LINE_5, dd_sequence="xy4"
    ).passes]
    assert with_dd[-1] == "dd", "DD must run last, after the optimizer"

    no_opt = [n for n, _ in build_pass_manager(
        TIFR_SUPERCONDUCTING, _LINE_5, do_optimize=False
    ).passes]
    assert "optimize" not in no_opt


def test_ordering_constraints_hold() -> None:
    """The order encodes fixed bugs; assert the constraints, not the exact list."""
    names = [n for n, _ in build_pass_manager(
        TIFR_SUPERCONDUCTING, _LINE_5, dd_sequence="xy4"
    ).passes]
    assert names.index("route") < names.index("lower"), "SWAPs must be lowered"
    assert names.index("lower") < names.index("optimize")
    assert names.index("optimize") < names.index("dd"), "optimizer would eat DD"


# ── metrics attribute cost to the pass that caused it ────────────────────


def test_metrics_record_every_pass() -> None:
    result = transpile_with_layout(_circuit(), TIFR_SUPERCONDUCTING, coupling_map=_LINE_5)
    assert isinstance(result.metrics, TranspileMetrics)
    assert [p.name for p in result.metrics.passes] == [
        "decompose", "route", "fuse", "lower", "optimize"
    ]


def test_metrics_gate_counts_match_the_real_circuit() -> None:
    c = _circuit()
    result = transpile_with_layout(c, TIFR_SUPERCONDUCTING, coupling_map=_LINE_5)
    m = result.metrics
    assert m.input_gate_count == len(c.gates)
    assert m.output_gate_count == len(result.circuit.gates)
    assert m.two_qubit_count == sum(1 for g in result.circuit.gates if len(g.qubits) == 2)
    assert m.passes[-1].gate_count == len(result.circuit.gates)


def test_swaps_are_counted_before_lowering_dissolves_them() -> None:
    """TIFR has no native SWAP, so counting after lowering would always report 0."""
    result = transpile_with_layout(_circuit(), TIFR_SUPERCONDUCTING, coupling_map=_LINE_5)

    assert result.metrics.swaps_inserted > 0
    assert "swap" not in {g.name for g in result.circuit.gates}


def test_no_routing_means_no_swaps() -> None:
    result = transpile_with_layout(Circuit(3).h(0).cx(0, 1), TIFR_SUPERCONDUCTING)
    assert result.metrics.swaps_inserted == 0


def test_metrics_format_is_readable() -> None:
    result = transpile_with_layout(_circuit(), TIFR_SUPERCONDUCTING, coupling_map=_LINE_5)
    text = result.metrics.format()
    assert "Transpile metrics" in text
    assert "SWAPs" in text
    for name in ("decompose", "route", "lower", "optimize"):
        assert name in text


# ── the refactor changed no behaviour ────────────────────────────────────


@pytest.mark.parametrize("use_lookahead", [False, True])
@pytest.mark.parametrize("dd", [None, "xy4"])
def test_transpile_output_is_unchanged_by_the_refactor(use_lookahead: bool, dd) -> None:
    """transpile() must still return a bare Circuit, identical to the result's."""
    c = _circuit()
    plain = transpile(c, TIFR_SUPERCONDUCTING, coupling_map=_LINE_5,
                      use_lookahead=use_lookahead, dd_sequence=dd)
    rich = transpile_with_layout(c, TIFR_SUPERCONDUCTING, coupling_map=_LINE_5,
                                 use_lookahead=use_lookahead, dd_sequence=dd)
    assert isinstance(plain, Circuit)
    assert plain == rich.circuit
    assert {g.name for g in plain.gates} <= set(TIFR_SUPERCONDUCTING.basis_gates)


# ── the manager is reusable on its own ───────────────────────────────────


def test_a_custom_pipeline_can_be_assembled_and_run() -> None:
    """The point of exposing PassManager: build a variant without rewriting transpile."""
    manager = PassManager(passes=(
        ("nothing", circuit_pass(lambda c: c)),
        ("double-h", circuit_pass(lambda c: c.h(0).h(0))),
    ))
    circuit, state, metrics = manager.run(Circuit(2).h(0))

    assert len(circuit.gates) == 3
    assert [p.name for p in metrics.passes] == ["nothing", "double-h"]
    assert metrics.passes[0].gate_count == 1
    assert metrics.passes[1].gate_count == 3
    assert state.final_layout == (0, 1)


def test_pass_state_defaults_to_the_identity_layout() -> None:
    _, state, _ = PassManager(passes=()).run(Circuit(4))
    assert state == PassState(final_layout=(0, 1, 2, 3))
