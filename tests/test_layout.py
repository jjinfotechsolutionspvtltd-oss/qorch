"""Layout chooses where qubits start, so routing has less to undo.

Two properties carry this pass. It must actually *reduce SWAPs* — a layout pass
that reorders qubits without saving anything is pure risk — and it must not
change what the circuit computes, which is the easier thing to get wrong because
a layout composes with routing's own permutation rather than replacing it.
"""

from __future__ import annotations

import pytest

from qorch import Circuit, LocalSimulator
from qorch.transpiler import (
    TIFR_SUPERCONDUCTING,
    CouplingMap,
    QubitQuality,
    apply_layout,
    dense_layout,
    interaction_graph,
    select_layout,
    transpile_with_layout,
    trivial_layout,
)

_LINE_5 = CouplingMap(TIFR_SUPERCONDUCTING.coupling_map)


def _physical_readout(circuit: Circuit, shots: int = 200) -> str:
    raw = Circuit(
        num_qubits=circuit.num_qubits,
        gates=circuit.gates,
        measured=tuple(range(circuit.num_qubits)),
        num_clbits=circuit.num_clbits,
    )
    counts = LocalSimulator(seed=5).run(raw, shots=shots).counts
    assert len(counts) == 1, f"expected a deterministic outcome, got {counts}"
    return next(iter(counts))


def _swaps(circuit: Circuit, method: str) -> int:
    result = transpile_with_layout(
        circuit, TIFR_SUPERCONDUCTING, coupling_map=_LINE_5, layout_method=method
    )
    return result.metrics.swaps_inserted


# ── it earns its place: fewer SWAPs ──────────────────────────────────────


@pytest.mark.parametrize("name,circuit", [
    ("repeated distant pair", Circuit(5).h(0).cx(0, 4).cx(0, 4).cx(0, 4)),
    ("star from one qubit", Circuit(5).h(0).cx(0, 1).cx(0, 2).cx(0, 3).cx(0, 4)),
    ("two distant pairs", Circuit(5).cx(0, 4).cx(1, 3).cx(0, 4).cx(1, 3)),
])
def test_dense_layout_reduces_swaps(name: str, circuit: Circuit) -> None:
    """Each of these pays for the identity placement; a good one should not."""
    trivial = _swaps(circuit, "trivial")
    dense = _swaps(circuit, "dense")
    assert dense < trivial, f"{name}: dense used {dense} SWAPs vs trivial {trivial}"


def test_layout_never_makes_a_well_placed_circuit_worse() -> None:
    """A GHZ chain already matches a line; layout must not disturb it."""
    ghz = Circuit(5).h(0).cx(0, 1).cx(1, 2).cx(2, 3).cx(3, 4)
    assert _swaps(ghz, "dense") <= _swaps(ghz, "trivial")


# ── it does not change what the circuit computes ─────────────────────────


@pytest.mark.parametrize("method", ["trivial", "dense", "noise-adaptive"])
def test_layout_preserves_semantics_end_to_end(method: str) -> None:
    """Decode raw physical wires through the composed layout and check the state.

    This is the test that catches a layout/routing composition error: routing
    reports where the qubit entering each *wire* ended up, so with a layout in
    front the two permutations must compose, not overwrite.
    """
    c = Circuit(5).x(0).x(4).cx(0, 4)          # → logical 1,0,0,0,0 after the CX
    result = transpile_with_layout(
        c, TIFR_SUPERCONDUCTING, coupling_map=_LINE_5, layout_method=method
    )

    bits = _physical_readout(result.circuit)
    decoded = "".join(bits[result.final_layout[q]] for q in range(5))
    assert decoded == "10000"


@pytest.mark.parametrize("method", ["trivial", "dense"])
def test_measured_distribution_is_unchanged(method: str) -> None:
    c = Circuit(5).h(0).cx(0, 4).measure(0, 1, 2, 3, 4)
    before = LocalSimulator(seed=3).run(c, shots=4000).counts
    routed = transpile_with_layout(
        c, TIFR_SUPERCONDUCTING, coupling_map=_LINE_5, layout_method=method
    ).circuit
    after = LocalSimulator(seed=3).run(routed, shots=4000).counts

    for key in set(before) | set(after):
        assert abs(before.get(key, 0) - after.get(key, 0)) / 4000 < 0.05


def test_composed_layout_is_still_a_permutation() -> None:
    result = transpile_with_layout(
        Circuit(5).cx(0, 4).cx(1, 3), TIFR_SUPERCONDUCTING,
        coupling_map=_LINE_5, layout_method="dense",
    )
    assert sorted(result.final_layout) == list(range(5))


# ── apply_layout on its own ──────────────────────────────────────────────


def test_apply_layout_relabels_gates_and_readout() -> None:
    c = Circuit(3).h(0).cx(0, 2).measure(0, 2)
    moved = apply_layout(c, (2, 1, 0))

    assert moved.gates[0].qubits == (2,)
    assert moved.gates[1].qubits == (2, 0)
    assert moved.measured == (2, 0)


def test_apply_layout_preserves_the_statevector_up_to_relabelling() -> None:
    c = Circuit(3).h(0).cx(0, 1)
    moved = apply_layout(c, (2, 1, 0))

    base = LocalSimulator(seed=1).run(c.measure(0, 1, 2), shots=2000).counts
    # Reading logical q from its new wire must reproduce the original counts.
    layout = (2, 1, 0)
    raw = LocalSimulator(seed=1).run(
        Circuit(3, gates=moved.gates, measured=tuple(layout[q] for q in range(3))),
        shots=2000,
    ).counts
    assert set(base) == set(raw)


def test_identity_layout_is_a_no_op() -> None:
    c = Circuit(3).h(0).cx(0, 2).measure(0, 2)
    assert apply_layout(c, (0, 1, 2)) == c


@pytest.mark.parametrize("bad", [(0, 0, 1), (0, 1), (0, 1, 5)])
def test_apply_layout_rejects_non_permutations(bad) -> None:
    """A non-permutation would silently drop or duplicate a qubit."""
    with pytest.raises(ValueError):
        apply_layout(Circuit(3).h(0), bad)


# ── layout selection ─────────────────────────────────────────────────────


def test_interaction_graph_counts_two_qubit_gates() -> None:
    c = Circuit(3).h(0).cx(0, 1).cx(1, 0).cx(1, 2)
    assert interaction_graph(c) == {(0, 1): 2, (1, 2): 1}


def test_trivial_layout_is_the_identity() -> None:
    assert trivial_layout(Circuit(4)) == (0, 1, 2, 3)


def test_layout_of_a_circuit_with_no_two_qubit_gates_is_trivial() -> None:
    """Nothing to optimize: no pair wants to be adjacent."""
    assert dense_layout(Circuit(4).h(0).x(1), _LINE_5) == (0, 1, 2, 3)


def test_layouts_are_permutations() -> None:
    c = Circuit(5).cx(0, 4).cx(1, 3).cx(2, 4)
    for method in ("trivial", "dense", "noise-adaptive"):
        layout = select_layout(method, c, _LINE_5)
        assert sorted(layout) == list(range(5))


def test_layout_is_deterministic() -> None:
    c = Circuit(5).cx(0, 4).cx(1, 3).cx(2, 4)
    assert dense_layout(c, _LINE_5) == dense_layout(c, _LINE_5)


def test_noise_adaptive_layout_avoids_a_bad_qubit() -> None:
    """With one clearly worse qubit, the busiest logical qubit should avoid it."""
    c = Circuit(5).h(0).cx(0, 1).cx(0, 2).cx(0, 3)
    quality = {q: QubitQuality(0.999) for q in range(5)}
    quality[1] = QubitQuality(0.5)          # qubit 1 is the well-connected but bad one

    layout = select_layout("noise-adaptive", c, _LINE_5, quality)
    busiest = 0                              # logical 0 touches every other qubit
    assert layout[busiest] != 1


def test_unknown_layout_method_lists_the_options() -> None:
    with pytest.raises(ValueError, match="unknown layout method"):
        select_layout("nope", Circuit(3).cx(0, 1), _LINE_5)


# ── pipeline integration ─────────────────────────────────────────────────


def test_layout_pass_appears_only_when_requested() -> None:
    from qorch.transpiler import build_pass_manager

    trivial = [n for n, _ in build_pass_manager(
        TIFR_SUPERCONDUCTING, _LINE_5, layout_method="trivial").passes]
    assert "layout" not in trivial

    dense = [n for n, _ in build_pass_manager(
        TIFR_SUPERCONDUCTING, _LINE_5, layout_method="dense").passes]
    assert dense.index("layout") < dense.index("route"), "layout must precede routing"


def test_layout_is_skipped_without_a_coupling_map() -> None:
    from qorch.transpiler import build_pass_manager

    names = [n for n, _ in build_pass_manager(
        TIFR_SUPERCONDUCTING, layout_method="dense").passes]
    assert "layout" not in names
