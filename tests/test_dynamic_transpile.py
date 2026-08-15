"""Dynamic circuits survive the full transpile pipeline.

Mid-circuit measurement, reset, and feed-forward used to be rejected by every
compiler stage, which meant a teleportation or repetition-code circuit could be
*simulated* but never *compiled* to a real gate set and topology. These tests pin
the two properties that make dynamic transpilation trustworthy:

  1. **Semantics are preserved** — the classical register distribution of a
     transpiled dynamic circuit is identical to the original's.
  2. **Ordering is preserved** — a feed-forward gate never floats ahead of the
     measurement that decides it, even under the reordering lookahead router.
"""

from __future__ import annotations

import math

from qorch import Circuit, LocalSimulator
from qorch.dynamic import repetition_code_circuit, teleportation_circuit
from qorch.ir import Gate, Measure, Reset
from qorch.transpiler import (
    TIFR_SUPERCONDUCTING,
    CouplingMap,
    decompose,
    optimize,
    route,
    route_lookahead,
    transpile,
)

_LINE_5 = CouplingMap(TIFR_SUPERCONDUCTING.coupling_map)


def _conditions(circuit: Circuit) -> list[tuple[tuple[int, int], ...] | None]:
    return [op.condition for op in circuit.gates]


# ── end-to-end semantic equivalence ──────────────────────────────────────


def test_repetition_code_still_corrects_after_transpile():
    """The flagship dynamic circuit compiles to a real device and behaves identically.

    |0> encoded, an X injected on q1: the syndrome must read 11 and the decoded
    logical bit must still be 0 — through decomposition, routing on a linear
    chain, and optimization.
    """
    c = repetition_code_circuit(error_qubit=1)
    native = transpile(c, TIFR_SUPERCONDUCTING, coupling_map=_LINE_5)

    assert native.is_dynamic
    assert LocalSimulator(seed=7).run(c, shots=200).counts == {"110": 200}
    assert LocalSimulator(seed=7).run(native, shots=200).counts == {"110": 200}


def test_transpiled_teleportation_still_teleports():
    """Bob's qubit holds |1> on every shot after compiling the circuit."""
    c = teleportation_circuit(state_prep=Circuit(1).x(0))
    native = transpile(c, TIFR_SUPERCONDUCTING, coupling_map=CouplingMap(((0, 1), (1, 0), (1, 2), (2, 1))))

    counts = LocalSimulator(seed=11).run(native, shots=200).counts
    assert sum(counts.values()) == 200
    assert all(key[2] == "1" for key in counts)   # c2 = the teleported bit


def test_transpile_preserves_classical_register_width():
    c = Circuit(3, num_clbits=2).h(0).measure_into(0, 0).x_if(1, 0, 1)
    out = transpile(c, TIFR_SUPERCONDUCTING, coupling_map=_LINE_5)
    assert out.num_clbits == 2


# ── decomposition ────────────────────────────────────────────────────────


def test_decompose_propagates_condition_onto_expansion():
    """A conditioned h becomes a conditioned rz-sx-rz, not an unconditional one."""
    c = Circuit(1, num_clbits=1).measure_into(0, 0).gate_if("h", (0,), ((0, 1),))
    out = decompose(c, TIFR_SUPERCONDUCTING)

    expansion = [op for op in out.gates if isinstance(op, Gate)]
    assert len(expansion) > 1                       # h really did decompose
    assert all(op.condition == ((0, 1),) for op in expansion)


def test_decompose_passes_measure_and_reset_through():
    c = Circuit(2, num_clbits=1).h(0).measure_into(0, 0).reset(0).x_if(1, 0, 1)
    out = decompose(c, TIFR_SUPERCONDUCTING)

    assert any(isinstance(op, Measure) and op.cbit == 0 for op in out.gates)
    assert any(isinstance(op, Reset) for op in out.gates)


def test_decompose_leaves_unconditional_gates_unconditional():
    c = Circuit(2, num_clbits=1).h(0).measure_into(0, 0).x_if(1, 0, 1)
    out = decompose(c, TIFR_SUPERCONDUCTING)
    assert None in _conditions(out)


# ── routing ──────────────────────────────────────────────────────────────


def test_route_remaps_measure_onto_the_swapped_wire():
    """After a SWAP moves logical q0, its measurement must follow it."""
    c = (
        Circuit(3, num_clbits=1)
        .cx(0, 2)              # not adjacent on a line → forces a SWAP
        .measure_into(0, 0)
    )
    out = route(c, CouplingMap(((0, 1), (1, 0), (1, 2), (2, 1))))

    swaps = [op for op in out.gates if isinstance(op, Gate) and op.name == "swap"]
    measures = [op for op in out.gates if isinstance(op, Measure)]
    assert swaps                                     # routing actually happened
    assert len(measures) == 1
    assert measures[0].cbit == 0                     # classical target untouched
    assert measures[0].qubits != (0,)                # qubit followed the permutation


def test_route_preserves_conditions_on_single_qubit_gates():
    c = Circuit(3, num_clbits=1).h(0).measure_into(0, 0).x_if(2, 0, 1)
    out = route(c, CouplingMap(((0, 1), (1, 0), (1, 2), (2, 1))))
    assert ((0, 1),) in _conditions(out)


def test_route_keeps_inserted_swaps_unconditional():
    """SWAPs define the layout permutation, so they must never be conditioned."""
    c = Circuit(3, num_clbits=1).measure_into(0, 0).gate_if("cx", (0, 2), ((0, 1),))
    out = route(c, CouplingMap(((0, 1), (1, 0), (1, 2), (2, 1))))

    swaps = [op for op in out.gates if isinstance(op, Gate) and op.name == "swap"]
    assert swaps
    assert all(op.condition is None for op in swaps)


def test_lookahead_router_keeps_feedforward_after_its_measurement():
    """The classical hazard edges stop a conditional gate from being hoisted."""
    c = (
        Circuit(3, num_clbits=1)
        .h(0)
        .cx(0, 1)
        .measure_into(0, 0)
        .x_if(2, 0, 1)         # no qubit dependency — only a classical one
    )
    out = route_lookahead(c, _LINE_5)

    ops = list(out.gates)
    measure_at = next(i for i, op in enumerate(ops) if isinstance(op, Measure))
    cond_at = next(i for i, op in enumerate(ops) if op.condition is not None)
    assert cond_at > measure_at


def test_lookahead_router_orders_measurement_after_prior_reader():
    """A measurement overwriting a bit stays after the gate that read it (WAR)."""
    c = (
        Circuit(2, num_clbits=1)
        .measure_into(0, 0)
        .x_if(1, 0, 1)
        .measure_into(1, 0)    # overwrites c0, must not overtake the x_if
    )
    out = route_lookahead(c, _LINE_5)

    ops = list(out.gates)
    cond_at = next(i for i, op in enumerate(ops) if op.condition is not None)
    last_measure_at = max(i for i, op in enumerate(ops) if isinstance(op, Measure))
    assert last_measure_at > cond_at


# ── optimization ─────────────────────────────────────────────────────────


def test_optimizer_does_not_cancel_across_a_measurement():
    """h · measure · h is not the identity — the measurement collapses the state."""
    c = Circuit(1, num_clbits=1).h(0).measure_into(0, 0).h(0)
    out = optimize(c)
    assert len(out.gates) == 3


def test_optimizer_cancels_identically_conditioned_pair():
    c = (
        Circuit(1, num_clbits=1)
        .measure_into(0, 0)
        .gate_if("x", (0,), ((0, 1),))
        .gate_if("x", (0,), ((0, 1),))
    )
    out = optimize(c)
    assert not [op for op in out.gates if isinstance(op, Gate)]


def test_optimizer_keeps_differently_conditioned_pair():
    """Conditioned on different bits, only one of the two may fire — no cancellation."""
    c = (
        Circuit(1, num_clbits=2)
        .measure_into(0, 0)
        .gate_if("x", (0,), ((0, 1),))
        .gate_if("x", (0,), ((1, 1),))
    )
    out = optimize(c)
    assert len([op for op in out.gates if isinstance(op, Gate)]) == 2


def test_optimizer_does_not_cancel_conditional_against_unconditional():
    c = Circuit(1, num_clbits=1).measure_into(0, 0).gate_if("x", (0,), ((0, 1),)).x(0)
    out = optimize(c)
    assert len([op for op in out.gates if isinstance(op, Gate)]) == 2


def test_optimizer_merges_conditioned_rotations():
    c = (
        Circuit(1, num_clbits=1)
        .measure_into(0, 0)
        .gate_if("rz", (0,), ((0, 1),), params=(math.pi / 4,))
        .gate_if("rz", (0,), ((0, 1),), params=(math.pi / 4,))
    )
    out = optimize(c)

    rotations = [op for op in out.gates if isinstance(op, Gate)]
    assert len(rotations) == 1
    assert abs(float(rotations[0].params[0]) - math.pi / 2) < 1e-12
    assert rotations[0].condition == ((0, 1),)      # merge keeps the condition


def test_optimizer_drops_conditional_identity():
    c = Circuit(1, num_clbits=1).measure_into(0, 0).gate_if("id", (0,), ((0, 1),))
    out = optimize(c)
    assert not [op for op in out.gates if isinstance(op, Gate)]
