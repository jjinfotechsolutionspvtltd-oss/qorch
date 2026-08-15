"""Euler fusion and commutation-aware cancellation.

The adjacency-only optimizer combines gates that already sit next to each other.
These two passes go further: a run of single-qubit gates is *one* rotation
however long it is, and an inverse pair separated by a gate it commutes past can
still annihilate.

Both rewrite circuits at a level where a subtle error produces a plausible wrong
distribution rather than a crash, so the primary test is unitary equivalence over
random circuits — checked against the simulator, not argued structurally.
"""

from __future__ import annotations

import math
import random

import pytest

from qorch import Circuit, LocalSimulator
from qorch.transpiler import (
    IIT_JODHPUR_ION_TRAP,
    TIFR_SUPERCONDUCTING,
    CouplingMap,
    cancel_commuting,
    fuse_single_qubit_runs,
    transpile_with_layout,
)
from qorch.transpiler.fusion import euler_gates, zyz_angles

_LINE_5 = CouplingMap(TIFR_SUPERCONDUCTING.coupling_map)
_ONE_QUBIT = ["h", "x", "y", "z", "sx", "t", "id"]
_ROTATIONS = ["rx", "ry", "rz"]


def _unitary(circuit: Circuit) -> list[complex]:
    n = circuit.num_qubits
    sim = LocalSimulator(use_numpy=False)
    out: list[complex] = []
    for j in range(1 << n):
        prep = Circuit(n)
        for q in range(n):
            if j & (1 << (n - 1 - q)):
                prep = prep.x(q)
        out.extend(sim._evolve(Circuit(n, gates=prep.gates + circuit.gates)))
    return out


def _same_up_to_phase(u, v, tol: float = 1e-9) -> bool:
    ref = None
    for a, b in zip(u, v):
        if abs(b) > 1e-9:
            ref = a / b
            break
    if ref is None:
        return False
    if abs(abs(ref) - 1.0) > tol:
        return False
    return all(abs(a - ref * b) < tol for a, b in zip(u, v))


def _random_circuit(rng: random.Random, n: int, depth: int) -> Circuit:
    c = Circuit(n)
    for _ in range(depth):
        roll = rng.random()
        if roll < 0.5 or n == 1:
            c = c._add(rng.choice(_ONE_QUBIT), rng.randrange(n))
        elif roll < 0.72:
            c = c._add(rng.choice(_ROTATIONS), rng.randrange(n),
                       params=(rng.uniform(-math.pi, math.pi),))
        else:
            a, b = rng.sample(range(n), 2)
            c = c._add("cx", a, b)
    return c


# ── the property that matters ────────────────────────────────────────────


def test_fusion_preserves_the_unitary_on_random_circuits() -> None:
    rng = random.Random(11)
    for _ in range(60):
        c = _random_circuit(rng, rng.randint(1, 3), rng.randint(1, 18))
        assert _same_up_to_phase(_unitary(c), _unitary(fuse_single_qubit_runs(c)))


def test_commutation_cancellation_preserves_the_unitary() -> None:
    rng = random.Random(12)
    for _ in range(60):
        c = _random_circuit(rng, rng.randint(1, 3), rng.randint(1, 18))
        rewritten = cancel_commuting(fuse_single_qubit_runs(c))
        assert _same_up_to_phase(_unitary(c), _unitary(rewritten))


def test_fusion_reduces_gate_count_substantially() -> None:
    rng = random.Random(11)
    before = after = 0
    for _ in range(60):
        c = _random_circuit(rng, rng.randint(1, 3), rng.randint(4, 18))
        before += len(c.gates)
        after += len(cancel_commuting(fuse_single_qubit_runs(c)).gates)
    assert after < before * 0.85, f"only reduced {before} → {after}"


# ── Euler decomposition ──────────────────────────────────────────────────


@pytest.mark.parametrize("name,params", [
    ("h", ()), ("x", ()), ("y", ()), ("z", ()), ("sx", ()), ("t", ()),
    ("rx", (0.7,)), ("ry", (-1.1,)), ("rz", (2.3,)),
])
def test_zyz_reproduces_each_gate(name: str, params) -> None:
    """Every single-qubit gate is some Rz·Ry·Rz; check it round-trips."""
    original = Circuit(1)._add(name, 0, params=params)
    from qorch.gates import gate_matrix

    rebuilt = Circuit(1, gates=tuple(euler_gates(gate_matrix(name, params), 0)))
    assert _same_up_to_phase(_unitary(original), _unitary(rebuilt))


def test_identity_fuses_to_nothing() -> None:
    fused = fuse_single_qubit_runs(Circuit(1).x(0).x(0))
    assert fused.gates == ()


def test_zyz_angles_of_identity_are_zero() -> None:
    alpha, beta, gamma = zyz_angles((1 + 0j, 0j, 0j, 1 + 0j))
    assert abs(beta) < 1e-12
    assert abs(math.remainder(alpha + gamma, 2 * math.pi)) < 1e-12


def test_a_long_run_collapses_to_at_most_three_rotations() -> None:
    c = Circuit(1)
    for name in ("h", "t", "sx", "t", "h", "z", "sx", "t", "h"):
        c = c._add(name, 0)
    assert len(fuse_single_qubit_runs(c).gates) <= 3


# ── boundaries fusion must respect ───────────────────────────────────────


def test_fusion_does_not_cross_a_two_qubit_gate() -> None:
    c = Circuit(2).h(0).cx(0, 1).h(0)
    fused = fuse_single_qubit_runs(c)
    assert any(g.name == "cx" for g in fused.gates)
    assert _same_up_to_phase(_unitary(c), _unitary(fused))


def test_fusion_does_not_cross_a_measurement() -> None:
    """Fusing across a measurement would change what the circuit computes."""
    c = Circuit(1, num_clbits=1).h(0).measure_into(0, 0).h(0)
    fused = fuse_single_qubit_runs(c)
    assert sum(1 for g in fused.gates if g.name == "h") == 2


def test_fusion_leaves_conditioned_gates_alone() -> None:
    c = Circuit(2, num_clbits=1).measure_into(0, 0).x_if(1, 0, 1).x_if(1, 0, 1)
    fused = fuse_single_qubit_runs(c)
    assert sum(1 for g in fused.gates if g.condition is not None) == 2


def test_fusion_leaves_unbound_parameters_alone() -> None:
    from qorch.ir import Parameter

    c = Circuit(1).rz(0, Parameter("theta")).h(0)
    assert fuse_single_qubit_runs(c) == c


# ── commutation ──────────────────────────────────────────────────────────


def test_rz_pair_cancels_through_a_cx_control() -> None:
    """rz on a control commutes past cx, so the pair meets and annihilates."""
    c = Circuit(2).rz(0, 0.4).cx(0, 1).rz(0, -0.4)
    out = cancel_commuting(c)
    assert [g.name for g in out.gates] == ["cx"]
    assert _same_up_to_phase(_unitary(c), _unitary(out))


def test_x_pair_cancels_through_a_cx_target() -> None:
    c = Circuit(2).x(1).cx(0, 1).x(1)
    out = cancel_commuting(c)
    assert [g.name for g in out.gates] == ["cx"]
    assert _same_up_to_phase(_unitary(c), _unitary(out))


def test_rz_does_not_cancel_through_a_cx_target() -> None:
    """rz does NOT commute with a CX target; cancelling here would be wrong."""
    c = Circuit(2).rz(1, 0.4).cx(0, 1).rz(1, -0.4)
    out = cancel_commuting(c)
    assert len(out.gates) == 3
    assert _same_up_to_phase(_unitary(c), _unitary(out))


def test_cancellation_does_not_cross_a_measurement() -> None:
    c = Circuit(1, num_clbits=1).x(0).measure_into(0, 0).x(0)
    assert len(cancel_commuting(c).gates) == 3


def test_gates_on_disjoint_qubits_always_commute() -> None:
    c = Circuit(2).h(0).x(1).h(0)
    assert [g.name for g in cancel_commuting(c).gates] == ["x"]


# ── pipeline integration ─────────────────────────────────────────────────


@pytest.mark.parametrize("target,coupling", [
    (TIFR_SUPERCONDUCTING, _LINE_5),
    (IIT_JODHPUR_ION_TRAP, None),
])
@pytest.mark.parametrize("name,build", [
    ("ghz", lambda: Circuit(5).h(0).cx(0, 1).cx(1, 2).cx(2, 3).cx(3, 4)),
    ("rotation run", lambda: Circuit(3).h(0).t(0).h(0).t(0).h(0).cx(0, 1)),
    ("long run", lambda: Circuit(2).h(0).t(0).sx(0).t(0).h(0).z(0).sx(0).t(0).h(0).cx(0, 1)),
])
def test_fusion_never_makes_the_final_circuit_worse(target, coupling, name, build) -> None:
    """Measured on what actually runs, not on the pre-lowering gate count.

    Collapsing a run into Rz-Ry-Rz is fewer gates as written, but a target with
    no native ry expands that ry into more gates than fusion saved. The pass
    keeps its rewrite only if it survives lowering — this is that guarantee.
    """
    c = build()
    off = transpile_with_layout(c, target, coupling_map=coupling, do_fusion=False)
    on = transpile_with_layout(c, target, coupling_map=coupling, do_fusion=True)

    assert len(on.circuit.gates) <= len(off.circuit.gates), (
        f"{name} on {target.name}: fusion grew the circuit "
        f"{len(off.circuit.gates)} → {len(on.circuit.gates)}"
    )
    assert {g.name for g in on.circuit.gates} <= set(target.basis_gates)


def test_fusion_helps_measurably_where_the_basis_suits_it() -> None:
    """On an rx/ry-native target, fusing a long run is a large win."""
    c = Circuit(2).h(0).t(0).sx(0).t(0).h(0).z(0).sx(0).t(0).h(0).cx(0, 1)
    off = transpile_with_layout(c, IIT_JODHPUR_ION_TRAP, do_fusion=False)
    on = transpile_with_layout(c, IIT_JODHPUR_ION_TRAP, do_fusion=True)
    assert len(on.circuit.gates) < len(off.circuit.gates) * 0.6


def test_fuse_runs_before_lowering() -> None:
    """Fusion emits rz/ry; running it after `lower` would leak non-native gates."""
    from qorch.transpiler import build_pass_manager

    names = [n for n, _ in build_pass_manager(
        TIFR_SUPERCONDUCTING, _LINE_5, do_fusion=True).passes]
    assert names.index("fuse") < names.index("lower")


def test_fusion_can_be_disabled() -> None:
    from qorch.transpiler import build_pass_manager

    names = [n for n, _ in build_pass_manager(
        TIFR_SUPERCONDUCTING, _LINE_5, do_fusion=False).passes]
    assert "fuse" not in names
