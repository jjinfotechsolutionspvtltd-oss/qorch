"""Every supported gate must decompose to every target, and mean the same thing.

Two properties, checked exhaustively over ``SUPPORTED_GATES × every gate set``:

  1. **Reachability** — decomposition succeeds and emits only native gates. A gate
     that is in ``SUPPORTED_GATES`` but has no path to a target's basis is a hole a
     user falls into with a bare ``ValueError``; ``rx`` was unreachable on the
     flagship superconducting set exactly this way.
  2. **Correctness** — the decomposition implements the *same unitary*, up to a
     global phase. Reachability alone would happily accept a rule that compiles
     cleanly and computes the wrong thing, which is the failure mode that matters
     here: no crash, just a wrong distribution.

The unitary is reconstructed column by column (``U|j⟩`` for each basis state ``j``)
rather than compared structurally, so a rule is judged by what it *does*, not by
how it is written.
"""

from __future__ import annotations

import math

import pytest

from qorch import Circuit, LocalSimulator
from qorch.ir import SUPPORTED_GATES, Gate
from qorch.transpiler import decompose
from qorch.transpiler.gateset import (
    CLIFFORD_T,
    DRDO_MIRAI,
    IIT_JODHPUR_ION_TRAP,
    TIFR_SUPERCONDUCTING,
)

GATE_SETS = [IIT_JODHPUR_ION_TRAP, TIFR_SUPERCONDUCTING, DRDO_MIRAI, CLIFFORD_T]

# Continuous sets represent any rotation exactly. Clifford+T is *discrete*: it can
# only reach angles that are multiples of π/4 exactly, and everything else is an
# approximation (Solovay–Kitaev). Holding it to exact equality would be demanding
# the mathematically impossible, so it is tested separately and by fidelity.
EXACT_GATE_SETS = [IIT_JODHPUR_ION_TRAP, TIFR_SUPERCONDUCTING, DRDO_MIRAI]

_TWO_QUBIT = {"cx", "swap", "ms"}
_ANGLE = {"rx", "ry", "rz"}

# Angles chosen to include the degenerate cases (0, π) where a naive rule can be
# accidentally right, plus irrational-ish ones where it cannot.
_THETAS = (0.0, 0.3, 1.1, math.pi / 2, math.pi, 2.7)

# Angles Clifford+T reaches exactly: integer multiples of π/4 (powers of T).
_DYADIC_THETAS = tuple(k * math.pi / 4 for k in range(8))


def _unitary(circuit: Circuit) -> list[complex]:
    """Full unitary, as the concatenation of columns U|j⟩ over basis states j."""
    n = circuit.num_qubits
    sim = LocalSimulator()
    out: list[complex] = []
    for j in range(1 << n):
        prep = Circuit(n)
        for q in range(n):
            if j & (1 << (n - 1 - q)):
                prep = prep.x(q)
        out.extend(sim._evolve(Circuit(n, gates=prep.gates + circuit.gates)))
    return out


def _equal_up_to_phase(u: list[complex], v: list[complex], tol: float = 1e-9) -> bool:
    """True if u = e^{iφ}·v for some φ. Global phase is unobservable."""
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


def _one_gate_circuit(name: str, theta: float) -> Circuit:
    n = 2 if name in _TWO_QUBIT else 1
    if name in _ANGLE:
        params: tuple[float, ...] = (theta,)
    elif name == "ms":
        params = (theta,)
    else:
        params = ()
    return Circuit(n, gates=(Gate(name, tuple(range(n)), params),))


@pytest.mark.parametrize("gate_set", GATE_SETS, ids=lambda g: g.name)
@pytest.mark.parametrize("gate", sorted(SUPPORTED_GATES))
def test_every_gate_reaches_every_target(gate: str, gate_set) -> None:
    """Decomposition succeeds and emits only gates the hardware implements."""
    out = decompose(_one_gate_circuit(gate, 0.7), gate_set)
    emitted = {g.name for g in out.gates}
    assert emitted <= set(gate_set.basis_gates), (
        f"{gate} → {gate_set.name} emitted non-native {emitted - set(gate_set.basis_gates)}"
    )


@pytest.mark.parametrize("gate_set", EXACT_GATE_SETS, ids=lambda g: g.name)
@pytest.mark.parametrize("gate", sorted(SUPPORTED_GATES))
def test_every_decomposition_preserves_the_unitary(gate: str, gate_set) -> None:
    """The lowered circuit computes the same thing, up to an unobservable phase."""
    thetas = _THETAS if gate in _ANGLE or gate == "ms" else (0.0,)
    for theta in thetas:
        source = _one_gate_circuit(gate, theta)
        lowered = decompose(source, gate_set)
        assert _equal_up_to_phase(_unitary(source), _unitary(lowered)), (
            f"{gate}(θ={theta:.4f}) → {gate_set.name} changed the unitary"
        )


@pytest.mark.parametrize("gate", sorted(SUPPORTED_GATES - _ANGLE))
def test_clifford_t_is_exact_for_non_rotation_gates(gate: str) -> None:
    """Every discrete gate has an exact Clifford+T form."""
    source = _one_gate_circuit(gate, 0.0)
    lowered = decompose(source, CLIFFORD_T)
    assert _equal_up_to_phase(_unitary(source), _unitary(lowered))


@pytest.mark.parametrize("gate", ["rx", "ry", "rz"])
def test_clifford_t_is_exact_on_pi_over_four_multiples(gate: str) -> None:
    """The angles Clifford+T can actually represent must be represented exactly.

    T is Rz(π/4), so integer multiples of π/4 are reachable by T-powers with no
    approximation. Anything less than exact here is a real bug, not a limit of
    the gate set.
    """
    for theta in _DYADIC_THETAS:
        source = _one_gate_circuit(gate, theta)
        lowered = decompose(source, CLIFFORD_T)
        assert _equal_up_to_phase(_unitary(source), _unitary(lowered), tol=1e-9), (
            f"{gate}(θ={theta:.4f}) is a π/4 multiple and must be exact"
        )


def test_identity_compiles_to_nothing() -> None:
    """id is the identity on every target — it should cost zero native gates."""
    for gate_set in GATE_SETS:
        assert decompose(Circuit(1).id(0), gate_set).gates == ()


def test_ms_survives_a_round_trip_through_cx() -> None:
    """ms → CX-based → back to ms-native preserves the entangler.

    Exercises both directions of the ion-trap/superconducting bridge, which is
    what makes a circuit written for one architecture runnable on the other.
    """
    source = Circuit(2, gates=(Gate("ms", (0, 1), (0.7,)),))
    via_cx = decompose(source, TIFR_SUPERCONDUCTING)
    back = decompose(via_cx, IIT_JODHPUR_ION_TRAP)

    assert {g.name for g in back.gates} <= set(IIT_JODHPUR_ION_TRAP.basis_gates)
    assert _equal_up_to_phase(_unitary(source), _unitary(back))


def test_clifford_t_arbitrary_angle_approximation_is_documented_but_weak() -> None:
    """Pins the *current*, poor approximation quality for arbitrary angles.

    Clifford+T cannot express an arbitrary rotation exactly, so some error is
    inherent. The size of it is not: ``_rz_to_clifford_t`` runs a BFS capped at
    depth 8 and settles for a single ``t`` on θ=0.3 — process fidelity ≈0.97 for
    one rotation, which compounds badly across a real circuit and is reported to
    the caller nowhere.

    This test asserts what is true today rather than what should be true, so that
    improving the synthesis makes it fail loudly and get tightened.
    """
    worst = 1.0
    for theta in (0.3, 1.1, 2.7):
        source = _one_gate_circuit("rz", theta)
        lowered = decompose(source, CLIFFORD_T)
        u, v = _unitary(source), _unitary(lowered)
        fidelity = abs(sum(a.conjugate() * b for a, b in zip(u, v))) / 2.0
        worst = min(worst, fidelity)
        assert not _equal_up_to_phase(u, v), "unexpectedly exact — tighten this test"

    assert worst > 0.95, "approximation got worse than the recorded baseline"
    assert worst < 0.999, (
        "synthesis improved past the recorded baseline — raise this bound and "
        "consider whether arbitrary angles can now be asserted near-exact"
    )


def test_unitary_check_would_catch_a_wrong_rule() -> None:
    """Guard the guard: the comparison must reject a plausible-but-wrong rule.

    rz(θ) → ry(-π/2) rx(θ) ry(π/2) has the correct shape and the wrong sign — the
    kind of error that compiles, runs, and returns a wrong answer. If this passes,
    the checks above prove nothing.
    """
    theta = 0.3
    correct = Circuit(1, gates=(Gate("rz", (0,), (theta,)),))
    wrong = Circuit(1, gates=(
        Gate("ry", (0,), (-math.pi / 2,)),
        Gate("rx", (0,), (theta,)),
        Gate("ry", (0,), (math.pi / 2,)),
    ))
    assert not _equal_up_to_phase(_unitary(correct), _unitary(wrong))
