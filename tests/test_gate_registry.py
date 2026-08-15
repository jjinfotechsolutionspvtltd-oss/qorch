"""One registry defines each gate; everything else derives from it.

Gate facts used to be restated in five places with nothing keeping them in sync,
and they drifted. The regression test below pins the bug that drift caused —
the optimizer believed ``sx`` was self-inverse, so it cancelled ``sx sx`` pairs
even though ``SX·SX = X``, turning a circuit that outputs 1 into one that
outputs 0.

The consistency tests check the derived views against the registry *and* against
the physics, so a wrong entry in the registry itself is caught rather than
faithfully propagated everywhere.
"""

from __future__ import annotations

import cmath
import math

import pytest

from qorch import Circuit, LocalSimulator
from qorch.gates import (
    ANGLE_INVERSE_GATES,
    CLIFFORD_GATES,
    GATES,
    ROTATION_GATES,
    SELF_INVERSE_GATES,
    SUPPORTED_GATE_NAMES,
    gate_def,
    gate_duration_ns,
    gate_matrix,
)
from qorch.ir import SUPPORTED_GATES
from qorch.transpiler import optimize
from qorch.transpiler.optimizer import _ROTATION_GATES, _SELF_INVERSE

_IDENTITY = (1 + 0j, 0j, 0j, 1 + 0j)


def _matmul(u, v):
    a, b, c, d = u
    e, f, g, h = v
    return (a * e + b * g, a * f + b * h, c * e + d * g, c * f + d * h)


def _close(u, v, tol: float = 1e-12) -> bool:
    return all(abs(x - y) < tol for x, y in zip(u, v))


# ── the regression that motivated the registry ───────────────────────────


def test_sx_is_not_self_inverse() -> None:
    """SX is a fourth root of the identity: SX·SX = X, not I."""
    sx = gate_matrix("sx")
    assert _close(_matmul(sx, sx), gate_matrix("x"))
    assert "sx" not in SELF_INVERSE_GATES


def test_optimizer_does_not_cancel_adjacent_sx_pairs() -> None:
    """The bug in full: optimizing this used to flip the measured result.

    ``sx sx`` is X, so the qubit must read 1 both before and after optimization.
    """
    c = Circuit(1).sx(0).sx(0).measure(0)
    before = LocalSimulator(seed=1).run(c, shots=500).counts
    after = LocalSimulator(seed=1).run(optimize(c), shots=500).counts

    assert before == {"1": 500}
    assert after == before, "optimizer changed the result of sx·sx"


# ── every self-inverse claim is checked against the matrix ───────────────


@pytest.mark.parametrize("name", sorted(n for n in SELF_INVERSE_GATES
                                        if GATES[n].arity == 1))
def test_self_inverse_single_qubit_gates_really_square_to_identity(name: str) -> None:
    m = gate_matrix(name)
    assert _close(_matmul(m, m), _IDENTITY), f"{name} is not self-inverse"


@pytest.mark.parametrize("name", sorted(n for n, g in GATES.items()
                                        if g.arity == 1 and g.num_params == 0
                                        and n not in SELF_INVERSE_GATES))
def test_non_self_inverse_gates_really_are_not(name: str) -> None:
    """Guard the other direction: a missing entry is as wrong as a false one."""
    m = gate_matrix(name)
    assert not _close(_matmul(m, m), _IDENTITY), f"{name} could be marked self-inverse"


@pytest.mark.parametrize("name", sorted(ROTATION_GATES))
def test_rotations_are_inverted_by_negating_the_angle(name: str) -> None:
    for theta in (0.3, 1.1, math.pi):
        product = _matmul(gate_matrix(name, (theta,)), gate_matrix(name, (-theta,)))
        assert _close(product, _IDENTITY, tol=1e-9)


def test_every_rotation_is_marked_angle_inverse() -> None:
    assert ROTATION_GATES <= ANGLE_INVERSE_GATES


# ── derived views actually derive ────────────────────────────────────────


def test_ir_supported_gates_comes_from_the_registry() -> None:
    assert SUPPORTED_GATES == SUPPORTED_GATE_NAMES


def test_optimizer_sets_come_from_the_registry() -> None:
    assert _ROTATION_GATES == ROTATION_GATES
    assert _SELF_INVERSE == {n for n in SELF_INVERSE_GATES if GATES[n].arity == 1}
    assert "sx" not in _SELF_INVERSE


def test_simulator_matrices_come_from_the_registry() -> None:
    from qorch.backends.simulator import _GATES_1Q, _gate_matrix

    for name, matrix in _GATES_1Q.items():
        assert _close(matrix, gate_matrix(name))
    assert _close(_gate_matrix("rz", (0.7,)), gate_matrix("rz", (0.7,)))


def test_ir_gate_algebra_sets_come_from_the_registry() -> None:
    from qorch.ir import _ANGLE_GATES, _SELF_INVERSE_GATES

    assert _SELF_INVERSE_GATES == SELF_INVERSE_GATES
    assert _ANGLE_GATES == ANGLE_INVERSE_GATES


# ── metadata sanity ──────────────────────────────────────────────────────


def test_t_is_the_only_non_clifford_discrete_gate() -> None:
    """T is what Clifford+T accounting is counting; nothing else discrete costs."""
    discrete = {n for n, g in GATES.items() if g.num_params == 0}
    assert discrete - CLIFFORD_GATES == {"t"}


def test_t_gate_matrix_is_the_pi_over_four_phase() -> None:
    assert _close(gate_matrix("t"), (1, 0, 0, cmath.exp(1j * math.pi / 4)))


@pytest.mark.parametrize("name", sorted(GATES))
def test_arity_matches_whether_a_matrix_is_available(name: str) -> None:
    """Two-qubit gates have no 2x2 matrix and must say so, not return nonsense."""
    definition = gate_def(name)
    if definition.arity == 1:
        assert definition.matrix is not None
        assert len(gate_matrix(name, (0.5,) * definition.num_params)) == 4
    else:
        with pytest.raises(ValueError, match="2-qubit gate"):
            gate_matrix(name)


def test_two_qubit_gates_cost_more_time_than_single_qubit_ones() -> None:
    """Durations are advisory, but the ordering must be physically sensible."""
    assert gate_duration_ns("cx") > gate_duration_ns("x")
    assert gate_duration_ns("swap") >= gate_duration_ns("cx")
    assert gate_duration_ns("rz") == 0.0, "rz is a frame change, not a pulse"


def test_unknown_gate_names_report_what_is_available() -> None:
    with pytest.raises(ValueError, match="unknown gate"):
        gate_def("nope")
