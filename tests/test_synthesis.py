"""Clifford+T rotation synthesis: accurate, honest about its error, and cheap.

The property that matters is not "it produced some gates" but the triple:

  - the word really does approximate the rotation (verified against the
    simulator, not against the synthesizer's own arithmetic),
  - the error it *reports* is the error it *achieved* — a synthesizer that
    silently misses its target is worse than one that fails loudly, because the
    T-count it produces feeds the fault-tolerant resource estimator, and
  - the T-count stays near-optimal, since T gates are the expensive resource
    the whole Clifford+T representation exists to count.
"""

from __future__ import annotations

import math
import random

import pytest

from qorch import Circuit
from qorch.backends.simulator import _gate_matrix
from qorch.ir import Gate
from qorch.transpiler import decompose
from qorch.transpiler.decompose import clifford_t_synthesis_error
from qorch.transpiler.gateset import CLIFFORD_T
from qorch.transpiler.synthesis import (
    DEFAULT_PRECISION,
    SynthesisResult,
    synthesize_rz,
)


def _matrix_of(circuit: Circuit) -> tuple[complex, ...]:
    """Multiply out a 1-qubit circuit, independently of the synthesizer's math."""
    m: tuple[complex, ...] = (1 + 0j, 0j, 0j, 1 + 0j)
    for g in circuit.gates:
        a = _gate_matrix(g.name, tuple(float(p) for p in g.params))
        m = (
            a[0] * m[0] + a[1] * m[2], a[0] * m[1] + a[1] * m[3],
            a[2] * m[0] + a[3] * m[2], a[2] * m[1] + a[3] * m[3],
        )
    return m


def _fidelity(u, v) -> float:
    return 0.5 * abs(sum(a.conjugate() * b for a, b in zip(u, v)))


def _achieved_error(theta: float, result: SynthesisResult) -> float:
    circuit = Circuit(1, gates=tuple(Gate(name, (0,)) for name in result.gates))
    return 1.0 - _fidelity(_matrix_of(circuit), _gate_matrix("rz", (theta,)))


# ── exactness where exactness is possible ────────────────────────────────


@pytest.mark.parametrize("k", range(8))
def test_pi_over_four_multiples_are_exact(k: int) -> None:
    """T is Rz(π/4), so these need no approximation at all."""
    theta = k * math.pi / 4
    result = synthesize_rz(theta)
    assert result.exact
    assert result.error == 0.0
    assert _achieved_error(theta, result) < 1e-12


def test_zero_angle_costs_nothing() -> None:
    assert synthesize_rz(0.0).gates == ()


def test_full_turn_reduces_to_identity() -> None:
    """Rz(2π) = -I, which is the identity up to global phase."""
    assert synthesize_rz(2 * math.pi).gates == ()


# ── the reported error is the real error ─────────────────────────────────


@pytest.mark.parametrize("theta", [0.3, 1.1, 2.7, -0.45, 0.12345, -2.9])
def test_reported_error_matches_measured_error(theta: float) -> None:
    """The headline property: no silent near-misses.

    Recomputed from the emitted gate names through the simulator's own gate
    matrices, so the synthesizer cannot mark its own homework.
    """
    result = synthesize_rz(theta)
    assert abs(result.error - _achieved_error(theta, result)) < 1e-9


@pytest.mark.parametrize("theta", [0.3, 1.1, 2.7, -0.45, 0.12345, -2.9])
def test_default_precision_is_met(theta: float) -> None:
    result = synthesize_rz(theta)
    assert result.error <= DEFAULT_PRECISION
    assert not result.exact


def test_accuracy_holds_across_many_random_angles() -> None:
    """A handful of angles can be lucky; a spread of them cannot."""
    rng = random.Random(11)
    worst = 0.0
    for _ in range(40):
        theta = rng.uniform(-math.pi, math.pi)
        result = synthesize_rz(theta)
        measured = _achieved_error(theta, result)
        assert abs(result.error - measured) < 1e-9
        worst = max(worst, measured)
    assert worst <= DEFAULT_PRECISION


def test_tighter_precision_is_honoured() -> None:
    """Asking for more accuracy delivers it, at a cost in T gates."""
    loose = synthesize_rz(0.3, precision=2e-3)
    tight = synthesize_rz(0.3, precision=3e-4)
    assert tight.error <= 3e-4
    assert tight.error < loose.error
    assert tight.t_count >= loose.t_count


# ── cost stays near-optimal ──────────────────────────────────────────────


def test_t_count_is_near_optimal() -> None:
    """An optimal synthesizer needs ~3·log₂(1/ε) T gates; stay within reach of it.

    The point of a *good* synthesis is not just accuracy — Solovay–Kitaev would
    reach the same error with thousands of T gates and wreck every downstream
    resource estimate. This is the test that would catch such a regression.
    """
    budget = 3 * math.log2(1 / DEFAULT_PRECISION) * 2      # 2x the optimal bound
    rng = random.Random(3)
    for _ in range(20):
        result = synthesize_rz(rng.uniform(-math.pi, math.pi))
        assert result.t_count <= budget, f"T-count {result.t_count} exceeds {budget:.0f}"


def test_synthesis_is_deterministic() -> None:
    """Same angle, same word — compilation must be reproducible."""
    assert synthesize_rz(0.31415).gates == synthesize_rz(0.31415).gates


# ── the error is visible from the circuit level ──────────────────────────


def test_circuit_level_error_is_zero_for_exact_angles() -> None:
    c = Circuit(1).rz(0, math.pi / 4).rz(0, math.pi / 2)
    assert clifford_t_synthesis_error(c) == 0.0


def test_circuit_level_error_is_reported_for_arbitrary_angles() -> None:
    c = Circuit(1).rz(0, 0.3)
    error = clifford_t_synthesis_error(c)
    assert 0 < error <= DEFAULT_PRECISION


def test_circuit_level_error_ignores_unbound_parameters() -> None:
    """An unbound angle has no error yet — it must not crash the estimate."""
    from qorch.ir import Parameter

    c = Circuit(1).rz(0, Parameter("theta"))
    assert clifford_t_synthesis_error(c) == 0.0


def test_resource_estimate_surfaces_synthesis_error() -> None:
    """The estimator's T-count is only as meaningful as the synthesis under it."""
    from qorch.resource_estimation import estimate_resources, format_estimate

    approx = estimate_resources(Circuit(1).rz(0, 0.3))
    assert approx.synthesis_error > 0
    assert "synth error" in format_estimate(approx).lower()

    exact = estimate_resources(Circuit(1).rz(0, math.pi / 4))
    assert exact.synthesis_error == 0.0
    assert "exact" in format_estimate(exact).lower()


# ── end-to-end through the real decomposer ───────────────────────────────


@pytest.mark.parametrize("theta", [0.3, 1.1, -2.9])
def test_decompose_emits_only_clifford_t_and_stays_accurate(theta: float) -> None:
    lowered = decompose(Circuit(1, gates=(Gate("rz", (0,), (theta,)),)), CLIFFORD_T)
    assert {g.name for g in lowered.gates} <= set(CLIFFORD_T.basis_gates)
    error = 1.0 - _fidelity(_matrix_of(lowered), _gate_matrix("rz", (theta,)))
    assert error <= DEFAULT_PRECISION
