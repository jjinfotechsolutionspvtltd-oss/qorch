"""The optional numpy kernel must agree exactly with the pure-Python one.

Two kernels computing the same thing is a correctness risk, not a feature: the
moment they disagree, results depend on whether numpy happened to be installed.
These tests pin agreement on random circuits covering every gate, and pin that
the pure-Python path still works entirely on its own — that fallback is what
makes ``dependencies = []`` a real claim rather than an aspiration.
"""

from __future__ import annotations

import math
import random

import pytest

from qorch import Circuit, LocalSimulator
from qorch.backends import numpy_kernel
from qorch.backends.simulator import _NUMPY_MIN_QUBITS

requires_numpy = pytest.mark.skipif(
    not numpy_kernel.is_available(), reason="numpy not installed"
)

_ONE_QUBIT = ["h", "x", "y", "z", "sx", "t", "id"]
_ROTATIONS = ["rx", "ry", "rz"]


def _random_circuit(rng: random.Random, num_qubits: int, depth: int) -> Circuit:
    c = Circuit(num_qubits)
    for _ in range(depth):
        roll = rng.random()
        if roll < 0.4 or num_qubits == 1:
            c = c._add(rng.choice(_ONE_QUBIT), rng.randrange(num_qubits))
        elif roll < 0.6:
            c = c._add(rng.choice(_ROTATIONS), rng.randrange(num_qubits),
                       params=(rng.uniform(-math.pi, math.pi),))
        else:
            a, b = rng.sample(range(num_qubits), 2)
            if rng.random() < 0.7:
                c = c._add(rng.choice(["cx", "swap"]), a, b)
            else:
                c = c.ms(a, b, rng.uniform(0.1, 1.0))
    return c


# ── the two kernels agree ────────────────────────────────────────────────


@requires_numpy
def test_kernels_agree_on_random_circuits() -> None:
    """Every gate, every arity, across many shapes — amplitudes must match."""
    rng = random.Random(7)
    worst = 0.0
    for _ in range(40):
        n = rng.randint(1, 6)
        circuit = _random_circuit(rng, n, rng.randint(1, 25))
        pure = LocalSimulator(use_numpy=False)._evolve(circuit)
        fast = LocalSimulator(use_numpy=True)._evolve(circuit)
        assert len(pure) == len(fast) == 2 ** n
        worst = max(worst, max(abs(a - b) for a, b in zip(pure, fast)))
    assert worst < 1e-12, f"kernels disagree by {worst:.2e}"


@requires_numpy
@pytest.mark.parametrize("gate,params", [
    ("h", ()), ("x", ()), ("y", ()), ("z", ()), ("sx", ()), ("t", ()), ("id", ()),
    ("rx", (0.7,)), ("ry", (-1.3,)), ("rz", (2.2,)),
])
def test_single_qubit_gates_match(gate: str, params) -> None:
    """Checked on qubit 1 of 3, so an axis-ordering error cannot hide."""
    c = Circuit(3).h(0).h(2)._add(gate, 1, params=params)
    assert _amplitudes_match(c)


@requires_numpy
@pytest.mark.parametrize("gate,args", [
    ("cx", (0, 2)), ("cx", (2, 0)), ("swap", (0, 2)), ("swap", (2, 0)),
])
def test_two_qubit_gates_match_in_both_operand_orders(gate: str, args) -> None:
    """Operand order matters for cx and is the easiest thing to get backwards."""
    c = Circuit(3).h(0).x(1)._add(gate, *args)
    assert _amplitudes_match(c)


@requires_numpy
def test_ms_gate_matches() -> None:
    assert _amplitudes_match(Circuit(3).h(0).ms(0, 2, 0.4))


def _amplitudes_match(circuit: Circuit, tol: float = 1e-12) -> bool:
    pure = LocalSimulator(use_numpy=False)._evolve(circuit)
    fast = LocalSimulator(use_numpy=True)._evolve(circuit)
    return all(abs(a - b) < tol for a, b in zip(pure, fast))


@requires_numpy
def test_measured_distributions_match() -> None:
    """Agreement at the level users actually observe, not just amplitudes."""
    c = Circuit(4).h(0).cx(0, 1).cx(1, 2).cx(2, 3).measure(0, 1, 2, 3)
    pure = LocalSimulator(seed=1, use_numpy=False).run(c, shots=4000).counts
    fast = LocalSimulator(seed=1, use_numpy=True).run(c, shots=4000).counts

    assert set(pure) == set(fast) == {"0000", "1111"}
    for key in pure:
        assert abs(pure[key] - fast[key]) / 4000 < 0.05


# ── the pure-Python fallback stands alone ────────────────────────────────


def test_pure_python_kernel_works_without_numpy() -> None:
    """The dependency-free path is the product, not a degraded mode."""
    c = Circuit(3).h(0).cx(0, 1).cx(1, 2).measure(0, 1, 2)
    counts = LocalSimulator(seed=2, use_numpy=False).run(c, shots=1000).counts
    assert set(counts) == {"000", "111"}
    assert sum(counts.values()) == 1000


def test_forcing_numpy_without_numpy_installed_is_an_error() -> None:
    if numpy_kernel.is_available():
        pytest.skip("numpy is installed; the error path cannot trigger")
    with pytest.raises(ValueError, match="numpy is not installed"):
        LocalSimulator(use_numpy=True)


# ── kernel selection is by circuit size, and is deliberate ───────────────


@requires_numpy
def test_small_circuits_use_the_python_kernel_by_default() -> None:
    """numpy is slower below the crossover, so auto mode must not reach for it."""
    sim = LocalSimulator()
    assert not sim._should_use_numpy(Circuit(_NUMPY_MIN_QUBITS - 1))


@requires_numpy
def test_large_circuits_use_the_numpy_kernel_by_default() -> None:
    sim = LocalSimulator()
    assert sim._should_use_numpy(Circuit(_NUMPY_MIN_QUBITS))


@requires_numpy
@pytest.mark.parametrize("forced", [True, False])
def test_an_explicit_choice_overrides_the_size_heuristic(forced: bool) -> None:
    sim = LocalSimulator(use_numpy=forced)
    assert sim._should_use_numpy(Circuit(2)) is forced
    assert sim._should_use_numpy(Circuit(12)) is forced


def test_selection_never_picks_numpy_when_it_is_unavailable() -> None:
    if numpy_kernel.is_available():
        pytest.skip("numpy is installed")
    assert not LocalSimulator()._should_use_numpy(Circuit(16))
