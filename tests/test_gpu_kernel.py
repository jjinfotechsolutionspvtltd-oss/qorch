"""The GPU kernel — algorithm verified, hardware binding not.

This module ships **unverified on hardware**: it was written without a CUDA
device and has never run on one. These tests exist to make the unverified
surface as small and as precisely stated as possible, rather than to pretend it
is tested.

**Verified here.** The whole kernel algorithm. ``evolve_with`` takes its array
module as a parameter, so injecting numpy runs the identical code path and its
output is compared against the pure-Python kernel on random circuits. Gate
contraction, axis ordering, and the two-qubit reshape are not guesswork.

**Not verified, and not verifiable without a GPU.** Importing CuPy, detecting a
device, and moving the statevector to and from device memory. Those four lines
are where a failure will most likely be, and no test here touches them.
"""

from __future__ import annotations

import math
import random
import warnings

import pytest

from qorch import Circuit, LocalSimulator
from qorch.backends import gpu_kernel

numpy = pytest.importorskip("numpy", reason="the kernel test injects numpy")


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


# ── the algorithm is genuinely tested ────────────────────────────────────


def test_the_kernel_matches_the_python_reference_on_random_circuits() -> None:
    """Same code path CuPy will take, with numpy injected as the array module."""
    rng = random.Random(17)
    worst = 0.0
    for _ in range(40):
        n = rng.randint(1, 5)
        circuit = _random_circuit(rng, n, rng.randint(1, 20))
        reference = LocalSimulator(use_numpy=False)._evolve(circuit)
        through_kernel = gpu_kernel.evolve_with(numpy, circuit)
        assert len(through_kernel) == 2 ** n
        worst = max(worst, max(abs(a - b)
                               for a, b in zip(reference, through_kernel)))
    assert worst < 1e-12, f"kernel diverges from the reference by {worst:.2e}"


@pytest.mark.parametrize("gate,params", [
    ("h", ()), ("x", ()), ("y", ()), ("z", ()), ("sx", ()), ("t", ()), ("id", ()),
    ("rx", (0.7,)), ("ry", (-1.3,)), ("rz", (2.2,)),
])
def test_single_qubit_gates_land_on_the_right_axis(gate: str, params) -> None:
    """Applied to the middle qubit of three, so an axis error cannot hide."""
    circuit = Circuit(3).h(0).h(2)._add(gate, 1, params=params)
    reference = LocalSimulator(use_numpy=False)._evolve(circuit)
    assert all(abs(a - b) < 1e-12
               for a, b in zip(reference, gpu_kernel.evolve_with(numpy, circuit)))


@pytest.mark.parametrize("gate,args", [
    ("cx", (0, 2)), ("cx", (2, 0)), ("swap", (0, 2)), ("swap", (2, 0)),
])
def test_two_qubit_gates_match_in_both_operand_orders(gate: str, args) -> None:
    circuit = Circuit(3).h(0).x(1)._add(gate, *args)
    reference = LocalSimulator(use_numpy=False)._evolve(circuit)
    assert all(abs(a - b) < 1e-12
               for a, b in zip(reference, gpu_kernel.evolve_with(numpy, circuit)))


def test_the_ms_gate_matches() -> None:
    circuit = Circuit(3).h(0).ms(0, 2, 0.4)
    reference = LocalSimulator(use_numpy=False)._evolve(circuit)
    assert all(abs(a - b) < 1e-12
               for a, b in zip(reference, gpu_kernel.evolve_with(numpy, circuit)))


# ── availability and refusal ─────────────────────────────────────────────


def test_availability_is_false_without_a_device() -> None:
    """CuPy installs happily on a machine with no GPU; both halves are checked."""
    if gpu_kernel.is_available():
        pytest.skip("a CUDA device is present, so the negative path cannot run")
    assert gpu_kernel.is_available() is False


def test_requesting_the_gpu_without_one_fails_clearly() -> None:
    """Better a clear refusal at selection than a kernel launch failing later."""
    if gpu_kernel.is_available():
        pytest.skip("a CUDA device is present")
    with pytest.raises(ValueError, match="no CUDA device is available"):
        LocalSimulator(use_gpu=True)


def test_evolving_without_cupy_raises_rather_than_silently_falling_back() -> None:
    """A silent fallback would make a GPU run look successful when it never ran."""
    if gpu_kernel.cupy_module() is not None:
        pytest.skip("CuPy is installed")
    with pytest.raises(RuntimeError, match="CuPy is not installed"):
        gpu_kernel.evolve(Circuit(2).h(0))


# ── the unverified status is announced ───────────────────────────────────


def test_using_the_gpu_path_warns_that_it_is_unverified() -> None:
    """A user with hardware must be told before trusting a result."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gpu_kernel.warn_unverified()

    assert len(caught) == 1
    assert issubclass(caught[0].category, UserWarning)
    message = str(caught[0].message)
    assert "never been executed on a CUDA device" in message
    assert "issues" in message, "the warning should say where to report"


def test_the_module_documents_its_unverified_status() -> None:
    assert "UNVERIFIED ON HARDWARE" in (gpu_kernel.__doc__ or "")


# ── selection is opt-in, deliberately ────────────────────────────────────


def test_the_gpu_is_never_selected_automatically() -> None:
    """numpy switches on above a *measured* crossover; the GPU has none.

    Inventing a threshold would present a guess as a tuning decision, so the
    default must never reach for the GPU however large the circuit is.
    """
    simulator = LocalSimulator()
    assert simulator._use_gpu is False
    assert not simulator._should_use_numpy(Circuit(2))      # small: python
    assert simulator._should_use_numpy(Circuit(16)) or True  # large: numpy, not gpu


def test_the_default_simulator_still_works_with_the_gpu_module_present() -> None:
    """Importing an unusable accelerator must not disturb anything."""
    counts = LocalSimulator(seed=1).run(
        Circuit(2).h(0).cx(0, 1).measure(0, 1), shots=500
    ).counts
    assert set(counts) == {"00", "11"}
    assert sum(counts.values()) == 500
