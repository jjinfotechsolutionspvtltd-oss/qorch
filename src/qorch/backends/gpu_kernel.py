"""CuPy statevector kernel — **never executed on a GPU by its authors.**

⚠️ UNVERIFIED ON HARDWARE ⚠️

This module was written and merged without a CUDA device available. Everything
below is reviewed, type-checked, and exercised against a numpy stand-in, but no
part of it has ever run on an actual GPU. Treat it as a starting point that
needs validating on your hardware, not as a tested feature.

**What is genuinely tested.** The kernel takes its array module as a parameter,
so the whole algorithm — gate contraction, axis ordering, the two-qubit
reshape — is exercised in CI with numpy injected, and its results are compared
against the pure-Python kernel. The maths is not guesswork.

**What is not tested, and could not be.** Importing CuPy. Detecting a device.
Moving the statevector to and from the GPU. Anything about performance. If this
breaks for you it will most likely be in those four places, and they are the
first things to check.

**Why it is opt-in and never automatic.** The numpy kernel switches on
automatically above a *measured* crossover of 8 qubits. No such measurement
exists here — a GPU's crossover depends on the card, the driver, and the
transfer cost, and inventing a threshold would be presenting a guess as a
tuning decision. So the GPU path runs only when explicitly asked for, and using
it emits a warning saying it is unverified.

If you have hardware and this works, the honest next step is to measure the
crossover and make selection automatic in the way the numpy path already is.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

from qorch.gates import gate_matrix
from qorch.ir import Circuit, bound_params

_CUPY: Any | None = None
_CHECKED = False

_UNVERIFIED_WARNING = (
    "qorch's GPU kernel has never been executed on a CUDA device by its "
    "authors. The algorithm is tested against numpy, but CuPy import, device "
    "detection, and host/device transfer are unverified. Please report what "
    "happens: https://github.com/jjinfotechsolutionspvtltd-oss/qorch/issues"
)


def cupy_module() -> Any | None:
    """Return CuPy if it is importable, else None. Imported once, lazily."""
    global _CUPY, _CHECKED
    if not _CHECKED:
        _CHECKED = True
        try:
            import cupy
        except ImportError:
            _CUPY = None
        else:
            _CUPY = cupy
    return _CUPY


def is_available() -> bool:
    """Whether CuPy imports *and* reports a usable device.

    Both halves matter: CuPy installs fine on a machine with no GPU and fails
    only when a kernel actually launches, which would turn a missing device into
    a confusing runtime error deep inside a simulation rather than a clear one
    at selection time.
    """
    cupy = cupy_module()
    if cupy is None:
        return False
    try:
        return bool(cupy.cuda.runtime.getDeviceCount())
    except Exception:                                # pragma: no cover - needs a GPU
        return False


def warn_unverified() -> None:
    """Emit the unverified-hardware warning. Called on every GPU run."""
    warnings.warn(_UNVERIFIED_WARNING, UserWarning, stacklevel=3)


# ── the kernel, parameterized by array module so it can be tested ────────


def _cx_matrix(xp: Any) -> Any:
    return xp.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
                    dtype=complex)


def _swap_matrix(xp: Any) -> Any:
    return xp.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
                    dtype=complex)


def _xx_matrix(xp: Any, theta: float) -> Any:
    c = math.cos(theta)
    s = -1j * math.sin(theta)
    return xp.array([[c, 0, 0, s], [0, c, s, 0], [0, s, c, 0], [s, 0, 0, c]],
                    dtype=complex)


def _apply_1q(xp: Any, psi: Any, matrix: Any, qubit: int) -> Any:
    out = xp.tensordot(matrix, psi, axes=([1], [qubit]))
    return xp.moveaxis(out, 0, qubit)


def _apply_2q(xp: Any, psi: Any, matrix: Any, q0: int, q1: int) -> Any:
    tensor = matrix.reshape(2, 2, 2, 2)
    out = xp.tensordot(tensor, psi, axes=([2, 3], [q0, q1]))
    return xp.moveaxis(out, [0, 1], [q0, q1])


def evolve_with(xp: Any, circuit: Circuit) -> list[complex]:
    """Evolve |0...0> using ``xp`` as the array module.

    Split out from :func:`evolve` precisely so it can be tested: passing numpy
    here runs the identical code path in CI, which is what makes the algorithm
    verified even though the CuPy binding is not.

    Bit convention matches the rest of the library — qubit 0 is the most
    significant bit, so a C-order reshape into ``(2,)*n`` puts qubit ``q`` on
    axis ``q``.
    """
    n = circuit.num_qubits
    psi = xp.zeros((2,) * n, dtype=complex)
    psi[(0,) * n] = 1.0

    for gate in circuit.gates:
        name = gate.name
        if name == "cx":
            psi = _apply_2q(xp, psi, _cx_matrix(xp), gate.qubits[0], gate.qubits[1])
        elif name == "swap":
            psi = _apply_2q(xp, psi, _swap_matrix(xp), gate.qubits[0], gate.qubits[1])
        elif name == "ms":
            theta = float(gate.params[0]) if gate.params else 0.0
            psi = _apply_2q(xp, psi, _xx_matrix(xp, theta),
                            gate.qubits[0], gate.qubits[1])
        else:
            matrix = xp.array(
                gate_matrix(name, bound_params(gate.params)), dtype=complex
            ).reshape(2, 2)
            psi = _apply_1q(xp, psi, matrix, gate.qubits[0])

    flat = psi.reshape(-1)
    # .get() copies device memory back to the host; numpy arrays have no such
    # method, which is exactly the boundary that cannot be tested without a GPU.
    host = flat.get() if hasattr(flat, "get") else flat
    return [complex(a) for a in host]


def evolve(circuit: Circuit) -> list[complex]:
    """Statevector of ``circuit`` on the GPU, as a plain Python list.

    Warns on every call: this path has never run on real hardware.
    """
    cupy = cupy_module()
    if cupy is None:
        raise RuntimeError(
            "GPU kernel requested but CuPy is not installed "
            "(pip install qorch[gpu])"
        )
    warn_unverified()
    return evolve_with(cupy, circuit)
