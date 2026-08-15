"""Optional numpy-vectorized statevector kernel.

The pure-Python kernel loops over all 2^n amplitudes in Python for every gate,
which is fine at 8 qubits and painful at 16. numpy applies a gate as a tensor
contraction over one or two axes, moving the loop into compiled code.

This is strictly an *optional accelerator*. numpy is not a dependency of the
core (``dependencies = []`` is an enforced invariant), so everything here is
reached only when numpy happens to be importable, and
:class:`~qorch.backends.simulator.LocalSimulator` keeps the pure-Python kernel
as a fallback that is always available and always correct.

Bit convention, matching the rest of the library: qubit 0 is the *most*
significant bit, so a bitstring reads leftmost = qubit 0. A C-order reshape of
the 2^n amplitude vector into ``(2,) * n`` puts qubit ``q`` on axis ``q``, which
is why the contraction below can index axes by qubit number directly.
"""

from __future__ import annotations

import math
from typing import Any

from qorch.gates import gate_matrix
from qorch.ir import Circuit, bound_params

_NUMPY: Any | None = None
_CHECKED = False


def numpy_module() -> Any | None:
    """Return numpy if it is importable, else None. Imported once, lazily."""
    global _NUMPY, _CHECKED
    if not _CHECKED:
        _CHECKED = True
        try:
            import numpy
        except ImportError:                      # pragma: no cover - env dependent
            _NUMPY = None
        else:
            _NUMPY = numpy
    return _NUMPY


def is_available() -> bool:
    return numpy_module() is not None


def _cx_matrix(np: Any) -> Any:
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ], dtype=complex)


def _swap_matrix(np: Any) -> Any:
    return np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ], dtype=complex)


def _xx_matrix(np: Any, theta: float) -> Any:
    c = math.cos(theta)
    s = -1j * math.sin(theta)
    return np.array([
        [c, 0, 0, s],
        [0, c, s, 0],
        [0, s, c, 0],
        [s, 0, 0, c],
    ], dtype=complex)


def _apply_1q(np: Any, psi: Any, matrix: Any, qubit: int) -> Any:
    """Contract a 2x2 gate with axis ``qubit``, then restore the axis order."""
    out = np.tensordot(matrix, psi, axes=([1], [qubit]))
    return np.moveaxis(out, 0, qubit)


def _apply_2q(np: Any, psi: Any, matrix: Any, q0: int, q1: int) -> Any:
    """Contract a 4x4 gate with axes (q0, q1).

    The matrix is in basis order 00, 01, 10, 11 with ``q0`` the high bit, so
    reshaping it to (2, 2, 2, 2) gives output axes (q0, q1) followed by input
    axes (q0, q1) — matching the axis pair fed to ``tensordot``.
    """
    tensor = matrix.reshape(2, 2, 2, 2)
    out = np.tensordot(tensor, psi, axes=([2, 3], [q0, q1]))
    return np.moveaxis(out, [0, 1], [q0, q1])


def evolve(circuit: Circuit) -> list[complex]:
    """Statevector of ``circuit`` starting from |0...0>, as a plain Python list.

    Returns a list rather than an ndarray so callers — sampling, tomography,
    anything downstream — are identical whichever kernel produced the state.
    """
    np = numpy_module()
    if np is None:                               # pragma: no cover - env dependent
        raise RuntimeError("numpy kernel requested but numpy is not installed")

    n = circuit.num_qubits
    psi = np.zeros((2,) * n, dtype=complex)
    psi[(0,) * n] = 1.0

    for gate in circuit.gates:
        name = gate.name
        if name == "cx":
            psi = _apply_2q(np, psi, _cx_matrix(np), gate.qubits[0], gate.qubits[1])
        elif name == "swap":
            psi = _apply_2q(np, psi, _swap_matrix(np), gate.qubits[0], gate.qubits[1])
        elif name == "ms":
            theta = float(gate.params[0]) if gate.params else 0.0
            psi = _apply_2q(np, psi, _xx_matrix(np, theta),
                            gate.qubits[0], gate.qubits[1])
        else:
            matrix = np.array(
                gate_matrix(name, bound_params(gate.params)), dtype=complex
            ).reshape(2, 2)
            psi = _apply_1q(np, psi, matrix, gate.qubits[0])

    return [complex(a) for a in psi.reshape(-1)]
