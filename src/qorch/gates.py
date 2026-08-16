"""The gate registry: one place that knows what each gate *is*.

Gate facts used to live in at least five places — the simulator's matrix table,
``ir.SUPPORTED_GATES``, ``ir._SELF_INVERSE_GATES`` and ``_ANGLE_GATES``, the
optimizer's ``_SELF_INVERSE`` and ``_ROTATION_GATES``, and the decomposition
rules. Nothing kept them consistent, and they drifted: the optimizer listed
``sx`` as self-inverse while the IR did not. Since ``SX·SX = X``, not identity,
the optimizer cancelled ``sx sx`` pairs and silently turned a circuit that
outputs 1 into one that outputs 0.

That is the cost of scattered metadata in a library whose failure mode is a
plausible wrong distribution rather than a crash. Everything here is derived
from a single :class:`GateDef` per gate, so a fact can only be stated once.

``duration_ns`` values are advisory defaults modelled on superconducting
hardware — a real device supplies its own through ``DeviceCalibration``. They
exist so a scheduling pass has something principled to order against.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from typing import Callable

# A 2x2 complex matrix, row-major.
Matrix2 = tuple[complex, complex, complex, complex]

_INV_SQRT2 = 1 / math.sqrt(2)


@dataclass(frozen=True)
class GateDef:
    """Everything the stack needs to know about one gate.

    ``self_inverse`` means ``G·G = I`` exactly — the property an optimizer needs
    to cancel an adjacent pair. It is *not* "G is its own inverse up to phase",
    and it is not implied by being a square root of something: ``sx`` is a
    fourth root of the identity, so two of them make X, not I.

    ``angle_inverse`` means the inverse is the same gate with a negated angle.
    """

    name: str
    arity: int
    num_params: int = 0
    is_clifford: bool = True
    self_inverse: bool = False
    angle_inverse: bool = False
    duration_ns: float = 35.0
    matrix: Callable[[tuple[float, ...]], Matrix2] | None = None

    @property
    def is_rotation(self) -> bool:
        """A single-qubit gate parameterized by one continuous angle."""
        return self.arity == 1 and self.num_params == 1


def _const(m: Matrix2) -> Callable[[tuple[float, ...]], Matrix2]:
    def build(_params: tuple[float, ...] = ()) -> Matrix2:
        return m
    return build


def _rx(params: tuple[float, ...] = ()) -> Matrix2:
    theta = params[0] if params else 0.0
    c, s = math.cos(theta / 2), -1j * math.sin(theta / 2)
    return (c, s, s, c)


def _ry(params: tuple[float, ...] = ()) -> Matrix2:
    theta = params[0] if params else 0.0
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return (complex(c), complex(-s), complex(s), complex(c))


def _rz(params: tuple[float, ...] = ()) -> Matrix2:
    theta = params[0] if params else 0.0
    return (cmath.exp(-1j * theta / 2), 0j, 0j, cmath.exp(1j * theta / 2))


GATES: dict[str, GateDef] = {
    "h": GateDef("h", 1, self_inverse=True,
                 matrix=_const((_INV_SQRT2, _INV_SQRT2, _INV_SQRT2, -_INV_SQRT2))),
    "x": GateDef("x", 1, self_inverse=True, matrix=_const((0, 1, 1, 0))),
    "y": GateDef("y", 1, self_inverse=True, matrix=_const((0, -1j, 1j, 0))),
    "z": GateDef("z", 1, self_inverse=True, matrix=_const((1, 0, 0, -1))),
    "id": GateDef("id", 1, self_inverse=True, matrix=_const((1, 0, 0, 1))),
    # SX is a *fourth* root of the identity: SX·SX = X, so it is NOT self-inverse.
    "sx": GateDef("sx", 1, matrix=_const((0.5 + 0.5j, 0.5 - 0.5j,
                                          0.5 - 0.5j, 0.5 + 0.5j))),
    # T is the canonical non-Clifford gate; its cost is the whole point of
    # Clifford+T accounting.
    "t": GateDef("t", 1, is_clifford=False,
                 matrix=_const((1, 0, 0, cmath.exp(1j * math.pi / 4)))),
    # Rotations: generally non-Clifford, inverted by negating the angle.
    # rz is a frame change on real superconducting hardware — free, hence 0 ns.
    "rx": GateDef("rx", 1, num_params=1, is_clifford=False,
                  angle_inverse=True, matrix=_rx),
    "ry": GateDef("ry", 1, num_params=1, is_clifford=False,
                  angle_inverse=True, matrix=_ry),
    "rz": GateDef("rz", 1, num_params=1, is_clifford=False,
                  angle_inverse=True, duration_ns=0.0, matrix=_rz),
    "cx": GateDef("cx", 2, self_inverse=True, duration_ns=300.0),
    "swap": GateDef("swap", 2, self_inverse=True, duration_ns=900.0),
    "ms": GateDef("ms", 2, num_params=1, is_clifford=False,
                  angle_inverse=True, duration_ns=200.0),
}


# ── derived views, so each fact is stated exactly once ───────────────────

SUPPORTED_GATE_NAMES: frozenset[str] = frozenset(GATES)
SELF_INVERSE_GATES: frozenset[str] = frozenset(
    n for n, g in GATES.items() if g.self_inverse
)
ANGLE_INVERSE_GATES: frozenset[str] = frozenset(
    n for n, g in GATES.items() if g.angle_inverse
)
ROTATION_GATES: frozenset[str] = frozenset(
    n for n, g in GATES.items() if g.is_rotation
)
CLIFFORD_GATES: frozenset[str] = frozenset(
    n for n, g in GATES.items() if g.is_clifford
)


def gate_def(name: str) -> GateDef:
    """Look up a gate, with an error naming what is actually available."""
    try:
        return GATES[name]
    except KeyError:
        raise ValueError(
            f"unknown gate {name!r}; supported: {sorted(GATES)}"
        ) from None


def gate_matrix(name: str, params: tuple[float, ...] = ()) -> Matrix2:
    """The 2x2 matrix of a single-qubit gate."""
    definition = gate_def(name)
    if definition.matrix is None:
        raise ValueError(
            f"{name!r} is a {definition.arity}-qubit gate and has no 2x2 matrix"
        )
    return definition.matrix(params)


def gate_duration_ns(name: str) -> float:
    """Advisory duration, for scheduling in the absence of calibration data."""
    return gate_def(name).duration_ns


def xx_matrix(theta: float) -> tuple[complex, ...]:
    """Mølmer–Sørensen entangler XX(θ) = exp(-iθ X⊗X), row-major 4×4.

    Basis order (q0 high bit, q1 low bit): 00, 01, 10, 11.

    Two-qubit matrices live here for the same reason the single-qubit ones do:
    this used to be private to the statevector simulator and imported from there
    by the Indian-QPU backend, which made a simulator internal load-bearing for
    a hardware adapter. Gate matrices belong to the gate registry.
    """
    c = math.cos(theta)
    s = -1j * math.sin(theta)
    return (
        c, 0, 0, s,
        0, c, s, 0,
        0, s, c, 0,
        s, 0, 0, c,
    )
