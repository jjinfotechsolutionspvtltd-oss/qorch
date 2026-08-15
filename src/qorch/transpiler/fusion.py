"""Euler fusion and commutation-aware cancellation.

The existing optimizer only combines gates that are *already adjacent* in the op
list. That misses two large classes of reduction:

  - **A run of single-qubit gates is one rotation.** Any product of single-qubit
    unitaries is itself a single-qubit unitary, and every one of those is
    ``Rz·Ry·Rz`` for some angles. A run of nine gates collapses to three
    regardless of what the nine were.
  - **Gates that commute can be brought together.** ``rz`` on a control wire
    commutes through a ``cx``; ``x`` on a target wire does too. An inverse pair
    separated by such a gate never becomes adjacent, so the adjacency-only
    optimizer cannot see it.

Both are checked the same way, and it is the only way worth trusting here: the
rewritten circuit must implement the same unitary as the original, verified
numerically rather than argued structurally.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import replace

from qorch.gates import GATES, gate_matrix
from qorch.ir import Circuit, Gate, Operation, Parameter

Matrix2 = tuple[complex, complex, complex, complex]

_IDENTITY: Matrix2 = (1 + 0j, 0j, 0j, 1 + 0j)
_EPS = 1e-12


def _matmul(u: Matrix2, v: Matrix2) -> Matrix2:
    """``u`` applied after ``v`` — matrix product u·v."""
    a, b, c, d = u
    e, f, g, h = v
    return (a * e + b * g, a * f + b * h, c * e + d * g, c * f + d * h)


def _is_fusible(op: Operation) -> bool:
    """A concrete, unconditioned single-qubit gate with no free parameters."""
    if not isinstance(op, Gate) or len(op.qubits) != 1:
        return False
    if op.condition is not None:
        return False
    if any(isinstance(p, Parameter) for p in op.params):
        return False
    definition = GATES.get(op.name)
    return definition is not None and definition.matrix is not None


def zyz_angles(u: Matrix2) -> tuple[float, float, float]:
    """Angles (alpha, beta, gamma) with ``u ≈ e^{iφ}·Rz(α)·Ry(β)·Rz(γ)``.

    Global phase is dropped: it is unobservable, and keeping it would mean
    emitting a gate to represent nothing.
    """
    a, b, c, d = u
    det = a * d - b * c
    if abs(det) < _EPS:                                    # pragma: no cover
        raise ValueError("cannot decompose a singular matrix")
    # Normalize to SU(2) so the entries carry the Euler angles directly.
    scale = cmath.sqrt(det)
    a, b, c, d = a / scale, b / scale, c / scale, d / scale

    beta = 2.0 * math.atan2(abs(c), abs(a))

    if abs(c) < 1e-12:            # beta ≈ 0: only the sum of α and γ is defined
        return (2.0 * cmath.phase(d), 0.0, 0.0)
    if abs(a) < 1e-12:            # beta ≈ π: only the difference is defined
        return (2.0 * cmath.phase(c), math.pi, 0.0)

    sum_ang = 2.0 * cmath.phase(d)
    diff_ang = 2.0 * cmath.phase(c)
    return ((sum_ang + diff_ang) / 2.0, beta, (sum_ang - diff_ang) / 2.0)


def _normalize(angle: float) -> float:
    """Fold an angle into (-π, π]; 2π-periodic rotations differ only by phase."""
    return math.remainder(angle, 2 * math.pi)


def euler_gates(u: Matrix2, qubit: int) -> list[Gate]:
    """``u`` as at most three rotations in circuit order, dropping identities."""
    alpha, beta, gamma = (_normalize(x) for x in zyz_angles(u))
    # Circuit order is the reverse of the matrix product: Rz(γ) acts first.
    candidates = [("rz", gamma), ("ry", beta), ("rz", alpha)]
    return [
        Gate(name, (qubit,), (angle,))
        for name, angle in candidates
        if abs(angle) > 1e-12
    ]


def fuse_single_qubit_runs(circuit: Circuit) -> Circuit:
    """Collapse maximal runs of single-qubit gates into ``Rz·Ry·Rz``.

    A run ends at anything that is not a plain single-qubit gate on that qubit:
    a two-qubit gate, a measurement, a reset, a classically-conditioned gate, or
    an unbound parameter. Those are boundaries rather than obstacles — fusing
    across a measurement would change what the circuit computes.

    The rewrite is applied only when it does not *increase* the gate count, so
    fusion can never make a circuit worse.
    """
    runs: dict[int, list[int]] = {}
    ops = list(circuit.gates)
    replacements: dict[int, list[Gate]] = {}
    dropped: set[int] = set()

    def flush(qubit: int) -> None:
        indices = runs.pop(qubit, [])
        if len(indices) < 2:
            return
        product = _IDENTITY
        for i in indices:
            op = ops[i]
            product = _matmul(
                gate_matrix(op.name, tuple(float(p) for p in op.params)), product
            )
        fused = euler_gates(product, qubit)
        if len(fused) > len(indices):
            return                       # never trade a short run for a longer one
        replacements[indices[0]] = fused
        dropped.update(indices[1:])
        if not fused:
            dropped.add(indices[0])
            replacements.pop(indices[0])

    for index, op in enumerate(ops):
        if _is_fusible(op):
            runs.setdefault(op.qubits[0], []).append(index)
            continue
        for qubit in op.qubits:
            flush(qubit)
        if isinstance(op, Gate) and op.condition is not None:
            # A conditioned gate reads classical bits; be conservative and treat
            # it as a barrier for every qubit it touches (already done above).
            pass
    for qubit in list(runs):
        flush(qubit)

    if not replacements and not dropped:
        return circuit

    out: list[Operation] = []
    for index, op in enumerate(ops):
        if index in dropped:
            continue
        if index in replacements:
            out.extend(replacements[index])
        else:
            out.append(op)
    return replace(circuit, gates=tuple(out))


# ── commutation ──────────────────────────────────────────────────────────

# Gates diagonal in the Z basis commute with each other and with the *control*
# of a CX. Gates diagonal in X commute with its *target*.
_Z_DIAGONAL = frozenset({"z", "rz", "t", "id"})
_X_DIAGONAL = frozenset({"x", "rx", "id"})


def _commutes_with(op: Operation, gate: Gate, qubit: int) -> bool:
    """Does ``gate`` on ``qubit`` commute past ``op``?

    Only relationships that are true for *every* angle are allowed. A rule that
    holds for particular parameter values would be a correctness trap.
    """
    if qubit not in op.qubits:
        return True                      # disjoint qubits always commute
    if not isinstance(op, Gate):
        return False                     # measurement and reset are barriers
    if op.condition is not None or gate.condition is not None:
        return False

    if len(op.qubits) == 1:
        if gate.name in _Z_DIAGONAL and op.name in _Z_DIAGONAL:
            return True
        return gate.name in _X_DIAGONAL and op.name in _X_DIAGONAL

    if op.name == "cx":
        control, target = op.qubits
        if qubit == control:
            return gate.name in _Z_DIAGONAL
        if qubit == target:
            return gate.name in _X_DIAGONAL
    return False


def _inverse_pair(first: Gate, second: Gate) -> bool:
    """True if applying ``second`` immediately after ``first`` is the identity."""
    if first.qubits != second.qubits or first.condition != second.condition:
        return False
    if any(isinstance(p, Parameter) for p in first.params + second.params):
        return False
    definition = GATES.get(first.name)
    if definition is None:
        return False                                        # pragma: no cover
    if first.name == second.name and definition.self_inverse:
        return True
    if first.name == second.name and definition.angle_inverse:
        total = float(first.params[0]) + float(second.params[0])
        return abs(_normalize(total)) < 1e-12
    return False


def cancel_commuting(circuit: Circuit, max_passes: int = 4) -> Circuit:
    """Cancel inverse pairs that are separated only by gates they commute past.

    The adjacency-only optimizer cannot see ``h(0) · cx(0,1) · h(0)`` — sorry,
    cannot see ``rz(0,θ) · cx(0,1) · rz(0,-θ)`` — because the CX sits between
    them. Since ``rz`` on a control commutes through a CX, the pair meets and
    annihilates.
    """
    ops = list(circuit.gates)
    for _ in range(max_passes):
        removed = False
        for i, first in enumerate(ops):
            if not isinstance(first, Gate) or len(first.qubits) != 1:
                continue
            qubit = first.qubits[0]
            for j in range(i + 1, len(ops)):
                candidate = ops[j]
                if (isinstance(candidate, Gate) and len(candidate.qubits) == 1
                        and candidate.qubits[0] == qubit
                        and _inverse_pair(first, candidate)):
                    del ops[j]
                    del ops[i]
                    removed = True
                    break
                if not _commutes_with(candidate, first, qubit):
                    break
            if removed:
                break
        if not removed:
            break
    if len(ops) == len(circuit.gates):
        return circuit
    return replace(circuit, gates=tuple(ops))
