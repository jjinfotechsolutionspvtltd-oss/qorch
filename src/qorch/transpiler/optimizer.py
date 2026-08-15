"""Circuit optimizer: gate cancellation + rotation merging + identity removal.

Runs multiple passes until the gate count stabilizes.

All passes are *adjacency-based*: two ops combine only when they sit next to each
other in the op list. That makes the optimizer dynamic-circuit safe for free — a
``Measure`` or ``Reset`` between two gates breaks their adjacency, so nothing
cancels across a state-collapsing operation. Conditioned gates combine only with
gates carrying an identical condition, where "both applied" and "neither applied"
are the only two possibilities.
"""

from __future__ import annotations

import math
from dataclasses import replace

from qorch.ir import Circuit, Gate, Operation, Parameter

# Gate pairs that cancel: (g1, g2) on the same qubits → identity
_SELF_INVERSE: frozenset[str] = frozenset({"h", "x", "y", "z", "sx"})

# Rotation gate types (angle-modulable)
_ROTATION_GATES: frozenset[str] = frozenset({"rx", "ry", "rz"})


def _combinable(prev: Operation, op: Operation) -> bool:
    """True if two adjacent ops are unitary gates under the same classical control."""
    return (
        isinstance(prev, Gate)
        and isinstance(op, Gate)
        and prev.condition == op.condition
    )


def _merge_rotations(gates: list[Operation]) -> list[Operation]:
    """Merge adjacent same-type rotations on the same qubit."""
    if not gates:
        return gates
    result: list[Operation] = [gates[0]]
    for g in gates[1:]:
        prev = result[-1]
        if (
            _combinable(prev, g)
            and g.name in _ROTATION_GATES
            and prev.name == g.name
            and prev.qubits == g.qubits
            and not isinstance(prev.params[0], Parameter)
            and not isinstance(g.params[0], Parameter)
        ):
            merged_angle = (float(prev.params[0]) + float(g.params[0])) % (2 * math.pi)
            eps = 1e-12
            if abs(merged_angle) < eps or abs(merged_angle - 2 * math.pi) < eps:
                result.pop()  # cancels to identity
            else:
                result[-1] = Gate(g.name, g.qubits, (merged_angle,), g.condition)
        else:
            result.append(g)
    return result


def _cancel_self_inverse(gates: list[Operation]) -> list[Operation]:
    """Cancel adjacent identical self-inverse gates on same qubits."""
    if not gates:
        return gates
    result: list[Operation] = [gates[0]]
    for g in gates[1:]:
        if not result:
            result.append(g)
            continue
        prev = result[-1]
        if not _combinable(prev, g):
            result.append(g)
            continue
        if (
            prev.name == g.name
            and prev.qubits == g.qubits
            and g.name in _SELF_INVERSE
        ):
            result.pop()
            continue
        if (
            prev.name == g.name
            and set(prev.qubits) == set(g.qubits) and len(g.qubits) == 2
            and g.name in ("cx", "swap")
        ):
            result.pop()
            continue
        result.append(g)
    return result


def _remove_idle_gates(gates: list[Operation]) -> list[Operation]:
    """Remove id gates and zero-angle rotations."""
    eps = 1e-12
    cleaned: list[Operation] = []
    for g in gates:
        if not isinstance(g, Gate):
            cleaned.append(g)
            continue
        # An identity is an identity whether or not it is classically controlled.
        if g.name == "id":
            continue
        if (
            g.name in _ROTATION_GATES
            and g.params
            and not isinstance(g.params[0], Parameter)
            and abs(float(g.params[0]) % (2 * math.pi)) < eps
        ):
            continue
        cleaned.append(g)
    return cleaned


def optimize(circuit: Circuit, max_passes: int = 10) -> Circuit:
    """Optimize a circuit by running cancellation + merging passes.

    Repeats until gate count stabilizes or ``max_passes`` is reached. Dynamic
    circuits are preserved: measurement, reset, and classical conditions survive
    untouched, and no simplification is allowed to cross them.
    """
    gates: list[Operation] = list(circuit.gates)
    for _ in range(max_passes):
        prev_len = len(gates)
        gates = _merge_rotations(gates)
        gates = _cancel_self_inverse(gates)
        gates = _remove_idle_gates(gates)
        if len(gates) == prev_len:
            break
    return replace(circuit, gates=tuple(gates))
