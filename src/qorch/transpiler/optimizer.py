"""Circuit optimizer: gate cancellation + rotation merging + identity removal.

Runs multiple passes until the gate count stabilizes.
"""

from __future__ import annotations

import math
from dataclasses import replace

from qorch.ir import Circuit, Gate

# Gate pairs that cancel: (g1, g2) on the same qubits → identity
_SELF_INVERSE: frozenset[str] = frozenset({"h", "x", "y", "z", "sx"})

# Rotation gate types (angle-modulable)
_ROTATION_GATES: frozenset[str] = frozenset({"rx", "ry", "rz"})


def _merge_rotations(gates: list[Gate]) -> list[Gate]:
    """Merge adjacent same-type rotations on the same qubit."""
    if not gates:
        return gates
    result: list[Gate] = [gates[0]]
    for g in gates[1:]:
        prev = result[-1]
        if (
            g.name in _ROTATION_GATES
            and prev.name == g.name
            and prev.qubits == g.qubits
        ):
            merged_angle = (prev.params[0] + g.params[0]) % (2 * math.pi)
            eps = 1e-12
            if abs(merged_angle) < eps or abs(merged_angle - 2 * math.pi) < eps:
                result.pop()  # cancels to identity
            else:
                result[-1] = Gate(g.name, g.qubits, (merged_angle,))
        else:
            result.append(g)
    return result


def _cancel_self_inverse(gates: list[Gate]) -> list[Gate]:
    """Cancel adjacent identical self-inverse gates on same qubits."""
    if not gates:
        return gates
    result: list[Gate] = [gates[0]]
    for g in gates[1:]:
        if not result:
            result.append(g)
            continue
        prev = result[-1]
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


def _remove_idle_gates(gates: list[Gate]) -> list[Gate]:
    """Remove id gates and zero-angle rotations."""
    eps = 1e-12
    cleaned: list[Gate] = []
    for g in gates:
        if g.name == "id":
            continue
        if g.name in _ROTATION_GATES and g.params and abs(g.params[0] % (2 * math.pi)) < eps:
            continue
        cleaned.append(g)
    return cleaned


def optimize(circuit: Circuit, max_passes: int = 10) -> Circuit:
    """Optimize a circuit by running cancellation + merging passes.

    Repeats until gate count stabilizes or ``max_passes`` is reached.
    """
    gates = list(circuit.gates)
    for _ in range(max_passes):
        prev_len = len(gates)
        gates = _merge_rotations(gates)
        gates = _cancel_self_inverse(gates)
        gates = _remove_idle_gates(gates)
        if len(gates) == prev_len:
            break
    return replace(circuit, gates=tuple(gates))
