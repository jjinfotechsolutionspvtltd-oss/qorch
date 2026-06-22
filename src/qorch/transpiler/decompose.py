"""Gate decomposition: lower any circuit to a target native gate set."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Callable

from qorch.ir import Circuit, Gate, SUPPORTED_GATES
from qorch.transpiler.gateset import IndianGateSet

# A decomposition rule takes (qubits, params) and returns a list of native Gates.
DecompRule = Callable[..., list[Gate]]

_DECOMP_CACHE: dict[tuple[str, frozenset[str]], DecompRule | None] = {}


def _h_to_sx_rz(sx_supported: bool) -> DecompRule:
    """h → rz(π/2) sx rz(π/2)"""
    def rule(*qubits: int, params: tuple[float, ...] = ()) -> list[Gate]:
        q = qubits[0]
        return [
            Gate("rz", (q,), (math.pi / 2,)),
            Gate("sx", (q,)),
            Gate("rz", (q,), (math.pi / 2,)),
        ]
    return rule


def _h_to_rx_ry(ms_supported: bool) -> DecompRule:
    """h → ry(π/4) rx(π) ry(-π/4) — variant for ion-trap native sets."""
    def rule(*qubits: int, params: tuple[float, ...] = ()) -> list[Gate]:
        q = qubits[0]
        return [
            Gate("ry", (q,), (math.pi / 4,)),
            Gate("rx", (q,), (math.pi,)),
            Gate("ry", (q,), (-math.pi / 4,)),
        ]
    return rule


def _x_to_sx3() -> DecompRule:
    """x → sx sx sx (when sx is native but x is not)."""
    def rule(*qubits: int, params: tuple[float, ...] = ()) -> list[Gate]:
        q = qubits[0]
        return [Gate("sx", (q,)), Gate("sx", (q,)), Gate("sx", (q,))]
    return rule


def _z_to_rz() -> DecompRule:
    """z → rz(π)"""
    def rule(*qubits: int, params: tuple[float, ...] = ()) -> list[Gate]:
        return [Gate("rz", (qubits[0],), (math.pi,))]
    return rule


def _cx_to_ms_rx() -> DecompRule:
    """cx → rx(π/2) @ ms(π/4) @ rx(-π/2) for ion-trap (MS-based).

    For MS gate: XX(θ) = exp(-i θ X⊗X).
    CX = (I⊗R_x(π/2)) · XX(π/4) · (I⊗R_x(-π/2))
    """
    def rule(*qubits: int, params: tuple[float, ...] = ()) -> list[Gate]:
        c, t = qubits[0], qubits[1]
        return [
            Gate("rx", (t,), (math.pi / 2,)),
            Gate("ms", (c, t), (math.pi / 4,)),
            Gate("rx", (t,), (-math.pi / 2,)),
        ]
    return rule


def _swap_to_cx() -> DecompRule:
    """swap → cx(0,1) cx(1,0) cx(0,1)"""
    def rule(*qubits: int, params: tuple[float, ...] = ()) -> list[Gate]:
        q0, q1 = qubits[0], qubits[1]
        return [
            Gate("cx", (q0, q1)),
            Gate("cx", (q1, q0)),
            Gate("cx", (q0, q1)),
        ]
    return rule


def _swap_to_ms() -> DecompRule:
    """swap → ms(π/4) rz(π/2) ms(π/4) for ion-trap."""
    def rule(*qubits: int, params: tuple[float, ...] = ()) -> list[Gate]:
        q0, q1 = qubits[0], qubits[1]
        return [
            Gate("ms", (q0, q1), (math.pi / 4,)),
            Gate("rz", (q0,), (math.pi / 2,)),
            Gate("rz", (q1,), (math.pi / 2,)),
            Gate("ms", (q0, q1), (math.pi / 4,)),
        ]
    return rule


# Registry of decomposition rules keyed by (gate_name, frozenset of native gates).
# None = gate is already native (passthrough).
DECOMPOSITION_RULES: dict[tuple[str, frozenset[str]], DecompRule | None] = {
    ("h", frozenset({"h"})): None,
    ("x", frozenset({"x"})): None,
    ("z", frozenset({"z"})): None,
    ("cx", frozenset({"cx"})): None,
    ("sx", frozenset({"sx"})): None,
    ("rz", frozenset({"rz"})): None,
    ("rx", frozenset({"rx"})): None,
    ("ry", frozenset({"ry"})): None,
    ("swap", frozenset({"swap"})): None,
    ("ms", frozenset({"ms"})): None,
    # h → rz sx rz
    ("h", frozenset({"rz", "sx"})): _h_to_sx_rz(True),
    ("h", frozenset({"rz", "sx", "cx"})): _h_to_sx_rz(True),
    # h → ry rx ry (ion-trap)
    ("h", frozenset({"ry", "rx"})): _h_to_rx_ry(True),
    ("h", frozenset({"ry", "rx", "ms"})): _h_to_rx_ry(True),
    # h → rx rz combination
    ("h", frozenset({"rx", "rz"})): _h_to_rx_ry(True),
    ("h", frozenset({"rx", "rz", "cx"})): _h_to_rx_ry(True),
    # x → sx sx sx
    ("x", frozenset({"sx"})): _x_to_sx3(),
    ("x", frozenset({"sx", "cx"})): _x_to_sx3(),
    ("x", frozenset({"sx", "rz"})): _x_to_sx3(),
    ("x", frozenset({"sx", "rz", "cx"})): _x_to_sx3(),
    # z → rz(π)
    ("z", frozenset({"rz"})): _z_to_rz(),
    ("z", frozenset({"rz", "cx"})): _z_to_rz(),
    ("z", frozenset({"rz", "sx"})): _z_to_rz(),
    ("z", frozenset({"rz", "sx", "cx"})): _z_to_rz(),
    # cx → rx ms rx (ion-trap)
    ("cx", frozenset({"rx", "ms"})): _cx_to_ms_rx(),
    ("cx", frozenset({"rx", "ry", "ms"})): _cx_to_ms_rx(),
    # swap → cx cx cx
    ("swap", frozenset({"cx"})): _swap_to_cx(),
    ("swap", frozenset({"cx", "sx", "rz"})): _swap_to_cx(),
    # swap → ms rz ms (ion-trap)
    ("swap", frozenset({"ms", "rz"})): _swap_to_ms(),
    ("swap", frozenset({"rx", "ms", "rz"})): _swap_to_ms(),
}


def _find_rule(gate_name: str, native_set: frozenset[str]) -> DecompRule | None:
    """Find best decomposition rule for a gate given available natives."""
    # Direct match
    key = (gate_name, native_set)
    if key in DECOMPOSITION_RULES:
        return DECOMPOSITION_RULES[key]
    # Check if gate is itself native
    if gate_name in native_set:
        return None
    # Check superset matches (native_set contains all required gates of a rule)
    for (g, req), rule in DECOMPOSITION_RULES.items():
        if g == gate_name and req <= native_set:
            return rule
    return None


def _passthrough(*qubits: int, params: tuple[float, ...] = ()) -> list[Gate]:
    return [Gate("ERROR", qubits, params)]


def _can_decompose(circuit: Circuit, target: IndianGateSet) -> bool:
    """Check if every gate in the circuit can be decomposed to the target."""
    native = frozenset(target.basis_gates)
    for g in circuit.gates:
        if g.name not in SUPPORTED_GATES:
            return False
        rule = _find_rule(g.name, native)
        if rule is None and g.name not in native:
            return False
    return True


def decompose(circuit: Circuit, target: IndianGateSet) -> Circuit:
    """Decompose ``circuit`` so every gate is in ``target.basis_gates``.

    Returns a new ``Circuit`` whose gates are all native to the target.
    """
    native = frozenset(target.basis_gates)
    new_gates: list[Gate] = []
    for g in circuit.gates:
        if g.name in native:
            new_gates.append(g)
            continue
        rule = _find_rule(g.name, native)
        if rule is None:
            raise ValueError(
                f"no decomposition rule for {g.name!r} -> {target.basis_gates}"
            )
        decomposed = rule(*g.qubits, params=g.params)
        new_gates.extend(decomposed)
    return replace(circuit, gates=tuple(new_gates))
