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


def _rx_to_h_rz_h(*qubits: int, params: tuple[float, ...] = ()) -> list[Gate]:
    q = qubits[0]
    theta = params[0] if params else 0.0
    return [Gate("h", (q,)), Gate("rz", (q,), (theta,)), Gate("h", (q,))]


def _ry_to_rz_rx_rz(*qubits: int, params: tuple[float, ...] = ()) -> list[Gate]:
    q = qubits[0]
    theta = params[0] if params else 0.0
    return [
        Gate("rz", (q,), (-math.pi / 2,)),
        Gate("rx", (q,), (theta,)),
        Gate("rz", (q,), (math.pi / 2,)),
    ]


def _x_to_h_zh(*qubits: int, params: tuple[float, ...] = ()) -> list[Gate]:
    q = qubits[0]
    return [Gate("h", (q,)), Gate("z", (q,)), Gate("h", (q,))]


def _y_to_via_t(*qubits: int, params: tuple[float, ...] = ()) -> list[Gate]:
    """y → S X S† where S = T², S† = T⁶."""
    q = qubits[0]
    return [Gate("t", (q,)), Gate("t", (q,)), Gate("x", (q,)),
            Gate("t", (q,)), Gate("t", (q,)), Gate("t", (q,)),
            Gate("t", (q,)), Gate("t", (q,)), Gate("t", (q,))]


def _z_to_tttt(*qubits: int, params: tuple[float, ...] = ()) -> list[Gate]:
    """z → T⁴ — decompose Z into Clifford+T."""
    q = qubits[0]
    return [Gate("t", (q,)), Gate("t", (q,)), Gate("t", (q,)), Gate("t", (q,))]


def _sx_to_h_thth(*qubits: int, params: tuple[float, ...] = ()) -> list[Gate]:
    """sx → H T² H — Rx(π/2) as H Rz(π/2) H."""
    q = qubits[0]
    return [Gate("h", (q,)), Gate("t", (q,)), Gate("t", (q,)), Gate("h", (q,))]


def _rz_to_clifford_t(*qubits: int, params: tuple[float, ...] = ()) -> list[Gate]:
    """Decompose Rz(θ) into H, T gates (Clifford+T).

    For θ = k·π/4 exactly: uses T^k.
    For arbitrary θ: uses a depth-limited BFS over H/T sequences.
    """
    q = qubits[0]
    theta = params[0] if params else 0.0
    target = theta * 4 / math.pi
    k = round(target)

    if abs(target - k) < 1e-12:
        k_mod = k % 8
        if k_mod == 0:
            return []
        gates: list[Gate] = []
        for _ in range(k_mod):
            gates.append(Gate("t", (q,)))
        return gates

    import cmath
    from collections import deque

    def _target_mat(angle: float) -> tuple[complex, ...]:
        return (cmath.exp(-1j * angle / 2), 0j, 0j, cmath.exp(1j * angle / 2))

    def _apply_h(m: tuple[complex, ...]) -> tuple[complex, ...]:
        a, b, c, d = m
        s = 1 / math.sqrt(2)
        return (s * a + s * c, s * b + s * d, s * a - s * c, s * b - s * d)

    def _apply_t(m: tuple[complex, ...]) -> tuple[complex, ...]:
        a, b, c, d = m
        e = cmath.exp(1j * math.pi / 8)
        return (a * 1 / e, b * 1 / e, c * e, d * e)

    target_mat = _target_mat(theta)
    best_seq: list[Gate] = []
    best_fid = -1.0

    qq: deque[tuple[tuple[complex, ...], list[Gate]]] = deque()
    qq.append(((1j, 0j, 0j, 1j), []))  # | to make complex

    while qq:
        mat, seq = qq.popleft()
        fid = 0.5 * abs(
            mat[0].conjugate() * target_mat[0] +
            mat[1].conjugate() * target_mat[1] +
            mat[2].conjugate() * target_mat[2] +
            mat[3].conjugate() * target_mat[3]
        )
        if fid > best_fid:
            best_fid = fid
            best_seq = seq

        if fid > 0.99999 or len(seq) >= 8:
            continue

        h_seq = seq + [Gate("h", (q,))]
        qq.append((_apply_h(mat), h_seq))
        t_seq = seq + [Gate("t", (q,))]
        qq.append((_apply_t(mat), t_seq))

    return best_seq if best_seq else [Gate("t", (q,))]


# ── Clifford+T decomposition rule helpers ─────────────────────────────────


def _count_t(gates: tuple[Gate, ...]) -> int:
    """Count T gates in a gate sequence."""
    return sum(1 for g in gates if g.name == "t")


def _t_depth(gates: tuple[Gate, ...], num_qubits: int) -> int:
    """Compute T-depth (minimum number of T-gate layers)."""
    t_layers: list[set[int]] = []
    for g in gates:
        if g.name == "t":
            qubit_set = set(g.qubits)
            placed = False
            for layer in t_layers:
                if not layer.intersection(qubit_set):
                    layer.update(qubit_set)
                    placed = True
                    break
            if not placed:
                t_layers.append(qubit_set)
    return len(t_layers)


def decompose_to_clifford_t(circuit: Circuit) -> tuple[Circuit, int, int]:
    """Decompose a circuit into Clifford+T and return (circuit, t_count, t_depth).

    Uses the Clifford+T gate set ({h, cx, t}) as target.
    """
    from qorch.transpiler.gateset import CLIFFORD_T
    result = decompose(circuit, CLIFFORD_T)
    tc = _count_t(result.gates)
    td = _t_depth(result.gates, result.num_qubits)
    return result, tc, td


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
    # ry → rz rx rz (general, for native sets with rz+rx)
    ("ry", frozenset({"rz", "rx"})): _ry_to_rz_rx_rz,
    ("ry", frozenset({"rz", "rx", "cx"})): _ry_to_rz_rx_rz,
    ("ry", frozenset({"rz", "rx", "ms"})): _ry_to_rz_rx_rz,
    ("ry", frozenset({"rz", "rx", "x"})): _ry_to_rz_rx_rz,
    ("ry", frozenset({"rz", "rx", "cx", "x"})): _ry_to_rz_rx_rz,
    # rx → h rz h (general)
    ("rx", frozenset({"rz", "h"})): _rx_to_h_rz_h,
    ("rx", frozenset({"rz", "h", "cx"})): _rx_to_h_rz_h,
    # ── Clifford+T rules (target: h, cx, t) ────────────────────────────────
    ("h", frozenset({"h", "cx", "t"})): None,
    ("cx", frozenset({"h", "cx", "t"})): None,
    ("t", frozenset({"h", "cx", "t"})): None,
    ("rz", frozenset({"h", "cx", "t"})): _rz_to_clifford_t,
    ("rx", frozenset({"h", "cx", "t"})): _rx_to_h_rz_h,
    ("ry", frozenset({"h", "cx", "t"})): _ry_to_rz_rx_rz,
    ("x", frozenset({"h", "cx", "t"})): _x_to_h_zh,
    ("y", frozenset({"h", "cx", "t"})): _y_to_via_t,
    ("z", frozenset({"h", "cx", "t"})): _z_to_tttt,
    ("sx", frozenset({"h", "cx", "t"})): _sx_to_h_thth,
    ("swap", frozenset({"h", "cx", "t"})): _swap_to_cx(),
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


def decompose(circuit: Circuit, target: IndianGateSet, _depth: int = 0) -> Circuit:
    """Decompose ``circuit`` so every gate is in ``target.basis_gates``.

    Recursively decomposes until all gates are native or max depth is reached.
    Returns a new ``Circuit`` whose gates are all native to the target.
    """
    if _depth > 20:
        raise ValueError(f"decomposition did not converge for target {target.basis_gates}")
    native = frozenset(target.basis_gates)
    new_gates: list[Gate] = []
    any_non_native = False
    for g in circuit.gates:
        if g.name in native:
            new_gates.append(g)
            continue
        any_non_native = True
        rule = _find_rule(g.name, native)
        if rule is None:
            raise ValueError(
                f"no decomposition rule for {g.name!r} -> {target.basis_gates}"
            )
        decomposed = rule(*g.qubits, params=g.params)
        new_gates.extend(decomposed)
    if not any_non_native:
        return replace(circuit, gates=tuple(new_gates))
    return decompose(replace(circuit, gates=tuple(new_gates)), target, _depth=_depth + 1)
