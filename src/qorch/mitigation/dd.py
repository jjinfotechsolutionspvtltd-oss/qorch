"""Dynamical Decoupling (DD): suppress decoherence by inserting refocusing pulses.

DD works by applying sequences of gates that average out environmental noise.
The simplest (and most widely used) is the XY-4 sequence: X — τ — Y — τ — X — τ — Y.

This module inserts DD sequences into idle periods of a circuit — gaps between
non-adjacent gates on the same qubit. Longer idle periods → more decoherence →
more benefit from DD.
"""

from __future__ import annotations

from dataclasses import replace

from qorch.ir import Circuit, Gate

# Supported DD sequences as tuples of (gate_name,)
DD_SEQUENCES: dict[str, tuple[tuple[str], ...]] = {
    "xy4": (("x",), ("y",), ("x",), ("y",)),
    "xy8": (("x",), ("y",), ("x",), ("y",), ("y",), ("x",), ("y",), ("x",)),
    "cpmg": (("y",), ("y",)),
    "hahn": (("x",), ("x",)),
}


def _find_idle_windows(
    circuit: Circuit,
    qubit: int,
    min_gap: int = 0,
) -> list[tuple[int, int]]:
    """Find idle windows on a qubit: (start_index, end_index) in the gate list.

    An idle window exists when a qubit is not involved in consecutive gates.
    Returns list of (insert_before_gate_index, num_consecutive_idle_slots).
    """
    if not circuit.gates:
        return [(0, 0)]

    ops = [i for i, g in enumerate(circuit.gates) if qubit in g.qubits]
    if len(ops) < 2:
        return []

    windows: list[tuple[int, int]] = []
    for i in range(len(ops) - 1):
        gap = ops[i + 1] - ops[i] - 1
        if gap > min_gap:
            windows.append((ops[i] + 1, gap))
    return windows


def insert_dd(
    circuit: Circuit,
    sequence: str = "xy4",
    qubits: tuple[int, ...] | None = None,
) -> Circuit:
    """Insert dynamical decoupling sequences during idle periods.

    For each target qubit, finds idle windows (gaps between consecutive gates
    where the qubit is not used) and fills them with DD refocusing pulses.

    ``sequence``: one of ``'xy4'``, ``'xy8'``, ``'cpmg'``, ``'hahn'``.
    ``qubits``: subset of qubits to protect (None = all qubits).
    """
    if sequence not in DD_SEQUENCES:
        raise ValueError(f"unknown DD sequence: {sequence!r}. Options: {list(DD_SEQUENCES)}")

    seq: tuple[tuple[str], ...] = DD_SEQUENCES[sequence]
    target_qubits = set(qubits) if qubits else set(range(circuit.num_qubits))

    new_gates = list(circuit.gates)

    # Work from last qubit to first so insertion indices stay valid
    for q in sorted(target_qubits, reverse=True):
        windows = _find_idle_windows(circuit, q)
        if not windows:
            # Pre-idle: insert DD before the first gate
            ops = [i for i, g in enumerate(circuit.gates) if q in g.qubits]
            if ops and ops[0] > 0:
                dd = [Gate(name, (q,)) for name, in seq]
                new_gates[ops[0]:ops[0]] = dd
            elif not ops:
                dd = [Gate(name, (q,)) for name, in seq]
                new_gates.extend(dd)
            continue

        # Insert DD in each idle window (reverse order so indices stay valid)
        for insert_pos, _ in reversed(windows):
            dd = [Gate(name, (q,)) for name, in seq]
            new_gates[insert_pos:insert_pos] = dd

    return replace(circuit, gates=tuple(new_gates))


def apply_dd_mitigation(
    circuit: Circuit,
    backend,
    sequence: str = "xy4",
    shots: int = 8192,
) -> dict[str, int]:
    """Run a circuit with DD inserted and return the measurement counts."""
    dd_circuit = insert_dd(circuit, sequence=sequence)
    return backend.run(dd_circuit, shots=shots).counts
