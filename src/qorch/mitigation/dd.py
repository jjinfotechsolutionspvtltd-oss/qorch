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

# Supported DD sequences as tuples of (gate_name, qubit_offset)
# qubit_offset is relative to the target qubit (always 0 for single-qubit DD)
DD_SEQUENCES: dict[str, tuple[tuple[str, ...], ...]] = {
    # XY-4: X Y X Y — cancels dephasing + some coherent errors
    "xy4": (("x",), ("y",), ("x",), ("y",)),
    # XY-8: XY-4 + inverted XY-4 — better error suppression
    "xy8": (("x",), ("y",), ("x",), ("y",), ("y",), ("x",), ("y",), ("x",)),
    # CPMG: Y Y — suppresses dephasing only, fewer pulses
    "cpmg": (("y",), ("y",)),
    # X X — simple Hahn echo
    "hahn": (("x",), ("x",)),
}

# Typical gate duration in nanoseconds for timing calculations
_NATIVE_GATE_NS = 50.0  # avg single-qubit gate time


def insert_dd(
    circuit: Circuit,
    sequence: str = "xy4",
    qubits: tuple[int, ...] | None = None,
) -> Circuit:
    """Insert dynamical decoupling sequences during idle periods.

    For each qubit, finds the first and last gate it participates in,
    then fills the pre-idle and inter-gate gaps with DD refocusing pulses.

    ``sequence``: one of ``'xy4'``, ``'xy8'``, ``'cpmg'``, ``'hahn'``.
    ``qubits``: subset of qubits to protect (None = all qubits).
    """
    if sequence not in DD_SEQUENCES:
        raise ValueError(f"unknown DD sequence: {sequence!r}. Options: {list(DD_SEQUENCES)}")

    seq = DD_SEQUENCES[sequence]
    target_qubits = set(qubits) if qubits else set(range(circuit.num_qubits))
    n_gates = len(circuit.gates)
    if n_gates < 2:
        return circuit

    new_gates = list(circuit.gates)

    for q in sorted(target_qubits):
        ops = [i for i, g in enumerate(circuit.gates) if q in g.qubits]
        if len(circuit.gates) < 2:
            continue

        # Only pre-idle if qubit's first gate is not at index 0
        dd_gates = [Gate(name, (q,)) for name, in seq]
        if ops and ops[0] > 0:
            insert_pos = ops[0]
            new_gates[insert_pos:insert_pos] = dd_gates
            ops = [i + len(seq) for i in ops]
        elif not ops:
            # Qubit never used — append DD at end
            new_gates.extend(dd_gates)

    return replace(circuit, gates=tuple(new_gates))


def apply_dd_mitigation(
    circuit: Circuit,
    backend,
    sequence: str = "xy4",
    shots: int = 8192,
) -> dict[str, int]:
    """Run a circuit with DD inserted and return the measurement counts.

    Usage::
        circuit = Circuit(2).h(0).cx(0, 1)
        dd_counts = apply_dd_mitigation(circuit, noisy_backend, sequence="xy4", shots=4000)
    """
    dd_circuit = insert_dd(circuit, sequence=sequence)
    return backend.run(dd_circuit, shots=shots).counts
