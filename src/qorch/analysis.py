"""Circuit analysis: depth, gate counts, estimated fidelity reporting."""

from __future__ import annotations


from qorch.ir import Circuit


def circuit_report(circuit: Circuit, gate_error_rate: float = 0.01) -> dict:
    """Return a dict of metrics for a circuit.

    Args:
        circuit: The circuit to analyze.
        gate_error_rate: Per-gate error probability (default 1%).
                         Used for estimated fidelity.

    Returns:
        Dict with keys: num_qubits, num_gates, depth, gate_counts,
        num_2q_gates, estimated_fidelity, readout_qubits.
    """
    gate_counts: dict[str, int] = {}
    for g in circuit.gates:
        gate_counts[g.name] = gate_counts.get(g.name, 0) + 1

    num_2q = sum(1 for g in circuit.gates if len(g.qubits) > 1)
    depth = _compute_depth(circuit)
    total_gates = len(circuit.gates)
    fidelity = (1 - gate_error_rate) ** total_gates if total_gates > 0 else 1.0

    return {
        "num_qubits": circuit.num_qubits,
        "num_gates": total_gates,
        "depth": depth,
        "gate_counts": dict(gate_counts),
        "num_2q_gates": num_2q,
        "estimated_fidelity": fidelity,
        "readout_qubits": len(circuit.readout_qubits),
    }


def _compute_depth(circuit: Circuit) -> int:
    """Compute circuit depth (critical path length).

    Simple greedy algorithm: track the last time step each qubit was used.
    """
    if not circuit.gates:
        return 0
    last_used: dict[int, int] = {}
    depth = 0
    for g in circuit.gates:
        qubits = g.qubits
        start = max(last_used.get(q, -1) for q in qubits) + 1
        for q in qubits:
            last_used[q] = start
        depth = max(depth, start + 1)
    return depth


def format_report(report: dict) -> str:
    """Format a circuit report as a human-readable string."""
    lines = [
        "Circuit Report",
        "==============",
        f"  Qubits:            {report['num_qubits']}",
        f"  Gates (total):     {report['num_gates']}",
        f"  Depth:             {report['depth']}",
        f"  2-qubit gates:     {report['num_2q_gates']}",
        f"  Readout qubits:    {report['readout_qubits']}",
        f"  Est. fidelity:     {report['estimated_fidelity']:.4f}",
        "",
        "Gate breakdown:",
    ]
    for name, count in sorted(report["gate_counts"].items()):
        lines.append(f"  {name}: {count}")
    return "\n".join(lines)
