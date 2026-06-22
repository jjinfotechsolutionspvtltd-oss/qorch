"""Tests for circuit analysis and reporting."""

from qorch import Circuit
from qorch.analysis import circuit_report, format_report, _compute_depth


def test_depth_empty():
    assert _compute_depth(Circuit(num_qubits=2)) == 0


def test_depth_single_gate():
    assert _compute_depth(Circuit(num_qubits=1).h(0)) == 1


def test_depth_parallel_gates():
    """h(0) and h(1) can run in parallel → depth 1."""
    c = Circuit(num_qubits=2)
    c = c._add("h", 0)._add("h", 1)
    assert _compute_depth(c) == 1


def test_depth_sequential():
    """h(0) then x(0) → depth 2."""
    c = Circuit(num_qubits=1).h(0).x(0)
    assert _compute_depth(c) == 2


def test_report_basic():
    c = Circuit(num_qubits=2).h(0).cx(0, 1)
    r = circuit_report(c)
    assert r["num_qubits"] == 2
    assert r["num_gates"] == 2
    assert r["depth"] == 2
    assert r["num_2q_gates"] == 1


def test_report_estimated_fidelity():
    """3 gates at 99% fidelity each → 0.99^3 = 0.970299."""
    c = Circuit(num_qubits=1).h(0).x(0).z(0)
    r = circuit_report(c, gate_error_rate=0.01)
    expected = 0.99 ** 3
    assert abs(r["estimated_fidelity"] - expected) < 1e-6


def test_report_gate_counts():
    c = Circuit(num_qubits=2).h(0).cx(0, 1).x(1)
    r = circuit_report(c)
    assert r["gate_counts"]["h"] == 1
    assert r["gate_counts"]["cx"] == 1
    assert r["gate_counts"]["x"] == 1


def test_format_report():
    c = Circuit(num_qubits=2).h(0).cx(0, 1)
    r = circuit_report(c)
    out = format_report(r)
    assert "Qubits" in out
    assert "Depth" in out
    assert "h: 1" in out or "h" in out
