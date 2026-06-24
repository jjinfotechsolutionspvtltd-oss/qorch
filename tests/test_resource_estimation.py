"""Fault-tolerant resource estimation from Clifford+T cost."""

from __future__ import annotations

from qorch import Circuit
from qorch.resource_estimation import estimate_resources, format_estimate


def test_estimate_basic_fields():
    c = Circuit(3).h(0).cx(0, 1).t(1).rz(2, 0.3)
    est = estimate_resources(c, physical_error_rate=1e-3)
    assert est.algorithm_qubits == 3
    assert est.t_count >= 1
    assert est.code_distance % 2 == 1  # surface-code distances are odd
    assert est.physical_qubits > est.algorithm_qubits
    assert est.runtime_seconds > 0.0


def test_lower_physical_error_needs_smaller_distance():
    c = Circuit(2).h(0).cx(0, 1).rz(0, 0.7).t(1)
    clean = estimate_resources(c, physical_error_rate=1e-4)
    noisy = estimate_resources(c, physical_error_rate=5e-3)
    # closer to threshold ⇒ larger distance ⇒ more physical qubits
    assert noisy.code_distance >= clean.code_distance
    assert noisy.physical_qubits >= clean.physical_qubits


def test_more_t_gates_increase_runtime():
    shallow = estimate_resources(Circuit(1).t(0), physical_error_rate=1e-3)
    deep = estimate_resources(
        Circuit(1).t(0).t(0).t(0).t(0).t(0), physical_error_rate=1e-3
    )
    assert deep.t_depth > shallow.t_depth
    assert deep.runtime_seconds >= shallow.runtime_seconds


def test_format_estimate_renders():
    est = estimate_resources(Circuit(2).h(0).cx(0, 1).t(0), physical_error_rate=1e-3)
    text = format_estimate(est)
    assert "Resource Estimate" in text
    assert "Code distance" in text
