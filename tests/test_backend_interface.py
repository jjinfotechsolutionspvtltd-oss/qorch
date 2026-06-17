"""HAL + IR + scheduler slice: the spine runs end-to-end on the local simulator."""

from __future__ import annotations

import math

import pytest

from qorch import Circuit, LocalSimulator, Scheduler, from_qasm3


def test_bell_state_is_correlated():
    """H on q0 then CX(0,1) yields the Bell state: only '00' and '11' appear."""
    bell = Circuit(num_qubits=2).h(0).cx(0, 1)
    result = LocalSimulator(seed=42).run(bell, shots=4000)

    assert set(result.counts) <= {"00", "11"}, "Bell state must be perfectly correlated"
    # both outcomes roughly equal (~50/50) within statistical tolerance
    p00 = result.counts.get("00", 0) / result.shots
    assert math.isclose(p00, 0.5, abs_tol=0.05)


def test_x_gate_flips_qubit():
    counts = LocalSimulator(seed=1).run(Circuit(num_qubits=1).x(0), shots=500).counts
    assert counts == {"1": 500}


def test_bit_ordering_leftmost_is_qubit_zero():
    """X on qubit 0 only -> '10' (leftmost char = qubit 0), per architecture §5."""
    counts = LocalSimulator(seed=1).run(Circuit(num_qubits=2).x(0), shots=100).counts
    assert counts == {"10": 100}


def test_qasm3_subset_round_trips_to_bell():
    qasm = """
    OPENQASM 3.0;
    qubit[2] q;
    h q[0];
    cx q[0], q[1];
    measure q[0];
    measure q[1];
    """
    circuit = from_qasm3(qasm)
    assert circuit.num_qubits == 2
    assert [g.name for g in circuit.gates] == ["h", "cx"]
    assert set(LocalSimulator(seed=7).run(circuit, shots=200).counts) <= {"00", "11"}


def test_circuit_rejects_out_of_range_qubit():
    with pytest.raises(ValueError):
        Circuit(num_qubits=1).x(3)


def test_scheduler_routes_to_capable_backend():
    sched = Scheduler(backends=[LocalSimulator(seed=3)])
    sched.submit(Circuit(num_qubits=1).x(0), shots=100)
    sched.submit(Circuit(num_qubits=2).h(0).cx(0, 1), shots=100)
    results = sched.run_all()
    assert len(results) == 2
    assert all(r.backend_name == "local-simulator" for r in results)
