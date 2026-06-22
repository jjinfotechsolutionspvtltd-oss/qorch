"""Tests that the simulator handles all gate types correctly."""

import math

from qorch import Circuit, LocalSimulator


def test_rx_pi_flips_qubit():
    """rx(π) should be equivalent to x."""
    sim = LocalSimulator(seed=1)
    rx = sim.run(Circuit(num_qubits=1).rx(0, math.pi), shots=500)
    x = sim.run(Circuit(num_qubits=1).x(0), shots=500)
    assert rx.counts.get("1", 0) > 480
    assert x.counts.get("1", 0) > 480


def test_ry_pi_flips_qubit():
    sim = LocalSimulator(seed=1)
    ry = sim.run(Circuit(num_qubits=1).ry(0, math.pi), shots=500)
    assert ry.counts.get("1", 0) > 480


def test_rz_pi_adds_phase():
    """rz(π) should be equivalent to z (up to global phase)."""
    sim = LocalSimulator(seed=1)
    circuit = Circuit(num_qubits=1).h(0).rz(0, math.pi).h(0)
    result = sim.run(circuit, shots=500)
    assert result.counts.get("1", 0) > 480  # H rz(π) H = X


def test_sx_squared_is_x():
    """sx sx = x (up to global phase)."""
    sim = LocalSimulator(seed=1)
    sx2 = sim.run(Circuit(num_qubits=1).sx(0).sx(0), shots=500)
    x = sim.run(Circuit(num_qubits=1).x(0), shots=500)
    assert sx2.counts.get("1", 0) > 480
    assert x.counts.get("1", 0) > 480


def test_z_noeffect_on_0():
    """z on |0> should leave state unchanged."""
    sim = LocalSimulator(seed=1)
    z = sim.run(Circuit(num_qubits=1).z(0), shots=500)
    assert z.counts == {"0": 500}


def test_id_passthrough():
    """id gate should not change the measurement distribution."""
    sim = LocalSimulator(seed=42)
    base = sim.run(Circuit(num_qubits=1).h(0), shots=10000)
    with_id = sim.run(Circuit(num_qubits=1).h(0).id(0), shots=10000)
    for key in ("0", "1"):
        assert abs(base.counts.get(key, 0) - with_id.counts.get(key, 0)) < 200


def test_param_gates_on_trajectory():
    """Parametric gates work under noisy trajectory simulation."""
    noisy = LocalSimulator(seed=1, gate_noise=None)
    _ = type(noisy._gate_noise).__new__(type(noisy._gate_noise))
    noisy._gate_noise = type(noisy._gate_noise)(depolarizing_prob=0.01)
    result = noisy.run(Circuit(num_qubits=1).rx(0, math.pi), shots=500)
    assert result.counts.get("1", 0) > 400


def test_h_on_transpiled_circuit():
    """Transpiled circuit (rz sx rz) should run on simulator."""
    from qorch.transpiler import decompose
    from qorch.transpiler.gateset import TIFR_SUPERCONDUCTING
    circuit = Circuit(num_qubits=1).h(0)
    decomposed = decompose(circuit, TIFR_SUPERCONDUCTING)
    sim = LocalSimulator(seed=1)
    result = sim.run(decomposed, shots=500)
    p0 = result.counts.get("0", 0) / 500
    assert abs(p0 - 0.5) < 0.1
