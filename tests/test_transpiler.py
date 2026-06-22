"""Tests for gate decomposition + qubit routing."""

from qorch import Circuit
from qorch.transpiler import (
    decompose, route, CouplingMap,
    IIT_JODHPUR_ION_TRAP, TIFR_SUPERCONDUCTING,
)


def test_decompose_h_to_rz_sx_rz():
    """h → rz(π/2) sx rz(π/2) for a superconducting native set."""
    circuit = Circuit(num_qubits=1).h(0)
    decomposed = decompose(circuit, TIFR_SUPERCONDUCTING)
    names = [g.name for g in decomposed.gates]
    assert names == ["rz", "sx", "rz"]
    assert all(g.name in TIFR_SUPERCONDUCTING.basis_gates for g in decomposed.gates)


def test_x_is_native_tifr_passthrough():
    """x is native in TIFR_SUPERCONDUCTING basis, so no decomposition needed."""
    circuit = Circuit(num_qubits=1).x(0)
    decomposed = decompose(circuit, TIFR_SUPERCONDUCTING)
    assert len(decomposed.gates) == 1
    assert decomposed.gates[0].name == "x"


def test_x_decomposes_when_not_native():
    """x decomposes when only sx is available."""
    from qorch.transpiler import IndianGateSet
    sx_only = IndianGateSet(
        name="sx-only",
        description="test: only sx native",
        basis_gates=("sx",),
    )
    circuit = Circuit(num_qubits=1).x(0)
    decomposed = decompose(circuit, sx_only)
    names = [g.name for g in decomposed.gates]
    assert names == ["sx", "sx", "sx"]


def test_decompose_z_to_rz_pi():
    circuit = Circuit(num_qubits=1).z(0)
    decomposed = decompose(circuit, TIFR_SUPERCONDUCTING)
    assert decomposed.gates[0].name == "rz"
    assert abs(decomposed.gates[0].params[0] - 3.14159) < 0.01


def test_decompose_cx_to_rx_ms_rx():
    """cx → rx(π/2) ms(π/4) rx(-π/2) for ion-trap."""
    circuit = Circuit(num_qubits=2).cx(0, 1)
    decomposed = decompose(circuit, IIT_JODHPUR_ION_TRAP)
    names = [g.name for g in decomposed.gates]
    assert names == ["rx", "ms", "rx"]
    assert all(g.name in IIT_JODHPUR_ION_TRAP.basis_gates for g in decomposed.gates)


def test_decompose_bell_state_superconducting():
    """Full Bell state decomposition to TIFR native gates."""
    bell = Circuit(num_qubits=2).h(0).cx(0, 1)
    decomposed = decompose(bell, TIFR_SUPERCONDUCTING)
    assert all(g.name in TIFR_SUPERCONDUCTING.basis_gates for g in decomposed.gates)
    # Should have: rz, sx, rz (for h) + cx (native)
    assert len(decomposed.gates) == 4


def test_cx_already_native_passthrough():
    circuit = Circuit(num_qubits=2).cx(0, 1)
    decomposed = decompose(circuit, TIFR_SUPERCONDUCTING)
    assert len(decomposed.gates) == 1
    assert decomposed.gates[0].name == "cx"


def test_routing_linear_topology():
    """SWAP insertion for non-adjacent qubits on a 5-qubit line."""
    edges = ((0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2), (3, 4), (4, 3))
    cmap = CouplingMap(edges=edges)
    # CX between qubits 0 and 3 — not directly connected, needs routing
    circuit = Circuit(num_qubits=5).cx(0, 3)
    routed = route(circuit, cmap)
    # Should insert SWAPs to bring them adjacent
    assert len(routed.gates) > 1


def test_routing_all_to_all_no_swaps():
    """Empty coupling map = all-to-all, no routing needed."""
    cmap = CouplingMap(edges=())
    circuit = Circuit(num_qubits=5).cx(0, 3)
    routed = route(circuit, cmap)
    assert len(routed.gates) == 1  # unchanged


def test_routing_preserves_bell_state():
    """Routed Bell state should still be correlated (entangled) on noiseless sim.

    After routing cx(0,3) on a 0-1-2-3 line, SWAPs move qubit 0 to physical 3
    and qubit 3 to physical 2, so the final entangled outcomes are 0000 and 0011.
    """
    from qorch import LocalSimulator
    edges = ((0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2))
    cmap = CouplingMap(edges=edges)
    bell = Circuit(num_qubits=4).h(0).cx(0, 3)
    routed = route(bell, cmap)
    result = LocalSimulator(seed=42).run(routed, shots=2000)
    assert set(result.counts) <= {"0000", "0011"}
