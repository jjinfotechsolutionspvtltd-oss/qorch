"""Tests for gate decomposition + qubit routing."""

from qorch import Circuit
from qorch.transpiler import (
    decompose, route, CouplingMap, route_lookahead,
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


# ── SabreSWAP lookahead router ───────────────────────────────────────────


def test_lookahead_baseline_no_routing_needed():
    cmap = CouplingMap(edges=((0, 1),))
    c = Circuit(2).h(0).cx(0, 1)
    routed = route_lookahead(c, cmap)
    assert len(routed.gates) == len(c.gates)


def test_lookahead_empty_edges_noop():
    cmap = CouplingMap(edges=())
    c = Circuit(2).cx(0, 1)
    routed = route_lookahead(c, cmap)
    assert len(routed.gates) == len(c.gates)


def test_lookahead_empty_circuit_noop():
    cmap = CouplingMap(edges=((0, 1),))
    c = Circuit(2)
    routed = route_lookahead(c, cmap)
    assert len(routed.gates) == 0


def test_lookahead_routes_through_linear_chain():
    cmap = CouplingMap(edges=((0, 1), (1, 0), (1, 2), (2, 1)))
    c = Circuit(3).h(0).cx(0, 2)
    routed = route_lookahead(c, cmap)
    names = [g.name for g in routed.gates]
    assert "swap" in names
    assert "cx" in names


def test_lookahead_disconnected_graph_raises():
    cmap = CouplingMap(edges=((0, 1),))
    c = Circuit(3).cx(0, 2)
    try:
        route_lookahead(c, cmap)
        assert False
    except ValueError:
        pass


def test_lookahead_preserves_semantics_via_sim():
    from qorch import LocalSimulator
    cmap = CouplingMap(edges=((0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)))
    c = Circuit(4).h(0).cx(0, 3)
    routed = route_lookahead(c, cmap)
    result = LocalSimulator(seed=42).run(routed, shots=1000)
    assert set(result.counts) <= {"0000", "0011"}


def test_lookahead_fewer_swaps_than_greedy():
    cmap = CouplingMap(edges=((0, 1), (1, 0), (0, 2), (2, 0), (1, 3), (3, 1), (2, 3), (3, 2)))
    c = Circuit(4).cx(0, 3).cx(1, 2).cx(0, 3)
    greedy_r = route(c, cmap)
    lookahead_r = route_lookahead(c, cmap)
    greedy_swaps = sum(1 for g in greedy_r.gates if g.name == "swap")
    lookahead_swaps = sum(1 for g in lookahead_r.gates if g.name == "swap")
    assert lookahead_swaps <= greedy_swaps


# ── Clifford+T decomposition ──────────────────────────────────────────


class TestCliffordT:
    def test_native_gates_passthrough(self):
        from qorch.transpiler.decompose import decompose_to_clifford_t
        c = Circuit(2).h(0).cx(0, 1).t(0)
        result, tc, td = decompose_to_clifford_t(c)
        assert len(result.gates) == len(c.gates)

    def test_rz_pi_over_4_to_one_t(self):
        from qorch.transpiler.decompose import decompose_to_clifford_t
        c = Circuit(1).rz(0, 3.14159 / 4)
        result, tc, td = decompose_to_clifford_t(c)
        assert tc == 1

    def test_rz_pi_over_2_to_two_t(self):
        from qorch.transpiler.decompose import decompose_to_clifford_t
        c = Circuit(1).rz(0, 3.14159 / 2)
        result, tc, td = decompose_to_clifford_t(c)
        assert tc == 2

    def test_x_decomposes_to_h_zh(self):
        from qorch.transpiler.decompose import decompose_to_clifford_t
        c = Circuit(1).x(0)
        result, tc, td = decompose_to_clifford_t(c)
        assert all(g.name in ("h", "cx", "t") for g in result.gates)

    def test_all_gates_become_clifford_t(self):
        from qorch.transpiler.decompose import decompose_to_clifford_t
        native = {"h", "cx", "t"}
        c = Circuit(4).h(0).x(1).y(2).z(3).sx(0).cx(1, 2).swap(2, 3).rz(0, 0.5).rx(1, 0.3)
        result, tc, td = decompose_to_clifford_t(c)
        for g in result.gates:
            assert g.name in native, f"{g.name} not in Clifford+T"
        assert tc >= 0

    def test_t_depth_parallel(self):
        from qorch.transpiler.decompose import decompose_to_clifford_t
        c = Circuit(3).t(0).t(1).t(2)
        result, tc, td = decompose_to_clifford_t(c)
        assert td == 1
        assert tc == 3

    def test_t_depth_sequential(self):
        from qorch.transpiler.decompose import decompose_to_clifford_t
        c = Circuit(1).t(0).t(0).t(0)
        result, tc, td = decompose_to_clifford_t(c)
        assert td == 3
        assert tc == 3
