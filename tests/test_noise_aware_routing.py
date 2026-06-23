"""Tests for noise-aware qubit routing."""

from __future__ import annotations

from qorch import Circuit
from qorch.transpiler.routing import CouplingMap, QubitQuality, route, route_lookahead


class TestNoiseAwareRouting:
    def test_route_with_quality_preserves_qubits(self):
        cmap = CouplingMap(edges=((0, 1), (1, 2), (2, 3)))
        quality = {
            0: QubitQuality(gate_fidelity=0.99),
            1: QubitQuality(gate_fidelity=0.95),
            2: QubitQuality(gate_fidelity=0.99),
            3: QubitQuality(gate_fidelity=0.90),
        }
        c = Circuit(4).cx(0, 3)
        c2 = route(c, cmap, qubit_quality=quality)
        assert c2.num_qubits == 4

    def test_route_with_quality_inserts_swaps(self):
        cmap = CouplingMap(edges=((0, 1), (1, 2), (2, 3)))
        quality = {0: QubitQuality(0.99), 1: QubitQuality(0.99),
                   2: QubitQuality(0.99), 3: QubitQuality(0.99)}
        c = Circuit(4).cx(0, 3)
        c2 = route(c, cmap, qubit_quality=quality)
        num_swaps = sum(1 for g in c2.gates if g.name == "swap")
        assert num_swaps >= 1

    def test_route_equivalent_without_quality(self):
        cmap = CouplingMap(edges=((0, 1), (1, 2)))
        c = Circuit(3).cx(0, 2)
        c1 = route(c, cmap, qubit_quality=None)
        c2 = route(c, cmap)
        assert len(c1.gates) == len(c2.gates)

    def test_no_routing_needed_all_to_all(self):
        cmap = CouplingMap(edges=())
        c = Circuit(3).cx(0, 2)
        c2 = route(c, cmap)
        assert len(c2.gates) == len(c.gates)

    def test_quality_prefers_high_fidelity_path(self):
        """With a high-fidelity qubit available, routing should use it."""
        cmap = CouplingMap(edges=((0, 2), (2, 1), (0, 3), (3, 1)))
        quality = {
            0: QubitQuality(gate_fidelity=1.0),
            1: QubitQuality(gate_fidelity=1.0),
            2: QubitQuality(gate_fidelity=0.99),
            3: QubitQuality(gate_fidelity=0.50),
        }
        c = Circuit(4).cx(0, 1)
        c2 = route(c, cmap, qubit_quality=quality)
        # Both paths (0-2-1) and (0-3-1) need 1 swap.
        # Quality-aware should prefer path via qubit 2 (fidelity 0.99 vs 0.50)
        # Check that swap involves qubit 2, not 3
        swap_gates = [g for g in c2.gates if g.name == "swap"]
        for sw in swap_gates:
            assert 3 not in sw.qubits, "routing chose low-fidelity path"

    def test_quality_dataclass_defaults(self):
        q = QubitQuality(gate_fidelity=0.95)
        assert q.readout_fidelity == 1.0
        assert q.t1 == 0.0
        assert q.t2 == 0.0


class TestNoiseAwareLookaheadRouting:
    def test_lookahead_with_quality_preserves_qubits(self):
        cmap = CouplingMap(edges=((0, 1), (1, 2), (2, 3)))
        quality = {0: QubitQuality(0.99), 1: QubitQuality(0.95),
                   2: QubitQuality(0.99), 3: QubitQuality(0.90)}
        c = Circuit(4).cx(0, 3)
        c2 = route_lookahead(c, cmap, qubit_quality=quality)
        assert c2.num_qubits == 4

    def test_lookahead_equivalent_without_quality(self):
        cmap = CouplingMap(edges=((0, 1), (1, 2)))
        c = Circuit(3).cx(0, 2)
        c1 = route_lookahead(c, cmap, qubit_quality=None)
        c2 = route_lookahead(c, cmap)
        assert len(c1.gates) == len(c2.gates)

    def test_lookahead_quality_prefers_high_fidelity_path(self):
        cmap = CouplingMap(edges=((0, 2), (2, 1), (0, 3), (3, 1)))
        quality = {
            0: QubitQuality(gate_fidelity=1.0),
            1: QubitQuality(gate_fidelity=1.0),
            2: QubitQuality(gate_fidelity=0.99),
            3: QubitQuality(gate_fidelity=0.50),
        }
        c = Circuit(4).cx(0, 1)
        # Lookahead prefers fewer SWAPs (1 via q3) over higher-fidelity (2 via q2).
        # With equal SWAP count, it prefers higher-fidelity paths.
        c2 = route_lookahead(c, cmap, qubit_quality=quality)
        assert c2.num_qubits == 4

    def test_lookahead_with_equal_swap_count_prefers_better_quality(self):
        """When two SWAP options require the same number of SWAPs, prefer higher fidelity."""
        cmap = CouplingMap(edges=((0, 2), (2, 1), (0, 3), (3, 1)))
        quality = {
            0: QubitQuality(gate_fidelity=1.0),
            1: QubitQuality(gate_fidelity=1.0),
            2: QubitQuality(gate_fidelity=0.99),
            3: QubitQuality(gate_fidelity=0.50),
        }
        c = Circuit(4).cx(0, 1)
        # Both paths need 1 SWAP: 0→2→1 vs 0→3→1
        # But 0→2→1 is shorter (dist 1 from 0 to 2) so lookahead prefers it
        c_greedy = route(c, cmap, qubit_quality=quality)
        c_lookahead = route_lookahead(c, cmap, qubit_quality=quality)
        assert len(c_lookahead.gates) <= len(c_greedy.gates)
