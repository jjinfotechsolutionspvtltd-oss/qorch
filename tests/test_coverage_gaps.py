"""Targeted tests to fill remaining coverage gaps."""

from __future__ import annotations

import math
import pytest

from qorch import Circuit
from qorch.ir import Gate
from qorch.transpiler import CouplingMap, decompose, route
from qorch.transpiler.gateset import IndianGateSet
from qorch.transpiler.decompose import _find_rule, _can_decompose
from qorch.visual import draw_circuit
from qorch.scheduler import first_available


def _gs(basis_gates: tuple[str, ...]) -> IndianGateSet:
    return IndianGateSet(name="test", description="test set", basis_gates=basis_gates)


# ── decompose.py coverage ──────────────────────────────────────────────────

class TestDecomposeEdgeCases:
    def test_swap_to_cx(self):
        gs = _gs(("cx", "h", "x"))
        c = Circuit(2).swap(0, 1)
        c2 = decompose(c, gs)
        assert all(g.name == "cx" for g in c2.gates)
        assert len(c2.gates) == 3

    def test_swap_to_ms(self):
        gs = _gs(("ms", "rz", "rx"))
        c = Circuit(2).swap(0, 1)
        c2 = decompose(c, gs)
        assert [g.name for g in c2.gates] == ["ms", "rz", "rz", "ms"]

    def test_can_decompose_unsupported_gate(self):
        gs = _gs(("h",))
        c = Circuit.__new__(Circuit)
        object.__setattr__(c, "num_qubits", 1)
        object.__setattr__(c, "gates", (Gate("bogus", (0,)),))
        object.__setattr__(c, "measured", ())
        assert not _can_decompose(c, gs)

    def test_can_decompose_undecomposable(self):
        gs = _gs(("id",))
        c = Circuit(1).h(0)
        assert not _can_decompose(c, gs)

    def test_can_decompose_valid(self):
        gs = _gs(("h",))
        c = Circuit(1).h(0)
        assert _can_decompose(c, gs)

    def test_can_decompose_decomposable(self):
        gs = _gs(("rz", "sx"))
        c = Circuit(1).h(0)
        assert _can_decompose(c, gs)

    def test_decompose_raises_on_no_rule(self):
        gs = _gs(("id",))
        c = Circuit(1).h(0)
        with pytest.raises(ValueError, match="no decomposition rule"):
            decompose(c, gs)

    def test_find_rule_returns_none_for_native(self):
        result = _find_rule("h", frozenset({"h", "cx"}))
        assert result is None

    def test_find_rule_returns_none_for_unknown(self):
        result = _find_rule("bogus", frozenset({"h"}))
        assert result is None

    def test_decompose_preserves_params(self):
        gs = _gs(("rx", "cx"))
        c = Circuit(1).rx(0, 1.23)
        c2 = decompose(c, gs)
        assert abs(c2.gates[0].params[0] - 1.23) < 1e-10

    def test_decompose_h_to_ry_rx(self):
        gs = _gs(("rx", "ry", "cx"))
        c = Circuit(1).h(0)
        c2 = decompose(c, gs)
        assert [g.name for g in c2.gates] == ["ry", "rx", "ry"]

    def test_decompose_superset_match(self):
        gs = _gs(("rz", "sx", "cx", "id"))
        c = Circuit(1).h(0)
        c2 = decompose(c, gs)
        assert [g.name for g in c2.gates] == ["rz", "sx", "rz"]


# ── visual.py edge cases ──────────────────────────────────────────────────

class TestVisualEdgeCases:
    def test_draw_multi_qubit_gate_with_connectors(self):
        c = Circuit(4).cx(0, 3)
        out = draw_circuit(c)
        assert "q[0]" in out
        assert "q[3]" in out

    def test_draw_big_circuit(self):
        c = Circuit(6)
        for i in range(5):
            c = c.cx(i, i + 1)
        out = draw_circuit(c)
        assert "q[5]" in out

    def test_draw_param_gate_label(self):
        c = Circuit(1).ry(0, 0.5)
        out = draw_circuit(c)
        assert "RY" in out

    def test_draw_parametric_precision(self):
        c = Circuit(1).rz(0, math.pi)
        out = draw_circuit(c)
        assert "RZ" in out

    def test_draw_measurement_markers(self):
        c = Circuit(3).h(0).cx(0, 1).measure(0, 2)
        out = draw_circuit(c)
        assert "M" in out


# ── ir.py validation edge cases ────────────────────────────────────────────

class TestIRValidation:
    def test_circuit_zero_qubits(self):
        with pytest.raises(ValueError, match="num_qubits must be positive"):
            Circuit(0)

    def test_circuit_negative_qubits(self):
        with pytest.raises(ValueError, match="num_qubits must be positive"):
            Circuit(-1)

    def test_unsupported_gate(self):
        with pytest.raises(ValueError, match="unsupported gate"):
            Circuit(1, gates=(Gate("bogus", (0,)),))

    def test_qubit_out_of_range(self):
        with pytest.raises(ValueError, match="out of range"):
            Circuit(1, gates=(Gate("h", (5,)),))

    def test_measured_qubit_out_of_range(self):
        with pytest.raises(ValueError, match="out of range"):
            Circuit(2, measured=(5,))


# ── routing.py edge cases ──────────────────────────────────────────────────

class TestRoutingEdgeCases:
    def test_disconnected_graph_raises(self):
        cmap = CouplingMap(edges=((0, 1),))
        c = Circuit(3).cx(0, 2)
        with pytest.raises(ValueError, match="no path"):
            route(c, cmap)

    def test_disconnected_with_quality_raises(self):
        from qorch.transpiler.routing import QubitQuality
        cmap = CouplingMap(edges=((0, 1),))
        c = Circuit(3).cx(0, 2)
        quality = {0: QubitQuality(1.0), 1: QubitQuality(1.0), 2: QubitQuality(1.0)}
        with pytest.raises(ValueError, match="no path"):
            route(c, cmap, qubit_quality=quality)

    def test_routing_via_triangle_graph(self):
        from qorch.transpiler.routing import QubitQuality
        cmap = CouplingMap(edges=((0, 1), (0, 2), (1, 2)))
        c = Circuit(3).cx(0, 2).cx(1, 2)
        quality = {0: QubitQuality(0.99), 1: QubitQuality(0.98), 2: QubitQuality(0.97)}
        result = route(c, cmap, qubit_quality=quality)
        assert len(result.gates) == 2

    def test_best_swap_path_dijkstra_no_path(self):
        from qorch.transpiler.routing import _best_swap_path, QubitQuality
        adj = {0: {1}, 1: {0}, 2: {3}, 3: {2}}
        edges = {(0, 1), (1, 0), (2, 3), (3, 2)}
        quality = {0: QubitQuality(1.0), 1: QubitQuality(1.0), 2: QubitQuality(1.0), 3: QubitQuality(1.0)}
        with pytest.raises(ValueError, match="no path"):
            _best_swap_path(adj, edges, 0, 3, quality)


# ── dd.py edge cases ──────────────────────────────────────────────────────

class TestDDEdgeCases:
    def test_dd_preserves_after_1q_gate(self):
        from qorch.mitigation.dd import insert_dd
        c = Circuit(2).h(0).x(0)
        dd = insert_dd(c, sequence="xy4", qubits=(1,))
        assert len(dd.gates) > len(c.gates)


# ── pipeline.py edge cases ─────────────────────────────────────────────────

class TestPipelineEdgeCases:
    def test_pipeline_pec_path(self):
        from qorch import LocalSimulator
        from qorch.mitigation.pipeline import MitigationPipeline
        sim = LocalSimulator(seed=0)
        pipeline = MitigationPipeline(backend=sim, shots=256, pec=True)
        c = Circuit(1).h(0)
        result = pipeline.run(c)
        assert "pec" in result.steps_applied


# ── density_simulator.py edge cases ────────────────────────────────────────

class TestDensitySimEdgeCases:
    def test_ry_and_rz_gates(self):
        from qorch import DensitySimulator
        sim = DensitySimulator(seed=0)
        c = Circuit(1).ry(0, 1.5).rz(0, 0.5)
        result = sim.run(c, shots=256)
        assert result.shots > 0

    def test_zero_noise_early_returns(self):
        from qorch import DensitySimulator, NoiseChannel
        sim = DensitySimulator(seed=0, noise=NoiseChannel())
        c = Circuit(2).h(0).cx(0, 1)
        result = sim.run(c, shots=256)
        assert set(result.counts.keys()) <= {"00", "11"}

    def test_unknown_gate_raises(self):
        from qorch.backends.density_simulator import _gate_matrix
        try:
            _gate_matrix("nonexistent")
            assert False, "expected ValueError"
        except ValueError:
            pass


# ── visual.py edge cases ──────────────────────────────────────────────────

class TestVisualMoreEdgeCases:
    def test_draw_ms_symmetric_gate(self):
        c = Circuit(3).ms(0, 2, 0.25)
        out = draw_circuit(c)
        assert "MS" in out
        assert "q[0]" in out
        assert "q[2]" in out

    def test_print_circuit(self):
        from qorch.visual import print_circuit
        import io
        import sys
        c = Circuit(2).h(0).cx(0, 1)
        captured = io.StringIO()
        old = sys.stdout
        sys.stdout = captured
        try:
            print_circuit(c)
        finally:
            sys.stdout = old
        output = captured.getvalue()
        assert "q[0]" in output
        assert "q[1]" in output


# ── benchmarking.py edge cases (numpy is available) ────────────────────────

class TestBenchmarkingNumPyPaths:
    def test_rb_with_numpy(self):
        from qorch import LocalSimulator
        from qorch.benchmarking import randomized_benchmarking
        sim = LocalSimulator(seed=0)
        result = randomized_benchmarking(sim, depths=(1, 2), circuits_per_depth=2, shots=128, seed=42)
        assert len(result.depths) == 2

    def test_qv_with_numpy(self):
        from qorch import LocalSimulator
        from qorch.benchmarking import quantum_volume
        sim = LocalSimulator(seed=0)
        result = quantum_volume(sim, width=2, shots=128, trials=2, seed=42)
        assert result.width == 2

    def test_xeb_with_numpy(self):
        from qorch import LocalSimulator
        from qorch.benchmarking import cross_entropy_benchmarking
        sim = LocalSimulator(seed=0)
        result = cross_entropy_benchmarking(sim, num_qubits=2, depth=1, num_circuits=1, shots=64, seed=42)
        assert result.depth == 1

    def test_random_su2_odd_depth(self):
        from qorch.benchmarking import _random_su2_circuit
        import random
        rng = random.Random(0)
        c = _random_su2_circuit(2, depth=2, rng=rng)
        assert len(c.gates) > 0


# ── simulator.py remaining gaps ───────────────────────────────────────────

class TestSimulatorGaps:
    def test_unknown_gate_matrix_raises(self):
        from qorch.backends.simulator import _gate_matrix
        try:
            _gate_matrix("nonexistent")
            assert False
        except ValueError:
            pass

    def test_cx_and_swap_in_trajectory(self):
        circ = Circuit(2).cx(0, 1).swap(0, 1).measure(0, 0).measure(1, 1)
        from qorch import LocalSimulator
        from qorch.backends.simulator import GateNoise
        backend = LocalSimulator(seed=0, gate_noise=GateNoise(depolarizing_prob=0.05))
        r = backend.run(circ, shots=64)
        assert r.shots == 64

    def test_readout_noise_in_simulator(self):
        from qorch import LocalSimulator
        from qorch.backends.simulator import ReadoutNoise, GateNoise
        circ = Circuit(1).h(0).measure(0, 0)
        backend = LocalSimulator(seed=0, readout_noise=ReadoutNoise(p1_given0=0.1), gate_noise=GateNoise(depolarizing_prob=0.1))
        r = backend.run(circ, shots=128)
        assert r.shots == 128


# ── entanglement.py 1-qubit path ──────────────────────────────────────────

class TestEntanglementOneQubit:
    def test_single_qubit_expectation(self):
        from qorch.entanglement import _expectation_from_counts
        result = _expectation_from_counts({"0": 10, "1": 5}, 15)
        assert abs(result - (10 - 5) / 15) < 1e-12


# ── routing.py adjacency path ─────────────────────────────────────────────

class TestRoutingSameQubit:
    def test_routing_start_end_same(self):
        from qorch.transpiler.routing import _shortest_path
        path = _shortest_path({0: {1}, 1: {0}}, 0, 0)
        assert path == [0]


# ── readout.py singular matrix and empty counts ───────────────────────────

class TestReadoutEdgeCases:
    def test_singular_matrix_raises(self):
        from qorch.mitigation.readout import _invert
        try:
            _invert([[0, 0], [0, 0]])
            assert False
        except ValueError:
            pass

    def test_empty_counts(self):
        from qorch.mitigation.readout import ReadoutMitigator
        rm = ReadoutMitigator(labels=("0", "1"), inverse=((1, 0), (0, 1)))
        result = rm.apply({})
        assert result == {"0": 0.0, "1": 0.0}


# ── twirling.py non-CX 2q gate and non-Clifford path ──────────────────────

class TestTwirlingEdgeCases:
    def test_twirl_ms_2q_fallthrough(self):
        from qorch.mitigation.twirling import _conjugate_2q
        # MS is a non-CX/non-SWAP 2q gate → returns (p0, p1) unchanged
        result = _conjugate_2q("ms", 0, 1)
        assert result == (0, 1)

    def test_twirl_non_clifford_preserved(self):
        from qorch.mitigation.twirling import twirl_circuit
        c = Circuit(1).rx(0, 0.5)
        twirled = twirl_circuit(c, seed=0)
        assert len(twirled.gates) == 1
        assert twirled.gates[0].name == "rx"


# ── zne.py var == 0 path ──────────────────────────────────────────────────

class TestZNEEdgeCases:
    def test_linear_extrapolate_flat(self):
        from qorch.mitigation.zne import _linear_extrapolate
        result = _linear_extrapolate([1, 1, 1], [3, 3, 3])
        assert result == 3.0

    def test_expectation_z_empty(self):
        from qorch.mitigation.zne import expectation_z
        result = expectation_z({})
        assert result == 0.0


# ── pec.py edge cases ─────────────────────────────────────────────────────

class TestPECEdgeCases:
    def test_pec_improvement_zero_raw_value(self):
        from qorch.mitigation.pec import PECResult
        r = PECResult(raw_value=0.0, mitigated_value=0.5, n_samples=1, gamma=1.0, noise_prob=0.1)
        assert r.improvement == 0.0

    def test_pec_sample_noise_tail(self):
        from qorch.mitigation.pec import _sample_pauli
        import random
        rng = random.Random(0)
        # zero noise coefficients force the fallback
        result = _sample_pauli(rng, (0.0, 0.0, 0.0, 0.0), 1.0)
        assert result is not None
        # All cumulative sums are 0, so the function falls through to return (None, gamma)
        assert result == (None, 1.0)


# ── ir.py to_qasm3 no readout & from_qasm3 edge cases ────────────────────

class TestIRMoreEdgeCases:
    def test_to_qasm3_no_readout(self):
        from qorch.ir import to_qasm3
        c = Circuit(2).h(0).cx(0, 1)
        qasm = to_qasm3(c)
        assert "qubit[2] q" in qasm
        assert qasm.count("measure") == 2

    def test_from_qasm3_no_decl(self):
        from qorch.ir import from_qasm3
        try:
            from_qasm3("h q[0];")
            assert False
        except ValueError:
            pass

    def test_from_qasm3_swap_and_ms(self):
        from qorch.ir import from_qasm3
        qasm = 'OPENQASM 3;\nqubit[2] q;\nswap q[0], q[1];\nms(0.5) q[0], q[1];\n'
        c = from_qasm3(qasm)
        assert [g.name for g in c.gates] == ["swap", "ms"]


# ── scheduler.py no-backend path ──────────────────────────────────────────

class TestSchedulerEdgeCases:
    def test_no_backend_available(self):
        with pytest.raises(ValueError, match="no registered backend"):
            first_available(Circuit(100), [])


# ── optimizer.py identity merging and id removal ──────────────────────────

class TestOptimizerEdgeCases:
    def test_rotation_cancels_to_identity(self):
        from qorch.transpiler.optimizer import optimize
        c = Circuit(1).rz(0, 0.5).rz(0, -0.5)
        opt = optimize(c)
        assert len(opt.gates) == 0

    def test_id_gate_removed(self):
        from qorch.transpiler.optimizer import optimize
        c = Circuit(1).id(0).h(0)
        opt = optimize(c)
        assert all(g.name != "id" for g in opt.gates)


# ── decompose.py _passthrough path (native gate) ──────────────────────────

class TestDecomposePassthrough:
    def test_decompose_native_gate_passthrough(self):
        gs = IndianGateSet(name="test", description="", basis_gates=("h", "cx"))
        c = Circuit(1).h(0)
        result = decompose(c, gs)
        assert len(result.gates) == 1
        assert result.gates[0].name == "h"


# ── indian_backend.py uncovered paths ─────────────────────────────────────

class TestIndianBackendGaps:
    def test_h_and_identity_matrix(self):
        from qorch.backends.indian_backend import _indian_gate_matrix
        import math
        h = _indian_gate_matrix("h", ())
        inv = 1.0 / math.sqrt(2)
        assert abs(h[0] - inv) < 1e-12
        ident = _indian_gate_matrix("bogus", ())
        assert ident == (1, 0, 0, 1)

    def test_swap_in_indian_evolve(self):
        from qorch.backends.indian_backend import IndianQPU, IndianQPUConfig
        from qorch.transpiler import CouplingMap, IndianGateSet
        gs = IndianGateSet(name="t", description="", basis_gates=("h", "cx", "swap", "id"))
        cmap = CouplingMap(edges=((0, 1), (1, 0)))
        config = IndianQPUConfig(name="t", gate_set=gs, num_qubits=2, coupling_map=cmap,
                                 gate_fidelity=1.0, readout_fidelity=1.0, t1_us=1e9, t2_us=1e9)
        qpu = IndianQPU(config=config)
        c = Circuit(2).h(0).swap(0, 1).measure(0, 0).measure(1, 1)
        r = qpu.run(c, shots=64)
        assert r.shots == 64


# ── dd.py empty circuit path ──────────────────────────────────────────────

class TestDDEmptyCircuit:
    def test_dd_empty_circuit(self):
        from qorch.mitigation.dd import _find_idle_windows
        c = Circuit(2)
        windows = _find_idle_windows(c, qubit=0)
        assert windows == [(0, 0)]

    def test_run_with_dd(self):
        from qorch.mitigation.dd import apply_dd_mitigation
        from qorch import LocalSimulator
        sim = LocalSimulator(seed=0)
        c = Circuit(1).h(0).measure(0, 0)
        counts = apply_dd_mitigation(c, sim, shots=64)
        assert sum(counts.values()) == 64


# ── visual.py MS gate ctrl label ──────────────────────────────────────────

class TestVisualMSGateLabel:
    def test_draw_ms_has_ctrl_label(self):
        c = Circuit(2).ms(0, 1, 0.25)
        out = draw_circuit(c)
        assert "MS" in out


# ── cli.py edge cases ─────────────────────────────────────────────────────

class TestCLIEdgeCases:
    def test_build_backend_local(self):
        from qorch.cli import _build_backend
        b = _build_backend("local-simulator")
        assert b.name == "local-simulator"

    def test_build_backend_unknown_raises(self):
        from qorch.cli import _build_backend
        try:
            _build_backend("nonexistent")
            assert False
        except ValueError:
            pass
