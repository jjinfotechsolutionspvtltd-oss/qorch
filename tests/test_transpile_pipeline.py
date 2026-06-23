"""Tests for the full transpile pipeline (decompose + route + DD + optimize)."""

from __future__ import annotations

from qorch import Circuit, transpile
from qorch.transpiler import (
    CouplingMap,
    DRDO_MIRAI,
    IIT_JODHPUR_ION_TRAP,
    TIFR_SUPERCONDUCTING,
)
from qorch.transpiler.routing import QubitQuality


class TestTranspilePipeline:
    def test_basic_decompose_only(self):
        c = Circuit(2).h(0).cx(0, 1)
        c2 = transpile(c, target=IIT_JODHPUR_ION_TRAP)
        assert c2.num_qubits == 2

    def test_decompose_and_route(self):
        cmap = CouplingMap(edges=((0, 1), (1, 2)))
        c = Circuit(3).h(0).cx(0, 2).h(2)
        c2 = transpile(c, target=TIFR_SUPERCONDUCTING, coupling_map=cmap)
        num_swaps = sum(1 for g in c2.gates if g.name == "swap")
        assert num_swaps > 0

    def test_decompose_route_dd(self):
        cmap = CouplingMap(edges=((0, 1), (1, 2)))
        c = Circuit(3).h(0).cx(0, 2).h(2)
        c2 = transpile(c, target=TIFR_SUPERCONDUCTING, coupling_map=cmap, dd_sequence="xy4")
        num_x = sum(1 for g in c2.gates if g.name == "x")
        num_y = sum(1 for g in c2.gates if g.name == "y")
        assert num_x > 0 or num_y > 0

    def test_full_pipeline_no_crash(self):
        cmap = CouplingMap(edges=((0, 1), (1, 2)))
        quality = {
            0: QubitQuality(gate_fidelity=0.99),
            1: QubitQuality(gate_fidelity=0.98),
            2: QubitQuality(gate_fidelity=0.97),
        }
        c = Circuit(3).h(0).cx(0, 1).cx(1, 2)
        c2 = transpile(c, target=IIT_JODHPUR_ION_TRAP, coupling_map=cmap,
                       qubit_quality=quality, dd_sequence="xy4", do_optimize=True)
        assert c2.num_qubits == 3
        assert len(c2.gates) > 0

    def test_drdo_mirai_pipeline(self):
        cmap = CouplingMap(edges=((0, 1), (1, 2), (2, 3), (3, 0)))
        c = Circuit(4).h(0).cx(0, 2).cx(1, 3).h(3)
        c2 = transpile(c, target=DRDO_MIRAI, coupling_map=cmap, dd_sequence="hahn")
        assert c2.num_qubits == 4

    def test_no_route_when_all_to_all(self):
        c = Circuit(3).h(0).cx(0, 1)
        c2 = transpile(c, target=TIFR_SUPERCONDUCTING, coupling_map=None)
        num_swaps = sum(1 for g in c2.gates if g.name == "swap")
        assert num_swaps == 0

    def test_optimize_flag(self):
        c = Circuit(2).x(0).x(0).h(0)
        c_opt = transpile(c, target=DRDO_MIRAI, do_optimize=True)
        c_noopt = transpile(c, target=DRDO_MIRAI, do_optimize=False)
        assert len(c_opt.gates) < len(c_noopt.gates)
