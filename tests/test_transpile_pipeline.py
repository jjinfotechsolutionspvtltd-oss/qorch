"""Tests for the full transpile pipeline (decompose + route + optimize + DD)."""

from __future__ import annotations

import pytest

from qorch import Circuit, LocalSimulator, transpile
from qorch.ir import Gate
from qorch.transpiler import (
    CouplingMap,
    DRDO_MIRAI,
    IIT_JODHPUR_ION_TRAP,
    TIFR_SUPERCONDUCTING,
)
from qorch.transpiler.routing import QubitQuality

_TARGETS = (TIFR_SUPERCONDUCTING, DRDO_MIRAI, IIT_JODHPUR_ION_TRAP)


def _routing_map(target) -> CouplingMap:
    """A topology to route the target on.

    Ion traps declare all-to-all connectivity, but a caller may still hand one an
    explicit chain — and a SWAP inserted there has to lower to native MS just
    like it does on a superconducting grid, so exercise that path too.
    """
    if target.coupling_map:
        return CouplingMap(target.coupling_map)
    return CouplingMap(((0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)))


def _assert_executable(circuit: Circuit, target, coupling_map: CouplingMap | None = None) -> None:
    """Every gate is native, and every 2-qubit gate sits on a real directed edge."""
    basis = set(target.basis_gates)
    offenders = sorted({g.name for g in circuit.gates if isinstance(g, Gate) and g.name not in basis})
    assert not offenders, f"non-native gates {offenders} for {target.name} (basis {target.basis_gates})"

    if coupling_map is not None:
        edges = set(coupling_map.edges)
        off_edge = [g for g in circuit.gates if len(g.qubits) == 2 and g.qubits not in edges]
        assert not off_edge, f"2-qubit gates off the coupling map: {off_edge}"


class TestTranspilePipeline:
    def test_basic_decompose_only(self):
        c = Circuit(2).h(0).cx(0, 1)
        c2 = transpile(c, target=IIT_JODHPUR_ION_TRAP)
        assert c2.num_qubits == 2

    def test_decompose_and_route(self):
        """Routing happens, and the SWAPs it costs are paid in native CX gates.

        TIFR has no native SWAP, so a surviving one would be unexecutable; the
        routing cost has to show up as extra CX instead.
        """
        cmap = CouplingMap(edges=((0, 1), (1, 2)))
        c = Circuit(3).h(0).cx(0, 2).h(2)
        c2 = transpile(c, target=TIFR_SUPERCONDUCTING, coupling_map=cmap)
        unrouted = transpile(c, target=TIFR_SUPERCONDUCTING, coupling_map=None)

        assert not [g for g in c2.gates if g.name == "swap"]
        cx = sum(1 for g in c2.gates if g.name == "cx")
        assert cx > sum(1 for g in unrouted.gates if g.name == "cx")

    def test_decompose_route_dd(self):
        """DD pulses survive the optimizer and come out in the native basis.

        xy4 is XYXY = I, so an optimizer running after DD insertion would cancel
        the whole sequence away — the pulses only do their job if they are still
        physically there.
        """
        cmap = CouplingMap(edges=((0, 1), (1, 2)))
        c = Circuit(3).h(0).cx(0, 2).h(2)
        c2 = transpile(c, target=TIFR_SUPERCONDUCTING, coupling_map=cmap, dd_sequence="xy4")

        assert sum(1 for g in c2.gates if g.name == "x") > 0
        assert {g.name for g in c2.gates} <= set(TIFR_SUPERCONDUCTING.basis_gates)

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


class TestTargetBasisConformance:
    """`transpile` promises an executable circuit — hold it to that literally.

    Routing runs *after* the first decomposition pass, so the SWAPs it inserts
    used to reach the output untouched: transpiling for TIFR (basis cx/sx/rz/x)
    produced a circuit containing ``swap``, which that hardware cannot run.
    """

    @pytest.mark.parametrize("target", _TARGETS, ids=lambda t: t.name)
    def test_every_gate_is_in_the_target_basis(self, target):
        cmap = _routing_map(target)
        c = Circuit(4).h(0).cx(0, 3).x(2).swap(1, 2).cx(1, 3).measure(0, 1, 2, 3)
        _assert_executable(transpile(c, target, coupling_map=cmap), target, cmap)

    @pytest.mark.parametrize("target", _TARGETS, ids=lambda t: t.name)
    def test_every_gate_is_native_under_the_lookahead_router(self, target):
        cmap = _routing_map(target)
        c = Circuit(4).h(0).cx(0, 3).cx(2, 0).h(3).measure(0, 1, 2, 3)
        out = transpile(c, target, coupling_map=cmap, use_lookahead=True)
        _assert_executable(out, target, cmap)

    @pytest.mark.parametrize("target", _TARGETS, ids=lambda t: t.name)
    @pytest.mark.parametrize("sequence", ("xy4", "xy8", "cpmg", "hahn"))
    def test_dd_pulses_are_lowered_to_the_target_basis(self, target, sequence):
        """DD emits literal x/y pulses after decomposition — they need lowering too."""
        cmap = _routing_map(target)
        c = Circuit(4).h(0).cx(0, 3).h(3).measure(0, 1, 2, 3)
        out = transpile(c, target, coupling_map=cmap, dd_sequence=sequence)
        _assert_executable(out, target, cmap)

    @pytest.mark.parametrize("target", _TARGETS, ids=lambda t: t.name)
    def test_native_without_the_optimizer_too(self, target):
        """Basis conformance is the lowering's job, not a side effect of optimization."""
        cmap = _routing_map(target)
        c = Circuit(4).h(0).cx(0, 3).measure(0, 1, 2, 3)
        out = transpile(c, target, coupling_map=cmap, do_optimize=False)
        _assert_executable(out, target, cmap)

    def test_no_swap_survives_for_a_target_without_one(self):
        """The original report, pinned verbatim."""
        cmap = CouplingMap(TIFR_SUPERCONDUCTING.coupling_map)
        c = Circuit(3).h(0).cx(0, 2).measure(0, 1, 2)
        out = transpile(c, TIFR_SUPERCONDUCTING, coupling_map=cmap)

        assert "swap" not in TIFR_SUPERCONDUCTING.basis_gates
        assert sorted({g.name for g in out.gates}) == ["cx", "rz", "sx"]


class TestDirectedCouplingMap:
    """A one-way coupling map is the case SWAP lowering can silently violate.

    ``swap → cx(a,b) cx(b,a) cx(a,b)`` needs both directions. When only ``(a,b)``
    is an edge, the middle CX has to be flipped by Hadamard conjugation — and
    those Hadamards then need lowering themselves.
    """

    # Strictly one-way: no reverse edges anywhere.
    _ONE_WAY = CouplingMap(((0, 1), (1, 2), (2, 3)))

    def test_routed_circuit_respects_edge_direction(self):
        c = Circuit(4).h(0).cx(0, 3).cx(1, 2).measure(0, 1, 2, 3)
        out = transpile(c, TIFR_SUPERCONDUCTING, coupling_map=self._ONE_WAY)
        _assert_executable(out, TIFR_SUPERCONDUCTING, self._ONE_WAY)

    def test_direction_fixing_preserves_semantics(self):
        """Flipping a CX must be an identity, not a rewiring."""
        c = Circuit(4).h(0).cx(0, 3).cx(1, 2).measure(0, 1, 2, 3)
        out = transpile(c, TIFR_SUPERCONDUCTING, coupling_map=self._ONE_WAY)

        assert (
            LocalSimulator(seed=5).run(out, shots=2000).counts
            == LocalSimulator(seed=5).run(c, shots=2000).counts
        )

    def test_lookahead_router_respects_edge_direction(self):
        c = Circuit(4).h(0).cx(0, 3).h(3).measure(0, 1, 2, 3)
        out = transpile(c, TIFR_SUPERCONDUCTING, coupling_map=self._ONE_WAY, use_lookahead=True)
        _assert_executable(out, TIFR_SUPERCONDUCTING, self._ONE_WAY)
